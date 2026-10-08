"""Offline contract tests. Toy causal backbone validates wiring, not LLM quality."""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from . import LIMSSRAdapter, MaskAwareDualPathAggregation, token_metric_loss, limssr_loss


class ToyTokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.ones(1, 3, dtype=torch.long)}


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 16)
        self.layer = nn.Linear(16, 16)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, **kwargs):
        # Prefix accumulation makes missing/fusion slots depend on preceding context.
        count = torch.arange(1, inputs_embeds.shape[1] + 1,
                             device=inputs_embeds.device)[None, :, None]
        hidden = self.layer(inputs_embeds.cumsum(1) / count)
        return SimpleNamespace(last_hidden_state=hidden)


class TransferTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.model = LIMSSRAdapter({'visual': 5, 'audio': 7, 'flow': 9},
                                   ToyBackbone(), ToyTokenizer(), dropout=0)
        self.features = {'visual': torch.randn(4, 3, 5),
                         'audio': torch.randn(4, 3, 7), 'flow': torch.randn(4, 3, 9)}

    def test_mixed_masks_match_separate_samples_and_no_missing_leakage(self):
        self.model.eval()
        mask = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]])
        output = self.model(self.features, mask)
        single = [self.model({k: v[i:i+1] for k, v in self.features.items()}, mask[i])
                  for i in range(4)]
        torch.testing.assert_close(output['prediction'], torch.cat([o['prediction'] for o in single]))
        poisoned = {k: v.clone() for k, v in self.features.items()}
        for j, name in enumerate(self.features):
            poisoned[name][mask[:, j] == 0] = float('nan')
        other = self.model(poisoned, mask)
        torch.testing.assert_close(output['prediction'], other['prediction'])

    def test_order_and_none_for_absent_modality(self):
        features = dict(self.features, audio=None)
        packed = self.model.pcmi(features, [1, 0, 1], ToyTokenizer(),
                                 self.model.backbone.get_input_embeddings())
        spans = packed['modal_spans']
        self.assertLess(spans[0][0], spans[2][0])
        self.assertLess(spans[2][0], spans[1][0])
        self.assertGreater(packed['fusion_start'], spans[1][1])
        self.assertEqual(self.model(features, [1, 0, 1])['prediction'].shape, (4, 1))

    def test_gradients_reach_all_components(self):
        output = self.model(self.features, [1, 0, 1])
        losses = limssr_loss(output, torch.rand(4))
        losses['total'].backward()
        for value in (self.model.pcmi.missing, self.model.pcmi.fusion_slots,
                      self.model.pcmi.projections[0].conv1.weight,
                      self.model.backbone.layer.weight, self.model.lmrf.role_weights,
                      self.model.mda.mix_parameter):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        self.assertIsNone(self.model.pcmi.projections[1].conv1.weight.grad)

    def test_invalid_contracts(self):
        for mask in ([0, 0, 0], [1, 2, 0], [1, 0]):
            with self.assertRaises(ValueError):
                self.model(self.features, mask)
        with self.assertRaises(ValueError):
            self.model(dict(self.features, audio=None), [1, 1, 0])
        with self.assertRaises(ValueError):
            limssr_loss(self.model(self.features, [1, 0, 1]), torch.rand(4, 2))

    def test_all_nonempty_combinations(self):
        self.model.eval()
        for bits in range(1, 8):
            output = self.model(self.features, [(bits >> j) & 1 for j in range(3)])
            self.assertTrue(torch.isfinite(output['prediction']).all())
            self.assertEqual(output['fusion_tokens'].shape, (4, 4, 16))

    def test_loss_matches_official_triplet_reference(self):
        tokens = torch.randn(3, 4, 16, requires_grad=True)
        flat = torch.nn.functional.normalize(tokens.reshape(-1, 16), dim=-1)
        dist = 1 - flat @ flat.T
        terms = []
        for i in range(12):
            positives = [j for j in range(12) if j != i and j % 4 == i % 4]
            negatives = [j for j in range(12) if j % 4 != i % 4]
            terms.append((dist[i, positives].max() - dist[i, negatives].min() + 1).relu())
        torch.testing.assert_close(token_metric_loss(tokens), torch.stack(terms).mean())
        constant = token_metric_loss(tokens[:, :2], mode='paper_eq17')
        torch.testing.assert_close(constant, torch.tensor(2.0))
        constant.backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_single_sample_and_single_token_metric(self):
        for shape in ((1, 4, 16), (3, 1, 16)):
            tokens = torch.randn(*shape, requires_grad=True)
            loss = token_metric_loss(tokens)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_arbitrary_modalities_and_unbounded_head(self):
        model = LIMSSRAdapter({'skeleton': 6, 'depth': 8}, ToyBackbone(),
                              ToyTokenizer(), out_dim=3, bounded_scores=False,
                              constrain_mix=True, dropout=0)
        output = model({'skeleton': torch.randn(2, 2, 6), 'depth': None}, [1, 0])
        self.assertEqual(output['prediction'].shape, (2, 3))
        module = MaskAwareDualPathAggregation(16, 2, dropout=0)
        result = module(torch.randn(2, 16), torch.randn(2, 2, 16), [[1, 0], [0, 1]])
        self.assertEqual(result['main_feature'].shape, (2, 16))


if __name__ == '__main__':
    unittest.main()
