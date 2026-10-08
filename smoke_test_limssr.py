"""Real tiny HF models, random weights, CPU smoke; external metrics skipped."""
import argparse
import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from PIL import Image
import pytorch_lightning as pl
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (PreTrainedTokenizerFast, BertConfig, BertModel,
                          Dinov2Config, Dinov2Model, ViTImageProcessor,
                          GPT2Config, Qwen2Config, Qwen2Model)
from models.model_github import TrainLanguageModel, TrainLanguageModelOneSample
from models.limssr_bridge import clinical_context_present


def prepare_assets(root):
    root.mkdir(parents=True, exist_ok=True)
    words = ['[UNK]', '[BOS]', '[EOS]', '[PAD]', '[SEP]', '[CLS]', '[INDICATION]',
             '[HISTORY]', '[FINDINGS]', 'normal', 'lungs', 'clear', 'heart', 'size',
             'cough', 'fever', '.', 'no', 'effusion', 'history', 'patient']
    backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token='[BOS]',
                                       eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]',
                                       sep_token='[SEP]', cls_token='[CLS]')
    for name in ('text', 'vision', 'decoder', 'reasoner'):
        (root / name).mkdir(exist_ok=True)
    tokenizer.save_pretrained(root / 'decoder')
    tokenizer.save_pretrained(root / 'reasoner')
    BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2,
                         num_attention_heads=4, intermediate_size=64)).save_pretrained(root / 'text')
    Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                             image_size=32, patch_size=8)).save_pretrained(root / 'vision')
    ViTImageProcessor(size={'height': 32, 'width': 32}).save_pretrained(root / 'vision')
    GPT2Config(vocab_size=len(tokenizer), n_embd=32, n_head=4, n_layer=2,
               n_positions=64, bos_token_id=tokenizer.bos_token_id,
               eos_token_id=tokenizer.eos_token_id).save_pretrained(root / 'decoder')
    Qwen2Model(Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=4,
                           num_key_value_heads=2, max_position_embeddings=512)).save_pretrained(root / 'reasoner')
    (root / 'views.json').write_text(json.dumps({'AP': 0, 'PA': 1, 'unk': 2}), encoding='utf-8')
    Image.new('RGB', (32, 32), color=(100, 120, 140)).save(root / 'current.png')
    Image.new('RGB', (32, 32), color=(140, 100, 120)).save(root / 'prior.png')
    examples = []
    for i in range(4):
        examples.append({'subject_id': i, 'study_id': i, 'id': str(i),
                         'findings': 'normal lungs clear .', 'findings_factual_serialization': ['normal lungs'],
                         'indication_pure': 'cough' if i % 2 == 0 else '', 'history_pure': '',
                         'anchor_scan': {'image_path': ['current.png'], 'view_position': ['AP']},
                         'prior_study': {'latest_study': {'image_path': 'prior.png', 'view_position': 'PA'}}
                                        if i % 2 == 0 else None})
    (root / 'annotation.json').write_text(json.dumps({k: examples for k in ('train', 'val', 'test')}), encoding='utf-8')
    args = {'hidden_size': 32, 'text_encoder_num_blocks': 2,
            'temporal_fusion_num_blocks': 1, 'perceiver_num_blocks': 1,
            'num_heads': 4, 'num_latents': 4, 'rad_dino_path': str(root / 'vision'),
            'cxr_bert_path': str(root / 'text'), 'distilgpt2_path': str(root / 'decoder'),
            'view_position_dict': str(root / 'views.json'), 'limssr_model_path': str(root / 'reasoner'),
            'use_limssr': True, 'limssr_tokens': 4, 'limssr_fusion_tokens': 4,
            'limssr_lora_rank': 2, 'encoder_max_length': 24, 'max_length': 10,
            'num_beams': 2, 'pt_lr': 1e-4, 'ft_lr': 1e-4, 'limssr_lr': 2e-4,
            'patience': 2, 'monitor_mode': 'min', 'monitor_metric': 'train_loss',
            'project_name': str(root / 'outputs'), 'batch_size': 2, 'num_workers': 0,
            'ann_path': str(root / 'annotation.json'), 'images_dir': str(root), 'print_step': 1}
    return args, tokenizer


class QuietLogger:
    def info(self, text):
        pass


class IntegrationTests(unittest.TestCase):
    args = tokenizer = root = None
    evidence = {}

    def setUp(self):
        torch.manual_seed(42)

    def build(self, enabled=True):
        args = copy.deepcopy(self.args)
        args['use_limssr'] = enabled
        # Only skip evaluators requiring external checkpoints, not model/training code.
        with patch.object(TrainLanguageModel, '_setup_report_metrics', return_value=None):
            return TrainLanguageModel(args, self.tokenizer, QuietLogger())

    def batch(self):
        return {'current_study': {'image': torch.randn(2, 3, 32, 32), 'view_position': ['AP', 'AP']},
                'prior_study': {'image': torch.randn(1, 3, 32, 32), 'view_position': ['PA'],
                                'pri_idx': [0], 'no_pri_idx': [1]},
                'clinical_context': ['[INDICATION] cough [SEP] [HISTORY]', '[INDICATION] [SEP] [HISTORY]'],
                'reference_report': ['[BOS] normal lungs clear . [EOS]', '[BOS] no effusion . [EOS]']}

    def test_full_forward_backward_and_optimizer(self):
        model = self.build()
        batch = self.batch()
        captured = []
        hook = model.text_decoder.encoder_decoder.decoder.register_forward_pre_hook(
            lambda m, inputs, kwargs: captured.append(kwargs['encoder_hidden_states'].shape), with_kwargs=True)
        loss = model(batch['current_study'], batch['clinical_context'], batch['reference_report'], batch['prior_study'])
        hook.remove()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(captured[0]), (2, 16, 32))
        self.assertEqual(model.limssr_last_mask.tolist(), [[True, True, True], [True, False, False]])
        optimizer = model.configure_optimizers()['optimizer']
        registered = {id(p) for group in optimizer.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in registered for p in model.parameters() if p.requires_grad))
        self.assertEqual(optimizer.param_groups[-1]['lr'], 2e-4)
        loss.backward()
        bridge = model.limssr_bridge
        for p in (bridge.adapter.pcmi.missing, bridge.adapter.pcmi.fusion_slots,
                  bridge.memory_projection[0].weight, bridge.adapter.lmrf.role_weights,
                  bridge.adapter.mda.mix_parameter):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        lora = [p for n, p in bridge.named_parameters() if 'lora_B' in n]
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora))
        self.assertFalse(hasattr(bridge.adapter.mda, 'main_head'))
        before = bridge.memory_projection[0].weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, bridge.memory_projection[0].weight))
        self.evidence.update(forward_loss=loss.detach().item(), decoder_memory_shape=list(captured[0]),
                             lora_gradients=True)

    def test_generation_all_optional_prior_combinations(self):
        model = self.build().eval()
        batch = self.batch()
        with torch.no_grad():
            for prior in (None, batch['prior_study']):
                for context in (['[INDICATION] [SEP] [HISTORY]'] * 2, batch['clinical_context']):
                    reports = model(batch['current_study'], context, prior_study=prior, mode='test')
                    self.assertEqual(len(reports), 2)
                    self.assertTrue(all(isinstance(report, str) for report in reports))
        self.evidence['beam_generation'] = True

    def test_baseline_disabled(self):
        model = self.build(False).eval()
        batch = self.batch()
        self.assertIsNone(model.limssr_bridge)
        self.assertFalse(any(k.startswith('limssr_bridge.') for k in model.state_dict()))
        with torch.no_grad():
            loss = model(batch['current_study'], batch['clinical_context'], batch['reference_report'], batch['prior_study'])
        self.assertTrue(torch.isfinite(loss))

    def test_single_study_checkpoint_generation(self):
        trained = self.build().eval()
        checkpoint = self.root / 'tiny_roundtrip.pt'
        torch.save(trained.state_dict(), checkpoint)
        inference = TrainLanguageModelOneSample(copy.deepcopy(self.args), self.tokenizer).eval()
        inference.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        with torch.no_grad():
            reports = inference(self.batch())
        self.assertEqual(len(reports), 2)
        self.evidence['strict_checkpoint_roundtrip'] = True

    def test_bridge_missing_and_text_padding_do_not_leak(self):
        bridge = self.build().eval().limssr_bridge
        base = torch.randn(2, 12, 32)
        visual = {'current_image': torch.randn(2, 17, 32),
                  'prior_image': torch.randn(2, 17, 32), 'prior_mask': torch.tensor([True, False])}
        context = torch.randn(2, 6, 32)
        text_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]])
        contexts = ['cough', '[INDICATION] [SEP] [HISTORY]']
        with torch.no_grad():
            first = bridge(base, visual, context, contexts, text_mask)[0]
            visual['prior_image'][1] = float('nan')
            context[0, 3:] = float('nan')
            context[1] = float('nan')
            other = bridge(base, visual, context, contexts, text_mask)[0]
        torch.testing.assert_close(first, other)
        self.assertFalse(clinical_context_present('[INDICATION] [SEP] [HISTORY]'))
        self.assertTrue(clinical_context_present('no acute disease'))

    def test_lightning_fit_real_dataloader(self):
        model = self.build()
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1,
                             limit_train_batches=1, limit_val_batches=0, num_sanity_val_steps=0,
                             logger=False, enable_checkpointing=False, enable_model_summary=False,
                             enable_progress_bar=False, default_root_dir=str(self.root / 'lightning'))
        trainer.fit(model)
        self.assertEqual(trainer.global_step, 1)
        self.evidence['lightning_training_steps'] = trainer.global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', default='D:/codex-cache/priorrg-smoke')
    parser.add_argument('--report', default='limssr_smoke_result.json')
    args = parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(2)
    torch.manual_seed(42)
    IntegrationTests.root = Path(args.work_dir).resolve()
    IntegrationTests.args, IntegrationTests.tokenizer = prepare_assets(IntegrationTests.root)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(IntegrationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'success': result.wasSuccessful(), 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'device': 'cpu', 'torch': torch.__version__, 'assets': str(IntegrationTests.root),
              'weights': 'random tiny models; not pretrained clinical weights',
              'evaluators': 'external clinical evaluators skipped', **IntegrationTests.evidence}
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
