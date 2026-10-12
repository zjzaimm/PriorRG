"""PCMI -> existing GPT-2 -> LMRF -> MDA for PriorRG report generation.

Adapted from LIMSSR: https://github.com/XuHuangbiao/LIMSSR.
Produces decoder memory instead of action-quality scores; no extra backbone.
"""
import re
import torch
from torch import nn
from torch.nn import functional as F


def _compress(features, length, mask=None):
    if features is None:
        return None
    if mask is None:
        return F.adaptive_avg_pool1d(features.transpose(1, 2), length).transpose(1, 2)
    rows = []
    for row, valid in zip(features, mask.bool()):
        row = row[valid]
        if not len(row):
            row = features.new_zeros(1, features.shape[-1])
        rows.append(F.adaptive_avg_pool1d(row.T.unsqueeze(0), length).transpose(1, 2))
    return torch.cat(rows)


class PCMI(nn.Module):
    """Prompt-guided missing-modality positions; observations precede missing slots."""
    names = ('current_image', 'prior_image', 'clinical_context')

    def __init__(self, hidden_dim, fusion_tokens=4):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                          nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            for _ in self.names])
        self.boundaries = nn.Parameter(torch.randn(3, 2, hidden_dim) * 0.02)
        self.missing = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.fusion_tokens = nn.Parameter(torch.randn(fusion_tokens, hidden_dim) * 0.02)

    def forward(self, features, observed, tokenizer, embedding):
        batch, length, _ = features[0].shape
        available = ', '.join(name for name, present in zip(self.names, observed) if present)
        absent = ', '.join(name for name, present in zip(self.names, observed) if not present) or 'none'
        prompt = (f'Observed modalities: {available}. Missing modalities: {absent}. '
                  'Infer useful latent features at the missing positions, then fuse all features '
                  'for chest X-ray report generation at the final fusion positions.')
        ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(features[0].device)
        prompt_features = embedding(ids).expand(batch, -1, -1)
        parts, spans = [prompt_features], [None] * 3
        position = prompt_features.shape[1]
        order = [i for i, present in enumerate(observed) if present]
        order += [i for i, present in enumerate(observed) if not present]
        for i in order:
            if observed[i]:
                if features[i] is None:
                    raise ValueError('Observed modality has no features')
                body = self.projections[i](features[i])
            else:
                body = self.missing[i].view(1, 1, -1).expand(batch, length, -1)
            start = self.boundaries[i, 0].view(1, 1, -1).expand(batch, 1, -1)
            end = self.boundaries[i, 1].view(1, 1, -1).expand(batch, 1, -1)
            parts.extend([start.to(prompt_features), body.to(prompt_features), end.to(prompt_features)])
            spans[i] = (position + 1, position + 1 + length)
            position += length + 2
        parts.append(self.fusion_tokens.to(prompt_features).unsqueeze(0).expand(batch, -1, -1))
        return torch.cat(parts, dim=1), spans, position


class LMRF(nn.Module):
    """Learn role weights over the language backbone's fusion positions."""
    def __init__(self, hidden_dim, fusion_tokens=4):
        super().__init__()
        self.role_weights = nn.Parameter(torch.ones(fusion_tokens))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens):
        weights = self.role_weights.softmax(dim=0).to(tokens.dtype)
        return self.norm((tokens * weights[None, :, None]).sum(dim=1))


class MDA(nn.Module):
    """Mask-conditioned main residual and confidence-weighted cross-modal path."""
    def __init__(self, hidden_dim, heads=8):
        super().__init__()
        self.delta = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.GELU(),
                                   nn.Linear(hidden_dim, hidden_dim))
        self.gate = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.Sigmoid())
        self.modal_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(3)])
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.modal_gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, 3))
        self.missing_confidence = nn.Parameter(torch.full((3,), 0.6))
        self.mix = nn.Parameter(torch.logit(torch.tensor(0.7)))

    def forward(self, main, modal, mask):
        mask = mask.to(main.dtype)
        conditioning = torch.cat([main, mask], dim=-1)
        main = main + self.gate(conditioning) * self.delta(conditioning)
        modal = torch.stack([norm(modal[:, i]) for i, norm in enumerate(self.modal_norms)], dim=1)
        attended, _ = self.attention(modal, modal, modal, need_weights=False)
        weights = self.modal_gate(modal.flatten(1)).softmax(dim=-1)
        confidence = mask + (1 - mask) * self.missing_confidence.sigmoid()
        auxiliary = (attended * (weights * confidence).unsqueeze(-1)).sum(dim=1)
        mix = self.mix.sigmoid()
        return mix * main + (1 - mix) * auxiliary


class LIMSSR(nn.Module):
    """Append one fused memory token; backbone/tokenizer are supplied by the caller."""
    def __init__(self, hidden_dim, fusion_tokens=4, modality_tokens=16):
        super().__init__()
        self.modality_tokens = modality_tokens
        self.pcmi = PCMI(hidden_dim, fusion_tokens)
        self.lmrf = LMRF(hidden_dim, fusion_tokens)
        self.mda = MDA(hidden_dim)

    def forward(self, memory, visual, text_features, contexts, text_mask, backbone, tokenizer):
        batch = memory.shape[0]
        if len(contexts) != batch or text_mask.shape != text_features.shape[:2]:
            raise ValueError('Contexts and text mask must match the feature batch')
        clinical_present = [bool(re.sub(
            r'\[INDICATION\]|\[HISTORY\]|\[SEP\]|\[PAD\]|\[CLS\]|<\|endoftext\|>',
            ' ', context or '').strip()) for context in contexts]
        mask = torch.stack([
            torch.ones(batch, device=memory.device, dtype=torch.bool),
            visual['prior_mask'].to(device=memory.device, dtype=torch.bool),
            torch.tensor(clinical_present, device=memory.device, dtype=torch.bool)], dim=1)
        features = [_compress(visual['current_image'], self.modality_tokens),
                    _compress(visual['prior_image'], self.modality_tokens),
                    _compress(text_features, self.modality_tokens, text_mask)]
        groups, inverse = torch.unique(mask, dim=0, return_inverse=True)
        outputs, positions = [], []
        for group_id, group in enumerate(groups):
            indices = (inverse == group_id).nonzero(as_tuple=True)[0]
            subset = [None if feature is None else feature.index_select(0, indices) for feature in features]
            sequence, spans, fusion_start = self.pcmi(
                subset, group.tolist(), tokenizer, backbone.get_input_embeddings())
            limit = getattr(backbone.config, 'max_position_embeddings', None)
            if limit is not None and sequence.shape[1] > limit:
                raise ValueError('PCMI sequence exceeds the language backbone context length')
            hidden = backbone(inputs_embeds=sequence,
                              attention_mask=torch.ones(sequence.shape[:2], device=sequence.device,
                                                        dtype=torch.long),
                              use_cache=False, return_dict=True).last_hidden_state
            hidden = hidden.to(self.lmrf.role_weights.dtype)
            main = self.lmrf(hidden[:, fusion_start:])
            modal = torch.stack([hidden[:, start:end].mean(dim=1) for start, end in spans], dim=1)
            outputs.append(self.mda(main, modal, mask.index_select(0, indices)))
            positions.append(indices)
        restore = torch.cat(positions).argsort()
        fused = torch.cat(outputs).index_select(0, restore).to(memory.dtype)
        return torch.cat([memory, fused.unsqueeze(1)], dim=1)
