"""LMRF adaptation: fusion slots -> existing causal GPT2 -> role-weighted token.

Reuses PriorRG's report-language backbone. No modality imputation, missingness
gates, new backbone, or auxiliary objective is introduced by this module.
"""
import torch
from torch import nn


class LMRF(nn.Module):
    def __init__(self, hidden_dim, num_fusion_tokens=4):
        super().__init__()
        if hidden_dim < 1 or num_fusion_tokens < 1:
            raise ValueError('hidden_dim and num_fusion_tokens must be positive')
        self.fusion_tokens = nn.Parameter(torch.empty(num_fusion_tokens, hidden_dim))
        nn.init.normal_(self.fusion_tokens, std=0.02)
        self.role_weights = nn.Parameter(torch.ones(num_fusion_tokens))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features, language_backbone):
        """[B,S,D] -> [B,S+1,D]; backbone is passed, not registered twice."""
        if features.ndim != 3 or features.shape[-1] != self.fusion_tokens.shape[-1]:
            raise ValueError('features must be [B,S,D] with the GPT2 hidden dimension')
        if features.shape[0] < 1 or features.shape[1] < 1:
            raise ValueError('features must have a nonempty batch and sequence')
        slots = self.fusion_tokens.to(features).unsqueeze(0).expand(features.shape[0], -1, -1)
        sequence = torch.cat([features, slots], dim=1)
        limit = getattr(language_backbone.config, 'max_position_embeddings', None)
        if limit is not None and sequence.shape[1] > limit:
            raise ValueError('feature sequence plus fusion tokens exceeds GPT2 context length')
        hidden = language_backbone(
            inputs_embeds=sequence,
            attention_mask=torch.ones(sequence.shape[:2], device=features.device, dtype=torch.long),
            use_cache=False, return_dict=True,
        ).last_hidden_state[:, -self.fusion_tokens.shape[0]:]
        weights = self.role_weights.softmax(dim=0).to(hidden.dtype)
        fused = self.norm((hidden * weights[None, :, None]).sum(dim=1))
        return torch.cat([features, fused.to(features).unsqueeze(1)], dim=1)
