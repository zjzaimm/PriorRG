"""Independent implementation of LIMSSR sections 3.4--3.7.

Inputs are fixed-length feature sequences, not raw media. No missing-modality
target, reconstruction teacher, dataset, checkpoint, or GPU is required by API.
"""
from typing import Mapping, Optional
import torch
from torch import nn
from torch.nn import functional as F


def _mask(mask, batch, modalities, device):
    value = torch.as_tensor(mask, device=device)
    if value.ndim == 1:
        value = value.unsqueeze(0).expand(batch, -1)
    if value.shape != (batch, modalities):
        raise ValueError(f"mask must be [{modalities}] or [{batch}, {modalities}]")
    if not ((value == 0) | (value == 1)).all():
        raise ValueError("mask must contain only 0 or 1")
    if not value.bool().any(dim=1).all():
        raise ValueError("at least one observed modality is required per sample")
    return value.bool()


class _Projection(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, norm):
        super().__init__()
        if norm not in ("layer", "batch"):
            raise ValueError("projection_norm must be layer or batch")
        self.norm = norm
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        factory = nn.LayerNorm if norm == "layer" else nn.BatchNorm1d
        self.norm1, self.norm2 = factory(hidden_dim), factory(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def normalize(self, x, layer):
        if self.norm == "layer":
            return layer(x.transpose(1, 2)).transpose(1, 2)
        return layer(x)

    def forward(self, x):
        x = self.normalize(self.conv1(x.transpose(1, 2)), self.norm1)
        x = self.dropout(F.relu(x))
        x = self.normalize(self.conv2(x), self.norm2)
        return x.transpose(1, 2)


class PromptGuidedModalityImputation(nn.Module):
    """PCMI: build prompt + observed blocks + missing blocks + fusion slots.

    Each call uses one shared [M] mask. LIMSSRAdapter groups different per-sample
    masks automatically. features[name]: [B,T,C_name] or None. All T are equal;
    padded/unequal sequences must be aligned upstream (no padding API here).
    embedding_layer and tokenizer belong to the caller's language backbone.
    """
    def __init__(self, modality_dims: Mapping[str, int], hidden_dim: int,
                 num_fusion_tokens=4, dropout=0.15,
                 task_description="action quality assessment", projection_norm="layer"):
        super().__init__()
        if not modality_dims or any(d <= 0 for d in modality_dims.values()):
            raise ValueError("modality_dims must be nonempty and positive")
        if hidden_dim <= 0 or num_fusion_tokens < 1:
            raise ValueError("hidden_dim and num_fusion_tokens must be positive")
        self.names = tuple(modality_dims)
        self.dims = dict(modality_dims)
        self.hidden_dim = hidden_dim
        self.task = task_description
        # Indexed modules allow arbitrary modality names without ModuleDict restrictions.
        self.projections = nn.ModuleList([
            _Projection(d, hidden_dim, dropout, projection_norm)
            for d in modality_dims.values()
        ])
        self.boundaries = nn.Parameter(torch.empty(len(self.names), 2, hidden_dim))
        self.missing = nn.Parameter(torch.empty(len(self.names), hidden_dim))
        self.fusion_slots = nn.Parameter(torch.empty(num_fusion_tokens, hidden_dim))
        for value in (self.boundaries, self.missing, self.fusion_slots):
            nn.init.normal_(value, std=0.02)

    def forward(self, features, mask, tokenizer, embedding_layer):
        if set(features) != set(self.names):
            raise ValueError("features keys must exactly match modality_dims")
        reference = next((x for x in features.values() if x is not None), None)
        if reference is None or reference.ndim != 3:
            raise ValueError("provide at least one [B,T,C] feature tensor")
        batch, time = reference.shape[:2]
        if batch < 1 or time < 1:
            raise ValueError("batch and sequence length must be positive")
        checked = _mask(mask, batch, len(self.names), reference.device)
        if not torch.equal(checked, checked[:1].expand_as(checked)):
            raise ValueError("PCMI needs a shared mask; use LIMSSRAdapter for [B,M]")
        present = checked[0].tolist()
        for index, name in enumerate(self.names):
            x = features[name]
            if present[index] and x is None:
                raise ValueError(f"observed modality {name!r} has no feature tensor")
            if x is not None and (x.shape != (batch, time, self.dims[name])
                                  or x.device != reference.device):
                raise ValueError(f"{name}: expected [B,T,{self.dims[name]}] on same device")
        available = ", ".join(n for n, p in zip(self.names, present) if p)
        absent = ", ".join(n for n, p in zip(self.names, present) if not p) or "none"
        prompt = (
            f"Given the available {available} features. The missing modalities are {absent}. "
            "Infer useful latent representations for missing modalities at the designated positions. "
            f"Integrate all multimodal features for {self.task}. "
            "Output fused multidimensional representations at the final feature positions."
        )
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        ids = encoded["input_ids"].to(reference.device)
        prompt_embeds = embedding_layer(ids).expand(batch, -1, -1)
        dtype = prompt_embeds.dtype
        blocks, spans = [prompt_embeds], [None] * len(self.names)
        position = prompt_embeds.shape[1]
        # Critical for a causal LLM: all observations precede inferred positions.
        order = [i for i, p in enumerate(present) if p]
        order += [i for i, p in enumerate(present) if not p]
        for index in order:
            start = self.boundaries[index, 0].to(dtype).view(1, 1, -1).expand(batch, 1, -1)
            end = self.boundaries[index, 1].to(dtype).view(1, 1, -1).expand(batch, 1, -1)
            if present[index]:
                body = self.projections[index](features[self.names[index]]).to(dtype)
            else:
                body = self.missing[index].to(dtype).view(1, 1, -1).expand(batch, time, -1)
            spans[index] = (position + 1, position + 1 + time)
            blocks.extend([start, body, end])
            position += time + 2
        slots = self.fusion_slots.to(dtype).unsqueeze(0).expand(batch, -1, -1)
        blocks.append(slots)
        inputs = torch.cat(blocks, dim=1)
        attention = torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)
        if "attention_mask" in encoded:
            attention[:, :prompt_embeds.shape[1]] = encoded["attention_mask"].to(inputs.device)
        return {"inputs_embeds": inputs, "attention_mask": attention,
                "modal_spans": spans, "fusion_start": position}


class MultidimensionalRepresentationFusion(nn.Module):
    """LMRF / equation (8): learn role weights over [B,K,D] fusion tokens."""
    def __init__(self, hidden_dim, num_fusion_tokens=4):
        super().__init__()
        self.role_weights = nn.Parameter(torch.ones(num_fusion_tokens))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, fusion_tokens):
        if fusion_tokens.ndim != 3 or fusion_tokens.shape[1] != self.role_weights.numel():
            raise ValueError("fusion_tokens must be [B,K,D] with configured K")
        weights = self.role_weights.softmax(dim=0)
        return self.norm((fusion_tokens * weights[None, :, None]).sum(dim=1))


class MaskAwareDualPathAggregation(nn.Module):
    """MDA / equations (9)--(14); usable independently on supplied representations.

    main_feature: [B,D]; modal_features: [B,M,D], including inferred missing
    representations. Missing confidence weights inferred content; it does not
    create such content by itself. Returns features so existing heads can be reused.
    """
    def __init__(self, hidden_dim, num_modalities, out_dim=1, dropout=0.15,
                 num_heads=4, bounded_scores=True, constrain_mix=False, feature_only=False):
        super().__init__()
        if hidden_dim < 2 or num_modalities < 1 or out_dim < 1:
            raise ValueError("invalid dimensions")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.modalities = num_modalities
        self.hidden_dim = hidden_dim
        self.bounded = bounded_scores
        self.constrain_mix = constrain_mix
        self.feature_only = feature_only
        self.delta = nn.Sequential(nn.Linear(hidden_dim + num_modalities, hidden_dim),
                                   nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.gate = nn.Sequential(nn.Linear(hidden_dim + num_modalities, hidden_dim), nn.Sigmoid())
        self.modal_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_modalities)])
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.modal_gate = nn.Sequential(nn.Linear(hidden_dim * num_modalities, hidden_dim),
                                       nn.GELU(), nn.Linear(hidden_dim, num_modalities))
        self.missing_confidence = nn.Parameter(torch.full((num_modalities,), 0.6))
        initial = torch.logit(torch.tensor(0.7)) if constrain_mix else torch.tensor(0.7)
        self.mix_parameter = nn.Parameter(initial)
        def head():
            return nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, out_dim))
        if not feature_only:
            self.main_head, self.aux_head = head(), head()

    def forward(self, main_feature, modal_features, mask):
        if main_feature.ndim != 2 or main_feature.shape[-1] != self.hidden_dim:
            raise ValueError("main_feature must be [B,D]")
        batch = main_feature.shape[0]
        if modal_features.shape != (batch, self.modalities, self.hidden_dim):
            raise ValueError("modal_features must be [B,M,D]")
        mask = _mask(mask, batch, self.modalities, main_feature.device).to(main_feature.dtype)
        conditioning = torch.cat([main_feature, mask], dim=-1)
        main = main_feature + self.gate(conditioning) * self.delta(conditioning)
        modal = torch.stack([norm(modal_features[:, i]) for i, norm in enumerate(self.modal_norms)], dim=1)
        attended, _ = self.attention(modal, modal, modal, need_weights=False)
        weights = self.modal_gate(modal.flatten(1)).softmax(dim=-1)
        confidence = mask + (1 - mask) * self.missing_confidence.sigmoid()
        aux = (attended * (weights * confidence).unsqueeze(-1)).sum(dim=1)
        mix = self.mix_parameter.sigmoid() if self.constrain_mix else self.mix_parameter
        if self.feature_only:
            return {"main_feature": main, "aux_feature": aux,
                    "combined_feature": mix * main + (1 - mix) * aux}
        pred_main, pred_aux = self.main_head(main), self.aux_head(aux)
        if self.bounded:
            pred_main, pred_aux = pred_main.sigmoid(), pred_aux.sigmoid()
        mix = self.mix_parameter.sigmoid() if self.constrain_mix else self.mix_parameter
        return {"prediction": mix * pred_main + (1 - mix) * pred_aux,
                "prediction_main": pred_main, "prediction_aux": pred_aux,
                "main_feature": main, "aux_feature": aux}


class LIMSSRAdapter(nn.Module):
    """Full PCMI -> language backbone -> LMRF -> MDA, with feature-only outputs.

    backbone must provide get_input_embeddings() and accept inputs_embeds,
    attention_mask, output_hidden_states, return_dict and use_cache.
    Pass an HF AutoModel (preferred) or AutoModelForCausalLM. All modules and
    features must be on one device. Create this module BEFORE the optimizer.
    """
    def __init__(self, modality_dims, backbone, tokenizer, hidden_dim: Optional[int] = None,
                 num_fusion_tokens=4, dropout=0.15, task_description="action quality assessment",
                 projection_norm="layer", out_dim=1, num_heads=4,
                 bounded_scores=True, constrain_mix=False, feature_only=False):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        if hidden_dim is None:
            hidden_dim = backbone.get_input_embeddings().weight.shape[-1]
        self.pcmi = PromptGuidedModalityImputation(modality_dims, hidden_dim, num_fusion_tokens,
                                                   dropout, task_description, projection_norm)
        self.lmrf = MultidimensionalRepresentationFusion(hidden_dim, num_fusion_tokens)
        self.mda = MaskAwareDualPathAggregation(hidden_dim, len(modality_dims), out_dim,
                                                dropout, num_heads, bounded_scores, constrain_mix, feature_only)

    def _shared_forward(self, features, mask):
        packed = self.pcmi(features, mask, self.tokenizer, self.backbone.get_input_embeddings())
        result = self.backbone(inputs_embeds=packed["inputs_embeds"],
                               attention_mask=packed["attention_mask"],
                               output_hidden_states=True, return_dict=True, use_cache=False)
        hidden = getattr(result, "last_hidden_state", None)
        if hidden is None:
            hidden = result.hidden_states[-1]
        # Keep adapter weights FP32 by default while accepting lower-precision LLM states.
        hidden = hidden.to(self.lmrf.role_weights.dtype)
        tokens = hidden[:, packed["fusion_start"]:]
        modal = torch.stack([hidden[:, start:end].mean(dim=1)
                             for start, end in packed["modal_spans"]], dim=1)
        out = self.mda(self.lmrf(tokens), modal, mask)
        out.update(fusion_tokens=tokens, modal_features=modal)
        return out

    def forward(self, features, mask):
        reference = next((x for x in features.values() if x is not None), None)
        if reference is None or reference.ndim != 3:
            raise ValueError("at least one [B,T,C] tensor is required")
        masks = _mask(mask, reference.shape[0], len(self.pcmi.names), reference.device)
        groups, inverse = torch.unique(masks, dim=0, return_inverse=True)
        outputs, positions = [], []
        for group_index, group_mask in enumerate(groups):
            indices = (inverse == group_index).nonzero(as_tuple=True)[0]
            subset = {name: None if x is None else x.index_select(0, indices)
                      for name, x in features.items()}
            outputs.append(self._shared_forward(subset, group_mask))
            positions.append(indices)
        restore = torch.cat(positions).argsort()
        return {key: torch.cat([out[key] for out in outputs], dim=0).index_select(0, restore)
                for key in outputs[0]}


def token_metric_loss(tokens, margin=1.0, mode="official_triplet"):
    """Explicitly distinguish official repository loss from literal paper Eq.17.

    official_triplet: same slot across different samples = positives; different
    slots = negatives; hardest mining with cosine distance. B<2 or K<2 -> zero.
    paper_eq17: literal within-sample max(sim)-min(sim)+margin, summed over K,
    averaged over B. For positive margin this cannot reach zero; at K=2 it is
    constant. It is NOT silently substituted with another diversity objective.
    """
    if tokens.ndim != 3 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
        raise ValueError("tokens must be nonempty [B,K,D]")
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    if mode not in ("official_triplet", "paper_eq17"):
        raise ValueError("unknown token metric mode")
    batch, slots, _ = tokens.shape
    if slots < 2 or (mode == "official_triplet" and batch < 2):
        return tokens.sum() * 0
    if mode == "paper_eq17":
        normed = F.normalize(tokens, dim=-1)
        similarity = normed @ normed.transpose(1, 2)
        diagonal = torch.eye(slots, device=tokens.device, dtype=torch.bool)[None]
        maximum = similarity.masked_fill(diagonal, -torch.inf).amax(dim=-1)
        minimum = similarity.masked_fill(diagonal, torch.inf).amin(dim=-1)
        return F.relu(maximum - minimum + margin).sum(dim=-1).mean()
    flat = F.normalize(tokens.reshape(batch * slots, -1), dim=-1)
    distance = 1 - flat @ flat.T
    labels = torch.arange(slots, device=tokens.device).repeat(batch)
    equal = labels[:, None] == labels[None, :]
    diagonal = torch.eye(batch * slots, device=tokens.device, dtype=torch.bool)
    positive = equal & ~diagonal
    negative = ~equal
    hardest_positive = distance.masked_fill(~positive, -torch.inf).amax(dim=-1)
    hardest_negative = distance.masked_fill(~negative, torch.inf).amin(dim=-1)
    return F.relu(hardest_positive - hardest_negative + margin).mean()


def limssr_loss(outputs, target, task_weight=10.0, consistency_weight=1.0,
                metric_weight=1.0, margin=1.0, metric_mode="official_triplet"):
    """Regression loss; normalized targets required when bounded_scores=True.

    For a different task, reuse token_metric_loss and adapt branch consistency to
    that task's output space instead of blindly applying this regression objective.
    """
    prediction = outputs["prediction"]
    if target.ndim == 1 and prediction.shape[-1] == 1:
        target = target.unsqueeze(-1)
    if target.shape != prediction.shape:
        raise ValueError("target shape must match prediction (no broadcasting)")
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    task = F.mse_loss(prediction, target)
    consistency = F.mse_loss(outputs["prediction_main"], outputs["prediction_aux"])
    metric = token_metric_loss(outputs["fusion_tokens"], margin, metric_mode)
    total = task_weight * task + consistency_weight * consistency + metric_weight * metric
    return {"total": total, "task": task, "consistency": consistency, "metric": metric}


def load_hf_backbone(model_path="Qwen/Qwen2.5-0.5B", use_lora=True, lora_rank=16):
    """Optional transformers/peft loader; no device_map or automatic device moves.

    Uses AutoModel to skip unused vocabulary logits. Base weights are frozen for
    LoRA, input embeddings trainable as in the source. Our special slots are
    separate Parameters, so no tokenizer vocabulary resize or LM head is needed.
    """
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    backbone = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        backbone = get_peft_model(backbone, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, r=lora_rank, lora_alpha=32,
            lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        backbone.get_input_embeddings().weight.requires_grad_(True)
    return backbone, tokenizer
