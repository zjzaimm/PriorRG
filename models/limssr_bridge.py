"""LIMSSR adaptation for PriorRG's report-generation stage.

No report targets enter the reasoning backbone. Current/prior visual sequences
and clinical context are separately represented, with true per-patient availability.
The original scalar scoring heads are replaced by report decoder memory tokens.
"""
import re
import torch
from torch import nn
from torch.nn import functional as F
from models.limssr import LIMSSRAdapter, load_hf_backbone, token_metric_loss


def add_limssr_arguments(parser):
    parser.add_argument('--use_limssr', action='store_true', help='Enable LIMSSR for report generation')
    parser.add_argument('--limssr_model_path', default='Qwen/Qwen2.5-0.5B',
                        help='HF model ID or local directory; independent of ckpt_zoo_dir')
    parser.add_argument('--limssr_tokens', type=int, default=16, help='Compressed sequence length per modality')
    parser.add_argument('--limssr_fusion_tokens', type=int, default=4)
    parser.add_argument('--limssr_lora_rank', type=int, default=16)
    parser.add_argument('--limssr_lr', type=float, default=2e-4, help='Learning rate for new reasoning modules')
    parser.add_argument('--limssr_no_lora', action='store_true', help='Use full backbone fine-tuning instead of LoRA')
    parser.add_argument('--limssr_consistency_weight', type=float, default=0.1)
    parser.add_argument('--limssr_metric_weight', type=float, default=0.01)


def clinical_context_present(context):
    """Detect empty fields after removing PriorRG's structural tokens.

    Text 'none', 'no acute disease', etc. remains observed information. Only None,
    whitespace and purely structural tokens are treated as genuinely absent.
    """
    if context is None:
        return False
    text = re.sub(r'\[INDICATION\]|\[HISTORY\]|\[SEP\]|\[PAD\]|\[CLS\]|<\|endoftext\|>',
                  ' ', str(context))
    return bool(text.strip())


def _compress(sequence, length, attention_mask=None):
    if sequence.ndim != 3 or sequence.shape[1] == 0:
        raise ValueError('expected nonempty [B,S,D] feature sequence')
    if attention_mask is None:
        return F.adaptive_avg_pool1d(sequence.transpose(1, 2), length).transpose(1, 2)
    if attention_mask.shape != sequence.shape[:2]:
        raise ValueError('text attention mask must match [B,S]')
    rows = []
    for row, valid in zip(sequence, attention_mask.bool()):
        row = row[valid]
        if row.shape[0] == 0:
            row = sequence.new_zeros(1, sequence.shape[-1])
        rows.append(F.adaptive_avg_pool1d(row.T.unsqueeze(0), length).transpose(1, 2))
    return torch.cat(rows, dim=0)


class PriorRGLIMSSRBridge(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tokens = args.get('limssr_tokens', 16)
        if self.tokens < 1:
            raise ValueError('limssr_tokens must be positive')
        self.consistency_weight = args.get('limssr_consistency_weight', 0.1)
        self.metric_weight = args.get('limssr_metric_weight', 0.01)
        if min(self.consistency_weight, self.metric_weight) < 0:
            raise ValueError('LIMSSR loss weights must be nonnegative')
        backbone, tokenizer = load_hf_backbone(
            args.get('limssr_model_path', 'Qwen/Qwen2.5-0.5B'),
            use_lora=not args.get('limssr_no_lora', False),
            lora_rank=args.get('limssr_lora_rank', 16))
        dims = {name: args['hidden_size']
                for name in ('current_image', 'prior_image', 'clinical_context')}
        self.adapter = LIMSSRAdapter(
            dims, backbone, tokenizer,
            num_fusion_tokens=args.get('limssr_fusion_tokens', 4),
            task_description='chest X-ray report generation using observed current image, '
                             'optional previous image and optional clinical context',
            feature_only=True, constrain_mix=True)
        llm_dim = backbone.get_input_embeddings().weight.shape[-1]
        self.memory_projection = nn.Sequential(nn.Linear(llm_dim, args['hidden_size']),
                                                nn.LayerNorm(args['hidden_size']))

    def forward(self, base_memory, visual_inputs, context_features, contexts, text_mask):
        batch = base_memory.shape[0]
        if len(contexts) != batch:
            raise ValueError('one clinical context is required per sample (empty string if absent)')
        current = _compress(visual_inputs['current_image'], self.tokens)
        prior = visual_inputs['prior_image']
        if prior is not None:
            prior = _compress(prior, self.tokens)
        context = _compress(context_features, self.tokens, text_mask)
        mask = torch.stack([
            torch.ones(batch, device=current.device, dtype=torch.bool),
            visual_inputs['prior_mask'].to(device=current.device, dtype=torch.bool),
            torch.tensor([clinical_context_present(x) for x in contexts], device=current.device),
        ], dim=1)
        out = self.adapter({'current_image': current, 'prior_image': prior,
                            'clinical_context': context}, mask)
        # Retain K reasoning slots while distributing the main-path residual correction.
        original_main = self.adapter.lmrf(out['fusion_tokens'])
        corrected_slots = out['fusion_tokens'] + (out['main_feature'] - original_main).unsqueeze(1)
        mix = self.adapter.mda.mix_parameter.sigmoid()
        memory = mix * corrected_slots + (1 - mix) * out['aux_feature'].unsqueeze(1)
        memory = self.memory_projection(memory).to(base_memory.dtype)
        losses = {}
        if self.training:
            losses['consistency'] = F.mse_loss(F.normalize(out['main_feature'], dim=-1),
                                               F.normalize(out['aux_feature'], dim=-1))
            losses['metric'] = token_metric_loss(out['fusion_tokens'], mode='official_triplet')
            losses['total'] = (self.consistency_weight * losses['consistency']
                               + self.metric_weight * losses['metric'])
        return torch.cat([base_memory, memory], dim=1), losses, mask
