"""Run from workspace root: python -m limssr_transfer.example

Downloads a Qwen backbone unless --model points to a local model directory.
For a real project, replace random tensors with its encoder's feature sequences.
"""
import argparse
import torch
from . import LIMSSRAdapter, load_hf_backbone, limssr_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone, tokenizer = load_hf_backbone(args.model)
    model = LIMSSRAdapter({'visual': 768, 'audio': 768, 'flow': 1024},
                          backbone, tokenizer).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    # Missing entries may be zero/NaN placeholders: the masked rows are not projected.
    features = {'visual': torch.randn(4, 8, 768, device=device),
                'audio': torch.randn(4, 8, 768, device=device),
                'flow': torch.randn(4, 8, 1024, device=device)}
    mask = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]], device=device)
    target = torch.rand(4, device=device)  # Normalize scores in your data pipeline.
    model.train()
    optimizer.zero_grad()
    output = model(features, mask)
    losses = limssr_loss(output, target)
    losses['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    print({name: float(value.detach()) for name, value in losses.items()})
    print('Feature for an existing head:', tuple(output['main_feature'].shape))


if __name__ == '__main__':
    main()
