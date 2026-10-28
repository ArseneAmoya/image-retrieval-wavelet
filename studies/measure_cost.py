"""
Parameters, FLOPs and encoding latency for the ICAART revision (2026-10-28).
Answers MMM reviewers R1/R3/R4 ("report parameters, FLOPs and encoding throughput
against single ViT-S and ViT-B").

Models (64 bits, inference mode, fp32, 224x224):
  - MF-DINO-H, N=1 : MultiDinoHashing, 4 x dinov2_vits14 + cross_attention_advanced
  - ViT-S/14       : DINOHashBaseline(dinov2_vits14)
  - ViT-B/14       : DINOHashBaseline(dinov2_vitb14)

FLOPs are counted with torch.utils.flop_counter.FlopCounterMode (matmul/conv FLOPs,
1 multiply-add = 2 FLOPs) for ONE image. Latency is the mean GPU forward time over
`--iters` runs after `--warmup` runs, at batch 1 and at batch `--bs` (throughput).
The SWT itself runs on the CPU in the data pipeline (pywt.swt2); its per-image cost
is measured separately with --swt.

Usage (repo root, GPU runtime):
    python studies/measure_cost.py --bs 64 --swt
"""
import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main.models import MultiDinoHashing, DINOHashBaseline  # noqa: E402


def build_models(nbits=64):
    fusion = {
        'use_all_tokens': False, 'type': 'cross_attention_advanced', 'output_dim': 384,
        'num_heads': 8, 'dropout': 0.1, 'num_queries': 1, 'sub_band_dropout_p': 0,
        'ortho_weight': 0.1,
    }
    mf = MultiDinoHashing(
        backbones_config=[{'name': 'dinov2_vits14', 'frozen': False}] * 4,
        fusion_config=fusion, binary_config={'nbits': nbits}, use_bn=True)
    vits = DINOHashBaseline('dinov2_vits14', embed_dim=384, binary_config={'nbits': nbits})
    vitb = DINOHashBaseline('dinov2_vitb14', embed_dim=768, binary_config={'nbits': nbits})
    # (name, model, input shape for one image)
    return [('MF-DINO-H (4 x ViT-S/14, N=1)', mf, (3, 4, 224, 224)),
            ('DINOv2 ViT-S/14', vits, (3, 224, 224)),
            ('DINOv2 ViT-B/14', vitb, (3, 224, 224))]


def count_flops(model, shape, device):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, *shape, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops()


def latency(model, shape, bs, device, warmup, iters):
    x = torch.randn(bs, *shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def swt_cost(n=200):
    import numpy as np
    import pywt
    img = np.random.rand(224, 224).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(n):
        for _c in range(3):  # one SWT per RGB channel, as in basic_swt
            pywt.swt2(img, 'haar', level=1, start_level=0)
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--swt', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))
    print(f"{'model':32s} {'params (M)':>11s} {'GFLOPs/img':>11s} {'ms (bs=1)':>10s} "
          f"{'img/s (bs=' + str(args.bs) + ')':>16s}")
    for name, model, shape in build_models():
        model = model.to(device).eval()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        gflops = count_flops(model, shape, device) / 1e9
        t1 = latency(model, shape, 1, device, args.warmup, args.iters) * 1e3
        tb = latency(model, shape, args.bs, device, args.warmup, args.iters)
        print(f'{name:32s} {params:11.2f} {gflops:11.2f} {t1:10.2f} {args.bs / tb:16.1f}')
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if args.swt:
        print(f'SWT (Haar, 1 level, 3 channels, 224x224, CPU): {swt_cost() * 1e3:.2f} ms/img')


if __name__ == '__main__':
    main()
