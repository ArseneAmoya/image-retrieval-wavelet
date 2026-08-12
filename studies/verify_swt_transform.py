"""Sanity-check the SWT transform: does it really produce 4 distinct sub-bands, and
does its output match what the networks expect as input?

Motivation: mflickr_single_band_ablation returned ~82% mAP for *every* band, including
the high-frequency ones. On MIRFLICKR the degenerate floor is high (~77% -- measured
from a run whose bit_balance had collapsed to exactly 0.0, i.e. dead bits, which still
scored map=0.7736 purely from tag co-occurrence), so "all bands ~82%" is consistent
either with "the detail bands carry little usable signal" or with "the bands are not
actually different by the time they reach the backbone". This script distinguishes the
two, and checks the tensor contract at the same time.

It deliberately builds the transform through the real code path
(`Getter().get_transform(...)` on the real YAML) rather than reimplementing it, so what
is measured is what training actually used.

Usage:
    python studies/verify_swt_transform.py --image /content/data/mirflickr/images/im1.jpg
    python studies/verify_swt_transform.py --image <path> --transform config/transform/basic_swt.yaml --split test
    python studies/verify_swt_transform.py --image <path> --out swt_check.png

Outputs a per-band statistics table, a set of contract checks, and (unless --no-plot) a
PNG showing the original image next to the 4 sub-bands.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main.getter import Getter  # noqa: E402

BAND_NAMES = ["LL (cA)", "LH (cH)", "HL (cV)", "HH (cD)"]

# What DINOv2 was pretrained on. The SWT transform configs contain no Normalize (and no
# ToTensor -- SWTTransform does the tensor conversion itself), so these are the numbers
# the input distribution is being compared against, not applied.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to one real dataset image")
    parser.add_argument("--transform", default="config/transform/basic_swt.yaml")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                         help="test is deterministic (CenterCrop); train includes RandomResizedCrop")
    parser.add_argument("--out", default="swt_check.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    tf_path = Path(args.transform)
    if not tf_path.is_absolute():
        tf_path = REPO_ROOT / tf_path
    cfg = OmegaConf.load(tf_path)

    transform = Getter().get_transform(cfg[args.split])
    img = Image.open(args.image).convert("RGB")
    out = transform(img)

    print(f"\nimage      : {args.image}")
    print(f"transform  : {tf_path.name} [{args.split}]")
    print(f"output     : shape={tuple(out.shape)}  dtype={out.dtype}")

    # ---- Contract checks -------------------------------------------------------
    print("\n--- tensor contract ---")
    ok = True

    if out.dim() != 4:
        print(f"FAIL: expected 4 dims [C, S, H, W], got {out.dim()}")
        ok = False
    else:
        C, S, H, W = out.shape
        print(f"  [C={C}, S={S}, H={H}, W={W}]")
        print(f"  {'OK  ' if C == 3 else 'FAIL'}: C == 3 (RGB)")
        print(f"  {'OK  ' if S == 4 else 'FAIL'}: S == 4 sub-bands")
        print(f"  {'OK  ' if H == W == 224 else 'WARN'}: spatial size {H}x{W} (backbones expect 224x224)")
        ok = ok and C == 3 and S == 4

        # This is the exact indexing SingleBandNet.forward and the SWT band split use:
        #   x[:, :, detail_index, :, :] on a batched [B, C, S, H, W]
        batched = out.unsqueeze(0)
        picked = batched[:, :, 0, :, :]
        print(f"  {'OK  ' if tuple(picked.shape) == (1, 3, H, W) else 'FAIL'}: "
              f"x[:, :, i, :, :] -> {tuple(picked.shape)} (must be [B, 3, H, W] for the backbone)")

    print(f"  {'OK  ' if out.dtype == torch.float32 else 'FAIL'}: dtype float32")
    print(f"  {'OK  ' if torch.isfinite(out).all() else 'FAIL'}: all values finite")

    # ---- Are the 4 bands actually different? -----------------------------------
    print("\n--- per-band statistics (over all 3 channels) ---")
    print(f"{'band':<10} {'min':>9} {'max':>9} {'mean':>9} {'std':>9}   vs ImageNet-normalized range")
    for s in range(out.shape[1]):
        b = out[:, s]
        # Where a *correctly normalized* input would sit, for reference:
        print(f"{BAND_NAMES[s]:<10} {b.min():>9.4f} {b.max():>9.4f} {b.mean():>9.4f} {b.std():>9.4f}")

    lo = min((0.0 - m) / sd for m, sd in zip(IMAGENET_MEAN, IMAGENET_STD))
    hi = max((1.0 - m) / sd for m, sd in zip(IMAGENET_MEAN, IMAGENET_STD))
    print(f"\n  For reference, an ImageNet-normalized input spans roughly [{lo:.2f}, {hi:.2f}]")
    print("  with mean~0 and std~1 per channel. The SWT configs contain no Normalize step,")
    print("  so whatever is printed above is what DINOv2 actually receives.")

    print("\n--- pairwise band difference (are they distinct at all?) ---")
    flat = out.reshape(out.shape[0], out.shape[1], -1).permute(1, 0, 2).reshape(out.shape[1], -1)
    norm = torch.nn.functional.normalize(flat, dim=-1)
    cos = norm @ norm.t()
    print("cosine similarity between raw sub-band tensors:")
    header = " " * 10 + "".join(f"{n.split()[0]:>9}" for n in BAND_NAMES)
    print(header)
    for i, name in enumerate(BAND_NAMES):
        print(f"{name.split()[0]:<10}" + "".join(f"{v:>9.4f}" for v in cos[i].tolist()))
    off = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
    print(f"\n  mean |off-diagonal| = {off.abs().mean():.4f}")
    if off.abs().mean() > 0.95:
        print("  -> WARNING: the sub-bands are nearly identical. The decomposition is not")
        print("     doing what the architecture assumes (check for a RawStackTransform mixup).")
    else:
        print("  -> the sub-bands are genuinely distinct tensors.")

    # ---- Visual check ----------------------------------------------------------
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib not available -- skipping the plot (use --no-plot to silence).")
            return

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))

        axes[0, 0].imshow(img.resize((224, 224)))
        axes[0, 0].set_title("original (resized)")
        axes[0, 0].axis("off")
        axes[1, 0].axis("off")

        for s in range(out.shape[1]):
            band = out[:, s].permute(1, 2, 0).numpy()   # [H, W, 3]

            # Row 0: each band on its OWN scale -- shows the structure it contains.
            b_min, b_max = band.min(), band.max()
            shown = (band - b_min) / (b_max - b_min + 1e-8)
            axes[0, s + 1].imshow(shown)
            axes[0, s + 1].set_title(f"{BAND_NAMES[s]}\nper-band scale [{b_min:.3f}, {b_max:.3f}]")
            axes[0, s + 1].axis("off")

            # Row 1: every band on a SHARED scale -- shows their true relative amplitude,
            # i.e. what the backbone actually sees before any normalization.
            g_min, g_max = out.min().item(), out.max().item()
            shown_shared = (band - g_min) / (g_max - g_min + 1e-8)
            axes[1, s + 1].imshow(shown_shared)
            axes[1, s + 1].set_title(f"{BAND_NAMES[s]}\nshared scale (true amplitude)")
            axes[1, s + 1].axis("off")

        fig.suptitle("Top row: each band rescaled independently (structure). "
                     "Bottom row: all bands on one shared scale (what the backbone receives).",
                     fontsize=13)
        plt.tight_layout()
        fig.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"\nWrote {args.out}")
        print("Read the BOTTOM row: if the three detail bands look like flat grey there while")
        print("LL looks like the image, the detail branches are being fed a near-constant")
        print("input and cannot contribute much regardless of the architecture.")


if __name__ == "__main__":
    main()
