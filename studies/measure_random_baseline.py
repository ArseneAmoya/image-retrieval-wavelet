"""What does a *random* hash code score under our exact evaluation protocol?

Context: a training run whose bit_balance had collapsed to exactly 0.0 (all 64 bits
constant across the gallery, so every Hamming distance is identical and the ranking is
arbitrary) still reported map=0.7736 on MIRFLICKR. Published MIRFLICKR-25K@all results
at 64 bits sit between ~0.65 and ~0.85, so knowing where the floor actually is changes
how every one of those numbers should be read -- including ours.

The mechanism to check is the relevance criterion. `CustomCalculator.label_comparison_fn`
(main/engine/accuracy_calculator.py) defines relevance as
`matmul(query_labels, reference_labels.T) > 0`, i.e. "shares at least one tag". With 38
tags and dense multi-label annotation, the fraction of (query, gallery) pairs that
qualify can be very high -- and the expected mAP of an arbitrary ranking is essentially
that fraction. This script measures it directly instead of arguing about it.

Three numbers are reported:
  1. relevance density -- the fraction of query/gallery pairs sharing >=1 tag. This is
     the theoretical floor: a ranking carrying no information scores about this.
  2. random-code mAP -- actual `calculate_maphashing` on uniformly random +-1 codes,
     through the real calculator, with the real top_k and distance metric.
  3. constant-code mAP -- every sample gets the *same* code (the degenerate
     bit_balance=0 case). Reproduces the collapsed-run situation exactly.

No GPU and no trained model needed: only the dataset's labels are used.

Usage:
    python studies/measure_random_baseline.py --dataset mflickr \
        --data-dir /content/data/mirflickr --k 19581
    python studies/measure_random_baseline.py --ckpt <path/to/rolling.ckpt>   # reuse a run's exact dataset config
"""
import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main.getter import Getter  # noqa: E402
import main.utils as lib  # noqa: E402
from main.engine.accuracy_calculator import CustomCalculator  # noqa: E402


def get_labels(dts):
    for attr in ("labels", "targets"):
        if hasattr(dts, attr):
            lab = getattr(dts, attr)
            return lab if torch.is_tensor(lab) else torch.tensor(lab)
    raise AttributeError(f"{type(dts).__name__} exposes neither .labels nor .targets")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default=None, help="Take the dataset config from a checkpoint (most faithful)")
    parser.add_argument("--dataset", default="mflickr", help="config/dataset/<name>.yaml, if --ckpt is not given")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--nbits", type=int, default=64)
    parser.add_argument("--k", type=int, default=19581, help="top_k, same as experience.evaluation.top_k")
    parser.add_argument("--n-query", type=int, default=None, help="Subsample queries for speed (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    getter = Getter()
    if args.ckpt:
        state = torch.load(lib.expand_path(args.ckpt), map_location="cpu", weights_only=False)
        ds_cfg = state["config"].dataset
    else:
        ds_cfg = OmegaConf.load(REPO_ROOT / "config" / "dataset" / f"{args.dataset}.yaml")
    if args.data_dir:
        ds_cfg.kwargs.data_dir = lib.expand_path(args.data_dir)

    dts = getter.get_dataset(None, "test", ds_cfg)
    if isinstance(dts, dict):
        query_dts = dts.get("test")
        gallery_dts = dts.get("gallery", dts.get("database", query_dts))
    else:
        query_dts, gallery_dts = dts, dts

    q_lab = get_labels(query_dts).float()
    r_lab = get_labels(gallery_dts).float()

    g = torch.Generator().manual_seed(args.seed)
    if args.n_query and args.n_query < q_lab.shape[0]:
        idx = torch.randperm(q_lab.shape[0], generator=g)[:args.n_query]
        q_lab = q_lab[idx]

    print(f"\nqueries : {tuple(q_lab.shape)}")
    print(f"gallery : {tuple(r_lab.shape)}")
    print(f"top_k   : {args.k}   nbits: {args.nbits}")

    # --- 1. relevance density -------------------------------------------------
    rel = (torch.matmul(q_lab, r_lab.t()) > 0).float()
    density = rel.mean().item()
    per_query = rel.mean(dim=1)
    print("\n--- relevance density (label_comparison_fn: shares >=1 tag) ---")
    print(f"  fraction of (query, gallery) pairs that count as relevant : {density:.4f}")
    print(f"  per-query: min={per_query.min():.4f}  median={per_query.median():.4f}  max={per_query.max():.4f}")
    print(f"  -> an uninformative ranking scores about {density:.4f} mAP by construction.")

    calc = CustomCalculator(k=args.k, device=torch.device("cpu"), distance_metric="hamming", with_faiss=False)

    # --- 2. random codes ------------------------------------------------------
    q_code = torch.randint(0, 2, (q_lab.shape[0], args.nbits), generator=g).float() * 2 - 1
    r_code = torch.randint(0, 2, (r_lab.shape[0], args.nbits), generator=g).float() * 2 - 1
    rnd = calc.calculate_maphashing(q_code, q_lab, r_code, r_lab, args.k)
    print("\n--- mAP with uniformly random +-1 codes ---")
    print(f"  maphashing = {rnd:.4f}")

    # --- 3. constant codes (the bit_balance=0 collapse) -----------------------
    const = torch.ones(1, args.nbits)
    q_const = const.repeat(q_lab.shape[0], 1)
    r_const = const.repeat(r_lab.shape[0], 1)
    cst = calc.calculate_maphashing(q_const, q_lab, r_const, r_lab, args.k)
    print("\n--- mAP with a single constant code for every sample (bit_balance = 0) ---")
    print(f"  maphashing = {cst:.4f}")

    print("\n=== reading this ===")
    print(f"  floor (no information) ~ {max(rnd, cst, density):.4f}")
    print("  Any reported mAP should be judged against that floor, not against 0.")
    print("  If the floor is close to published MIRFLICKR@all numbers, the benchmark's")
    print("  headroom is small and per-method differences deserve seeds + CIs before")
    print("  being called improvements -- ours included.")


if __name__ == "__main__":
    main()
