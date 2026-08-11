"""Measure query-token orthogonality from a trained checkpoint (Reviewer #3).

Loads `net_state` from the given checkpoint file, extracts the fusion head's
learnable query tokens, and reports the Gram matrix of the L2-normalized queries.
Perfect orthogonality => identity Gram (off-diagonals at 0). No data or GPU needed.

Usage:
    python studies/measure_query_orthogonality.py path/to/weights/rolling.ckpt
    python studies/measure_query_orthogonality.py run1/weights/epoch_50.ckpt run2/weights/rolling.ckpt
"""
import argparse
import sys

import torch
import torch.nn.functional as F

from ckpt_resolve import resolve_ckpt_pattern

QUERY_KEY = "fusion_head.query_tokens"


def gram_stats(query_tokens):
    """query_tokens: [1, N, D] or [N, D] -> (gram NxN, mean|offdiag|, max|offdiag|)."""
    q = query_tokens.squeeze(0).float()
    q = F.normalize(q, p=2, dim=-1)
    gram = q @ q.t()
    n = gram.size(0)
    off = gram[~torch.eye(n, dtype=torch.bool)]
    return gram, off.abs().mean().item(), off.abs().max().item()


def measure(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net_state = state.get("net_state", state)
    if QUERY_KEY not in net_state:
        sys.exit(f"'{QUERY_KEY}' not found in {ckpt_path} -- is this a MultiDinoHashing "
                 f"checkpoint with a cross-attention fusion head?")
    gram, mean_off, max_off = gram_stats(net_state[QUERY_KEY])
    epoch = state.get("epoch", "?")

    torch.set_printoptions(precision=3, sci_mode=False)
    print(f"checkpoint : {ckpt_path}")
    print(f"epoch      : {epoch}")
    print(f"gram matrix (L2-normalized queries):\n{gram}")
    print(f"mean |off-diagonal| : {mean_off:.4f}")
    print(f"max  |off-diagonal| : {max_off:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpts", nargs="+", help="path(s) to checkpoint file(s)")
    args = parser.parse_args()

    for i, ckpt_path in enumerate(args.ckpts):
        if i:
            print()
        measure(resolve_ckpt_pattern(ckpt_path))


if __name__ == "__main__":
    main()
