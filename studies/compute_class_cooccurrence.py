"""
One-off script: dump a dataset's multi-hot label matrix to a .npy file, for
use as `label_matrix` by HashLossV3 / build_hybrid_centers (see
main/losses/center_init.py).

This is intentionally separate from the loss class: the loss is
constructed from config/loss/*.yaml kwargs only (main/getter.py's
get_loss() never receives the dataset), so the co-occurrence statistics
have to be precomputed once and pointed to by path.

Usage (from the repo root, same environment as the rest of the pipeline):

    python -m scripts.compute_class_cooccurrence \
        --dataset VOC2012Hashing \
        --data_dir /content/voc2012 \
        --mode train \
        --out data/voc_train_label_matrix.npy

The output is the raw (N, C) multi-hot label matrix, not the co-occurrence
matrix itself — HashLossV3 computes NPMI from it at load time via
main.losses.center_init.compute_npmi_matrix. Saving the raw label matrix
(rather than a precomputed C x C matrix) means the same file can later
support other statistics if needed, at the cost of a few KB more disk space
for VOC-sized C.
"""
import argparse
import os

import numpy as np
import torch

from main import datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset class name as it appears in main/datasets (e.g. VOC2012Hashing).",
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Same data_dir you pass in config/dataset/<name>.yaml.",
    )
    parser.add_argument(
        "--mode", default="train",
        help="Dataset split/mode to extract labels from (default: train — "
             "match this to whatever split the loss's proxies actually "
             "train against).",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output .npy path for the (N, C) multi-hot label matrix.",
    )
    args = parser.parse_args()

    dataset_cls = getattr(datasets, args.dataset)
    dts = dataset_cls(data_dir=args.data_dir, mode=args.mode, transform=None, download=False)

    labels = dts.labels
    if len(labels) == 0:
        raise RuntimeError(f"{args.dataset} (mode={args.mode}) produced 0 labels.")

    if torch.is_tensor(labels[0]):
        label_matrix = torch.stack(list(labels)).numpy()
    else:
        label_matrix = np.asarray(labels)

    if label_matrix.ndim != 2:
        raise RuntimeError(
            f"Expected a (N, C) multi-hot label matrix, got shape {label_matrix.shape}. "
            f"{args.dataset} may store single-label class indices rather than multi-hot "
            f"vectors — this script currently assumes multi-hot (VOC-style) labels."
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.save(args.out, label_matrix.astype(np.float32))

    n, c = label_matrix.shape
    avg_labels_per_sample = label_matrix.sum(axis=1).mean()
    print(f"Saved label matrix: {args.out}")
    print(f"  N={n} samples, C={c} classes, avg active labels/sample={avg_labels_per_sample:.2f}")
    print(f"  per-class sample counts: {label_matrix.sum(axis=0).astype(int).tolist()}")


if __name__ == "__main__":
    main()
