"""Fold every `*_per_epoch.csv` in this folder into one queryable summary table.

Rationale: `evaluate_all_checkpoints.py` writes one CSV per study, each holding
every saved epoch of every run. That is the right raw format but a poor one for
comparing across studies. This script produces `all_runs_metrics.csv`: one row
per run, with the best epoch (by --metric) and the final epoch side by side, plus
whatever swept parameters can be recovered from the run directory name.

It is idempotent -- rerun it after dropping a new `*_per_epoch.csv` here and the
summary regenerates from scratch. Nothing is appended by hand, so the summary can
never drift from the raw files.

Usage:
    python studies/results/consolidate_metrics.py
    python studies/results/consolidate_metrics.py --metric map_level0
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# Hydra writes run dirs as "{study}_{k1}={v1},{k2}={v2}" with the overrides sorted
# alphabetically by full key, so parse by key name rather than by position.
PARAM_PATTERNS = {
    "seed": r"seed=([0-9]+)",
    "ortho_weight": r"ortho_weight=([0-9.]+)",
    "detail_index": r"detail_index=([0-9]+)",
    "num_queries": r"num_queries=([0-9]+)",
    "query_scale_init": r"query_scale_init=([0-9.]+)",
    "fusion_type": r"fusion_config\.type=([a-z_]+)",
    "model": r"model=([a-z0-9_]+)",
    "sub_batch": r"sub_batch=([0-9]+)",
    "wavelet": r"SWTTransform\.wavelet=([a-z0-9.]+)",
}

METRIC_COLS = ["maphashing_level0", "map_level0", "bit_balance_level0", "worst_bit_balance_level0"]


def parse_params(run_name):
    out = {}
    for key, pat in PARAM_PATTERNS.items():
        m = re.search(pat, run_name)
        out[key] = m.group(1) if m else ""
    return out


def study_from_filename(path):
    return os.path.basename(path).replace("_per_epoch.csv", "")


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="maphashing_level0",
                         help="Metric used to pick the best epoch (default: the hashing-literature mAP)")
    parser.add_argument("--out", default=os.path.join(HERE, "all_runs_metrics.csv"))
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(HERE, "*_per_epoch.csv")))
    if not files:
        print(f"No *_per_epoch.csv found in {HERE}")
        return

    rows = []
    for path in files:
        study = study_from_filename(path)
        by_run = defaultdict(list)
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                by_run[r["run"]].append(r)

        for run, recs in sorted(by_run.items()):
            recs.sort(key=lambda r: int(r["epoch"]))
            scored = [r for r in recs if to_float(r.get(args.metric)) is not None]
            if not scored:
                print(f"  {study} / {run}: no '{args.metric}' column, skipped")
                continue

            best = max(scored, key=lambda r: to_float(r[args.metric]))
            final = recs[-1]

            row = {"study": study, "run": run, "n_epochs_evaluated": len(recs)}
            row.update(parse_params(run))
            row["best_epoch"] = best["epoch"]
            row["final_epoch"] = final["epoch"]
            for col in METRIC_COLS:
                row[f"best_{col}"] = best.get(col, "")
                row[f"final_{col}"] = final.get(col, "")
            bf = to_float(best.get(args.metric))
            ff = to_float(final.get(args.metric))
            row["best_minus_final"] = f"{bf - ff:.4f}" if (bf is not None and ff is not None) else ""
            rows.append(row)

    fieldnames = (["study", "run"] + list(PARAM_PATTERNS)
                  + ["n_epochs_evaluated", "best_epoch", "final_epoch", "best_minus_final"]
                  + [f"best_{c}" for c in METRIC_COLS]
                  + [f"final_{c}" for c in METRIC_COLS])

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Wrote {args.out}  ({len(rows)} runs from {len(files)} study file(s), best epoch by {args.metric})\n")
    w = max(len(r["run"]) for r in rows)
    print(f"{'run':<{min(w, 70)}} {'best_ep':>8} {'best':>9} {'final':>9} {'b-f':>8}")
    for r in rows:
        print(f"{r['run'][:70]:<{min(w, 70)}} {r['best_epoch']:>8} "
              f"{to_float(r[f'best_{args.metric}']):>9.4f} {to_float(r[f'final_{args.metric}']):>9.4f} "
              f"{r['best_minus_final']:>8}")


if __name__ == "__main__":
    main()
