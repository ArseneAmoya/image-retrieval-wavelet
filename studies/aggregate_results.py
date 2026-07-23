"""
Aggregates results for a study launched via run_plan.py.

Finds every run directory for the study, reads each run's saved config (from
weights/rolling.ckpt) to recover the swept parameter values it was run with (no name
parsing needed), pulls the principal metric plus any extra TensorBoard-logged scalars,
and reports mean +/- std grouped by every swept param except the seed.

Usage:
    python studies/aggregate_results.py studies/bn_ablation_voc.yaml
    python studies/aggregate_results.py studies/bn_ablation_voc.yaml \
        --metrics bit_balance worst_bit_balance --csv results.csv
"""
import argparse
import statistics
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def load_plan(plan_path):
    with open(plan_path, "r") as f:
        return yaml.safe_load(f)


def resolve_log_dir(plan):
    log_dir = plan["base_overrides"].get("experience.log_dir", ".")
    path = Path(log_dir).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_run(run_dir):
    ckpt_path = run_dir / "weights" / "rolling.ckpt"
    if not ckpt_path.is_file():
        return None
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def read_last_scalar(run_dir, tag):
    if not HAS_TENSORBOARD:
        return None
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        return None
    acc = EventAccumulator(str(logs_dir), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    events = acc.Scalars(tag)
    return events[-1].value if events else None


def collect_runs(plan, extra_metrics):
    log_dir = resolve_log_dir(plan)
    sweep_keys = list(plan["sweep"].keys())
    principal_metric = plan["base_overrides"].get("experience.principal_metric", "map_level0")
    eval_split = plan["base_overrides"].get("experience.eval_split", "test")

    rows = []
    for run_dir in sorted(log_dir.glob(f"{plan['study_name']}_*")):
        ckpt = load_run(run_dir)
        if ckpt is None:
            print(f"  skipping {run_dir.name}: no checkpoint yet")
            continue

        cfg = ckpt["config"]
        row = {"run": run_dir.name, "score": ckpt.get("best_score"), "epoch": ckpt.get("epoch")}
        for key in sweep_keys:
            row[key] = OmegaConf.select(cfg, key)

        for metric_name in extra_metrics:
            row[metric_name] = read_last_scalar(run_dir, f"{eval_split.title()}/Evaluation/{metric_name}")

        rows.append(row)

    return rows, sweep_keys, principal_metric


def aggregate(rows, sweep_keys, extra_metrics):
    seed_key = next((k for k in sweep_keys if k.endswith("seed")), None)
    group_keys = [k for k in sweep_keys if k != seed_key]

    groups = {}
    for row in rows:
        group_key = tuple(row[k] for k in group_keys)
        groups.setdefault(group_key, []).append(row)

    summary = []
    for group_key, group_rows in groups.items():
        entry = dict(zip(group_keys, group_key))
        entry["n_seeds"] = len(group_rows)
        for metric_name in ["score"] + extra_metrics:
            values = [r[metric_name] for r in group_rows if r.get(metric_name) is not None]
            entry[f"{metric_name}_mean"] = statistics.mean(values) if values else None
            entry[f"{metric_name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(entry)

    return summary, group_keys


def print_table(summary, group_keys, extra_metrics):
    metric_cols = ["score"] + extra_metrics
    headers = group_keys + ["n_seeds"] + [f"{m}_mean +/- std" for m in metric_cols]
    print(" | ".join(headers))
    for entry in summary:
        cells = [str(entry[k]) for k in group_keys] + [str(entry["n_seeds"])]
        for metric_name in metric_cols:
            mean, std = entry[f"{metric_name}_mean"], entry[f"{metric_name}_std"]
            cells.append(f"{mean:.4f} +/- {std:.4f}" if mean is not None else "n/a")
        print(" | ".join(cells))


def write_csv(path, summary, group_keys, extra_metrics):
    import csv
    metric_cols = ["score"] + extra_metrics
    fieldnames = group_keys + ["n_seeds"] + [f"{m}_mean" for m in metric_cols] + [f"{m}_std" for m in metric_cols]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in summary:
            writer.writerow({k: entry.get(k) for k in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=str, help="Path to the YAML experiment plan used to launch the study")
    parser.add_argument("--metrics", nargs="*", default=["bit_balance", "worst_bit_balance"],
                         help="Extra TensorBoard-logged metric names to pull, on top of the principal metric")
    parser.add_argument("--csv", type=str, default=None, help="Optional path to also write the summary as CSV")
    args = parser.parse_args()

    if not HAS_TENSORBOARD and args.metrics:
        print("tensorboard package not available: extra metrics will be reported as n/a "
              "(only the checkpointed principal metric will be shown).")

    plan = load_plan(args.plan)
    rows, sweep_keys, principal_metric = collect_runs(plan, args.metrics)

    if not rows:
        print(f"No completed runs found for study '{plan['study_name']}'.")
        return

    print(f"\nFound {len(rows)} run(s). Principal metric: {principal_metric}\n")
    summary, group_keys = aggregate(rows, sweep_keys, args.metrics)
    print_table(summary, group_keys, args.metrics)

    if args.csv:
        write_csv(args.csv, summary, group_keys, args.metrics)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
