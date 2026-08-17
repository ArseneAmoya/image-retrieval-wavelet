"""
Batch-evaluates every saved epoch checkpoint for a study's runs, meant to run
on a separate/cheaper GPU after training (see MIRFLICKR_DIAGNOSTIC_PLAN.md,
section 6.1: fast_eval is not trustworthy for epoch selection, so the real
per-epoch test signal has to come from here instead of from the training-time
`test_eval_freq` curve or from assuming the final epoch is best).

For each run directory belonging to the study (found the same way
aggregate_results.py finds them: {log_dir}/{study_name}_*), evaluates every
weights/epoch_*.ckpt present, using evaluate.py's own load_and_evaluate() so
model/dataset reconstruction is identical to the existing single-checkpoint
tool. Reports a full per-epoch table plus, per run, the best epoch by
--metric (default map_level0, matching experience.principal_metric).

Usage:
    python studies/evaluate_all_checkpoints.py studies/mflickr_lph_vs_ortho_multiseed.yaml \
        --set test --bs 256 --k 19581 --distance-metric hamming \
        --csv results_lph_vs_ortho_per_epoch.csv

Run one study at a time. GPU required (same as evaluate.py). If a run has no
weights/epoch_*.ckpt (e.g. save_model was off, or it hasn't reached its first
save point yet), it's skipped with a warning rather than failing the batch.

RESUME BEHAVIOUR (added after a power outage lost an in-progress batch, 2026-08-17):
if --csv points at a file that already exists, its (run, epoch) pairs are loaded
first and skipped -- no GPU time is spent re-evaluating a checkpoint already on
record. Every new row is appended and flushed to disk immediately after that
checkpoint's evaluation finishes, not buffered until the whole study is done, so a
second interruption only ever costs the one checkpoint being evaluated when it
happens. To resume an interrupted batch, re-run the exact same command (same
--csv path) -- already-done work is skipped automatically, nothing extra to pass.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import evaluate as evaluate_module  # noqa: E402  (repo-root evaluate.py)

EPOCH_RE = re.compile(r"epoch_(\d+)\.ckpt$")


def load_plan(plan_path):
    with open(plan_path, "r") as f:
        return yaml.safe_load(f)


def resolve_log_dir(plan, log_dir_override=None):
    log_dir = log_dir_override or plan["base_overrides"].get("experience.log_dir", ".")
    path = Path(log_dir).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def find_run_dirs(plan, log_dir_override=None):
    log_dir = resolve_log_dir(plan, log_dir_override)
    prefix = f"{plan['study_name']}_"
    # Recursive on purpose: hydra.run.dir is "{log_dir}/{experiment_name}/outputs"
    # (config/default.yaml), and experience.log_dir is then re-resolved *again*
    # relative to that already-nested cwd when checkpoints are saved -- so the
    # real weights/ dir can end up nested under an extra
    # "{run_name}/outputs/experiments_runs/{run_name}/weights" path instead of
    # directly under "{log_dir}/{run_name}/weights". Don't assume a fixed depth:
    # find every directory literally named "weights" whose immediate parent
    # starts with the study prefix, dedupe, and use that as the run dir (still
    # safe if log_dir is shared with unrelated studies -- the prefix check
    # still applies, just at whatever depth the match occurs).
    seen = set()
    run_dirs = []
    if log_dir.is_dir():
        for weights_dir in log_dir.rglob("weights"):
            run_dir = weights_dir.parent
            if run_dir.name.startswith(prefix):
                resolved = run_dir.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    run_dirs.append(run_dir)
    return sorted(run_dirs, key=lambda p: p.name)


def find_epoch_checkpoints(run_dir):
    weights_dir = run_dir / "weights"
    if not weights_dir.is_dir():
        return []
    ckpts = []
    for ckpt in weights_dir.glob("epoch_*.ckpt"):
        m = EPOCH_RE.search(ckpt.name)
        if m:
            ckpts.append((int(m.group(1)), ckpt))
    return sorted(ckpts, key=lambda pair: pair[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=str, help="Path to the YAML experiment plan used to launch the study")
    parser.add_argument("--log-dir", type=str, default=None,
                         help="Override experience.log_dir from the plan -- use this when checkpoints "
                              "live somewhere else than the yaml says (e.g. a Drive mount path that "
                              "differs between the training session and this eval session, or a shared "
                              "folder holding runs from several unrelated studies -- run dirs are still "
                              "filtered by the '{study_name}_' prefix, so mixing is safe).")
    parser.add_argument("--set", type=str, default="test")
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--nw", type=int, default=10)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=19581)
    parser.add_argument("--distance-metric", type=str, default="hamming")
    parser.add_argument("--metric", type=str, default="maphashing_level0",
                         help="Metric used to pick the best epoch per run (default: maphashing_level0, the "
                              "hashing-literature mAP@topk convention -- calculate_maphashing in "
                              "accuracy_calculator.py -- as opposed to map_level0, the generic torchmetrics "
                              "RetrievalMAP used as experience.principal_metric. The two occasionally disagree "
                              "on which epoch is best; maphashing_level0 is what the paper should report.)")
    parser.add_argument("--csv", type=str, default=None, help="Path to write the full per-epoch table. If it "
                         "already exists, its (run, epoch) rows are loaded and skipped -- this is also how you "
                         "resume an interrupted batch, see the module docstring.")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    run_dirs = find_run_dirs(plan, args.log_dir)
    if not run_dirs:
        print(f"No run directories found for study '{plan['study_name']}' under "
              f"{resolve_log_dir(plan, args.log_dir)}.")
        return

    extra_cols = ["map_level0", "bit_balance_level0", "worst_bit_balance_level0"]
    fieldnames = ["run", "epoch", args.metric] + [c for c in extra_cols if c != args.metric]

    # Load whatever was already evaluated in a prior (possibly interrupted) run of
    # this exact command, keyed by (run, epoch) so nothing gets re-evaluated.
    done = {}
    csv_exists = bool(args.csv) and Path(args.csv).exists()
    if csv_exists:
        with open(args.csv, newline="") as f:
            for row in csv.DictReader(f):
                for col in fieldnames:
                    if col not in ("run", "epoch") and row.get(col) not in (None, ""):
                        row[col] = float(row[col])
                row["epoch"] = int(row["epoch"])
                done[(row["run"], row["epoch"])] = row
        print(f"Resuming from {args.csv}: {len(done)} checkpoint(s) already evaluated, will be skipped.")

    csv_file = open(args.csv, "a" if csv_exists else "w", newline="") if args.csv else None
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames) if csv_file else None
    if csv_writer and not csv_exists:
        csv_writer.writeheader()
        csv_file.flush()

    all_rows = []
    try:
        for run_dir in run_dirs:
            ckpts = find_epoch_checkpoints(run_dir)
            if not ckpts:
                print(f"skipping {run_dir.name}: no weights/epoch_*.ckpt found")
                continue

            print(f"\n=== {run_dir.name} ({len(ckpts)} checkpoints) ===")
            run_rows = []
            for epoch, ckpt_path in ckpts:
                key = (run_dir.name, epoch)
                if key in done:
                    row = done[key]
                    run_rows.append(row)
                    print(f"  epoch {epoch:>3} -> already evaluated, skipping ({args.metric}="
                          f"{row.get(args.metric)})")
                    continue

                metrics = evaluate_module.load_and_evaluate(
                    path=str(ckpt_path),
                    set=args.set,
                    bs=args.bs,
                    nw=args.nw,
                    data_dir=args.data_dir,
                    k=args.k,
                    distance_metric=args.distance_metric,
                )
                split_metrics = metrics.get(args.set, {})
                row = {"run": run_dir.name, "epoch": epoch}
                row[args.metric] = split_metrics.get(args.metric)
                for col in extra_cols:
                    if col != args.metric:
                        row[col] = split_metrics.get(col)
                run_rows.append(row)
                print(f"  epoch {epoch:>3} -> " + ", ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in row.items() if k not in ("run", "epoch")
                ))

                # Flushed immediately: a second interruption only loses the checkpoint
                # currently being evaluated, never anything already printed above.
                if csv_writer:
                    csv_writer.writerow(row)
                    csv_file.flush()

            valid = [r for r in run_rows if r.get(args.metric) is not None]
            if valid:
                best = max(valid, key=lambda r: r[args.metric])
                final = run_rows[-1]
                print(f"  best epoch: {best['epoch']} ({args.metric}={best[args.metric]:.4f}) "
                      f"| final epoch {final['epoch']}: {args.metric}={final.get(args.metric)}")
                if best["epoch"] != final["epoch"]:
                    gap = best[args.metric] - (final.get(args.metric) or 0)
                    print(f"  NOTE: final epoch is NOT the best epoch (gap={gap:.4f}) -- "
                          f"report the best-epoch number or revisit test_eval_freq.")

            all_rows.extend(run_rows)
    finally:
        if csv_file:
            csv_file.close()

    if args.csv:
        print(f"\n{args.csv} is up to date ({len(all_rows)} rows across {len(run_dirs)} run(s)).")


if __name__ == "__main__":
    main()
