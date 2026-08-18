"""Extracts per-epoch training loss curves from a raw single_experiment_runner.py
multirun log (the kind saved by studies/run_plan.py's stdout, e.g. the "N=1 final
headline" run covering all 8 mflickr/voc/coco jobs in one file).

Written because that log needs to go into an appendix (training curves), and
hand-transcribing ~400 epoch rows out of a multi-thousand-line log by eye is exactly
the kind of silent-error risk this project has spent a lot of effort avoiding
elsewhere -- a regex parser is deterministic and reviewable instead.

What it extracts, per line matching tqdm's end-of-epoch summary format
("100% <n>/<n> [...], HashLoss=<x>, Ortho_Loss=<x>, total_loss=<x>"):
  - a running epoch counter, reset to 1 every time a new "Command i/j:" or
    HYDRA job-launch line is seen (so each dispatched job's epochs are numbered
    1..max_iter independently, matching what save_model/checkpoint numbering means)
  - the run name, taken from the nbits/embedding_size override pair on the most
    recent "HYDRA] #0 :" launch line (falls back to "unknown_run" if none seen yet,
    which should only happen if the log was truncated before its first job header)
  - hash_loss, ortho_loss, total_loss as floats

Usage:
    python studies/parse_training_log.py "training log N=1.txt" \
        --out studies/results/final_headline_training_curves.csv
"""
import argparse
import csv
import re

EPOCH_LINE = re.compile(
    r"HashLoss=(?P<hash_loss>[-\d.eE+]+),\s*Ortho_Loss=(?P<ortho_loss>[-\d.eE+]+),\s*"
    r"total_loss=(?P<total_loss>[-\d.eE+]+)"
)
# Matches e.g. "...nbits=32 loss.0.kwargs.embedding_size=32 experience.experiment_name=..."
# on the "[HYDRA] \t#0 : ..." launch line to recover which job is about to run.
NBITS_LINE = re.compile(r"model\.kwargs\.binary_config\.nbits=(?P<nbits>\d+)")
DATASET_LINE = re.compile(r"(?:^| )dataset=(?P<dataset>\w+)")
LAUNCH_MARKER = re.compile(r"\[HYDRA\]\s+#\d+\s*:")


def parse(path):
    rows = []
    current_run = "unknown_run"
    epoch = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if LAUNCH_MARKER.search(line):
                nbits_m = NBITS_LINE.search(line)
                dataset_m = DATASET_LINE.search(line)
                dataset = dataset_m.group("dataset") if dataset_m else "unknown"
                nbits = nbits_m.group("nbits") if nbits_m else "?"
                current_run = f"{dataset}_{nbits}bits"
                epoch = 0
                continue
            m = EPOCH_LINE.search(line)
            if m:
                epoch += 1
                rows.append({
                    "run": current_run,
                    "epoch": epoch,
                    "hash_loss": float(m.group("hash_loss")),
                    "ortho_loss": float(m.group("ortho_loss")),
                    "total_loss": float(m.group("total_loss")),
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_path")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    rows = parse(args.log_path)
    if not rows:
        print("No epoch lines matched -- check the log format / EPOCH_LINE regex.")
        return

    runs = sorted(set(r["run"] for r in rows))
    print(f"Parsed {len(rows)} epoch rows across {len(runs)} run(s):")
    for run in runs:
        run_rows = [r for r in rows if r["run"] == run]
        print(f"  {run}: {len(run_rows)} epochs, "
              f"final hash_loss={run_rows[-1]['hash_loss']}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "epoch", "hash_loss", "ortho_loss", "total_loss"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
