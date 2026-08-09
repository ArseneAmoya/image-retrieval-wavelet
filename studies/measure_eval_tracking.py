"""Read the pilot run's TensorBoard logs and answer the two questions
`mflickr_pilot_eval_tracking.yaml` was built for:

  1. Is the final epoch's test mAP close to the best test mAP observed during
     training, or does the model peak earlier and decay? (Validates -- or not
     -- the "test_eval_freq = max_iter, report the final epoch" policy used by
     the other 6 MIRFLICKR studies.)
  2. Does the cheap `fast_eval` signal (500-image train-subset self-retrieval)
     actually track the real, expensive test mAP trajectory closely enough to
     trust as a monitoring proxy, or does it diverge?

Usage:
    python studies/measure_eval_tracking.py experiments_runs/mflickr_pilot_eval_tracking_*/
    python studies/measure_eval_tracking.py <run_dir> --metric map_level0
    python studies/measure_eval_tracking.py <run_dir> --list-tags   # if --metric isn't found
    python studies/measure_eval_tracking.py <run_dir> --plot out.png  # optional, needs matplotlib
"""
import argparse
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    sys.exit("This script needs the `tensorboard` package: pip install tensorboard")


def load_scalars(logs_dir, tag):
    acc = EventAccumulator(str(logs_dir), size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None
    return {e.step: e.value for e in acc.Scalars(tag)}


def list_evaluation_tags(logs_dir):
    acc = EventAccumulator(str(logs_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return sorted(t for t in acc.Tags().get("scalars", []) if "Evaluation" in t)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx ** 0.5 * vy ** 0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=str, help="Path to the run directory (contains logs/ and weights/)")
    parser.add_argument("--metric", type=str, default="map_level0")
    parser.add_argument("--list-tags", action="store_true", help="Just print available */Evaluation/* tags and exit")
    parser.add_argument("--plot", type=str, default=None, help="Optional path to save a PNG comparison plot")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    logs_dir = run_dir / "logs"
    if not logs_dir.is_dir():
        sys.exit(f"No logs/ directory under {run_dir}")

    if args.list_tags:
        for tag in list_evaluation_tags(logs_dir):
            print(tag)
        return

    test_tag = f"Test/Evaluation/{args.metric}"
    fast_tag = f"Fast/Evaluation/{args.metric}"
    test_series = load_scalars(logs_dir, test_tag)
    fast_series = load_scalars(logs_dir, fast_tag)

    if test_series is None:
        print(f"Tag not found: {test_tag}")
        print("Available */Evaluation/* tags:")
        for tag in list_evaluation_tags(logs_dir):
            print(f"  {tag}")
        sys.exit(1)

    print(f"=== {test_tag} ===")
    for step in sorted(test_series):
        print(f"  epoch {step:>3}: {test_series[step]:.4f}")

    final_epoch = max(test_series)
    final_value = test_series[final_epoch]
    best_epoch = max(test_series, key=test_series.get)
    best_value = test_series[best_epoch]
    gap = best_value - final_value
    gap_pct = (gap / best_value * 100) if best_value else 0.0

    print()
    print(f"Final epoch ({final_epoch}): {final_value:.4f}")
    print(f"Best epoch  ({best_epoch}): {best_value:.4f}")
    print(f"Gap (best - final): {gap:.4f}  ({gap_pct:.2f}% relative)")
    if best_epoch == final_epoch:
        print("-> Final epoch IS the best epoch observed. 'Final-epoch-only' reporting is safe here.")
    elif gap_pct < 1.0:
        print("-> Final epoch is within 1% of the best observed epoch -- negligible, treat as converged.")
    else:
        print(f"-> Final epoch trails the best observed epoch by {gap_pct:.2f}%. Consider a safety-net "
              f"eval point (e.g. epoch {best_epoch}) or reducing max_iter, rather than eval-only-at-the-end.")

    if fast_series is None:
        print()
        print(f"Tag not found: {fast_tag} -- can't compare against fast_eval.")
        return

    print()
    print(f"=== {fast_tag} ===")
    for step in sorted(fast_series):
        print(f"  epoch {step:>3}: {fast_series[step]:.4f}")

    common_epochs = sorted(set(test_series) & set(fast_series))
    if len(common_epochs) < 2:
        print("\nNot enough overlapping epochs between test and fast eval to correlate.")
        return

    test_vals = [test_series[e] for e in common_epochs]
    fast_vals = [fast_series[e] for e in common_epochs]
    r = pearson(test_vals, fast_vals)

    print()
    print(f"Pearson correlation (test vs fast, {len(common_epochs)} overlapping epochs): "
          f"{r:.4f}" if r is not None else "Pearson correlation: n/a (degenerate series)")
    if r is not None:
        if r > 0.8:
            print("-> Strong agreement: fast_eval is a trustworthy cheap proxy for monitoring the other jobs.")
        elif r > 0.4:
            print("-> Moderate agreement: fast_eval tracks the general trend but shouldn't replace real test "
                  "eval for anything quantitative -- fine as a divergence/collapse canary only.")
        else:
            print("-> Weak/no correlation: fast_eval (train-subset self-retrieval) is not a reliable proxy for "
                  "test-set generalization here -- treat it purely as a sanity check for loss divergence, not "
                  "as a signal of retrieval quality.")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n--plot requested but matplotlib isn't installed; skipping.")
            return
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(sorted(test_series), [test_series[e] for e in sorted(test_series)],
                 marker="o", label=f"Test/{args.metric}", color="tab:blue")
        ax1.plot(sorted(fast_series), [fast_series[e] for e in sorted(fast_series)],
                 marker="s", label=f"Fast/{args.metric}", color="tab:orange")
        ax1.axvline(final_epoch, color="gray", linestyle="--", alpha=0.5, label="final epoch")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel(args.metric)
        ax1.legend()
        ax1.set_title("Test eval vs fast eval over training")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"\nSaved plot to {args.plot}")


if __name__ == "__main__":
    main()
