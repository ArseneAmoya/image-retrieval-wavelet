"""Check, before burning GPU, which overrides in a study actually affect the trained model.

Why this exists: the study YAMLs split their overrides into "protected" (must match the
reference exactly) and "flexible: pure engineering/compute knobs, don't affect the
trained model". That second label was inherited from the older VOC studies and taken on
trust. It was wrong for at least one key:

    experience.sub_batch

With batch_size=96, sub_batch=96 takes base_update.py's single-pass path while
sub_batch=48 takes _gradient_cached_optimization. The loss stays full-batch, but the
model contains nn.BatchNorm1d and BN normalizes on the CURRENT microbatch in train mode
-- 48 samples instead of 96 -- while running_mean/running_var receive 4 updates per step
(2 passes x 2 microbatches) instead of 1. Those running stats are what eval mode uses,
so every reported metric moves. A num_queries=1 run at sub_batch=48 is therefore not
paired-comparable with a num_queries=4 reference at 96, and comparing them silently
mixes two effects.

This script answers "where is this key actually read?" with grep evidence instead of
assumption. It does not decide for you -- it shows you the call sites and flags the ones
that sit in model/loss/training code, which is where a supposedly-neutral knob would
have to be examined by hand.

Usage:
    python studies/audit_overrides.py studies/mflickr_num_queries_1_multiseed.yaml
    python studies/audit_overrides.py studies/*.yaml --compare studies/mflickr_lph_vs_ortho_multiseed.yaml
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where a hit means "this can change the weights that get trained".
TRAINING_PATHS = ("main/models/", "main/losses/", "main/engine/base_update",
                  "main/engine/train", "main/engine/memory", "main/getter")
# Where a hit means "this only changes what gets measured or logged".
REPORTING_PATHS = ("main/engine/evaluate", "main/engine/accuracy_calculator",
                   "main/engine/batch_map", "main/engine/get_knn")

# Keys whose neutrality has been checked by hand. Anything not listed is unverified.
VERDICTS = {
    "experience.sub_batch": ("AFFECTS TRAINING",
                             "microbatching changes BatchNorm's batch statistics and its "
                             "running stats; must match across compared runs"),
    "experience.num_workers": ("neutral", "DataLoader parallelism only"),
    "experience.eval_bs": ("neutral", "evaluation batching only, no gradient"),
    "experience.log_dir": ("neutral", "output path"),
    "experience.save_model": ("neutral", "checkpoint frequency"),
    "experience.train_eval_freq": ("neutral",
                                   "eval_split is test, so train-set eval never feeds "
                                   "best_score; TensorBoard only"),
    "experience.test_eval_freq": ("reporting",
                                  "changes WHICH epoch is reported, not what is trained; "
                                  "use evaluate_all_checkpoints.py to decouple"),
    "experience.fast_eval_freq": ("reporting", "monitoring subset only"),
    "experience.fast_eval_size": ("reporting", "monitoring subset only"),
    "experience.clip_grad": ("AFFECTS TRAINING", "gradient clipping changes the updates"),
    "experience.seed": ("AFFECTS TRAINING", "intended -- this is the repetition axis"),
    "experience.batch_map_proxy": ("AFFECTS TRAINING", "changes the loss path; verify"),
    "experience.dsch_train": ("AFFECTS TRAINING", "selects a different training loop"),
    "experience.hooks_configs.active": ("neutral", "instrumentation only"),
}


def leaf(key):
    return key.split(".")[-1]


def grep(term):
    try:
        out = subprocess.run(
            ["grep", "-rn", "--include=*.py", term, "main/", "run.py", "single_experiment_runner.py"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        ).stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []
    return [l for l in out.strip().split("\n") if l]


def classify(hits):
    training = [h for h in hits if any(p in h for p in TRAINING_PATHS)]
    reporting = [h for h in hits if any(p in h for p in REPORTING_PATHS)]
    return training, reporting


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def audit(path, compare=None):
    plan = load(path)
    base = plan.get("base_overrides", {}) or {}
    sweep = plan.get("sweep", {}) or {}

    print(f"\n{'=' * 78}\n{os.path.basename(path)}  (study_name: {plan.get('study_name')})\n{'=' * 78}")
    print(f"swept: {', '.join(sweep) if sweep else '(none)'}")

    if compare:
        ref = load(compare)
        ref_base = ref.get("base_overrides", {}) or {}
        diffs = []
        for k, v in base.items():
            if k in ref_base and ref_base[k] != v:
                diffs.append((k, ref_base[k], v))
        print(f"\n--- differences vs {os.path.basename(compare)} (excluding swept keys) ---")
        if not diffs:
            print("  none")
        for k, ref_v, v in diffs:
            verdict, why = VERDICTS.get(k, ("UNVERIFIED", "no hand-checked verdict recorded"))
            mark = "!!" if verdict.startswith("AFFECTS") or verdict == "UNVERIFIED" else "  "
            print(f"  {mark} {k}: {ref_v!r} -> {v!r}")
            print(f"       {verdict}: {why}")
        print("\n  '!!' = this difference can change the trained model, so runs of the two")
        print("        studies are NOT paired-comparable on that axis.")

    print("\n--- where each experience.* override is consumed ---")
    for key in sorted(k for k in base if k.startswith("experience.")):
        verdict, why = VERDICTS.get(key, ("UNVERIFIED", "no hand-checked verdict recorded"))
        hits = grep(leaf(key))
        training, reporting = classify(hits)
        mark = "!!" if verdict.startswith("AFFECTS") else ("??" if verdict == "UNVERIFIED" else "  ")
        print(f"\n{mark} {key} = {base[key]!r}   [{verdict}]")
        print(f"     {why}")
        if training:
            print(f"     read in training code ({len(training)}):")
            for h in training[:4]:
                print(f"       {h.strip()[:110]}")
        if not training and not reporting and hits:
            print(f"     read in ({len(hits)}): {hits[0].strip()[:110]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plans", nargs="+")
    parser.add_argument("--compare", default=None,
                         help="Reference study to diff against -- use the one whose runs you "
                              "intend to compare results with")
    args = parser.parse_args()

    paths = []
    for p in args.plans:
        paths.extend(sorted(glob.glob(p)))
    for p in paths:
        audit(p, args.compare)


if __name__ == "__main__":
    main()
