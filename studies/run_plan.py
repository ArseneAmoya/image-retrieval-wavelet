"""
Launches a study defined in a YAML experiment plan (see bn_ablation_voc.yaml for the
schema) via Hydra's built-in --multirun, reusing config/hydra/launcher/ray_launcher.yaml
for parallel dispatch on the Ray cluster.

Each job's experience.experiment_name is derived from Hydra's own
`${hydra:job.override_dirname}` resolver, restricted to only the swept keys (the static
base_overrides are excluded from the name), so every combination lands in its own
log_dir/<experiment_name>/ without any manual naming.

Usage:
    python studies/run_plan.py studies/bn_ablation_voc.yaml
    python studies/run_plan.py studies/bn_ablation_voc.yaml --dry-run
"""
import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def format_override_value(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_override_value(v) for v in value) + "]"
    return str(value)


def load_plan(plan_path):
    with open(plan_path, "r") as f:
        plan = yaml.safe_load(f)
    for required in ("study_name", "base_overrides", "sweep"):
        if required not in plan:
            raise ValueError(f"Experiment plan {plan_path} is missing required key '{required}'")
    return plan


def build_command(plan):
    base_overrides = plan["base_overrides"]
    sweep = plan["sweep"]

    base_args = [f"{k}={format_override_value(v)}" for k, v in base_overrides.items()]
    sweep_args = [
        f"{k}=" + ",".join(format_override_value(v) for v in values)
        for k, values in sweep.items()
    ]

    # Only the swept keys should show up in each job's auto-generated name.
    exclude_keys = list(base_overrides.keys()) + ["experience.experiment_name"]
    exclude_arg = "hydra.job.config.override_dirname.exclude_keys=[" + ",".join(exclude_keys) + "]"
    name_arg = f"experience.experiment_name={plan['study_name']}_${{hydra:job.override_dirname}}"

    return (
        [sys.executable, "single_experiment_runner.py", "-m", "hydra/launcher=ray_launcher"]
        + base_args
        + sweep_args
        + [name_arg, exclude_arg]
    )


def preview_job_names(plan):
    """Approximates Hydra's default override_dirname formatting (sorted key=value pairs,
    comma-joined) so you can sanity-check names before actually launching anything."""
    sweep = plan["sweep"]
    keys = sorted(sweep.keys())
    names = []
    for combo in itertools.product(*(sweep[k] for k in keys)):
        dirname = ",".join(f"{k}={format_override_value(v)}" for k, v in zip(keys, combo))
        names.append(f"{plan['study_name']}_{dirname}")
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=str, help="Path to a YAML experiment plan")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and job names without launching")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    command = build_command(plan)
    names = preview_job_names(plan)

    print(f"Study '{plan['study_name']}': {len(names)} jobs")
    for name in names:
        print(f"  - {name}")
    print("\nCommand:")
    print(" ".join(command))

    if args.dry_run:
        return

    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
