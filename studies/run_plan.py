"""
Launches a study defined in a YAML experiment plan (see bn_ablation_voc.yaml for the
schema) via Hydra's built-in --multirun. By default this uses Hydra's plain sequential
sweeper (no extra plugin needed) -- the right choice on a single-GPU machine, since Ray
would just serialize the jobs anyway and requires hydra-ray-launcher to be installed. Pass
--ray (or set use_ray: true in the plan) to dispatch via config/hydra/launcher/ray_launcher.yaml
instead, e.g. for a multi-GPU/multi-node cluster.

Each job's experience.experiment_name is derived from Hydra's own
`${hydra:job.override_dirname}` resolver, restricted to only the swept keys (the static
base_overrides are excluded from the name), so every combination lands in its own
log_dir/<experiment_name>/ without any manual naming.

Usage:
    python studies/run_plan.py studies/bn_ablation_voc.yaml
    python studies/run_plan.py studies/bn_ablation_voc.yaml --dry-run
    python studies/run_plan.py studies/bn_ablation_voc.yaml --ray
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
    for required in ("study_name", "base_overrides"):
        if required not in plan:
            raise ValueError(f"Experiment plan {plan_path} is missing required key '{required}'")
    if "sweep" not in plan and "sweep_zip" not in plan:
        raise ValueError(f"Experiment plan {plan_path} needs either 'sweep' or 'sweep_zip'")
    zip_lists = plan.get("sweep_zip") or {}
    lengths = {k: len(v) for k, v in zip_lists.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"All 'sweep_zip' lists must have the same length (they are paired position by "
            f"position, not crossed). Got: {lengths}"
        )
    return plan


def zip_combos(plan):
    """`sweep` is a cartesian product; `sweep_zip` pairs its keys position by position.

    Needed whenever two overrides must co-vary rather than be crossed -- e.g. comparing
    exactly (num_queries=4, sub_batch=48) against (num_queries=1, sub_batch=96) without
    also training the two off-diagonal combinations. Hydra's -m only ever products
    comma-separated lists, so each zipped combination is dispatched as its own multirun
    with single-valued lists; job naming via ${hydra:job.override_dirname} is unaffected.
    """
    zip_lists = plan.get("sweep_zip") or {}
    if not zip_lists:
        return [{}]
    keys = list(zip_lists)
    n = len(zip_lists[keys[0]])
    return [{k: zip_lists[k][i] for k in keys} for i in range(n)]


def build_command(plan, use_ray, zip_fixed=None):
    base_overrides = plan["base_overrides"]
    sweep = plan.get("sweep") or {}
    zip_fixed = zip_fixed or {}

    base_args = [f"{k}={format_override_value(v)}" for k, v in base_overrides.items()]
    # Zipped keys are passed as single-valued lists so they still appear in
    # override_dirname (and therefore in the run's name) like any swept key.
    sweep_args = [
        f"{k}=" + ",".join(format_override_value(v) for v in values)
        for k, values in sweep.items()
    ] + [f"{k}={format_override_value(v)}" for k, v in zip_fixed.items()]

    # Only the swept keys should show up in each job's auto-generated name.
    exclude_keys = list(base_overrides.keys()) + ["experience.experiment_name"]
    exclude_arg = "hydra.job.config.override_dirname.exclude_keys=[" + ",".join(exclude_keys) + "]"
    name_arg = f"experience.experiment_name={plan['study_name']}_${{hydra:job.override_dirname}}"

    command = [sys.executable, "single_experiment_runner.py", "-m"]
    if use_ray:
        command.append("hydra/launcher=ray_launcher")

    return command + base_args + sweep_args + [name_arg, exclude_arg]


def preview_job_names(plan):
    """Approximates Hydra's default override_dirname formatting (sorted key=value pairs,
    comma-joined) so you can sanity-check names before actually launching anything."""
    sweep = plan.get("sweep") or {}
    keys = sorted(sweep.keys())
    names = []
    for zf in zip_combos(plan):
        for combo in itertools.product(*(sweep[k] for k in keys)) if keys else [()]:
            pairs = dict(zip(keys, combo))
            pairs.update(zf)
            dirname = ",".join(f"{k}={format_override_value(pairs[k])}" for k in sorted(pairs))
            names.append(f"{plan['study_name']}_{dirname}")
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=str, help="Path to a YAML experiment plan")
    parser.add_argument("--dry-run", action="store_true", help="Print the command and job names without launching")
    parser.add_argument("--ray", action="store_true", help="Dispatch via hydra/launcher=ray_launcher instead of Hydra's plain sequential sweeper")
    args = parser.parse_args()

    plan = load_plan(args.plan)
    use_ray = args.ray or plan.get("use_ray", False)
    names = preview_job_names(plan)
    combos = zip_combos(plan)
    commands = [build_command(plan, use_ray, zf) for zf in combos]

    print(f"Study '{plan['study_name']}': {len(names)} jobs")
    for name in names:
        print(f"  - {name}")
    for i, command in enumerate(commands):
        print(f"\nCommand{f' {i + 1}/{len(commands)}' if len(commands) > 1 else ''}:")
        print(" ".join(command))

    if args.dry_run:
        return

    for command in commands:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
