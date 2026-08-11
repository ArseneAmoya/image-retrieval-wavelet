"""Shared checkpoint path/glob resolution for the diagnostic scripts.

Two reasons this exists instead of passing raw path strings straight to
torch.load:

1. Quoted glob patterns (common in Colab `!python ... "path/*pattern*"`
   cells) don't get shell-expanded -- the literal '*' reaches Python and
   torch.load fails with a confusing FileNotFoundError. Globbing here makes
   quoting always safe.
2. hydra.run.dir re-resolves experience.log_dir a second time relative to its
   own already-nested cwd (config/default.yaml), so real weights/ dirs
   sometimes end up nested an extra level, e.g.
   "{run_name}/outputs/experiments_runs/{run_name}/weights/..." instead of
   "{run_name}/weights/...". If the plain pattern matches nothing and
   contains "/weights/", retry with a recursive "**" inserted right before it.
"""
import glob
import os
import sys
from pathlib import Path


def _order_independent_search(pattern):
    """Last-resort fallback for when the pattern's wildcard-separated tokens
    aren't in the same order Hydra actually wrote them in. override_dirname
    sorts overrides alphabetically by full key (e.g. "experience.seed" before
    "model.kwargs...ortho_weight"), which is often NOT the order a human
    naturally writes a pattern in (e.g. "*ortho_weight=0.0*seed=111*" when the
    real dirname has seed first) -- a plain glob requires tokens in the exact
    order written, so this silently returns zero matches instead of erroring.

    Finds the deepest wildcard-free ancestor directory from the pattern, walks
    everything below it, and accepts any file whose full path contains every
    non-wildcard fragment of the pattern, in any order.
    """
    parts = Path(pattern).parts
    fixed_parts = []
    for part in parts:
        if any(ch in part for ch in "*?["):
            break
        fixed_parts.append(part)
    root = Path(*fixed_parts) if fixed_parts else Path(".")
    if not root.is_dir():
        return []

    rest = str(Path(pattern)).replace(str(root), "", 1)
    tokens = [t.strip("/") for t in rest.split("*") if t.strip("/")]
    if not tokens:
        return []

    found = []
    for path in root.rglob("*"):
        if path.is_file():
            s = str(path)
            if all(tok in s for tok in tokens):
                found.append(s)
    return found


def resolve_ckpt_pattern(pattern):
    if os.path.isfile(pattern):
        return pattern
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches and "/weights/" in pattern:
        prefix, filename = pattern.rsplit("/weights/", 1)
        matches = sorted(glob.glob(f"{prefix}/**/weights/{filename}", recursive=True))
    if not matches:
        matches = sorted(_order_independent_search(pattern))
    if not matches:
        sys.exit(f"No checkpoint found matching '{pattern}' (tried a direct glob, a "
                  f"recursive variant for nested hydra output dirs, and an "
                  f"order-independent token search). Check the path/pattern.")
    if len(matches) > 1:
        sys.exit(f"Pattern '{pattern}' matched {len(matches)} checkpoints, expected exactly one:\n  "
                  + "\n  ".join(matches) + "\nNarrow the pattern (e.g. add more of the override string).")
    return matches[0]
