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


def resolve_ckpt_pattern(pattern):
    if os.path.isfile(pattern):
        return pattern
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches and "/weights/" in pattern:
        prefix, filename = pattern.rsplit("/weights/", 1)
        matches = sorted(glob.glob(f"{prefix}/**/weights/{filename}", recursive=True))
    if not matches:
        sys.exit(f"No checkpoint found matching '{pattern}' (also tried a recursive "
                  f"variant for nested hydra output dirs). Check the path/pattern.")
    if len(matches) > 1:
        sys.exit(f"Pattern '{pattern}' matched {len(matches)} checkpoints, expected exactly one:\n  "
                  + "\n  ".join(matches) + "\nNarrow the pattern (e.g. add more of the override string).")
    return matches[0]
