"""Robust `torch.hub.load` for the DINOv2 backbones.

Problem this solves: `torch.hub.load('facebookresearch/dinov2', name)` contacts
github.com *before* looking at the local cache. `_get_cache_or_reload` calls
`_parse_repo_info`, which, when no branch is given, opens
`https://github.com/facebookresearch/dinov2/tree/main/` purely to discover the
default branch name. So a transient network hiccup kills the job even when the
repo is already cached in ~/.cache/torch/hub, which is exactly what happened
mid-sweep on Colab:

    http.client.RemoteDisconnected: Remote end closed connection without response

Two fixes, both applied here:

1. Pin the branch (`facebookresearch/dinov2:main`). With an explicit ref,
   `_parse_repo_info` returns immediately and the cached checkout is used with
   no network access at all. This alone makes every load after the first one
   offline-safe.
2. `skip_validation=True` suppresses the second network call (the GitHub API
   check that the ref exists), and a short retry loop covers the genuinely
   first, uncached download.

Every DINOv2 call site in main/models/ goes through `load_dinov2` so that one
sweep failing at job 2 of 3 after hours of GPU time cannot happen again for
this reason.
"""
import time

import torch

DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_REF = "main"


def load_dinov2(model_name, retries=3, backoff=5.0, **kwargs):
    """Load a DINOv2 backbone, preferring the local hub cache.

    Args:
        model_name: e.g. 'dinov2_vits14', 'dinov2_vitb14'.
        retries: attempts for the first (uncached) download.
        backoff: seconds between attempts, multiplied by the attempt index.
    """
    repo = f"{DINOV2_REPO}:{DINOV2_REF}"
    last_err = None

    for attempt in range(retries):
        try:
            return torch.hub.load(repo, model_name, skip_validation=True, **kwargs)
        except Exception as err:  # network, partial cache, GitHub 5xx...
            last_err = err
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))

    raise RuntimeError(
        f"Could not load '{model_name}' from '{repo}' after {retries} attempts. "
        f"If the machine is offline, pre-populate the cache once with network access "
        f"(the checkout lands in ~/.cache/torch/hub/facebookresearch_dinov2_main and is "
        f"reused offline afterwards). Last error: {last_err!r}"
    ) from last_err
