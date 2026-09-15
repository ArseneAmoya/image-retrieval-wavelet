import os

import torch


def save_proxy_snapshot(save_dir, experiment_name, epoch, criterion):
    """Saves each loss's raw `proxies` parameter to its own small .pt file, once per epoch.

    Why this exists: `main/engine/chepoint.py::checkpoint()` only ever saves `net.state_dict()`
    (backbone + hash head) at `experience.save_model`-epoch intervals -- the loss/criterion
    module (where HashLoss/HashLossV2/HashLossV3's `self.proxies` actually lives) is never
    included, so none of the ablation runs so far (hash-loss-v1-vs-v2-mflickr-results.md)
    have any raw proxy trajectory to go back and analyze, only the scalar
    `diag_min_proxy_hamming_distance` (the single closest pair's distance -- not which pair,
    and not the full pairwise picture).

    Opt-in and tied to the same `log_proxy_diagnostics` kwarg every HashLoss variant already
    exposes, so it costs nothing for runs that don't ask for it. When enabled, the saved
    tensors (num_classes x embedding_size, ~10 KB each here -- 38x64 floats) are cheap enough
    to keep every epoch rather than only at checkpoint intervals, so post-hoc analysis (full
    pairwise distance/similarity matrix over training, which proxy indices are involved in a
    fusion event, correlating fused pairs with label co-occurrence, clustering, MDS/t-SNE, ...)
    isn't limited by how coarse `experience.save_model` happens to be set for a given study.

    Files land at `<save_dir>/proxies/<experiment_name>/epoch_{NNN}.pt`, each holding
    `{"epoch": epoch, "proxies": <cpu tensor, raw/unbounded -- apply the loss's own tanh/
    normalize at analysis time to match what forward() actually saw>, "loss_name": ...}`.
    """
    for crit, _ in criterion:
        if not getattr(crit, "log_proxy_diagnostics", False):
            continue
        proxies = getattr(crit, "proxies", None)
        if proxies is None:
            continue

        out_dir = os.path.join(save_dir, "proxies", experiment_name)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "proxies": proxies.detach().cpu(),
                "loss_name": crit.__class__.__name__,
            },
            os.path.join(out_dir, f"epoch_{epoch:03d}.pt"),
        )
