"""
Center-based hashing loss, decoupled from how the centers are produced.

A whole family of deep-hashing methods share the exact same training objective —
push tanh(embedding) onto a fixed per-class binary center, plus a quantization
term — and differ only in *where the centers come from*:

  - CSQ (Yuan et al., CVPR 2020)  -> centers are rows of a Hadamard matrix.
  - SHC (Chen et al., TOIS 2025)  -> centers come from their two-stage pipeline
                                     (class-similarity matrix, then an ALM
                                     optimisation under a minimum-distance
                                     constraint). Their released `train.py`
                                     literally defines `class CSQLoss` for
                                     stage 3 — the loss is CSQ's, only the
                                     centers change.

So this module implements the objective once and takes the centers as an input.
`centers="hadamard"` reproduces CSQ; `centers=<path>` uses any precomputed
(num_classes, bit) matrix of +/-1, e.g. SHC's stage-2 output.

The loss only ever sees `(embeddings, labels)`, so it is backbone-agnostic by
construction: DINOv2, ResNet, anything that hands it an embedding of the right
width.

Differences from the ad-hoc `CSQAdapter` in csq_loss.py, all deliberate:

  1. The centers are a registered buffer, so they follow `.to(device)` and land
     in `state_dict()`. In CSQAdapter they are a plain attribute pinned to cuda
     at construction time, and are absent from checkpoints.
  2. `quant_weight` is actually honoured. CSQAdapter reads
     `self.criterion.lambda_param`, which `CSQLoss` never defines, so the
     `hasattr` guard always fails and the weight is silently always 1e-4.
  3. The Bernoulli fallback used when 2*bit < num_classes is seeded. Unseeded,
     it draws different centers on every run — with 38 classes this triggers at
     16 bits (6 of the 38 classes), which would add run-to-run variance to a
     code-length sweep that has nothing to do with the model.
  4. Center geometry is logged (`center_min_hamming`, `center_mean_hamming`).
     This is what catches the silent failure where an externally supplied center
     file is identical to the Hadamard initialisation — i.e. where "SHC" is
     really just CSQ under another name.
  5. The center term is written in its logits form,
     `BCEWithLogits(2z, 0.5*(c+1))`, which is *identically* equal to CSQ's
     `BCE(0.5*(tanh(z)+1), 0.5*(c+1))` because `0.5*(tanh(z)+1) == sigmoid(2z)`.
     Same loss, same gradients; but `binary_cross_entropy` refuses to run under
     autocast and raises outright, and once `tanh` saturates to exactly ±1
     (which it does above |z|≈9 in fp32, and for most of a batch in fp16) the
     plain form is silently wrong: PyTorch clamps `log` at -100, so the loss
     stays finite but takes the wrong value and the gradient is exactly zero.
     Measured on a saturated logit z=20, target 0: BCE returns 100.0 with
     gradient 0.0, BCEWithLogits returns the correct 40.0 with gradient 2.0.
     The logits form has neither problem.
  6. A relative center path is resolved against the repo root as a fallback.
     Hydra changes the working directory at job runtime, so `centers:
     data/shc_centers_mflickr_64.pt` would otherwise be looked up inside the
     job's output directory and fail.

Multi-label handling is CSQ's own published rule, not an adaptation: the target
for a sample is the bit-wise majority vote over the centers of its active
classes, with ties broken by a fixed random vector.
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import hadamard

# <repo>/main/losses/center_hash_loss.py -> <repo>
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve(path):
    """Resolve a centers path, tolerating Hydra's job-time chdir.

    Hydra (version_base="1.1") changes the working directory to the job's
    output directory before the loss is built, so a path written relative to
    the repo root in config/loss/center_hash.yaml would not be found. Try the
    path as given first, then relative to the repo root.
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    candidate = os.path.join(_REPO_ROOT, path)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"centers file not found: tried {os.path.abspath(path)!r} (cwd, which "
        f"Hydra may have changed) and {candidate!r} (repo root). Generate it "
        f"with `python -m studies.generate_shc_centers`."
    )


def hadamard_centers(num_classes, bit, seed=None):
    """CSQ's Algorithm 1: Hadamard rows, with a Bernoulli fallback for the
    classes left over when 2*bit < num_classes."""
    H_K = hadamard(bit)
    H_2K = np.concatenate((H_K, -H_K), 0)
    centers = torch.from_numpy(H_2K[:num_classes]).float()

    if H_2K.shape[0] < num_classes:
        rng = random.Random(seed)
        full = torch.empty(num_classes, bit)
        full[:H_2K.shape[0]] = centers
        for _ in range(20):
            for index in range(H_2K.shape[0], num_classes):
                ones = torch.ones(bit)
                ones[rng.sample(range(bit), bit // 2)] = -1
                full[index] = ones
            d = pairwise_hamming(full)
            off = d[~torch.eye(num_classes, dtype=torch.bool)]
            if off.min() > bit / 4 and off.mean() >= bit / 2:
                break
        centers = full

    return centers


def pairwise_hamming(centers):
    """(C, bit) matrix of +/-1 -> (C, C) Hamming distances."""
    bit = centers.shape[1]
    return (bit - centers @ centers.t()) / 2


class CenterHashLoss(nn.Module):
    """BCE onto fixed per-class binary centers, plus a quantization term.

    Args:
        num_classes: number of classes C.
        embedding_size: code length in bits; must match the head's output width.
        centers: "hadamard" for CSQ's construction, or a path to a .pt/.npy file
            holding a (C, bit) or (bit, C) matrix of +/-1 (SHC's stage 2 emits
            (bit, C), which is transposed automatically).
        quant_weight: weight of the quantization term (CSQ's lambda, 1e-4).
        multi_label: True applies CSQ's majority-vote rule over the active
            classes; False assigns each sample to its argmax class's center.
        centers_seed: seed for the Bernoulli fallback (ignored when unused).
        log_diagnostics: also report center-geometry statistics each step.
    """

    takes_embeddings = True

    def __init__(self, num_classes=38, embedding_size=64, centers="hadamard",
                 quant_weight=1e-4, multi_label=True, centers_seed=0,
                 log_diagnostics=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.quant_weight = quant_weight
        self.multi_label = multi_label
        self.log_diagnostics = log_diagnostics
        self.last_components = {}

        if isinstance(centers, str) and centers == "hadamard":
            C = hadamard_centers(num_classes, embedding_size, seed=centers_seed)
            self.centers_source = "hadamard"
        elif isinstance(centers, str):
            # Resolve first, so the startup log names the file actually loaded.
            self.centers_source = _resolve(centers)
            C = self._load_centers(self.centers_source)
        else:
            C = torch.as_tensor(centers, dtype=torch.float32)
            self.centers_source = "tensor"

        C = self._validate(C)
        # A buffer, not a Parameter: the centers are fixed, but they must follow
        # the module across devices and be saved with the checkpoint.
        self.register_buffer("centers", C)
        # Fixed tie-break vector, also a buffer so it is identical on reload.
        g = torch.Generator().manual_seed(centers_seed if centers_seed is not None else 0)
        self.register_buffer(
            "tie_break", torch.randint(2, (embedding_size,), generator=g).float()
        )

        d = pairwise_hamming(self.centers)
        off = d[~torch.eye(num_classes, dtype=torch.bool)]
        print(f"[CenterHashLoss] centers={self.centers_source} "
              f"shape={tuple(self.centers.shape)} "
              f"hamming min={off.min().item():.1f} mean={off.mean().item():.2f} "
              f"max={off.max().item():.1f}")

    def _load_centers(self, path):
        if path.endswith(".npy"):
            return torch.from_numpy(np.load(path)).float()
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            for key in ("centers", "hash_centers", "H", "B"):
                if key in obj:
                    obj = obj[key]
                    break
            else:
                raise ValueError(
                    f"{path} is a dict without a recognised centers key "
                    f"(tried centers/hash_centers/H/B); got {list(obj.keys())}"
                )
        if isinstance(obj, np.ndarray):
            obj = torch.from_numpy(obj)
        return obj.float()

    def _validate(self, C):
        if C.shape == (self.embedding_size, self.num_classes) and \
                self.embedding_size != self.num_classes:
            C = C.t().contiguous()   # SHC's stage 2 returns (bit, C)
        if C.shape != (self.num_classes, self.embedding_size):
            raise ValueError(
                f"centers must be ({self.num_classes}, {self.embedding_size}) "
                f"or its transpose, got {tuple(C.shape)}"
            )
        uniq = torch.unique(C)
        if not torch.all((uniq == 1) | (uniq == -1)):
            raise ValueError(
                f"centers must contain only +1/-1, found values {uniq[:8].tolist()}"
            )
        return C

    def label2center(self, y):
        if not self.multi_label:
            return self.centers[y.argmax(dim=1)]
        # CSQ's published multi-label rule: bit-wise majority vote over the
        # centers of the active classes, ties broken by a fixed random vector.
        center_sum = y @ self.centers
        tie = self.tie_break.repeat(center_sum.shape[0], 1)
        center_sum = torch.where(center_sum == 0, tie.to(center_sum.dtype), center_sum)
        return 2 * (center_sum > 0).float() - 1

    def forward(self, embeddings, labels, **kwargs):
        target = self.label2center(labels.float())

        # CSQ writes the center term as BCE(0.5*(tanh(z)+1), 0.5*(c+1)). That is
        # *identically* BCEWithLogits(2z, 0.5*(c+1)), because
        #     0.5*(tanh(z)+1) = 0.5*((e^z - e^-z)/(e^z + e^-z) + 1)
        #                     = e^z/(e^z + e^-z) = 1/(1 + e^-2z) = sigmoid(2z).
        # Same loss, same gradients. The logits form is used because the plain
        # form fails twice under model.kwargs.with_autocast=True:
        #   - torch.nn.functional.binary_cross_entropy refuses to autocast and
        #     raises outright ("unsafe to autocast");
        #   - once tanh saturates to exactly +/-1 (above |z| ~ 9 in fp32, and for
        #     most of a batch in fp16), BCE is silently wrong rather than loud:
        #     PyTorch clamps log at -100, so the value is finite but incorrect
        #     and the gradient is exactly 0. On z=20 with target 0, BCE gives
        #     100.0 / grad 0.0 where the true values are 40.0 / grad 2.0.
        #     BCEWithLogits is computed in log-sum-exp form and gets both right.
        # embeddings is cast to fp32 for the same saturation reason; the cast is
        # differentiable, so autocast/GradScaler handle the backward pass as usual.
        z = embeddings.float()
        center_loss = nn.functional.binary_cross_entropy_with_logits(
            2.0 * z, 0.5 * (target.float() + 1)
        )

        # tanh is still what defines the code, so the quantization term and the
        # diagnostics below are computed from it -- in fp32, same reason.
        u = torch.tanh(z)
        quant_loss = (u.abs() - 1).pow(2).mean()

        self.last_components = {
            "center_bce": center_loss.detach(),
            "quant": quant_loss.detach(),
        }
        if self.log_diagnostics:
            d = pairwise_hamming(self.centers)
            off = d[~torch.eye(self.num_classes, dtype=torch.bool, device=d.device)]
            self.last_components.update({
                "center_min_hamming": off.min().detach(),
                "center_mean_hamming": off.mean().detach(),
                "mean_abs_code": u.abs().mean().detach(),
            })

        return center_loss + self.quant_weight * quant_loss

    def step(self):
        """No learnable parameters — the centers are fixed. Kept so the training
        loop can call .step() on every criterion uniformly."""
        return None
