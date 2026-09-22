"""LoRA (Low-Rank Adaptation) injection for the DINOv2 backbone.

Standard, library-free LoRA: freezes the backbone's own weights, then wraps
each targeted nn.Linear with a frozen base layer + a small trainable
low-rank update (B @ A, B initialized to zero so the model behaves exactly
like the pretrained backbone at step 0 -- the LoRA delta only grows as
training moves it away from zero).

No third-party LoRA/PEFT dependency required: DINOv2's official hub
implementation (facebookresearch/dinov2:main, loaded via torch.hub.load --
see run.py and main/models/hub_utils.py::load_dinov2) uses plain nn.Linear
for attn.qkv, attn.proj, mlp.fc1, mlp.fc2 in every transformer block, so
simple named_modules() substring matching is enough to find and replace
them -- no need to reach into xformers-specific internals.

Used by main/models/dino_baseline.py::DINOHashBaseline when a `lora_config`
kwarg is passed (see config/model/dino_hashing_lora.yaml). Does NOT touch
the training loop (main/engine/base_update.py) or the optimizer builder
(main/getter.py::get_optimizer) -- get_optimizer already skips any
`requires_grad=False` parameter when building param groups, so freezing the
backbone's base weights here is enough to exclude them automatically; the
newly added lora_A/lora_B parameters (requires_grad=True by default) are
picked up the same way any other trainable parameter would be.
"""
import math

import torch
import torch.nn as nn


# Substrings identifying which nn.Linear layers get a LoRA adapter, keyed by
# the 'scope' study parameter (config/model/dino_hashing_lora.yaml). Matched
# against the *submodule* name within the backbone (e.g. 'blocks.3.attn.qkv'),
# combined with an isinstance(nn.Linear) check in inject_lora() so e.g.
# patch_embed's Conv2d ('patch_embed.proj') can never match despite sharing
# the 'proj' substring -- the 'attn.' prefix on 'attn.proj' makes the intent
# explicit even before that guard.
LORA_SCOPES = {
    "attn_only": ("attn.qkv", "attn.proj"),
    "attn_mlp": ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"),
}


class LoRALinear(nn.Module):
    """Wraps a pretrained nn.Linear `base` (frozen) with a trainable
    low-rank update: y = base(x) + dropout(x) @ A^T @ B^T * (alpha / rank).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert rank > 0, "LoRA rank must be > 0"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_features, out_features = base.in_features, base.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # A: kaiming-uniform (same convention as the reference LoRA paper's
        # down-projection init). B: zero-init, so B @ A == 0 at step 0 --
        # this module is a pure pass-through wrapper around `base` until
        # training actually moves lora_B away from zero.
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + lora_out * self.scaling

    def extra_repr(self):
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"


def inject_lora(root_module: nn.Module, scope: str, rank: int, alpha: float = None, dropout: float = 0.05):
    """Freezes every parameter already in `root_module`, then replaces each
    nn.Linear submodule whose name contains one of LORA_SCOPES[scope] with a
    LoRALinear wrapper.

    alpha defaults to 2 * rank when not given, so the alpha/rank scaling
    factor stays constant (=2) across a rank sweep instead of silently
    shrinking as rank grows.

    Returns the number of layers wrapped. Raises if that count is 0: a
    silent no-op injection would make a "LoRA run" quietly identical to a
    frozen-backbone run with a linear head on top -- exactly the kind of
    silent numerical bug this project has already hit twice (autocast/BCE
    saturation, then double-tanh in SharedDinoHashing), so this fails loudly
    instead of producing an uninterpretable result.
    """
    if scope not in LORA_SCOPES:
        raise ValueError(f"Unknown LoRA scope '{scope}', expected one of {list(LORA_SCOPES)}")
    target_substrings = LORA_SCOPES[scope]
    alpha = alpha if alpha is not None else 2 * rank

    for p in root_module.parameters():
        p.requires_grad = False

    n_wrapped = 0
    for name, module in list(root_module.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(s in name for s in target_substrings):
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = root_module.get_submodule(parent_name) if parent_name else root_module
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        n_wrapped += 1

    if n_wrapped == 0:
        available = sorted({n for n, m in root_module.named_modules() if isinstance(m, nn.Linear)})
        raise RuntimeError(
            f"LoRA injection matched 0 nn.Linear layers for scope='{scope}' "
            f"(substrings={target_substrings}). Backbone module names likely "
            f"don't match the expected DINOv2 hub layout. nn.Linear modules "
            f"actually found: {available[:10]}{'...' if len(available) > 10 else ''}"
        )
    return n_wrapped
