"""Diagnose a suspected BatchNorm+tanh saturation collapse from a checkpoint alone.

Trigger: test-split metrics that are bit-for-bit identical across several eval
points (e.g. epoch 5 and epoch 10 in mflickr_pilot_eval_tracking), together with
`bit_balance_level0 == worst_bit_balance_level0 == 0.0` exactly -- i.e. every one
of the `nbits` hash bits is constant across the whole gallery (sign() never
flips), and training has stopped moving the model at all.

Mechanism this checks for: `MultiDinoHashing.forward` returns
`self.bn(self.hash_fc(fused_embedding))` during training, and `HashLoss.forward`
immediately applies `torch.tanh(embeddings)` to that BN output before computing
the proxy BCE + quantization loss. `nn.BatchNorm1d` has learnable per-channel
affine `weight` (init 1) and `bias` (init 0) with no constraint on their
magnitude. If a channel's learned `weight` grows large, or its `bias` drifts far
from 0, that bit's pre-tanh value saturates tanh (`tanh(3) = 0.995`) for
virtually every input regardless of content -- `d/dx tanh(x) -> 0` there, so
gradients through that bit vanish and `sign()` of it freezes to a constant,
exactly matching bit_balance=0 and frozen metrics.

No GPU/data needed, just the saved checkpoint (CPU-only, reads `net_state`).

Usage:
    python studies/diagnose_bn_collapse.py path/to/epoch_5.ckpt path/to/epoch_10.ckpt
"""
import argparse
import sys

import torch


def diagnose(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net_state = state.get("net_state", state)
    epoch = state.get("epoch", "?")

    missing = [k for k in ("bn.weight", "bn.bias", "bn.running_mean", "bn.running_var", "hash_fc.weight")
               if k not in net_state]
    if missing:
        sys.exit(f"{ckpt_path}: missing expected keys {missing} -- is this a MultiDinoHashing "
                  f"checkpoint with use_bn=true? Available keys (sample): "
                  f"{list(net_state.keys())[:10]}...")

    weight = net_state["bn.weight"].float()
    bias = net_state["bn.bias"].float()
    running_mean = net_state["bn.running_mean"].float()
    running_var = net_state["bn.running_var"].float()
    hash_fc_norm = net_state["hash_fc.weight"].float().norm().item()

    # Pre-tanh value BN would emit for an "average" sample (x ~= running_mean):
    # (x - running_mean)/sqrt(running_var+eps) * weight + bias ~= bias.
    # tanh saturates hard past |x| ~= 3 (tanh(3) = 0.995, gradient = 1-tanh(x)^2 ~= 0.01).
    saturated_bias = (bias.abs() > 3).sum().item()
    saturated_weight = (weight.abs() > 5).sum().item()
    n_bits = weight.numel()

    print(f"=== {ckpt_path} (epoch {epoch}) ===")
    print(f"bn.weight  : mean={weight.mean():.3f}  max|.|={weight.abs().max():.3f}  "
          f"bits with |weight|>5 : {saturated_weight}/{n_bits}")
    print(f"bn.bias    : mean={bias.mean():.3f}  max|.|={bias.abs().max():.3f}  "
          f"bits with |bias|>3   : {saturated_bias}/{n_bits}")
    print(f"running_mean: mean={running_mean.mean():.3f}  max|.|={running_mean.abs().max():.3f}")
    print(f"running_var : mean={running_var.mean():.3f}  min={running_var.min():.3f}")
    print(f"hash_fc.weight norm: {hash_fc_norm:.3f}")

    verdict_bits = max(saturated_bias, saturated_weight)
    if verdict_bits >= n_bits * 0.8:
        print(f"-> {verdict_bits}/{n_bits} bits look saturated (|bias|>3 or |weight|>5): "
              f"consistent with a BN+tanh saturation collapse.")
    elif verdict_bits > 0:
        print(f"-> {verdict_bits}/{n_bits} bits look saturated -- partial collapse, or an "
              f"early/ongoing drift toward one.")
    else:
        print("-> No individual bit looks saturated by this simple bias/weight check. "
              "The bit_balance=0 / frozen-metrics symptom may have another cause (e.g. "
              "check the quantization vs BCE loss balance, or whether gradients are "
              "flowing into hash_fc/bn at all -- compare this checkpoint's hash_fc.weight "
              "norm against an earlier one to see if it's genuinely not updating).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpts", nargs="+", help="path(s) to checkpoint file(s), e.g. epoch_5.ckpt epoch_10.ckpt")
    args = parser.parse_args()

    for i, ckpt_path in enumerate(args.ckpts):
        if i:
            print()
        diagnose(ckpt_path)

    if len(args.ckpts) > 1:
        print()
        print("If bn.weight/bn.bias/hash_fc.weight norms above are ~identical across "
              "checkpoints, the model genuinely stopped updating between them (not just "
              "a metric-computation artifact).")


if __name__ == "__main__":
    main()
