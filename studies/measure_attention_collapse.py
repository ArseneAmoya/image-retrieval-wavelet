"""Direct, mechanistic evidence on the orthogonality regularization (Reviewer #3).

Two questions, both answered from real forward passes on real data (not inferred
from downstream mAP):

  1. Attention collapse: for each of the `num_queries` learnable queries, how much
     attention mass (post-softmax, from nn.MultiheadAttention) lands on each of the
     4 subband tokens (LL, LH, HL, HH)? If collapse is present, most queries put
     almost all their mass on LL regardless of ortho_weight.
  2. Read-out diversity (independent of #1): even if the attention *distribution*
     collapses onto LL, do the per-band *projected* embeddings (`kv_list`, i.e. the
     tokens being attended to) become more mutually diverse when ortho_weight > 0?
     This is measured with the mean pairwise cosine similarity between the 4
     projected band embeddings, averaged over the batch -- a lower number means
     more diverse (less redundant) per-band content.

Both diagnostics reuse the exact model-loading path from evaluate.py
(`Getter().get_model` + `net.load_state_dict(state["net_state"])`), so there is no
risk of silently reconstructing a different architecture than the one actually
trained. This script is read-only: it does not modify the model, the training code,
or the loss computation, and requires no gradient.

Usage (run from the repo root, after `pip install -e .`):
    python studies/measure_attention_collapse.py \
        --ckpt no_ortho=experiments_runs/mflickr_lph_vs_ortho_multiseed_.../weights/rolling.ckpt \
        --ckpt ortho=experiments_runs/mflickr_lph_vs_ortho_multiseed_.../weights/rolling.ckpt \
        --set test --n-batches 5 --bs 64

Each --ckpt is `label=path`; pass as many as you want (e.g. one per ortho_weight
value from the lambda2 sweep) and they'll all be printed side by side.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from main.getter import Getter  # noqa: E402
import main.utils as lib  # noqa: E402
from ckpt_resolve import resolve_ckpt_pattern  # noqa: E402

BAND_NAMES = ["LL", "LH", "HL", "HH"]


class Capture:
    """Accumulates attn_weights and per-band projected embeddings across batches."""

    def __init__(self):
        self.attn_weights = []   # list of [B, num_queries, num_bands]
        self.band_embeds = []    # list of [num_bands][B, D] -> transposed at the end
        self.qk_inputs = []      # list of (query, key) as actually passed to self.attn

    def attn_pre_hook(self, module, args, kwargs):
        # CrossAttentionBottleneckHeadAdvanced.forward calls
        # self.attn(query=q, key=kv, value=kv) with keyword args, so read kwargs
        # first and fall back to positional. Needed to recompute the raw
        # pre-softmax scores (see score_stats) -- if those scores are tiny,
        # softmax returns a near-uniform distribution and the attention is
        # effectively inactive, which is a real property of the trained model
        # rather than a measurement artifact.
        q = kwargs.get("query", args[0] if len(args) > 0 else None)
        k = kwargs.get("key", args[1] if len(args) > 1 else None)
        if q is not None and k is not None:
            self.qk_inputs.append((q.detach().cpu(), k.detach().cpu()))

    def attn_hook(self, module, input, output):
        # Patched (see force_per_head_attn below) to call with need_weights=True,
        # average_attn_weights=False -- fusion heads use num_heads=8
        # (config/model/multidino_attention_hashing_ortho.yaml via
        # main/models/multi_dino_attention.py's num_heads default), and the
        # nn.MultiheadAttention default (average_attn_weights=True) averages
        # across all 8 heads before returning, which can make genuinely
        # per-head-specialized attention look artificially uniform. Shape here
        # is [B, num_heads, L=num_queries, S=num_bands].
        attn_w = output[1]
        if attn_w is None:
            raise RuntimeError(
                "attn_weights is None -- the installed nn.MultiheadAttention must be "
                "called with need_weights=True (the default); check torch version."
            )
        if attn_w.dim() != 4:
            raise RuntimeError(
                f"Expected per-head attn_weights of shape [B, num_heads, L, S] (4D) -- "
                f"got shape {tuple(attn_w.shape)}. force_per_head_attn's monkeypatch of "
                f"average_attn_weights=False may not have taken effect; check the "
                f"installed torch version's nn.MultiheadAttention signature."
            )
        self.attn_weights.append(attn_w.detach().cpu())

    def make_projection_hook(self, band_idx):
        def hook(module, input, output):
            if len(self.band_embeds) <= band_idx:
                self.band_embeds.extend([[] for _ in range(band_idx + 1 - len(self.band_embeds))])
            self.band_embeds[band_idx].append(output.detach().cpu())
        return hook


def load_model_and_data(ckpt_path, set_name, data_dir):
    ckpt_path = resolve_ckpt_pattern(lib.expand_path(ckpt_path))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]

    getter = Getter()
    cfg.model.kwargs.with_autocast = False
    net = getter.get_model(cfg.model)
    net.load_state_dict(state["net_state"])
    net.eval()
    if torch.cuda.is_available():
        net.cuda()

    if data_dir is not None:
        cfg.dataset.kwargs.data_dir = lib.expand_path(data_dir)

    transform = getter.get_transform(cfg.transform.test)
    dts = getter.get_dataset(transform, set_name, cfg.dataset)
    # dataset getter returns {"test": ..., "gallery": ...} for MIRFlickrHashing at
    # mode="test" -- use whichever split was asked for (query images by default).
    if isinstance(dts, dict):
        dts = dts.get(set_name, dts.get("test"))

    ortho_weight = float(cfg.model.kwargs.fusion_config.ortho_weight)
    num_queries = int(cfg.model.kwargs.fusion_config.num_queries)
    return net, dts, ortho_weight, num_queries, state.get("epoch", "?")


def score_stats(attn_module, qk_inputs):
    """Recompute the raw pre-softmax attention scores exactly as
    nn.MultiheadAttention does internally: project q and k through in_proj,
    split into heads, scale by 1/sqrt(head_dim), then q @ k^T.

    The point is to distinguish two very different situations that both show
    up as "uniform attention" downstream:
      - scores have real spread but happen to average out -> attention is
        doing something, the summary was just hiding it;
      - scores are numerically tiny (|score| << 1) -> softmax over them is
        mathematically forced toward uniform (softmax of near-equal logits),
        i.e. the attention module is effectively inactive and the queries
        cannot differentiate the bands at all.

    Note the correct scaling here is 1/sqrt(head_dim) (what PyTorch applies),
    NOT an arbitrary constant -- multiplying raw scores by an ad-hoc factor
    will manufacture non-uniform attention that the trained model never
    actually computes.
    """
    embed_dim = attn_module.embed_dim
    num_heads = attn_module.num_heads
    head_dim = embed_dim // num_heads
    W = attn_module.in_proj_weight.detach().cpu()
    b = attn_module.in_proj_bias
    b = b.detach().cpu() if b is not None else torch.zeros(3 * embed_dim)

    W_q, W_k = W[:embed_dim], W[embed_dim:2 * embed_dim]
    b_q, b_k = b[:embed_dim], b[embed_dim:2 * embed_dim]

    all_scores = []
    for q, k in qk_inputs:
        qp = F.linear(q, W_q, b_q)   # [B, L, D]
        kp = F.linear(k, W_k, b_k)   # [B, S, D]
        B, L, _ = qp.shape
        S = kp.shape[1]
        qp = qp.view(B, L, num_heads, head_dim).transpose(1, 2)   # [B, H, L, hd]
        kp = kp.view(B, S, num_heads, head_dim).transpose(1, 2)   # [B, H, S, hd]
        scores = torch.matmul(qp, kp.transpose(-2, -1)) / (head_dim ** 0.5)  # [B, H, L, S]
        all_scores.append(scores)

    scores = torch.cat(all_scores, dim=0)
    # Spread *within* each softmax row is what actually determines whether the
    # output is uniform -- a large constant offset shared by every band cancels
    # out in softmax, so absolute magnitude alone isn't the right diagnostic.
    row_spread = (scores.max(dim=-1).values - scores.min(dim=-1).values)
    return {
        "mean_abs": scores.abs().mean().item(),
        "std": scores.std().item(),
        "min": scores.min().item(),
        "max": scores.max().item(),
        "mean_row_spread": row_spread.mean().item(),
        "max_row_spread": row_spread.max().item(),
    }


def force_per_head_attn(attn_module):
    """Monkeypatch attn_module.forward so every call is made with
    need_weights=True, average_attn_weights=False, regardless of what the
    fusion head's own forward() passes. Returns a restore() callback.
    """
    original_forward = attn_module.forward

    def patched_forward(*args, **kwargs):
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False
        return original_forward(*args, **kwargs)

    attn_module.forward = patched_forward
    return lambda: setattr(attn_module, "forward", original_forward)


def run_diagnostic(label, ckpt_path, set_name, data_dir, n_batches, bs, nw):
    net, dts, ortho_weight, num_queries, epoch = load_model_and_data(ckpt_path, set_name, data_dir)

    fusion_head = net.fusion_head if not hasattr(net, "module") else net.module.fusion_head
    restore_attn = force_per_head_attn(fusion_head.attn)
    capture = Capture()
    handles = [
        fusion_head.attn.register_forward_hook(capture.attn_hook),
        fusion_head.attn.register_forward_pre_hook(capture.attn_pre_hook, with_kwargs=True),
    ]
    for i, proj in enumerate(fusion_head.projections):
        handles.append(proj.register_forward_hook(capture.make_projection_hook(i)))

    loader = DataLoader(dts, batch_size=bs, shuffle=True, num_workers=nw)
    device = next(net.parameters()).device

    seen = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            net(x)
            seen += 1
            if seen >= n_batches:
                break

    for h in handles:
        h.remove()
    restore_attn()

    all_attn_per_head = torch.cat(capture.attn_weights, dim=0)  # [N, num_heads, num_queries, num_bands]
    n_heads = all_attn_per_head.shape[1]
    mean_attn_per_head = all_attn_per_head.mean(dim=0)  # [num_heads, num_queries, num_bands]

    # Do individual heads specialize on different bands, even if the
    # head-average (what average_attn_weights=True would have returned) looks
    # uniform? Per head, per-query LL share and entropy.
    eps = 1e-12
    per_head_entropy = -(mean_attn_per_head * (mean_attn_per_head + eps).log()).sum(dim=-1)  # [num_heads, num_queries]
    per_head_ll_share = mean_attn_per_head[:, :, 0].mean(dim=-1)  # [num_heads], averaged over queries

    # Head-averaged view, for continuity with what average_attn_weights=True
    # would have reported (and to compare against it directly).
    all_attn = all_attn_per_head.mean(dim=1)  # [N, num_queries, num_bands]
    mean_attn = all_attn.mean(dim=0)  # [num_queries, num_bands]
    n_q = mean_attn.shape[0]

    band_stack = [torch.cat(embeds, dim=0) for embeds in capture.band_embeds]  # 4 x [N, D]
    band_stack = torch.stack(band_stack, dim=0)  # [4, N, D]
    band_norm = F.normalize(band_stack, p=2, dim=-1)
    # pairwise cosine similarity between bands, averaged over the batch dimension
    cos_matrix = torch.einsum("and,bnd->abn", band_norm, band_norm).mean(dim=-1)  # [4, 4]
    off_diag = cos_matrix[~torch.eye(4, dtype=torch.bool)]

    # Do queries just all collapse onto the *same* attention pattern (real functional
    # redundancy, independent of whether that pattern happens to peak on LL), or do
    # they differ from each other even while individually favoring LL? Measured on the
    # per-sample attention rows (not the batch-mean) so it isn't washed out by averaging.
    q_norm = F.normalize(all_attn, p=2, dim=-1)  # [N, num_queries, num_bands]
    query_cos = torch.einsum("nqd,nrd->qrn", q_norm, q_norm).mean(dim=-1)  # [num_queries, num_queries]
    q_off_diag = query_cos[~torch.eye(n_q, dtype=torch.bool)]

    # Entropy of each query's mean attention distribution: log(4) = 1.386 is uniform
    # (no collapse), 0 is a one-hot spike on a single band (full collapse).
    eps = 1e-12
    per_query_entropy = -(mean_attn * (mean_attn + eps).log()).sum(dim=-1)  # [num_queries]

    return {
        "label": label,
        "epoch": epoch,
        "ortho_weight": ortho_weight,
        "num_queries": num_queries,
        "n_images": all_attn.shape[0],
        "n_heads": n_heads,
        "mean_attn": mean_attn,
        "cos_matrix": cos_matrix,
        "mean_band_cos": off_diag.mean().item(),
        "query_cos": query_cos,
        "mean_query_cos": q_off_diag.mean().item(),
        "per_query_entropy": per_query_entropy,
        "mean_attn_per_head": mean_attn_per_head,
        "per_head_entropy": per_head_entropy,
        "per_head_ll_share": per_head_ll_share,
        "score_stats": score_stats(fusion_head.attn, capture.qk_inputs),
        "query_token_norm": fusion_head.query_tokens.detach().cpu().norm(dim=-1).flatten(),
    }


def print_report(results):
    torch.set_printoptions(precision=3, sci_mode=False)
    for r in results:
        print("=" * 78)
        print(f"{r['label']}  (epoch={r['epoch']}, ortho_weight={r['ortho_weight']}, "
              f"num_queries={r['num_queries']}, n_heads={r['n_heads']}, n_images={r['n_images']})")
        print("-" * 78)
        ss = r["score_stats"]
        print("Raw pre-softmax attention scores (recomputed exactly as")
        print("nn.MultiheadAttention does: in_proj, split heads, scale by 1/sqrt(head_dim)):")
        print(f"  mean|score|={ss['mean_abs']:.4f}  std={ss['std']:.4f}  "
              f"range=[{ss['min']:.4f}, {ss['max']:.4f}]")
        print(f"  within-row spread (max-min across the 4 bands, what softmax actually sees):")
        print(f"    mean={ss['mean_row_spread']:.4f}  max={ss['max_row_spread']:.4f}")
        print(f"  -> a within-row spread << 1 mathematically forces near-uniform softmax:")
        print(f"     the attention module cannot differentiate the bands, regardless of ortho.")
        qn = r["query_token_norm"]
        print(f"  query_token L2 norms: " + ", ".join(f"{v:.4f}" for v in qn.tolist()))
        print()
        print("Per-head LL share and entropy (averaged over queries) -- checks whether")
        print("individual heads specialize on different bands even if the head-averaged")
        print("view below (what average_attn_weights=True, the nn.MultiheadAttention")
        print("default, would have reported) looks uniform:")
        for hi in range(r["mean_attn_per_head"].shape[0]):
            ll = r["per_head_ll_share"][hi].item()
            ent_mean = r["per_head_entropy"][hi].mean().item()
            print(f"head{hi}   LL share={ll:.3f}   mean entropy={ent_mean:.3f} (max={torch.log(torch.tensor(4.0)).item():.3f})")
        head_ll_shares = r["per_head_ll_share"]
        print(f"-> LL share across heads: min={head_ll_shares.min().item():.3f}, "
              f"max={head_ll_shares.max().item():.3f}, std={head_ll_shares.std().item():.4f} "
              f"(near-zero std means heads really do behave alike, not just the average)")
        print()
        print("Head-averaged view (equivalent to average_attn_weights=True):")
        print("Attention mass: rows = query, cols = subband [LL, LH, HL, HH]")
        header = "        " + "  ".join(f"{b:>6}" for b in BAND_NAMES)
        print(header)
        for qi, row in enumerate(r["mean_attn"]):
            print(f"query{qi}  " + "  ".join(f"{v:6.3f}" for v in row.tolist())
                  + f"   entropy={r['per_query_entropy'][qi].item():.3f} (max={torch.log(torch.tensor(4.0)).item():.3f})")
        ll_share = r["mean_attn"][:, 0].mean().item()
        print(f"-> mean mass on LL across all queries: {ll_share:.3f}  "
              f"(1/4 = 0.250 would be uniform)")
        print()
        print("Inter-query redundancy: cosine similarity between queries' per-sample")
        print("attention rows (not the batch mean) -- low off-diagonal means queries")
        print("differ from each other even if each individually favors LL; high means")
        print("real functional collapse (queries behave identically, not just LL-heavy).")
        for qi, row in enumerate(r["query_cos"]):
            print(f"query{qi}  " + "  ".join(f"{v:6.3f}" for v in row.tolist()))
        print(f"-> mean |off-diagonal| inter-query cosine similarity: {r['mean_query_cos']:.4f}")
        print()
        print("Per-band read-out diversity (KV projections, before attention):")
        print("cosine similarity matrix [LL, LH, HL, HH] x [LL, LH, HL, HH]")
        for bi, row in enumerate(r["cos_matrix"]):
            print(f"{BAND_NAMES[bi]:>4}  " + "  ".join(f"{v:6.3f}" for v in row.tolist()))
        print(f"-> mean |off-diagonal| band cosine similarity: {r['mean_band_cos']:.4f}  "
              f"(lower = more diverse read-out content)")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", action="append", required=True,
                         help="label=path to a checkpoint, repeatable")
    parser.add_argument("--set", type=str, default="test", help="test (query) or gallery/database split")
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset.kwargs.data_dir")
    parser.add_argument("--n-batches", type=int, default=5, help="Number of batches to average over")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--nw", type=int, default=4)
    args = parser.parse_args()

    results = []
    for spec in args.ckpt:
        if "=" not in spec:
            sys.exit(f"--ckpt must be 'label=path', got: {spec}")
        label, path = spec.split("=", 1)
        print(f"Loading {label} <- {path} ...")
        results.append(run_diagnostic(label, path, args.set, args.data_dir, args.n_batches, args.bs, args.nw))

    print_report(results)


if __name__ == "__main__":
    main()
