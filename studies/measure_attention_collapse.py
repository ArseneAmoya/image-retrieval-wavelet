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

BAND_NAMES = ["LL", "LH", "HL", "HH"]


class Capture:
    """Accumulates attn_weights and per-band projected embeddings across batches."""

    def __init__(self):
        self.attn_weights = []   # list of [B, num_queries, num_bands]
        self.band_embeds = []    # list of [num_bands][B, D] -> transposed at the end

    def attn_hook(self, module, input, output):
        # nn.MultiheadAttention forward -> (attn_output, attn_weights); need_weights
        # defaults to True and average_attn_weights defaults to True, so weights come
        # back already averaged over heads, shape [B, L=num_queries, S=num_bands].
        attn_w = output[1]
        if attn_w is None:
            raise RuntimeError(
                "attn_weights is None -- the installed nn.MultiheadAttention must be "
                "called with need_weights=True (the default); check torch version."
            )
        self.attn_weights.append(attn_w.detach().cpu())

    def make_projection_hook(self, band_idx):
        def hook(module, input, output):
            if len(self.band_embeds) <= band_idx:
                self.band_embeds.extend([[] for _ in range(band_idx + 1 - len(self.band_embeds))])
            self.band_embeds[band_idx].append(output.detach().cpu())
        return hook


def load_model_and_data(ckpt_path, set_name, data_dir):
    state = torch.load(lib.expand_path(ckpt_path), map_location="cpu", weights_only=False)
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


def run_diagnostic(label, ckpt_path, set_name, data_dir, n_batches, bs, nw):
    net, dts, ortho_weight, num_queries, epoch = load_model_and_data(ckpt_path, set_name, data_dir)

    fusion_head = net.fusion_head if not hasattr(net, "module") else net.module.fusion_head
    capture = Capture()
    handles = [fusion_head.attn.register_forward_hook(capture.attn_hook)]
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

    all_attn = torch.cat(capture.attn_weights, dim=0)  # [N, num_queries, num_bands]
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
        "mean_attn": mean_attn,
        "cos_matrix": cos_matrix,
        "mean_band_cos": off_diag.mean().item(),
        "query_cos": query_cos,
        "mean_query_cos": q_off_diag.mean().item(),
        "per_query_entropy": per_query_entropy,
    }


def print_report(results):
    torch.set_printoptions(precision=3, sci_mode=False)
    for r in results:
        print("=" * 78)
        print(f"{r['label']}  (epoch={r['epoch']}, ortho_weight={r['ortho_weight']}, "
              f"num_queries={r['num_queries']}, n_images={r['n_images']})")
        print("-" * 78)
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
