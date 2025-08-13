# retrieval_report_fixed.py
"""
Compare two retrieval models (SmoothAP vs Cross-Entropy) on CIFAR using FAISS (CPU).
- Extract embeddings with each model
- Build two FAISS indices (cosine)
- Score per-query (AP, R-precision, P@k)
- Select 10 queries where SmoothAP dominates (with class diversity)
- Save side-by-side panels: top-k retrieved (row1=SmoothAP, row2=CE)

Usage (example):
python retrieval_report_fixed.py \
  --model1-dir /path/to/smoothap.ckpt \
  --model2-dir /path/to/crossentropy.ckpt \
  --bs 256 --nw 8 \
  --out-dir retrieval_panels --k-vis 10 --n-queries 10 \
  --device auto --metric cosine --min-margin 0.10

Notes:
- model1 is assumed to be the SmoothAP model; model2 is the CE model.
- We try to fetch class names from the dataset if available (dts.classes).
"""

import os
import csv
import faiss
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Optional

import argparse

from roadmap.getter import Getter
import roadmap.utils as lib

# ====== CONFIG ======
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)


def torch_l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, dim=1)


def extract_embeddings(model, dataloader, device="cpu", normalize=True):
    model.to(device).eval()
    feats, labels, imgs = [], [], []
    # inference_mode is slightly faster than no_grad
    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            f = model(x)                     # shape: [B, D] or tuple(..., embedding)
            if isinstance(f, (list, tuple)):
                f = f[0]
            f = f.detach().cpu()
            feats.append(f)
            labels.append(y.clone())
            imgs.append(x.detach().cpu())
    feats = torch.cat(feats, dim=0)
    if normalize:
        feats = torch_l2_normalize(feats)
    labels = torch.cat(labels, dim=0).long()
    imgs = torch.cat(imgs, dim=0)           # tensors normalisés CIFAR
    return feats.numpy().astype("float32"), labels.numpy(), imgs


def build_faiss_index(embeds: np.ndarray, metric="cosine"):
    d = embeds.shape[1]
    xb = embeds.copy()
    if metric == "cosine":
        faiss.normalize_L2(xb)              # inner product <=> cosine
        index = faiss.IndexFlatIP(d)
    elif metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError("metric must be 'cosine' or 'l2'")
    index.add(xb)
    return index


def faiss_search(index, queries: np.ndarray, k: int, metric="cosine"):
    xq = queries.copy()
    if metric == "cosine":
        faiss.normalize_L2(xq)
    D, I = index.search(xq, k)
    return D, I


def average_precision_from_relevances(rel: np.ndarray) -> float:
    """rel: array binaire ordonnée par rang (1 si pertinent)."""
    idx = np.where(rel == 1)[0]
    if len(idx) == 0:
        return 0.0
    precisions = [(rel[:i+1].sum()) / (i+1) for i in idx]
    return float(np.mean(precisions))


def r_precision(rel: np.ndarray, R: int) -> float:
    if R <= 0: return 0.0
    R = min(R, len(rel))
    return float(rel[:R].sum() / R)


def precisions_at_k(rel: np.ndarray, ks=(1,5,10,20)) -> Dict[int, float]:
    out = {}
    for k in ks:
        k = min(k, len(rel))
        out[k] = float(rel[:k].sum() / k)
    return out


def count_class_occurrences(labels, cls):
    return int((labels == cls).sum())


def unnormalize(img_t: torch.Tensor, mean=CIFAR_MEAN, std=CIFAR_STD):
    m = torch.tensor(mean).view(3,1,1)
    s = torch.tensor(std).view(3,1,1)
    return img_t * s + m


def plot_panel(query_img, gt_text, row1_imgs, row1_ok, row2_imgs, row2_ok,
               title1="SmoothAP", title2="CrossEntropy", k=10, out_path=None):
    # figure: query on the left, then k columns for results (2 rows)
    cols = k
    fig_h = 3.2
    fig_w = max(10, k*1.2)
    fig = plt.figure(figsize=(fig_w, fig_h))
    plt.suptitle(f"Query: {gt_text}", y=0.99)

    def to_np(img):
        img = img.detach().cpu()
        img = unnormalize(img).clamp(0,1)
        return np.transpose(img.numpy(), (1,2,0))

    # Query inset
    axq = plt.axes([0.01, 0.25, 0.10, 0.5])
    axq.imshow(to_np(query_img))
    axq.set_title("Query")
    axq.axis("off")

    # Row1: SmoothAP
    for i in range(cols):
        ax = plt.axes([0.13 + i*(0.86/cols), 0.55, (0.86/cols), 0.4])
        ax.imshow(to_np(row1_imgs[i]))
        ax.set_title(f"{title1} #{i+1}", fontsize=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_linewidth(3)
            spine.set_color("green" if row1_ok[i] else "red")

    # Row2: Cross-Entropy
    for i in range(cols):
        ax = plt.axes([0.13 + i*(0.86/cols), 0.08, (0.86/cols), 0.4])
        ax.imshow(to_np(row2_imgs[i]))
        ax.set_title(f"{title2} #{i+1}", fontsize=8)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_linewidth(3)
            spine.set_color("green" if row2_ok[i] else "red")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def per_query_eval(db_labels: np.ndarray,
                   retrieved_ids: np.ndarray,
                   q_idx: int,
                   topk: int) -> Dict[str, float]:
    # Retire TOUTE occurrence du self-match (sécurité) puis tronque à topk
    ranks = retrieved_ids.copy()
    ranks = ranks[ranks != q_idx]
    ranks = ranks[:topk]
    gt = db_labels[q_idx]
    rel = (db_labels[ranks] == gt).astype(np.int32)
    ap = average_precision_from_relevances(rel)
    R = count_class_occurrences(db_labels, gt) - 1  # on exclut la requête
    rprec = r_precision(rel, R)
    pks = precisions_at_k(rel, ks=(1,5,10,20))
    out = {"AP": ap, "R": rprec}
    out.update({f"P@{k}": v for k,v in pks.items()})
    out["hits"] = int(rel.sum())
    return out


def diversify(indices: List[int], labels: np.ndarray, per_class_limit=2) -> List[int]:
    bucket = defaultdict(int)
    kept = []
    for i in indices:
        c = int(labels[i])
        if bucket[c] < per_class_limit:
            kept.append(i)
            bucket[c] += 1
    return kept


def save_selection_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def run_report(
    test_loader,
    smoothap_model,
    ce_model,
    out_dir: str = "retrieval_panels",
    k_vis: int = 10,
    n_queries: int = 10,
    metric: str = "cosine",
    device: str = "cpu",
    min_margin: float = 0.10,   # marge AP minimale pour mettre en valeur SmoothAP
    per_class_limit: int = 2,
    class_names: Optional[List[str]] = None
):
    # 1) Embeddings
    embeds1, labels, imgs = extract_embeddings(smoothap_model, test_loader, device=device, normalize=(metric=="cosine"))
    embeds2, _labels2, _ = extract_embeddings(ce_model,       test_loader, device=device, normalize=(metric=="cosine"))
    assert np.array_equal(labels, _labels2), "Les labels doivent correspondre entre les deux passages."
    N = len(labels)

    # 2) Indices FAISS (DB = tout le split)
    idx1 = build_faiss_index(embeds1, metric=metric)
    idx2 = build_faiss_index(embeds2, metric=metric)

    # 3) Recherche top-K large (borné par N)
    K_eval = min(max(200, k_vis + 50), N)
    D1, I1 = faiss_search(idx1, embeds1, K_eval, metric=metric)
    D2, I2 = faiss_search(idx2, embeds2, K_eval, metric=metric)

    # 4) Scores par requête + deltas
    perq_1, perq_2, deltas = [], [], []
    for qi in range(N):
        m1q = per_query_eval(labels, I1[qi], qi, topk=K_eval-1)
        m2q = per_query_eval(labels, I2[qi], qi, topk=K_eval-1)
        perq_1.append(m1q)
        perq_2.append(m2q)
        deltas.append({
            "q": qi,
            "label": int(labels[qi]),
            "dAP": m1q["AP"] - m2q["AP"],
            "dR":  m1q["R"]  - m2q["R"],
            "dP@1": m1q["P@1"] - m2q["P@1"],
            "dP@5": m1q["P@5"] - m2q["P@5"],
            "dP@10": m1q["P@10"] - m2q["P@10"]
        })

    # 5) Ordre: dAP puis dP@5, dP@1
    order = sorted(range(N),
                   key=lambda i: (deltas[i]["dAP"], deltas[i]["dP@5"], deltas[i]["dP@1"]),
                   reverse=True)
    # filtre marge minimale
    order_margin = [i for i in order if deltas[i]["dAP"] >= min_margin]
    # diversité par classe
    chosen = diversify(order_margin, labels, per_class_limit=per_class_limit)

    # fallback si pas assez d'éléments: relaxer diversité puis marge
    if len(chosen) < n_queries:
        # relaxer diversité (reprendre au-delà de per_class_limit)
        extra = [i for i in order_margin if i not in chosen]
        chosen += extra
    if len(chosen) < n_queries:
        # relaxer la marge (prendre le meilleur restant)
        extra2 = [i for i in order if i not in chosen]
        chosen += extra2
    chosen = chosen[:n_queries]

    # 6) Panneaux visuels + CSV
    os.makedirs(out_dir, exist_ok=True)
    csv_rows = []
    for qi in chosen:
        qimg = imgs[qi]
        # ids récupérés (numpy) -> torch index pour imgs
        row1_ids = I1[qi]
        row2_ids = I2[qi]
        # Enlever toutes les occurrences du self-match
        row1_ids = row1_ids[row1_ids != qi][:k_vis]
        row2_ids = row2_ids[row2_ids != qi][:k_vis]

        # torch indexing
        row1_ids_t = torch.from_numpy(row1_ids).long()
        row2_ids_t = torch.from_numpy(row2_ids).long()

        gt = int(labels[qi])
        gt_text = (class_names[gt] if (class_names and gt < len(class_names)) else str(gt))
        row1_ok = (labels[row1_ids] == gt).tolist()
        row2_ok = (labels[row2_ids] == gt).tolist()

        plot_panel(
            query_img=qimg,
            gt_text=gt_text,
            row1_imgs=imgs[row1_ids_t],
            row1_ok=row1_ok,
            row2_imgs=imgs[row2_ids_t],
            row2_ok=row2_ok,
            k=k_vis,
            out_path=os.path.join(out_dir, f"q{qi}_cls{gt}_panel.png")
        )

        r = {
            "q_idx": qi,
            "label_id": gt,
            "label_name": gt_text,
            "SmoothAP_AP": perq_1[qi]["AP"],
            "CE_AP": perq_2[qi]["AP"],
            "dAP": deltas[qi]["dAP"],
            "SmoothAP_R": perq_1[qi]["R"],
            "CE_R": perq_2[qi]["R"],
            "dR": deltas[qi]["dR"],
            "SmoothAP_P@1": perq_1[qi]["P@1"],
            "CE_P@1": perq_2[qi]["P@1"],
            "dP@1": deltas[qi]["dP@1"],
            "SmoothAP_P@5": perq_1[qi]["P@5"],
            "CE_P@5": perq_2[qi]["P@5"],
            "dP@5": deltas[qi]["dP@5"],
            "SmoothAP_P@10": perq_1[qi]["P@10"],
            "CE_P@10": perq_2[qi]["P@10"],
            "dP@10": deltas[qi]["dP@10"],
        }
        csv_rows.append(r)

    save_selection_csv(os.path.join(out_dir, "selection_summary.csv"), csv_rows)

    print(f"==> {len(chosen)} requêtes sélectionnées (sur {N}).")
    for qi in chosen:
        print(f"[q={qi} | y={labels[qi]}] dAP={deltas[qi]['dAP']:.3f} "
              f"dR={deltas[qi]['dR']:.3f} dP@1={deltas[qi]['dP@1']:.3f} "
              f"dP@5={deltas[qi]['dP@5']:.3f}")

    return {
        "chosen": chosen,
        "per_query_smoothap": perq_1,
        "per_query_ce": perq_2,
        "deltas": deltas
    }


def decide_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1-dir", type=str, required=True, help="path to SmoothAP checkpoint")
    parser.add_argument("--model2-dir", type=str, required=True, help="path to Cross-Entropy checkpoint")
    parser.add_argument("--bs", type=int, default=256, help="Batch size for DataLoader")
    parser.add_argument("--nw", type=int, default=8, help="Num workers for DataLoader")
    parser.add_argument("--out-dir", type=str, default="retrieval_panels")
    parser.add_argument("--k-vis", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=10)
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "l2"])
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--min-margin", type=float, default=0.10)
    args = parser.parse_args()

    device = decide_device(args.device)

    # Logs
    lib.LOGGER.info(f"Evaluating: {args.model1_dir} (SmoothAP)  vs  {args.model2_dir} (CE)")

    # Load states
    path1 = lib.expand_path(args.model1_dir)
    path2 = lib.expand_path(args.model2_dir)
    state1 = torch.load(path1, map_location='cpu')
    state2 = torch.load(path2, map_location='cpu')
    cfg1 = state1["config"]
    cfg2 = state2["config"]

    # Models
    getter = Getter()
    lib.LOGGER.info(f"Loading models from {path1} and {path2}")
    net1 = getter.get_model(cfg1.model)
    net2 = getter.get_model(cfg2.model)
    net1.load_state_dict(state1["net_state"])
    net2.load_state_dict(state2["net_state"])

    if torch.cuda.is_available() and device.startswith("cuda"):
        if torch.cuda.device_count() > 1:
            net1 = torch.nn.DataParallel(net1).cuda()
            net2 = torch.nn.DataParallel(net2).cuda()
        else:
            net1 = net1.cuda()
            net2 = net2.cuda()
    net1.eval()
    net2.eval()

    # Data
    transform = getter.get_transform(cfg1.transform.test)
    dts = getter.get_dataset(transform, "test", cfg1.dataset)
    dataloader = torch.utils.data.DataLoader(
        dts,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.nw,
        pin_memory=True,
    )

    class_names = None
    if hasattr(dts, "classes"):
        class_names = list(dts.classes)

    run_report(
        test_loader=dataloader,
        smoothap_model=net1,
        ce_model=net2,
        out_dir=args.out_dir,
        k_vis=args.k_vis,
        n_queries=args.n_queries,
        metric=args.metric,
        device=device,
        min_margin=args.min_margin,
        class_names=class_names
    )


if __name__ == "__main__":
    main()
