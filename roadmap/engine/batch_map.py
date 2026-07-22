import random

import torch

from .accuracy_calculator import CustomCalculator
from .make_subset import make_subset


def build_batch_map_calculator(distance_metric, device):
    """Lightweight CustomCalculator for a cheap self-retrieval mAP estimate on a
    single minibatch (query == reference == the batch itself)."""
    metric_name = "maphashing" if distance_metric == "hamming" else "map"
    calculator = CustomCalculator(
        exclude=["NMI", "AMI"],
        k="max_bin_count",
        with_faiss=False,
        distance_metric=distance_metric,
        device=device,
    )
    return calculator, metric_name


def compute_batch_map(calculator, metric_name, embeddings, labels):
    with torch.no_grad():
        embeddings = embeddings.detach()
        if labels.ndim == 2 and labels.size(1) == 1:
            labels = labels.view(-1)
        result = calculator.get_accuracy(
            query=embeddings,
            query_labels=labels,
            reference=embeddings,
            reference_labels=labels,
            embeddings_come_from_same_source=True,
            include=[metric_name],
        )
    return result[metric_name]


def build_fast_eval_subset(dataset, size, min_per_class=2, seed=0):
    """Fixed, stratified self-retrieval subsample of `dataset`, built once so
    it can be re-evaluated cheaply and consistently across epochs."""
    rng = random.Random(seed)

    label_to_idx = {}
    for idx, label in enumerate(dataset.labels):
        label_to_idx.setdefault(label, []).append(idx)

    eligible_classes = [idx_list for idx_list in label_to_idx.values() if len(idx_list) >= min_per_class]
    rng.shuffle(eligible_classes)

    selected = []
    for idx_list in eligible_classes:
        if len(selected) >= size:
            break
        selected.extend(idx_list)
    selected = selected[:size]

    subset = make_subset(dataset, selected)
    if hasattr(subset, "_at_R"):
        del subset._at_R

    return subset
