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
    it can be re-evaluated cheaply and consistently across epochs.

    Groups by `dataset.instance_dict` ({class/tag index: [image indices]}),
    which every dataset class already builds correctly via get_instance_dict()
    (see VOC2012Hashing / MIRFlickrHashing). This does NOT group by
    `dataset.labels` directly: labels are multi-hot float tensors for
    multi-label datasets (VOC, MIRFLICKR), and a tensor's default `__hash__`
    is identity-based, not value-based -- grouping by raw label silently
    turned every image into its own singleton "class" (nothing satisfied
    `len(idx_list) >= min_per_class`), leaving `eligible_classes` empty and
    crashing downstream in `compute_all_embeddings` with
    `UnboundLocalError: all_q` on an empty dataloader.

    A single image can belong to multiple groups here (multiple active tags),
    so selection is deduplicated as it goes.
    """
    rng = random.Random(seed)

    if not hasattr(dataset, "instance_dict"):
        raise AttributeError(
            f"{type(dataset).__name__} has no `instance_dict` -- build_fast_eval_subset "
            "needs {class_or_tag_idx: [image_indices]} grouping (see get_instance_dict() "
            "on VOC2012Hashing / MIRFlickrHashing for the expected shape)."
        )

    eligible_groups = [idx_list for idx_list in dataset.instance_dict.values() if len(idx_list) >= min_per_class]
    rng.shuffle(eligible_groups)

    selected = []
    seen = set()
    for idx_list in eligible_groups:
        if len(selected) >= size:
            break
        for idx in idx_list:
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)
    selected = selected[:size]

    if not selected:
        raise ValueError(
            f"build_fast_eval_subset found no eligible groups (>= {min_per_class} members) "
            f"in {type(dataset).__name__}.instance_dict -- fast_eval_freq should be disabled "
            "(-1) for this dataset rather than silently evaluating on an empty subset."
        )

    subset = make_subset(dataset, selected)
    if hasattr(subset, "_at_R"):
        del subset._at_R

    return subset
