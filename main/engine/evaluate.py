import os
from collections import defaultdict

import torch
from pytorch_metric_learning import testers
import pytorch_metric_learning.utils.common_functions as c_f
from tqdm import tqdm

import main.utils as lib
from .accuracy_calculator import get_accuracy_calculator


class GlobalEmbeddingSpaceTester(testers.GlobalEmbeddingSpaceTester):

    def label_levels_to_evaluate(self, query_labels):
        num_levels_available = query_labels.shape[1]
        if self.label_hierarchy_level == "all":
            return range(num_levels_available)
        elif isinstance(self.label_hierarchy_level, int):
            assert self.label_hierarchy_level < num_levels_available
            return [self.label_hierarchy_level]
        elif c_f.is_list_or_tuple(self.label_hierarchy_level):
            # assert max(self.label_hierarchy_level) < num_levels_available
            return self.label_hierarchy_level

    def compute_all_embeddings(self, dataloader, trunk_model, embedder_model):
        if len(dataloader.dataset) == 0:
            raise ValueError(
                "compute_all_embeddings got an empty dataset -- this used to fail later "
                "with an opaque 'UnboundLocalError: all_q' once the loop below ran zero "
                "iterations. Check whatever built this split (e.g. build_fast_eval_subset)."
            )
        s, e = 0, 0
        with torch.no_grad():
            lib.LOGGER.info("Computing embeddings")
            # added the option of disabling TQDM
            for i, data in enumerate(tqdm(dataloader, disable=os.getenv('TQDM_DISABLE'))):
                img, label = self.data_and_label_getter(data)
                #print(f"Batch {i} - {img.shape} - {label.shape}")
                label = c_f.process_label(label, "all", self.label_mapper)
                q = self.get_embeddings_for_eval(trunk_model, embedder_model, img)
                q = q.cpu()
                label = label.cpu()
                if label.dim() == 1:
                    label = label.unsqueeze(1)
                if i == 0:
                    labels = torch.zeros(
                        len(dataloader.dataset),
                        label.size(1),
                        device=torch.device("cpu"), #self.data_device,
                        dtype=label.dtype,
                    )
                    all_q = torch.zeros(
                        len(dataloader.dataset),
                        q.size(1),
                        device=torch.device("cpu"), #self.data_device,
                        dtype=q.dtype,
                    )
                #print(f"Batch {i} - {q.shape} - {label.shape}")
                e = s + q.size(0)
                all_q[s:e] = q
                labels[s:e] = label
                s = e
        return all_q, labels


def get_tester(
    normalize_embeddings=False,
    batch_size=64,
    num_workers=16,
    pca=None,
    exclude_ranks=None,
    k=5000,
    **kwargs,
):
    calculator = get_accuracy_calculator(
        exclude_ranks=exclude_ranks,
        k=k,
        device=torch.device("cpu"),
        **kwargs,
    )

    return GlobalEmbeddingSpaceTester(
        normalize_embeddings=normalize_embeddings,
        data_and_label_getter=get_data,
        batch_size=batch_size,
        dataloader_num_workers=num_workers,
        accuracy_calculator=calculator,
        data_device=None,
        pca=pca,
    )
def get_data(batch):
    return batch["image"].cuda(), batch["label"]

def _build_dataset_dict_and_splits(train_dataset, val_dataset, test_dataset, custom_eval):
    """Factored out of evaluate() so evaluate_multi_k() builds the exact same
    dataset_dict/splits_to_eval without duplicating (and risking drift from)
    this branching logic."""
    at_R = 0
    dataset_dict = {}
    splits_to_eval = []
    if train_dataset is not None:
        dataset_dict["train"] = train_dataset
        splits_to_eval.append(('train', ['train']))
        at_R = max(at_R, train_dataset.my_at_R)

    if val_dataset is not None:
        dataset_dict["val"] = val_dataset
        splits_to_eval.append(('val', ['val']))
        at_R = max(at_R, val_dataset.my_at_R)

    if test_dataset is not None:
        if isinstance(test_dataset, dict):
            if 'gallery' in test_dataset:
                dataset_dict.update(test_dataset)
                splits_to_eval.append(('test', ['gallery']))
                at_R = max(at_R, test_dataset['test'].my_at_R, test_dataset['gallery'].my_at_R)
            elif 'distractor' in test_dataset:
                dataset_dict.update(test_dataset)
                splits_to_eval.append(('test', ['test', 'distractor']))
                at_R = max(at_R, test_dataset['test'].my_at_R, test_dataset['distractor'].my_at_R)
        elif isinstance(test_dataset, list):
            for dts in test_dataset:
                dataset_dict.update(dts)
                names = list(dts.keys())
                at_R = max(at_R, list(dts.values())[0].my_at_R, list(dts.values())[1].my_at_R)
                splits_to_eval.append((
                    names[0] if names[0].startswith("query") else names[1],
                    [names[0] if names[0].startswith("gallery") else names[1]]
                ))
        else:
            dataset_dict["test"] = test_dataset
            splits_to_eval.append(('test', ['test']))
            at_R = max(at_R, test_dataset.my_at_R)

    if custom_eval is not None:
        dataset_dict = custom_eval["dataset"]
        splits_to_eval = custom_eval["splits"]

    return dataset_dict, splits_to_eval, at_R


@lib.get_set_random_state
def evaluate(
    net,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    epoch=None,
    tester=None,
    custom_eval=None,
    **kwargs
):
    dataset_dict, splits_to_eval, at_R = _build_dataset_dict_and_splits(
        train_dataset, val_dataset, test_dataset, custom_eval,
    )

    if tester is None:
        # next lines usefull when computing only the mAP@R and small recall values
        # if ('k' not in kwargs) and (at_R != 0):
        #     kwargs["k"] = at_R + 1
        tester = get_tester(**kwargs)

    return tester.test(
        dataset_dict=dataset_dict,
        epoch=f"{epoch}",
        trunk_model=net,
        splits_to_eval=splits_to_eval,
    )


@lib.get_set_random_state
def evaluate_multi_k(
    net,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    epoch=None,
    custom_eval=None,
    k_list=(5000,),
    **kwargs
):
    """
    Same dataset_dict/splits_to_eval construction as evaluate(), but computes
    embeddings (the expensive DINOv2 forward pass over the whole query+gallery
    set) exactly ONCE and reuses them for every k in k_list, rebuilding only
    the (cheap: knn + argsort over already-computed embeddings) accuracy
    calculator per k.

    Added 2026-08-18 for COCO's mAP@5000-and-mAP@ALL protocol: running
    evaluate_all_checkpoints.py --k 5000 then --k <db_size> as two separate
    invocations would re-embed the same checkpoint's full test+gallery set
    twice for no reason. With this, one call to load_and_evaluate_multi_k
    covers both.

    Returns {k: {split_name: {metric_name: value, ...}, ...}, ...} -- one
    full metrics dict per k, same shape load_and_evaluate()/evaluate() return
    for a single k.
    """
    dataset_dict, splits_to_eval, _at_R = _build_dataset_dict_and_splits(
        train_dataset, val_dataset, test_dataset, custom_eval,
    )

    k_list = list(k_list)
    tester = get_tester(k=k_list[0], **kwargs)
    trunk_model = net
    embedder_model = torch.nn.Identity()
    trunk_model.eval()
    embedder_model.eval()

    splits_to_eval, splits_to_compute_embeddings = tester.get_splits_to_compute_embeddings(
        dataset_dict, splits_to_eval,
    )
    lib.LOGGER.info(f"Computing embeddings once, reused for k in {k_list}")
    embeddings_and_labels = tester.get_all_embeddings_for_all_splits(
        dataset_dict, trunk_model, embedder_model, splits_to_compute_embeddings, None,
    )

    # Rebuilding a whole tester per k just to steal its accuracy_calculator is
    # wasteful-looking but is actually cheap (no GPU work, no forward pass)
    # and guarantees this stays in sync with get_tester()'s own kwarg-splitting
    # logic instead of duplicating it here.
    calculators_by_k = {k: get_tester(k=k, **kwargs).accuracy_calculator for k in k_list}

    results_by_k = {}
    for k, calculator in calculators_by_k.items():
        tester.accuracy_calculator = calculator
        all_accuracies = defaultdict(dict)
        for query_split_name, reference_split_names in splits_to_eval:
            all_accuracies[query_split_name]["epoch"] = f"{epoch}"
            tester.reference_split_names[query_split_name] = reference_split_names
            tester.do_knn_and_accuracies(
                all_accuracies[query_split_name],
                embeddings_and_labels,
                query_split_name,
                reference_split_names,
            )
        results_by_k[k] = dict(all_accuracies)
        lib.LOGGER.info(f"k={k} done: " + ", ".join(
            f"{split}.{metric}={val:.4f}" if isinstance(val, float) else f"{split}.{metric}={val}"
            for split, mtrc in all_accuracies.items() for metric, val in mtrc.items() if metric != "epoch"
        ))

    del embeddings_and_labels
    return results_by_k
