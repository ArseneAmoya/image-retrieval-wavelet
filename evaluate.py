import logging
import argparse

import torch
import numpy as np

from main.getter import Getter
import main.utils as lib
import main.engine as eng


def load_and_evaluate(
    path,
    set,
    bs,
    nw,
    data_dir=None,
    **kwargs
):
    lib.LOGGER.info(f"Evaluating : \033[92m{path}\033[0m")
    state = torch.load(lib.expand_path(path), map_location='cpu', weights_only=False)
    cfg = state["config"]

    lib.LOGGER.info("Loading model...")
    cfg.model.kwargs.with_autocast = True
    net = Getter().get_model(cfg.model)
    net.load_state_dict(state["net_state"])
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.cuda()
    net.eval()

    if data_dir is not None:
        cfg.dataset.kwargs.data_dir = lib.expand_path(data_dir)

    getter = Getter()
    transform = getter.get_transform(cfg.transform.test)
    if hasattr(cfg.experience, 'split') and (cfg.experience.split is not None):
        assert isinstance(cfg.experience.split, int)
        dts = getter.get_dataset(None, 'all', cfg.dataset)
        splits = eng.get_splits(dts.labels, dts.super_labels, cfg.experience.kfold, random_state=cfg.experience.split_random_state)
        dts = eng.make_subset(dts, splits[cfg.experience.split]['train' if set == 'train' else 'val'], transform, set)
        lib.LOGGER.info(dts)
    else:
        dts = getter.get_dataset(transform, set, cfg.dataset)

    lib.LOGGER.info("Dataset created...")

    metrics = eng.evaluate(
        net=net,
        test_dataset=dts,
        epoch=state["epoch"],
        batch_size=bs,
        num_workers=nw,
        exclude=["mean_reciprocal_rank", "mean_average_precision",
                              "precision_at_1","recall_at_1","r_precision",'rpr', 'pr_rc', "recall_at_1000", "recall_at_100",
                              "recall_at_10", "recall_at_16", "recall_at_20", "recall_at_30", "recall_at_32", "recall_at_4", "recall_at_8",
                                "recall_at_2", "recall_at_10"],
        k=kwargs.get('k', 5000),
        distance_metric=kwargs.get('distance_metric', 'cosine')
    )

    lib.LOGGER.info("Evaluation completed...")
    for split, mtrc in metrics.items():
        for k, v in mtrc.items():
            if k == 'epoch':
                continue
            lib.LOGGER.info(f"{split} --> {k} : {np.around(v*100, decimals=2)}")

    return metrics


def load_and_evaluate_multi_k(
    path,
    set,
    bs,
    nw,
    data_dir=None,
    k_list=(5000,),
    **kwargs
):
    """
    Same checkpoint/model/dataset loading as load_and_evaluate(), but calls
    eng.evaluate_multi_k() instead of eng.evaluate() so the embedding forward
    pass runs once and every k in k_list is evaluated off the same
    embeddings. Use this instead of calling load_and_evaluate() once per k
    (e.g. for COCO's mAP@5000 + mAP@ALL, which need two different k's on the
    same checkpoint).

    Returns {k: metrics_dict, ...} -- metrics_dict has the same shape
    load_and_evaluate() returns for a single k.
    """
    lib.LOGGER.info(f"Evaluating (multi-k={list(k_list)}) : \033[92m{path}\033[0m")
    state = torch.load(lib.expand_path(path), map_location='cpu', weights_only=False)
    cfg = state["config"]

    lib.LOGGER.info("Loading model...")
    cfg.model.kwargs.with_autocast = True
    net = Getter().get_model(cfg.model)
    net.load_state_dict(state["net_state"])
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.cuda()
    net.eval()

    if data_dir is not None:
        cfg.dataset.kwargs.data_dir = lib.expand_path(data_dir)

    getter = Getter()
    transform = getter.get_transform(cfg.transform.test)
    if hasattr(cfg.experience, 'split') and (cfg.experience.split is not None):
        assert isinstance(cfg.experience.split, int)
        dts = getter.get_dataset(None, 'all', cfg.dataset)
        splits = eng.get_splits(dts.labels, dts.super_labels, cfg.experience.kfold, random_state=cfg.experience.split_random_state)
        dts = eng.make_subset(dts, splits[cfg.experience.split]['train' if set == 'train' else 'val'], transform, set)
        lib.LOGGER.info(dts)
    else:
        dts = getter.get_dataset(transform, set, cfg.dataset)

    lib.LOGGER.info("Dataset created...")

    results_by_k = eng.evaluate_multi_k(
        net=net,
        test_dataset=dts,
        epoch=state["epoch"],
        batch_size=bs,
        num_workers=nw,
        exclude=["mean_reciprocal_rank", "mean_average_precision",
                              "precision_at_1","recall_at_1","r_precision",'rpr', 'pr_rc', "recall_at_1000", "recall_at_100",
                              "recall_at_10", "recall_at_16", "recall_at_20", "recall_at_30", "recall_at_32", "recall_at_4", "recall_at_8",
                                "recall_at_2", "recall_at_10"],
        k_list=k_list,
        distance_metric=kwargs.get('distance_metric', 'cosine'),
    )

    lib.LOGGER.info("Multi-k evaluation completed...")
    for k, metrics in results_by_k.items():
        for split, mtrc in metrics.items():
            for name, v in mtrc.items():
                if name == 'epoch':
                    continue
                lib.LOGGER.info(f"k={k} {split} --> {name} : {np.around(v*100, decimals=2)}")

    return results_by_k


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, nargs='+', help='Path.s to checkpoint')
    parser.add_argument("--parse-file", default=False, action='store_true', help='allows to pass a .txt file with several models to evaluate')
    parser.add_argument("--set", type=str, default='test', help='Set on which to evaluate')
    parser.add_argument("--bs", type=int, default=128, help='Batch size for DataLoader')
    parser.add_argument("--nw", type=int, default=10, help='Num workers for DataLoader')
    parser.add_argument("--data-dir", type=str, default=None, help='Possible override of the datadir in the dataset config')
    parser.add_argument("--metric-dir", type=str, default=None, help='Path in which to store the metrics')
    parser.add_argument("--k", type=int, default=2047, help='k for the k-NN evaluation')
    parser.add_argument("--distance-metric", type=str, default="cosine", help='distance metric for the k-NN evaluation')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
    )

    if args.parse_file:
        with open(args.config[0], 'r') as f:
            paths = f.read().split('\n')
            paths.remove("")
        args.config = paths

    for path in args.config:
        metrics = load_and_evaluate(
            path=path,
            set=args.set,
            bs=args.bs,
            nw=args.nw,
            data_dir=args.data_dir,
            k=args.k,
            distance_metric=args.distance_metric,
        )
        print()
        print()

        if args.metric_dir is not None:
            with open(args.metric_dir, 'a') as f:
                f.write(path)
                f.write("\n")
                for split, mtrc in metrics.items():
                    for k, v in mtrc.items():
                        if k == 'epoch':
                            continue
                        f.write(f"{split} --> {k} : {np.around(v*100, decimals=2)}\n")
                f.write("\n\n")
