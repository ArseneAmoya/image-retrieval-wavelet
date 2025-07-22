import logging
import argparse

import torch
import numpy as np

from roadmap.getter import Getter
import roadmap.utils as lib
import roadmap.engine as eng

from torch import nn
from torchsummary import summary
import matplotlib.pyplot as plt
#from models import *

#from Smooth_AP_loss import SmoothAP

#import datasets as data
from tqdm import tqdm
#from model import IndLeftRightDisparity
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def generate_alphas(dataloader, model_ft):
    """
    Parameters
    ----------
    files : list
        The list of images directories to index.
    model :
    Returns
    -------
    X: Tensor componed of features vectors of images.

    """
    model_ft.eval()
    summary(model_ft, (3, 32, 32))
    #print(model_ft)
#    datas = data.EvaluationDataset(files)
#    dataloader = torch.utils.data.DataLoader(datas, batch_size=1)
    output = 0
    with torch.no_grad():

        for inputs in tqdm(dataloader):
            img = inputs["image"].to(device)#inp = [a.to(device) for a in inputs]#inp = [a.to(device) for a in inputs]#
            #print(inp[0].shape, len(inp))
            output += model_ft.backbone.alphas(img).sum(0)#.reshape(1,2048)

        output /= len(dataloader.dataset)
    return output


def load_and_evaluate(
    path,
    set,
    bs,
    nw,
    data_dir=None,
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
        dataloader = torch.utils.data.DataLoader(
            dts,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
        )
    
    if isinstance(dataloader, torch.utils.data.DataLoader):
        lib.LOGGER.info(f"Dataset size: {len(dataloader.dataset)}")
        lib.LOGGER.info(f"Computing alphas")
        output = generate_alphas(dataloader, net)
        print(output)

        


    
    

    lib.LOGGER.info("Alphas computed...")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, nargs='+', help='Path.s to checkpoint')
    parser.add_argument("--parse-file", default=False, action='store_true', help='allows to pass a .txt file with several models to evaluate')
    parser.add_argument("--set", type=str, default='test', help='Set on which to evaluate')
    parser.add_argument("--bs", type=int, default=128, help='Batch size for DataLoader')
    parser.add_argument("--nw", type=int, default=10, help='Num workers for DataLoader')
    parser.add_argument("--data-dir", type=str, default=None, help='Possible override of the datadir in the dataset config')
    parser.add_argument("--metric-dir", type=str, default=None, help='Path in which to store the metrics')
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
        alphas = load_and_evaluate(
            path=path,
            set=args.set,
            bs=args.bs,
            nw=args.nw,
            data_dir=args.data_dir,
        )
        print()
        print()
