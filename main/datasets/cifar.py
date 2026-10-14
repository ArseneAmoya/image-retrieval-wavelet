import os
from torchvision.datasets.cifar import CIFAR100
from .base_dataset import BaseDataset
import numpy as np

class CifarDataset(BaseDataset):
    def __init__(self, data_dir, mode, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        train = (mode == "train")
        print(f"Loading CIFAR100 dataset in {mode} mode with transform: {transform}")
        root = self._resolve_root(data_dir)
        # download=True is a no-op when <root>/cifar-100-python already exists and passes
        # torchvision's md5 check: drop the extracted archive there to skip the (slow) download.
        dataset = CIFAR100(root=root, train=train, download=True, transform=transform)
        self.paths = np.arange(len(dataset))
        self.labels = np.array(dataset.targets)

        self.dataset = dataset  # Store for __getitem__

        self.get_instance_dict()

    @staticmethod
    def _resolve_root(data_dir):
        """data_dir from the config, made absolute w.r.t. the launch directory.

        Hydra (version_base 1.1) chdirs into the run dir, so a relative path used to land
        under <log_dir>/<experiment>/outputs and was wiped by `rm -rf /content/experiments`.
        Previously the root was hard-coded to "../../data/cifar100" and data_dir was ignored.
        """
        data_dir = os.path.expanduser(os.path.expandvars(data_dir or "data/cifar"))
        if not os.path.isabs(data_dir):
            try:
                from hydra.utils import to_absolute_path
                data_dir = to_absolute_path(data_dir)
            except Exception:
                data_dir = os.path.abspath(data_dir)
        return data_dir

    def __getitem__(self, idx):
        img, label = self.dataset[self.paths[idx]]
        return {"image": img, "label": label}