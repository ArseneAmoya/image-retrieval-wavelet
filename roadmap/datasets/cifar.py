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
        dataset = CIFAR100(root=self.data_dir, train=train, download=True)
        self.paths = np.arange(len(dataset))
        self.labels = np.array(dataset.targets)

        self.dataset = dataset  # Store for __getitem__

        self.get_instance_dict()

    def __getitem__(self, idx):
        img, label = self.dataset[self.paths[idx]]
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": label}