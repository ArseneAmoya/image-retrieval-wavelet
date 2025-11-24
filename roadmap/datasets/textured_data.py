import os

import numpy as np
from torchvision import datasets

from .base_dataset import BaseDataset


class TexturedDataset(BaseDataset):

    def __init__(self, data_dir, mode, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        dataset = {"train": datasets.ImageFolder(os.path.join(self.data_dir, 'train')),
                     "test": datasets.ImageFolder(os.path.join(self.data_dir, 'test'))
                }
        
        if mode == 'all':
            train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'train'))
            test_dataset = datasets.ImageFolder(os.path.join(self.data_dir, 'test'))
            dataset = datasets.ConcatDataset([train_dataset, test_dataset])
            self.imgs = train_dataset.imgs + test_dataset.imgs
        else:
            dataset = dataset[mode]
            self.imgs = dataset.imgs
        paths = np.array([a for (a, b) in self.imgs])
        labels = np.array([b for (a, b) in self.imgs])

        self.paths = []
        self.labels = []
        for lb, pth in zip(labels, paths):
            self.paths.append(pth)
            self.labels.append(lb)

        self.get_instance_dict()
