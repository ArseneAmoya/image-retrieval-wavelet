from torchvision import datasets
import scipy.io
import numpy as np
import os

from .base_dataset import BaseDataset

class StanfordDog12Dataset(BaseDataset):
    def __init__(self, data_dir, mode, transform=None, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        if mode == "train":
            mat = scipy.io.loadmat(os.path.join(self.data_dir, 'train_list.mat'))
        elif mode == "test":
            mat = scipy.io.loadmat(os.path.join(self.data_dir, 'test_list.mat'))
        elif mode == "all":
            mat = scipy.io.loadmat(os.path.join(self.data_dir, 'file_list.mat'))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        paths = np.array([os.path.join(self.data_dir, 'Images', str(a[0][0])) for a in mat['file_list']])
        labels = np.array([int(a[0]) - 1 for a in mat['labels']])  # Convert to zero-based index
        self.paths = paths
        self.labels = labels
        self.get_instance_dict()



      