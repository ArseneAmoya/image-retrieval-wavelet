from torchvision.datasets import CIFAR100
from .base_dataset import BaseDataset
import numpy as np
from PIL import Image

class Cifar100RetrievalDataset(BaseDataset):
    def __init__(self, data_dir, mode, transform=None, seed=42, **kwargs):
        """
        Args:
            data_dir (str): path to the data.
            mode (str): 'train' (10k), 'query' (5k), or 'gallery' (45k).
            transform (callable, optional): transform to apply.
            seed (int): shuffle seed, ensures train/query/gallery never overlap.
        """
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        print(f"Loading full CIFAR100 dataset for custom split ({mode})...")
        train_set = CIFAR100(root=data_dir, train=True, download=True)
        test_set = CIFAR100(root=data_dir, train=False, download=True)

        all_data = np.concatenate([train_set.data, test_set.data], axis=0)
        all_targets = np.concatenate([train_set.targets, test_set.targets], axis=0)

        total_images = len(all_data)
        assert total_images == 60000, f"Expected 60000 images, found {total_images}"

        np.random.seed(seed)
        perm = np.random.permutation(total_images)

        n_train = 10000
        n_query = 5000
        n_gallery = 45000

        if mode == 'train':
            indices = perm[:n_train]
        elif mode == 'query':
            indices = perm[n_train : n_train + n_query]
        elif mode in ['gallery', 'database']:
            indices = perm[n_train + n_query :]
        else:
            raise ValueError(f"Mode must be 'train', 'query', or 'gallery'. Got {mode}")

        print(f"-> Split '{mode}': {len(indices)} images selected.")

        self.data = all_data[indices]
        self.labels = all_targets[indices]
        self.paths = np.arange(len(indices))

        self.get_instance_dict()

    def __getitem__(self, idx):
        img_arr = self.data[idx]
        label = self.labels[idx]

        img = Image.fromarray(img_arr)

        if self.transform:
            img = self.transform(img)

        return {"image": img, "label": label, "index": idx}
