import numpy as np
from torchvision.datasets import CIFAR10
from PIL import Image
from .base_dataset import BaseDataset

class Cifar10Retrieval(BaseDataset):
    def __init__(self, data_dir, mode, transform=None, seed=42, **kwargs):
        """
        "54k Database" protocol, out of 60 000 images total:
        - Query (Test) : 100/class (indices 0-100)   -> disjoint
        - Train        : 500/class (indices 200-700) -> disjoint
        - Database     : 5400/class (indices 100-200 and 700-end) -> 54k images
        - Val          : 100/class (indices 100-200) -> included in Database
        """
        super().__init__(**kwargs)
        self.mode = mode
        self.transform = transform

        train_set = CIFAR10(root=data_dir, train=True, download=True)
        test_set = CIFAR10(root=data_dir, train=False, download=True)

        data = np.concatenate([train_set.data, test_set.data], axis=0)
        targets = np.concatenate([train_set.targets, test_set.targets], axis=0)

        np.random.seed(seed)

        indices_split = {
            'query': [],
            'val': [],
            'train': [],
            'database': []
        }

        class_indices = {i: [] for i in range(10)}
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)

        for label in range(10):
            indices = np.array(class_indices[label])
            perm = np.random.permutation(len(indices))

            indices_split['query'].extend(indices[perm[:100]])

            val_indices = indices[perm[100:200]]
            indices_split['val'].extend(val_indices)

            indices_split['train'].extend(indices[perm[200:700]])

            # Database = Val (100-200) + everything after Train (700+); excludes Test and Train.
            rest_indices = indices[perm[700:]]

            db_indices = np.concatenate([val_indices, rest_indices])
            indices_split['database'].extend(db_indices)

        target_mode = mode
        if mode == 'test': target_mode = 'query'
        if mode == 'gallery': target_mode = 'database'

        if target_mode not in indices_split:
             raise ValueError(f"Mode inconnu: {mode}")

        final_indices = indices_split[target_mode]

        print(f"-> CIFAR-10 ({target_mode}): {len(final_indices)} images.")
        if target_mode == 'database':
            print("   (Note: Contains Validation images but excludes Train & Test)")

        self.data = data[final_indices]
        self.labels = targets[final_indices]
        self.classes = list(range(10))
        self.paths = np.arange(len(final_indices))

        self.get_instance_dict()

    def __getitem__(self, idx):
        img_arr = self.data[idx]
        label = self.labels[idx]
        img = Image.fromarray(img_arr)
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": label, "index": idx}
