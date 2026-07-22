import os
import glob
import random
from .base_dataset import BaseDataset

class ImageNet100Hashing(BaseDataset):
    """
    Hashing setup for ImageNet-100: merges the train.X1-4 folders and
    subsamples 130 images per class for training.
    """
    def __init__(self, data_dir, mode, transform=None, train_samples_per_class=130, seed=333, **kwargs):
        super().__init__(**kwargs)

        assert mode in ["train", "query", "gallery", "val", "test", "database"], f"Mode : {mode} unknown"

        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        val_dir = os.path.join(data_dir, 'val.X')
        self.classes = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        paths = []
        labels = []

        random.seed(seed)

        for cls_name in self.classes:
            label = self.class_to_idx[cls_name]

            if self.mode in ['query', 'val', 'test']:
                cls_paths = glob.glob(os.path.join(val_dir, cls_name, '*.*'))
                paths.extend(cls_paths)
                labels.extend([label] * len(cls_paths))

            elif self.mode in ['train', 'database', 'gallery']:
                train_paths = []
                for i in range(1, 5):
                    train_folder = os.path.join(data_dir, f'train.X{i}', cls_name)
                    if os.path.exists(train_folder):
                        train_paths.extend(glob.glob(os.path.join(train_folder, '*.*')))

                train_paths = sorted(train_paths)

                if self.mode == 'train':
                    random.shuffle(train_paths)
                    train_paths = train_paths[:train_samples_per_class]

                paths.extend(train_paths)
                labels.extend([label] * len(train_paths))

        self.paths = paths
        self.labels = labels

        self.super_labels_name = [self.classes[lbl] for lbl in self.labels]
        self.super_labels = self.labels.copy()

        self.get_instance_dict()
        if hasattr(self, 'get_super_dict'):
             self.get_super_dict()
