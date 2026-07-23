import configparser
import os
import platform

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_class_names(dataset):
    """
    get class names of dataset.
    """
    with open(f"{os.path.dirname(__file__)}/_datasets/{dataset}/concepts.txt", "r") as fp:
        lines = fp.read().splitlines()
    return np.array(lines)


class MyDataset(Dataset):
    """
    Common dataset for DeepHashing.

    Args
        root(str): Directory of all datasets.
        dataset(str): Dataset name.
        usage(str): train/query/dbase.
        transform(callable, optional): Transform images.
    """

    def __init__(self, root, dataset, usage, transform=None):
        assert dataset in ["cifar", "flickr", "coco", "nuswide"]
        self.name = dataset

        assert usage in ["train", "query", "dbase"]
        self.usage = usage

        if not os.path.exists(root):
            print(f"root not exists: {root}")
            root = os.path.dirname(__file__) + "/_datasets"
            print(f"root will use: {root}")

        xxx_dir = os.path.join(root, f"{dataset}")
        img_dir = f"{xxx_dir}/images"
        # img_loc = os.path.join(img_dir, "images_location.txt")
        ini_loc = os.path.join(img_dir, "images_location.ini")
        if os.path.exists(ini_loc):
            # self.img_dir = open(img_loc, "r").readline()
            config = configparser.ConfigParser()
            config.read(ini_loc)
            self.img_dir = config["DEFAULT"][platform.system()]
        else:
            self.img_dir = img_dir
        self.transform = build_default_trans(usage) if transform is None else transform

        # Read files
        self.data = [
            (x.split()[0], np.array(x.split()[1:], dtype=np.float32))
            for x in open(os.path.join(xxx_dir, f"{usage}.txt"), "r").readlines()
        ]

    def __getitem__(self, index):
        file_name, label = self.data[index]
        with open(os.path.join(self.img_dir, file_name), "rb") as fp:
            img = Image.open(fp).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, index

    def __len__(self):
        return len(self.data)

    def get_all_labels(self):
        return torch.from_numpy(np.vstack([x[1] for x in self.data]))


def get_class_num(dataset):
    r = {"cifar": 10, "flickr": 38, "nuswide": 21, "coco": 80}[dataset]
    return r


def get_topk(dataset):
    r = {"cifar": None, "flickr": None, "nuswide": 5000, "coco": None}[dataset]
    return r


def build_default_trans(usage, resize_size=256, crop_size=224):
    if usage == "train":
        # step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
        step = [transforms.RandomCrop(crop_size), transforms.RandomHorizontalFlip()]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose(
        [transforms.Resize(resize_size)]
        + step
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_loader(root, dataset, usage, transform, **kwargs):
    ds_cls = kwargs.pop("ds_cls", MyDataset)
    dset = ds_cls(root, dataset, usage, transform)

    verbose = kwargs.pop("verbose", True)
    if verbose:
        print(f"{usage} set length: {len(dset)}")

    shuffle = kwargs.pop("shuffle", usage == "train")
    if shuffle:
        loader = DataLoader(dset, shuffle=True, **kwargs)
    else:
        # generator=torch.Generator(): to keep torch.get_rng_state() unchanged!
        # https://discuss.pytorch.org/t/does-a-dataloader-change-random-state-even-when-shuffle-argument-is-false/92569/4
        loader = DataLoader(dset, generator=torch.Generator(), **kwargs)

    return loader


def build_loaders(root, dataset, trans_train=None, trans_test=None, **kwargs):
    bl_fnc = kwargs.pop("bl_fnc", build_loader)

    loaders = []
    for usage in ["train", "query", "dbase"]:
        loaders.append(
            bl_fnc(
                root,
                dataset,
                usage,
                trans_train if usage == "train" else trans_test,
                **kwargs,
            )
        )

    return loaders


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     dataset = "coco"
#     batch_size = 2
#     trans = transforms.Compose([transforms.ToTensor()])
#     loader, _, _ = build_loaders("./_datasets", dataset, trans, trans, batch_size=batch_size, num_workers=1)
#     print("topk", get_topk(dataset))
#     print("num_classes", get_class_num(dataset))
#     class_names = get_class_names(dataset)
#     print("-" * 10)
#     for images, labels, _ in loader:
#         images = images.numpy()
#         images = images.transpose(0, 2, 3, 1)

#         fig, axes = plt.subplots(1, batch_size)
#         for i in range(labels.size(0)):
#             axes[i].imshow(images[i])
#             axes[i].set_title(class_names[labels[i].nonzero().squeeze()])  # 显示标签
#             axes[i].axis("off")  # 关闭坐标轴
#         plt.show()

#         break
