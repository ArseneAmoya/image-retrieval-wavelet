from torchvision.datasets import CIFAR100
from .base_dataset import BaseDataset
import numpy as np
from PIL import Image

class Cifar100RetrievalDataset(BaseDataset):
    def __init__(self, data_dir, mode, transform=None, seed=42, **kwargs):
        """
        Args:
            data_dir (str): Chemin vers les données.
            mode (str): 'train' (10k), 'query' (5k), ou 'gallery' (45k).
            transform (callable, optional): Transformation à appliquer.
            seed (int): Graine pour le mélange aléatoire (assure que train/query/gallery ne se chevauchent pas).
        """
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        # 1. Charger l'intégralité de CIFAR-100 (Train + Test)
        # CIFAR-100 total = 50 000 (train) + 10 000 (test) = 60 000 images
        print(f"Loading full CIFAR100 dataset for custom split ({mode})...")
        train_set = CIFAR100(root=data_dir, train=True, download=True)
        test_set = CIFAR100(root=data_dir, train=False, download=True)

        # 2. Fusionner les données et les labels
        # .data est un numpy array (N, 32, 32, 3)
        # .targets est une liste
        all_data = np.concatenate([train_set.data, test_set.data], axis=0)
        all_targets = np.concatenate([train_set.targets, test_set.targets], axis=0)
        
        total_images = len(all_data)
        assert total_images == 60000, f"Expected 60000 images, found {total_images}"

        # 3. Mélange déterministe (Seed fixe)
        # C'est crucial pour que le 'query' set soit toujours le même et distinct du 'train' set
        np.random.seed(seed)
        perm = np.random.permutation(total_images)

        # 4. Définir les bornes des splits
        n_train = 10000
        n_query = 5000
        n_gallery = 45000
        
        # 5. Sélectionner les indices selon le mode
        if mode == 'train':
            indices = perm[:n_train]
        elif mode == 'query':
            indices = perm[n_train : n_train + n_query]
        elif mode in ['gallery', 'database']:
            indices = perm[n_train + n_query :]
        else:
            raise ValueError(f"Mode must be 'train', 'query', or 'gallery'. Got {mode}")

        print(f"-> Split '{mode}': {len(indices)} images selected.")

        # 6. Stocker les sous-ensembles
        self.data = all_data[indices]
        self.labels = all_targets[indices]
        self.paths = np.arange(len(indices))  # Indices virtuels pour BaseDataset

        # Générer le dictionnaire des instances (requis par BaseDataset pour certains samplers)
        self.get_instance_dict()

    def __getitem__(self, idx):
        # Récupération de l'image brute (numpy array) et du label
        img_arr = self.data[idx]
        label = self.labels[idx]

        # Conversion en PIL Image (Nécessaire pour torchvision transforms)
        img = Image.fromarray(img_arr)

        # Application des transformations
        if self.transform:
            img = self.transform(img)

        # Retour conforme à votre structure BaseDataset / CifarDataset
        return {"image": img, "label": label, "index": idx}