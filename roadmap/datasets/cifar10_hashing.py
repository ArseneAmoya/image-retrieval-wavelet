import numpy as np
from torchvision.datasets import CIFAR10
from PIL import Image
from .base_dataset import BaseDataset

class Cifar10Retrieval(BaseDataset):
    def __init__(self, data_dir, mode, transform=None, seed=42, **kwargs):
        """
        Protocole "54k Database" :
        - Total : 60 000 images.
        - Query (Test) : 100/classe (Indices 0-100) -> DISJOINT
        - Train        : 500/classe (Indices 200-700) -> DISJOINT
        - Database     : 5400/classe (Indices 100-200 ET 700-Fin) -> 54k images
        - Val          : 100/classe (Indices 100-200) -> INCLUS DANS DATABASE
        """
        super().__init__(**kwargs)
        self.mode = mode
        self.transform = transform

        # 1. Chargement et Fusion
        train_set = CIFAR10(root=data_dir, train=True, download=True)
        test_set = CIFAR10(root=data_dir, train=False, download=True)
        
        data = np.concatenate([train_set.data, test_set.data], axis=0)
        targets = np.concatenate([train_set.targets, test_set.targets], axis=0)
        
        # 2. Séparation des indices
        np.random.seed(seed)
        
        indices_split = {
            'query': [],    # Test Set
            'val': [],      # Validation Set (Query pour la valid)
            'train': [],    # Training Set
            'database': []  # Gallery
        }
        
        class_indices = {i: [] for i in range(10)}
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)
            
        # 3. Découpage avec chevauchement Val/Database
        for label in range(10):
            indices = np.array(class_indices[label])
            perm = np.random.permutation(len(indices))
            
            # --- A. Test (Query) : 0 à 100 (100 images) ---
            # Strictement disjoint
            indices_split['query'].extend(indices[perm[:100]])
            
            # --- B. Validation : 100 à 200 (100 images) ---
            # Servira de Query pendant la validation
            val_indices = indices[perm[100:200]]
            indices_split['val'].extend(val_indices)
            
            # --- C. Train : 200 à 700 (500 images) ---
            # Strictement disjoint
            indices_split['train'].extend(indices[perm[200:700]])
            
            # --- D. Database : 100 à 200 ET 700 à la fin (5400 images) ---
            # Contient le Val (100-200) + le Reste (700+)
            # On exclut uniquement Test (0-100) et Train (200-700)
            rest_indices = indices[perm[700:]]
            
            # Fusion pour la database
            db_indices = np.concatenate([val_indices, rest_indices])
            indices_split['database'].extend(db_indices)
            
        # 4. Sélection selon le mode
        target_mode = mode
        if mode == 'test': target_mode = 'query'
        if mode == 'gallery': target_mode = 'database'
        
        if target_mode not in indices_split:
             raise ValueError(f"Mode inconnu: {mode}")

        final_indices = indices_split[target_mode]
        
        # Affichage pour vérification
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