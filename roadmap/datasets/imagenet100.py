import os
import glob
import random
from .base_dataset import BaseDataset

class ImageNet100Hashing(BaseDataset):
    """
    Dataset optimisé pour le setup de Hashing sur ImageNet-100.
    Gère la fusion des dossiers train.X1-4 et l'extraction de 130 images pour le train.
    Respecte l'héritage de BaseDataset de ROADMAP.
    """
    def __init__(self, data_dir, mode, transform=None, train_samples_per_class=130, seed=333, **kwargs):
        # Initialisation de la classe parente (crucial pour get_instance_dict)
        super().__init__(**kwargs)
        
        # ROADMAP utilise généralement train, query, et gallery (ou database)
        assert mode in ["train", "query", "gallery", "val", "test", "database"], f"Mode : {mode} unknown"
        
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        
        # 1. Identifier les 100 classes via le dossier val.X
        val_dir = os.path.join(data_dir, 'val.X')
        self.classes = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # ROADMAP attend ces noms exacts de variables
        paths = []
        labels = []
        
        # Fixer la seed pour assurer la reproductibilité du sous-échantillonnage
        random.seed(seed)
        
        # 2. Remplir les listes d'images selon le mode
        for cls_name in self.classes:
            label = self.class_to_idx[cls_name]
            
            if self.mode in ['query', 'val', 'test']:
                # Les requêtes viennent de val.X
                cls_paths = glob.glob(os.path.join(val_dir, cls_name, '*.*'))
                paths.extend(cls_paths)
                labels.extend([label] * len(cls_paths))
                
            elif self.mode in ['train', 'database', 'gallery']:
                # On collecte depuis les 4 dossiers train.X
                train_paths = []
                for i in range(1, 5):
                    train_folder = os.path.join(data_dir, f'train.X{i}', cls_name)
                    if os.path.exists(train_folder):
                        train_paths.extend(glob.glob(os.path.join(train_folder, '*.*')))
                        
                # Tri déterministe avant le mélange
                train_paths = sorted(train_paths)
                
                if self.mode == 'train':
                    # Extraction des 130 images
                    random.shuffle(train_paths)
                    train_paths = train_paths[:train_samples_per_class]
                    
                paths.extend(train_paths)
                labels.extend([label] * len(train_paths))

        self.paths = paths
        self.labels = labels

        # Si vous avez besoin de super labels (hiérarchie), vous pouvez les ajouter ici.
        # Pour ImageNet par défaut, on peut utiliser le nom du dossier comme super_label_name.
        self.super_labels_name = [self.classes[lbl] for lbl in self.labels]
        self.super_labels = self.labels.copy()

        # Appel des méthodes de la classe parente BaseDataset
        self.get_instance_dict()
        if hasattr(self, 'get_super_dict'):
             self.get_super_dict()

    # Plus besoin de définir __len__ et __getitem__ ici ! 
    # BaseDataset s'en charge généralement en lisant self.paths et self.labels,
    # et gère l'ouverture avec PIL et le renvoi de (img, label, idx).