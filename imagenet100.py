import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset

class ImageNet100Hashing(Dataset):
    """
    Dataset optimisé pour le setup de Hashing sur ImageNet-100.
    Gère la fusion des dossiers train.X1-4 et l'extraction de 130 images.
    """
    def __init__(self, data_dir, split='train', transform=None, train_samples_per_class=130, seed=333):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # 1. Identifier les 100 classes (Le dossier val.X contient forcément les 100)
        val_dir = os.path.join(data_dir, 'val.X')
        # On trie pour garantir que la classe 'n0...' ait toujours le même label (0 à 99)
        self.classes = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        # Fixer la seed pour que les 130 images d'entraînement soient toujours les mêmes !
        random.seed(seed)
        
        # 2. Remplir les listes d'images selon le split demandé
        for cls_name in self.classes:
            label = self.class_to_idx[cls_name]
            
            if split in ['query', 'val', 'test']:
                # Les requêtes viennent uniquement de val.X
                paths = glob.glob(os.path.join(val_dir, cls_name, '*.*'))
                self.image_paths.extend(paths)
                self.labels.extend([label] * len(paths))
                
            elif split in ['train', 'database']:
                # On collecte les images depuis les 4 dossiers train.X
                train_paths = []
                for i in range(1, 5):
                    train_folder = os.path.join(data_dir, f'train.X{i}', cls_name)
                    if os.path.exists(train_folder):
                        train_paths.extend(glob.glob(os.path.join(train_folder, '*.*')))
                        
                # On trie pour avoir un ordre déterministe avant de mélanger
                train_paths = sorted(train_paths)
                
                if split == 'train':
                    # C'est ici qu'on applique la règle d'or du papier (130 images)
                    random.shuffle(train_paths)
                    train_paths = train_paths[:train_samples_per_class]
                    
                self.image_paths.extend(train_paths)
                self.labels.extend([label] * len(train_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Conversion RGB indispensable car certaines images ImageNet sont en Noir & Blanc
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        # ROADMAP attend généralement un tuple (image, label, index) ou un dict. 
        # Si votre pipeline CIFAR renvoyait juste (img, label), laissez tel quel.
        # S'il y a des erreurs d'unpacking dans ROADMAP, renvoyez : return img, label, idx
        return img, label