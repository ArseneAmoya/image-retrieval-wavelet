import os
import torch
from collections import defaultdict
from PIL import Image
from .base_dataset import BaseDataset 

class MIRFlickrHashing(BaseDataset):
    def __init__(self, data_dir, mode='train', transform=None, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        # Mapping des modes ROADMAP vers les fichiers textes fournis
        if mode == 'train':
            txt_file = 'train.txt'
        elif mode in ['query', 'val', 'test']:
            txt_file = 'test.txt'
        elif mode in ['database', 'gallery']:
            txt_file = 'database.txt'
        else:
            raise ValueError(f"Mode inconnu: {mode}")

        list_path = os.path.join(data_dir, txt_file)
        # MIRFlickr a toutes ses images dans un sous-dossier 'images'
        img_folder = os.path.join(data_dir, 'images')

        self.paths = []
        self.labels = []

        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                
                # parts[0] = nom de l'image (ex: im1.jpg), parts[1:] = les 38 bits de labels
                img_name = parts[0]
                target = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                
                self.paths.append(os.path.join(img_folder, img_name))
                self.labels.append(target)

        self.get_instance_dict()

    def get_instance_dict(self):
        self.instance_dict = defaultdict(list)
        for img_idx, target_tensor in enumerate(self.labels):
            active_classes = torch.where(target_tensor == 1.0)[0].tolist()
            for cls_idx in active_classes:
                self.instance_dict[cls_idx].append(img_idx)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224), (0, 0, 0)) # Sécurité fichiers corrompus
            
        if self.transform is not None:
            img = self.transform(img)
            
        target = self.labels[idx].clone().detach().float()
        return {"image": img, "label": target, "path": path}
    def __len__(self):
        return len(self.paths)

class COCOHashing(BaseDataset):
    def __init__(self, data_dir, mode='train', transform=None, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        if mode == 'train':
            txt_file = 'train.txt'
        elif mode in ['query', 'val', 'test']:
            txt_file = 'test.txt'
        elif mode in ['database', 'gallery']:
            txt_file = 'database.txt'
        else:
            raise ValueError(f"Mode inconnu: {mode}")

        list_path = os.path.join(data_dir, txt_file)

        self.paths = []
        self.labels = []

        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                
                img_name = parts[0]
                target = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
                
                # Contrairement à MIRFlickr, le txt de COCO inclut déjà les sous-dossiers
                # ex: "train2014/COCO_train2014_0000.jpg"
                self.paths.append(os.path.join(data_dir, img_name))
                self.labels.append(target)

        self.get_instance_dict()

    def get_instance_dict(self):
        self.instance_dict = defaultdict(list)
        for img_idx, target_tensor in enumerate(self.labels):
            active_classes = torch.where(target_tensor == 1.0)[0].tolist()
            for cls_idx in active_classes:
                self.instance_dict[cls_idx].append(img_idx)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            # Sécurité si une image COCO est corrompue
            img = Image.new('RGB', (256, 256), (0, 0, 0)) 
            
        if self.transform is not None:
            img = self.transform(img)
            
        target = self.labels[idx].clone().detach().float()
        return {"image": img, "label": target, "path": path}

    def __len__(self):
        return len(self.paths)