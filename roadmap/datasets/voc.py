import os
import torch
import torchvision # <-- Nouvel import indispensable
import xml.etree.ElementTree as ET
from collections import defaultdict
from PIL import Image
from .base_dataset import BaseDataset 

class VOC2012Hashing(BaseDataset):
    """
    Dataset PASCAL VOC 2012 (Multi-Label) 100% compatible ROADMAP.
    Inclut le téléchargement automatique (Plug & Play).
    """
    VOC_CLASSES = (
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    )

    def __init__(self, data_dir, mode='train', transform=None, download=True, **kwargs):
        # 1. Appel crucial de la classe parente
        super().__init__(**kwargs)
        
        # --- NOUVEAU : BLOC DE TÉLÉCHARGEMENT AUTOMATIQUE ---
        if download:
            # On utilise le module natif de PyTorch juste pour télécharger et extraire.
            # S'il détecte que le dossier VOCdevkit existe déjà, il passe cette étape instantanément.
            print(f"Vérification/Téléchargement de VOC-2012 dans {data_dir}...")
            torchvision.datasets.VOCDetection(root=data_dir, year='2012', image_set='train', download=True)
            torchvision.datasets.VOCDetection(root=data_dir, year='2012', image_set='val', download=True)
        # ----------------------------------------------------

        # Le downloader natif crée automatiquement l'arborescence VOCdevkit/VOC2012
        self.data_dir = os.path.join(data_dir, 'VOCdevkit', 'VOC2012')
        self.transform = transform
        self.mode = mode
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.VOC_CLASSES)}
        
        # Mapping du mode ROADMAP vers les splits VOC
        if mode in ['train', 'gallery', 'database']:
            image_set = 'train'
        elif mode in ['query', 'val', 'test']:
            image_set = 'val'
        else:
            raise ValueError(f"Mode inconnu: {mode}")

        splits_dir = os.path.join(self.data_dir, 'ImageSets', 'Main')
        split_f = os.path.join(splits_dir, f'{image_set}.txt')
        
        with open(split_f, "r") as f:
            image_names = [x.strip() for x in f.readlines()]

        images_dir = os.path.join(self.data_dir, 'JPEGImages')
        annotations_dir = os.path.join(self.data_dir, 'Annotations')

        paths = []
        labels = []

        # On parse les XML à l'initialisation
        for img_name in image_names:
            paths.append(os.path.join(images_dir, f"{img_name}.jpg"))
            
            xml_path = os.path.join(annotations_dir, f"{img_name}.xml")
            xml_tree = ET.parse(xml_path)
            
            target = torch.zeros(len(self.VOC_CLASSES), dtype=torch.float32)
            has_valid = False
            
            for obj in xml_tree.getroot().findall('object'):
                name = obj.find('name').text
                difficult = int(obj.find('difficult').text)
                if difficult == 0 and name in self.class_to_idx:
                    target[self.class_to_idx[name]] = 1.0
                    has_valid = True
                    
            if not has_valid:
                for obj in xml_tree.getroot().findall('object'):
                    if obj.find('name').text in self.class_to_idx:
                        target[self.class_to_idx[obj.find('name').text]] = 1.0
            
            labels.append(target)

        self.paths = paths
        self.labels = labels

        # Création du dictionnaire compatible ROADMAP
        self.get_instance_dict()

    def get_instance_dict(self):
        self.instance_dict = defaultdict(list)
        for img_idx, target_tensor in enumerate(self.labels):
            active_classes = torch.where(target_tensor == 1.0)[0].tolist()
            for cls_idx in active_classes:
                self.instance_dict[cls_idx].append(img_idx)

    def __getitem__(self, idx):
        """
        Surcharge explicite pour garantir l'intégrité du tenseur Multi-Hot
        tout en respectant le format dictionnaire exigé par ROADMAP.
        """
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        # Tenseur multi-hot en Float
        target = self.labels[idx].clone().detach().float()
        
        # --- LA CORRECTION EST ICI ---
        # ROADMAP attend un dictionnaire avec ces clés exactes
        return {
            "image": img,
            "label": target,
            "path": path
        }