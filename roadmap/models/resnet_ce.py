import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

class ResNetCE(nn.Module):
    def __init__(self, num_classes, dropout=0.5, pretrained=True, freeze_bn=True, **kwargs):
        super().__init__()
        
        # 1. Charger ResNet50 pré-entraîné
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        
        # 2. Garder uniquement l'extracteur de features (jusqu'à avgpool inclus)
        # ResNet50 structure: children()[:-1] retire la couche FC finale
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048  # Fixe pour ResNet50
        
        # 3. Tête de classification spécifique au papier
        # Dropout (0.5) -> Linear (2048 -> num_classes)
        # Pas de BN supplémentaire ici pour CUB !
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # 4. Initialisation à Zéro (Critique)
        nn.init.constant_(self.classifier.weight, 0)
        nn.init.constant_(self.classifier.bias, 0)

        # Flag pour gérer le mode des Batch Norm
        self.freeze_bn = freeze_bn

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten (Batch, 2048)

        if self.training:
            # TRAINING: Retourne les logits pour la CrossEntropy
            x = self.dropout(x)
            logits = self.classifier(x)
            return logits
        else:
            # EVAL: Retourne les features normalisées L2 pour le calcul de distance (Recall)
            return F.normalize(x, p=2, dim=1)

    def train(self, mode=True):
        """
        Surcharge critique : même en mode train=True, 
        les couches BN doivent rester en mode eval si freeze_bn est activé.
        """
        super().train(mode)
        if self.freeze_bn and mode:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()