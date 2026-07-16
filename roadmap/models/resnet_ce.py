import math
import os

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

class ResNetHashing(nn.Module):
    def __init__(self, num_bits, pretrained=True, freeze_bn=True, **kwargs):
        super().__init__()
        
        # 1. Charger ResNet50 pré-entraîné
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        
        # 2. Garder uniquement l'extracteur de features (jusqu'à avgpool inclus)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048  # Fixe pour ResNet50
        
        # 3. Tête de hashing spécifique au papier
        #self.dropout = nn.Dropout(p=dropout)
        

    
        self.hash_layer = nn.Linear(self.feature_dim, num_bits)
        
        # Initialisation à Zéro (Critique)
        nn.init.xavier_uniform_(self.hash_layer.weight)
        nn.init.constant_(self.hash_layer.bias, 0)

        # Flag pour gérer le mode des Batch Norm
        self.freeze_bn = freeze_bn

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten (Batch, 2048)
        hash_logits = self.hash_layer(x)

        if self.training:
            # TRAINING: Retourne les logits pour la perte de hashing
            # x = self.dropout(x)
            return torch.tanh(hash_logits)
        else:
            return torch.sign(hash_logits)  # Binarisation pour l'évaluation

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
class ResNetHashingAlpha(ResNetHashing):
    def __init__(self, num_bits, pretrained=True, freeze_bn=False, **kwargs):
        super().__init__(num_bits, pretrained, freeze_bn, **kwargs)
        self.alpha = 1.0

    def set_alpha(self, epoch):
        """
        Méthode de continuation inspirée de HashNet
        """
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
    

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten (Batch, 2048)
        hash_logits = self.hash_layer(x)

        if self.training:
            return torch.tanh(self.alpha * hash_logits)
        else:
            return torch.sign(hash_logits)  # Binarisation pour l'évaluation

class ResNet50(nn.Module):
    def __init__(self, n_bits, pretrained=True, **kwargs):
        super().__init__()

        self.frozen = kwargs.pop("frozen", False)
        self.double = kwargs.pop("double", False)
        self.normalize = kwargs.pop("normalize", False)
        self.layernorm = kwargs.pop("layernorm", False)
        self.tanh = kwargs.pop("tanh", False)

        # 定义本地预训练权重路径
        local_weights_path = "/dk0/home/user011/project/DSpH/pretrained/resnet50-0676ba61.pth"

        # 加载不带预训练权重的模型
        self.backbone = models.resnet50(weights=None)

        # 如果pretrained为True，加载本地权重
        if pretrained:
            if os.path.exists(local_weights_path):
                print(f"Loading local pretrained weights from: {local_weights_path}")
                state_dict = torch.load(local_weights_path, map_location='cpu')
                self.backbone.load_state_dict(state_dict)
            else:
                print(f"Warning: Local weights not found at {local_weights_path}")
                print("Using ImageNet pre-trained weights from torchvision...")
                weights = models.ResNet50_Weights.IMAGENET1K_V1
                self.backbone = models.resnet50(weights=weights)

        self.dim_feature = self.backbone.fc.in_features

        # 替换最后的全连接层
        self.backbone.fc = nn.Linear(self.dim_feature, n_bits)

        # 初始化新的全连接层
        nn.init.xavier_uniform_(self.backbone.fc.weight)
        nn.init.zeros_(self.backbone.fc.bias)

        # IDML
        # nn.init.kaiming_normal_(self.backbone.fc.weight, mode="fan_out")
        # nn.init.constant_(self.backbone.fc.bias, 0)

        if self.double:
            self.backbone.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))

        if self.frozen:
            for module in filter(lambda m: isinstance(m, nn.BatchNorm2d), self.backbone.modules()):
                module.eval()
                module.train = lambda _: None

        if self.layernorm:
            # elementwise_affine=False means no learnable parameters
            self.layer_norm = nn.LayerNorm(n_bits, elementwise_affine=False)

    def forward(self, x: torch.Tensor):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        if self.double:
            x = self.backbone.avgpool(x) + self.backbone.maxpool2(x)
        else:
            x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        if self.layernorm:
            x = self.layer_norm(x)

        if self.normalize:
            x = F.normalize(x, dim=-1)

        return x
    
class ResNet50Mod(nn.Module):
    def __init__(self, n_bits, pretrained, **kwargs):
        super().__init__()
        self.tanh = kwargs.pop("tanh", False)
        self.net = ResNet50(n_bits, pretrained, **kwargs)
        self.alpha = 1.0

    def epoch_step(self, epoch):
        """
        continuation methods from HashNet
        """
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
    def set_alpha(self, epoch):
        self.epoch_step(epoch)

    def forward(self, x):
        x = self.net(x)
        if self.tanh:
            x = torch.tanh(self.alpha * x)
        return x