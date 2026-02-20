import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DetailTesterNet(nn.Module):
    def __init__(self, backbone_name='resnet50', detail_index=1, output_dim=64, **kwargs):
        """
        Args:
            backbone_name: 'resnet50', 'convnextv2_tiny.fcmae', ou 'dinov2_vits14'
            detail_index: 1 pour LH, 2 pour HL, 3 pour HH (0 étant LL/l'image globale)
            output_dim: Dimension finale du hachage (ex: 64 bits)
        """
        super().__init__()
        self.detail_index = detail_index
        
        # 1. Chargement du Backbone SOTA
        if 'dinov2' in backbone_name:
            # Le champion ViT
            self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
            dim = self.backbone.embed_dim
        else:
            # Le champion ConvNet (ResNet) ou ConvNeXt
            # timm gère automatiquement la création et retire la tête de classification avec num_classes=0
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
            self.backbone.forward = self.backbone.forward_features if hasattr(self.backbone, 'forward_features') else self.backbone.forward
            dim = self.backbone.num_features

        # Global Pooling (nécessaire pour ResNet/ConvNext, DINO sort déjà un vecteur 1D)
        self.pool = nn.AdaptiveAvgPool2d(1) if not 'dinov2' in backbone_name else nn.Identity()

        # 2. Tête de Hachage (Standard pour l'expérience)
        self.bn = nn.BatchNorm1d(dim)
        self.hash_fc = nn.Linear(dim, output_dim)
        
        # Init propre
        nn.init.normal_(self.hash_fc.weight, std=0.01)
        nn.init.constant_(self.hash_fc.bias, 0)

    def forward(self, x):
        # 1. Isolation des détails
        # x est supposé être la sortie de SWT : shape (B, Canaux=3, Sous-bandes=4, H, W)
        # On extrait la sous-bande cible (ex: index 1 pour LH)
        # Shape devient : (B, 3, H, W) -> Parfait pour les réseaux pré-entraînés
        detail_band = x[:, :, self.detail_index, :, :]
        
        # 2. Passage dans le réseau
        if isinstance(self.backbone, timm.models.resnet.ResNet) or 'convnext' in self.backbone.__class__.__name__.lower():
            # Pour TIMM CNNs
            feat_map = self.backbone(detail_band)
            feat = self.pool(feat_map).flatten(1)
        else:
            # Pour DINO
            out = self.backbone(detail_band)
            feat = out['x_norm_clstoken'] if isinstance(out, dict) else out
            
        # 3. Hachage (Identique à votre MultiDino)
        feat = self.bn(feat)
        logits = self.hash_fc(feat)
        
        if self.training:
            return torch.tanh(logits)
        else:
            return torch.sign(logits)