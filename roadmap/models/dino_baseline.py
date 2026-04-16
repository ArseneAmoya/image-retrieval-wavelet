import torch
import torch.nn as nn

class DINOHashBaseline(nn.Module):
    def __init__(self, dino_backbone='dinov2_vits14', embed_dim=384, binary_config={'nbits': 64}):
        super().__init__()
    
        self.backbone = torch.hub.load('facebookresearch/dinov2', dino_backbone)
        
        
        
        self.hash_head = nn.Sequential(
            nn.Linear(embed_dim, binary_config['nbits'], bias=False),
            nn.BatchNorm1d(binary_config['nbits']), 
            nn.Tanh() 
        )

    def forward(self, x):
        """
        x: batch d'images RGB de taille [B, 3, 224, 224]
        """
        features = self.backbone(x)
        
        continuous_hash = self.hash_head(features)

        if not self.training:
            continuous_hash = torch.sign(continuous_hash)

        return continuous_hash
    
    def get_binary_code(self, x):
        """
        Fonction à utiliser UNIQUEMENT lors de l'inférence (pour calculer le mAP)
        """
        continuous_hash = self.forward(x)
        return torch.sign(continuous_hash)