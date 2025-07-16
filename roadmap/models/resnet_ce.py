import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchsummary import summary
import torch.nn.functional as F

class ResNetCE(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, pretrained: bool = False ):
        super().__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc = nn.Linear(2048, embed_dim)
        self.layer_norm = nn.BatchNorm1d(2048)
        nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        #x = self.pooling(x)

        x = torch.flatten(x, 1)
        x = self.layer_norm(x)
        embed = self.fc(x)

        if self.training:
            logits = self.classifier(embed)
            return logits
        else:
            x = F.normalize(embed, dim=1, p=2)

            return x

