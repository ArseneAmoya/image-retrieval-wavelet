import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from torchsummary import summary
import torch.nn.functional as F

class ResNetCE(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, pretrained: bool = False ):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # Remove the final fully connected layer
        #self.fc = nn.Linear(2048, embed_dim)
        self.layer_norm = nn.BatchNorm1d(512)
        nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        #x = self.pooling(x)

        x = torch.flatten(x, 1)
        x = self.layer_norm(x)
        #embed = self.fc(x)

        if self.training:
            logits = self.classifier(x)
            return logits
        else:
            x = F.normalize(x, dim=1, p=2)

            return x

