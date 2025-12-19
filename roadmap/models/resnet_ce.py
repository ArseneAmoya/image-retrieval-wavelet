import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from torchsummary import summary
import torch.nn.functional as F
import re

class ResNetCE(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, pretrained: bool = False, backbone_name: str = 'resnet50'):
        super().__init__()
        try:
            self.backbone = getattr(models, backbone_name)(weights= f"ResNet{re.search(r'(\d+)', backbone_name).group(1)}_Weights.IMAGENET1K_V1" if pretrained else None)
        except AttributeError:
            raise ValueError(f"Backbone '{backbone_name}' is not available in torchvision.models")
        self.backbone.fc = nn.Identity()  # Remove the final fully connected layer
        self.fc = nn.Linear(2048, embed_dim)

        # self.classifier = nn.Sequential(nn.Dropout(0.5),
        #                                 nn.Linear(embed_dim, num_classes))
        # nn.init.constant_(self.classifier[1].bias, 0)
        # nn.init.constant_(self.classifier[1].weight, 0)

    def forward(self, x) -> torch.Tensor:
        x = self.backbone(x)
        #x = self.pooling(x)

        x = torch.flatten(x, 1)
        #x = self.layer_norm(x)
        #embed = self.fc(x)

        if self.training:
            logits = self.fc(x) #self.classifier(x)
            return logits
        else:
            x = F.normalize(x, dim=1, p=2)

            return x

