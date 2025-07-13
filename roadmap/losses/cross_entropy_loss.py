import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxCrossEntropy(nn.Module):
    def __init__(self, embed_size: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_size, num_classes, bias=False)
        nn.init.xavier_normal_(self.fc.weight)  # optionnel, déjà proche du défaut

    def forward(self, embed: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.fc(embed)           # [batch, num_classes]
        loss   = F.cross_entropy(logits, targets)
        return loss
