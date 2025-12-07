import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class MultiCrossEntropyLoss(nn.Module):
    takes_embeddings = True
    takes_logits = True

    def __init__(self,  weights: list = [1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        self.loss_fns = nn.ModuleList([CrossEntropyLoss() for _ in range(len(weights))])
        self.weights = weights

    def forward(self, preds, targets) -> torch.Tensor:
        total_loss = 0.0
        for i,  weight in enumerate(self.weights):
            loss = self.loss_fns[i](preds[i], targets)
            total_loss += weight * loss
        return total_loss
