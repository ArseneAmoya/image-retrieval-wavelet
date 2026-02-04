import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_entropy_loss import CrossEntropy as CrossEntropyLoss

class MultiCrossEntropyLoss(nn.Module):
    takes_embeddings = True
    takes_logits = True

    def __init__(self,  weights: list = [1.0, 1.0, 1.0, 1.0], label_smoothing: float = 0.1):
        super().__init__()
        self.loss_fns = nn.ModuleList([CrossEntropyLoss(label_smoothing=label_smoothing) for _ in range(len(weights))])
        self.weights = weights

    def forward(self, preds, targets) -> torch.Tensor:
        total_loss = 0.0
        for i,  weight in enumerate(self.weights):
            loss = self.loss_fns[i](preds[i], targets)
            total_loss += weight * loss
        return total_loss/len(self.weights)
    def __str__(self):
        return f"MultiCrossEntropyLoss with weights {self.weights} and label_smoothing {self.loss_fns[0].label_smoothing}"
