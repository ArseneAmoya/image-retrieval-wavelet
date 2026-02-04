import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_entropy_loss import CrossEntropy as CrossEntropyLoss
from .smooth_rank_ap import (
    HeavisideAP,
    SmoothAP,
    SupAP,
)
import sys
from .calibration_loss import CalibrationLoss

import roadmap.utils as lib

class MultiEmbeddingLoss(nn.Module):
    takes_embeddings = True
    def __init__(self,  weights: list = [1.0, 1.0, 1.0, 1.0], loss_name="SupAP", **kwargs):
        super().__init__()
        self.loss_fns = nn.ModuleList([getattr(sys.modules[__name__], loss_name)(**kwargs) for _ in range(len(weights))])
        self.weights = weights
        if hasattr(self.loss_fns[0], 'takes_embeddings'):
            self.forward = self.direct_forward
        else:
            self.forward = self.matmul_forward
    def direct_forward(self, preds, targets) -> torch.Tensor:
        total_loss = 0.0
        for i,  weight in enumerate(self.weights):
            loss = self.loss_fns[i](preds[i], targets)
            total_loss += weight * loss
        return total_loss/len(self.weights)
    def matmul_forward(self, preds, targets) -> torch.Tensor:
        total_loss = 0.0
        preds = [F.normalize(pred, p=2, dim=1) for pred in preds]
        targets = lib.create_label_matrix(targets)
        for i,  weight in enumerate(self.weights):
            loss = self.loss_fns[i](torch.mm(preds[i], preds[i].t()), targets)
            total_loss += weight * loss
        return total_loss/len(self.weights)

    def forward(self, preds, targets) -> torch.Tensor:
        pass
    def __str__(self):
        return f"MultiEmbedding loss with weights {self.weights} and loss function {self.loss_fns[0]}"
