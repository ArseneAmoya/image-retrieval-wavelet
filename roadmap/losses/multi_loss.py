import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from .. import utils as lib
from .cross_entropy_loss import CrossEntropy
from .blackbox_ap import BlackBoxAP
from .calibration_loss import CalibrationLoss
from .fast_ap import FastAP
from .softbin_ap import SoftBinAP
from .pair_loss import PairLoss
from .multi_ce_loss import MultiCrossEntropyLoss
from .multi_embedding_loss import MultiEmbeddingLoss
from .arcface_loss import ArcFaceLoss
from .smooth_rank_ap import (
    HeavisideAP,
    SmoothAP,
    SupAP,
)
from .distillation_loss import FeatureDistillationLoss
class MultiLoss(nn.Module):
    takes_embeddings = True

    def __init__(self, criterion: list, weights: list = None):
        """
        criterion: list of loss configs per branch (one list per branch).
        weights: global weight per branch (e.g. [1.0, 1.0, 1.0, 1.0]).
        """
        super().__init__()

        if weights is None:
            weights = [1.0] * len(criterion)

        self.branch_weights = weights

        self.losses = nn.ModuleList()
        self.per_loss_weights = []

        for i, branch_config in enumerate(criterion):
            modules, internal_weights = self.get_loss_for_branch(branch_config)

            self.losses.append(modules)
            self.per_loss_weights.append(internal_weights)

    def get_loss_for_branch(self, config):
        modules = nn.ModuleList()
        weights = []

        for crit in config:
            loss_cls = getattr(sys.modules[__name__], crit.name)
            loss_instance = loss_cls(**crit.kwargs)

            weight = crit.weight
            lib.LOGGER.info(f"Adding {crit.name} with weight {weight}")

            modules.append(loss_instance)
            weights.append(weight)

        return modules, weights

    def forward(self, embeddings, labels):
        total_loss = 0.0

        for i, branch_emb in enumerate(embeddings):
            if i >= len(self.losses): break

            branch_total = 0.0
            branch_weight = self.branch_weights[i]

            for j, criterion in enumerate(self.losses[i]):
                crit_weight = self.per_loss_weights[i][j]

                if hasattr(criterion, 'requires_all_branches') and getattr(criterion, 'requires_all_branches'):
                    # Provide the full list of embeddings so the loss can access teacher/student branches.
                    loss = criterion(embeddings, labels.view(-1))
                elif hasattr(criterion, 'takes_embeddings') and criterion.takes_embeddings:
                    loss = criterion(branch_emb, labels.view(-1))
                else:
                    scores = torch.mm(branch_emb, branch_emb.t())
                    label_matrix = lib.create_label_matrix(labels, labels)
                    loss = criterion(scores, label_matrix)

                if isinstance(loss, (tuple, list)):
                    loss = loss[0]

                if loss.dim() > 0:
                    loss = loss.mean()

                branch_total += crit_weight * loss

            total_loss += branch_weight * branch_total

        return total_loss
