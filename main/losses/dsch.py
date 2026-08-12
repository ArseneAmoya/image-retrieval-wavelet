import torch
import torch.nn.functional as F
from torch import nn


class SCHLoss(nn.Module):
    """DSCH loss.

    Note on the input contract: `BI_BI = (k - batch @ batch.T) / 2` is the Hamming
    distance formula, which is only correct if `batch` holds values in [-1, 1]. Models
    whose forward already squashes (e.g. `resnet50_tanh`) satisfy that; models that
    emit raw logits and rely on the loss to squash (e.g. `DINOHashBaseline`,
    `MultiDinoHashing`, whose companion `HashLoss` applies `torch.tanh` itself) do not
    -- with raw logits `batch @ batch.T` is unbounded and the bounds below become
    meaningless.

    `apply_tanh` defaults to False so every existing config keeps its exact current
    behaviour; set it to True when pairing this loss with a raw-logit model.
    """

    takes_embeddings = True

    def __init__(self, n_bits=64, alpha=0.1, beta=0.1, apply_tanh=False, *args, **kwargs):
        super().__init__()
        self.n_bits = n_bits
        self.alpha = alpha
        self.beta = beta
        self.apply_tanh = apply_tanh
        self.max_func = torch.nn.ReLU()

    def forward(self, batch, labels):
        if self.apply_tanh:
            batch = torch.tanh(batch)
        batch_size = labels.size(0)
        k = self.n_bits
        S = F.normalize(labels).mm(F.normalize(labels).t())

        lambda_ = (1 - S) * k / 2
        lambda_u = lambda_
        lambda_l = lambda_ - 3  # τ is 3
        lambda_l[lambda_l <= 0] = 0
        lambda_l[S == 0] = k / 2

        W_l = torch.ones(batch_size, batch_size).to(labels.device)
        W_u = torch.ones(batch_size, batch_size).to(labels.device)
        W_l[S == 1] = 0
        W_l[S == 0] = self.beta
        W_u[S == 0] = 0
        W_u[S == 1] = self.alpha

        BI_BI = (k - batch.mm(batch.t())) / 2

        # lower bound
        loss1 = torch.norm(self.max_func(lambda_l - BI_BI) * W_l) / (batch_size * batch_size)

        # upper bound
        loss2 = torch.norm(self.max_func(BI_BI - lambda_u) * W_u) / (batch_size * batch_size)

        return loss1 + loss2
