import torch
import torch.nn.functional as F
from torch import nn


class SCHLoss(nn.Module):
    takes_embeddings = True
    def __init__(self, n_bits=64, alpha=0.1, beta=0.1, *args, **kwargs):
        super().__init__()
        self.n_bits = n_bits
        self.alpha = alpha
        self.beta = beta
        self.max_func = torch.nn.ReLU()

    def forward(self, batch, labels):
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
