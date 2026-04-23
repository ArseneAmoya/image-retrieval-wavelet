import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def get_optimizer_for_loss(loss_module, optim_cfg):
    """Initialise l'optimiseur interne pour les paramètres de la perte."""
    if isinstance(optim_cfg, dict):
        opt_name = optim_cfg.get('name', 'AdamW')
        opt_kwargs = optim_cfg.get('kwargs', {})
    else:
        opt_name = getattr(optim_cfg, 'name', 'AdamW')
        opt_kwargs = getattr(optim_cfg, 'kwargs', {}) or {}

    optimizer = getattr(optim, opt_name)(loss_module.parameters(), **opt_kwargs)
    return optimizer


class HHFAdapter(nn.Module):
    """
        Implementation of HHF loss (Hinge Hashing Function) from the paper "Hinge Hashing Function for Efficient Image Retrieval" (CVPR 2022).
        Adapted from https://github.com/JerryXu0129/HHF/blob/main/model.py
    """
    takes_embeddings = True

    def __init__(self, num_classes=20, embedding_size=64, alpha=15.0, delta=0.1, threshold=0.0, beta=0.1, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.threshold = threshold
        self.beta = beta
        self.num_classes = num_classes

        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        default_opt = {'name': 'AdamW', 'kwargs': {'lr': 1e-4, 'weight_decay': 1e-4}}
        optim_cfg = kwargs.get('optimizer', default_opt)
        self.loss_optimizer = get_optimizer_for_loss(self, optim_cfg)

    def forward(self, embeddings, labels, **kwargs):
        x = torch.tanh(embeddings)

        cos = F.normalize(x, p=2, dim=1).mm(F.normalize(self.proxies, p=2, dim=1).t())

        pos_exp = torch.exp(self.alpha * F.relu(1 - self.delta - cos)) - 1
        neg_exp = torch.exp(self.alpha * F.relu(cos - self.threshold - self.delta)) - 1

        P_sim_sum = torch.where(labels == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(labels == 0, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        non_zero_pos = len(torch.nonzero(labels.sum(dim=0) != 0))
        non_zero_pos = non_zero_pos if non_zero_pos > 0 else 1 # Sécurité

        pos_term = torch.log(1 + P_sim_sum).sum() / non_zero_pos
        neg_term = torch.log(1 + N_sim_sum).sum() / self.num_classes

        loss_semantic = pos_term + neg_term

        loss_quant = torch.sum(torch.norm(torch.sign(x) - x, dim=1).pow(2)) / x.shape[0]

        return loss_semantic + (self.beta * loss_quant)

    def step(self):
        """Mise à jour des proxies à la fin de chaque batch."""
        self.loss_optimizer.step()
        print("optimizer step for HHFAdapter proxies")
        self.loss_optimizer.zero_grad()
        
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        sd['optimizer_state'] = self.loss_optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        optimizer_state = state_dict.pop('optimizer_state', None)
        super().load_state_dict(state_dict, strict)
        if optimizer_state is not None:
            self.loss_optimizer.load_state_dict(optimizer_state)