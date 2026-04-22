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

class HashLoss(nn.Module):
    """
    Implémentation SOTA (GSPH/CSQ) avec optimiseur autonome pour les proxies.
    """
    takes_embeddings = True

    def __init__(self, num_classes=20, embedding_size=64, quant_weight=0.1, scale=15.0, **kwargs):
        super().__init__()
        self.quant_weight = quant_weight
        self.scale = scale
        
        # Proxies (Centres apprenables pour chaque classe)
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.proxies)
        
        # Configuration de l'optimiseur interne (avec des valeurs par défaut robustes)
        default_opt = {'name': 'AdamW', 'kwargs': {'lr': 1e-4, 'weight_decay': 1e-4}}
        optim_cfg = kwargs.get('optimizer', default_opt)
        self.loss_optimizer = get_optimizer_for_loss(self, optim_cfg)

    def forward(self, embeddings, labels, **kwargs):
        # Normalisation L2
        embeddings = torch.tanh(embeddings)  # Forcer les embeddings dans [-1, 1]
        norm_emb = F.normalize(embeddings, p=2, dim=1)
        norm_proxies = F.normalize(self.proxies, p=2, dim=1)
        
        # Similarité Cosinus + Scale
        sim_matrix = torch.matmul(norm_emb, norm_proxies.t())
        logits = sim_matrix * self.scale
        
        # 1. Perte BCE pour le Multi-Label
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        # 2. Perte de Quantification (Forçage binaire)
        quant_loss = torch.mean(torch.abs(torch.abs(embeddings) - 1.0))
        
        return bce_loss + (self.quant_weight * quant_loss)

    def step(self):
        """Met à jour les proxies indépendamment du réseau principal."""
        self.loss_optimizer.step()
        self.loss_optimizer.zero_grad()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Sauvegarde l'état des proxies ET de leur optimiseur."""
        sd = super().state_dict(destination, prefix, keep_vars)
        sd['optimizer_state'] = self.loss_optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        """Restaure l'état des proxies et de l'optimiseur lors de la reprise."""
        optimizer_state = state_dict.pop('optimizer_state', None)
        super().load_state_dict(state_dict, strict)
        if optimizer_state is not None:
            self.loss_optimizer.load_state_dict(optimizer_state)