import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import center_init

def get_optimizer_for_loss(loss_module, optim_cfg):
    params = list(loss_module.parameters())
    if len(params) == 0:
        return None
 
    if isinstance(optim_cfg, dict):
        opt_name = optim_cfg.get('name', 'AdamW')
        opt_kwargs = optim_cfg.get('kwargs', {})
    else:
        opt_name = getattr(optim_cfg, 'name', 'AdamW')
        opt_kwargs = getattr(optim_cfg, 'kwargs', {}) or {}
 
    optimizer = getattr(optim, opt_name)(params, **opt_kwargs)
    return optimizer

class HashLoss(nn.Module):
    """GSPH/CSQ-style proxy hashing loss with its own internal optimizer."""
    takes_embeddings = True

    def __init__(self, num_classes=20, embedding_size=64, quant_weight=0.1, scale=15.0,
                 log_proxy_diagnostics=False, **kwargs):
        super().__init__()
        self.quant_weight = quant_weight
        self.scale = scale
        self.log_proxy_diagnostics = log_proxy_diagnostics

        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.proxies)
        self.last_components = {}

        default_opt = {'name': 'AdamW', 'kwargs': {'lr': 1e-4, 'weight_decay': 1e-4}}
        optim_cfg = kwargs.get('optimizer', default_opt)
        self.loss_optimizer = get_optimizer_for_loss(self, optim_cfg)

    def forward(self, embeddings, labels, **kwargs):
        raw_embeddings = embeddings
        embeddings = torch.tanh(embeddings)
        norm_emb = F.normalize(embeddings, p=2, dim=1)
        norm_proxies = F.normalize(self.proxies, p=2, dim=1)

        sim_matrix = torch.matmul(norm_emb, norm_proxies.t())
        logits = sim_matrix * self.scale

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        quant_loss = torch.mean(torch.abs(torch.abs(embeddings) - 1.0))

        self.last_components = {
            "bce": bce_loss.detach(),
            "quant": quant_loss.detach(),
        }
        if self.log_proxy_diagnostics:
            diag = diagnostic_stats(raw_embeddings, self.proxies, labels)
            self.last_components.update({f"diag_{k}": v for k, v in diag.items()})

        return bce_loss + (self.quant_weight * quant_loss)

    def step(self):
        self.loss_optimizer.step()
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



class HashLossV2(nn.Module):
 
    takes_embeddings = True
 
    def __init__(self, num_classes=20, embedding_size=64, scale=2.0,
                 proxy_polarization_weight=0.05, log_proxy_diagnostics=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.scale = scale
        self.proxy_polarization_weight = proxy_polarization_weight
        self.log_proxy_diagnostics = log_proxy_diagnostics

        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.proxies)
        self.last_components = {}

        default_opt = {'name': 'AdamW', 'kwargs': {'lr': 1e-4, 'weight_decay': 1e-4}}
        optim_cfg = kwargs.get('optimizer', default_opt)
        self.loss_optimizer = get_optimizer_for_loss(self, optim_cfg)

    def forward(self, embeddings, labels, **kwargs):
        h = torch.tanh(embeddings)
        proxies_bounded = torch.tanh(self.proxies)

        logits = (h @ proxies_bounded.t()) / self.embedding_size * self.scale  # (N, C)

        bit_bce = F.binary_cross_entropy_with_logits(logits, labels.float())

        proxy_polarization = torch.mean((proxies_bounded.abs() - 1.0) ** 2)

        self.last_components = {
            "bit_bce": bit_bce.detach(),
            "proxy_polarization": proxy_polarization.detach(),
        }
        if self.log_proxy_diagnostics:
            diag = diagnostic_stats(embeddings, self.proxies, labels)
            self.last_components.update({f"diag_{k}": v for k, v in diag.items()})

        loss = bit_bce + self.proxy_polarization_weight * proxy_polarization
        return loss
 
    def step(self):
        if self.loss_optimizer is not None:
            self.loss_optimizer.step()
            self.loss_optimizer.zero_grad()
 
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        if self.loss_optimizer is not None:
            sd['optimizer_state'] = self.loss_optimizer.state_dict()
        return sd
 
    def load_state_dict(self, state_dict, strict=True):
        optimizer_state = state_dict.pop('optimizer_state', None)
        super().load_state_dict(state_dict, strict)
        if optimizer_state is not None and self.loss_optimizer is not None:
            self.loss_optimizer.load_state_dict(optimizer_state)
 
 
class HashLossV3(nn.Module):
    """
    Same forward pass as HashLossV2 (tanh-bounded proxies + scaled BCE +
    polarization — this is what fixed the cosine-normalization/quantization
    mismatch: no F.normalize, so per-coordinate magnitude survives into the
    quantization signal).

    What changes here is *only* the proxy initialization: instead of
    `nn.init.xavier_uniform_` (random, unconstrained — the thing diagnosed
    as enabling proxy collapse), proxies start at positions derived from
    real inter-class structure via main.losses.center_init.build_hybrid_centers
    (currently: NPMI label co-occurrence -> classical MDS; semantic LLM
    embeddings are a planned addition, not wired up yet).

    `freeze_centers` lets you run the two variants HashLossV2 already
    supports (trainable) against a frozen-center version for comparison:
      - freeze_centers=False (default): centers are structurally
        initialized but still gradient-learned, same as V2/HashLoss.
      - freeze_centers=True: centers stay exactly at their initialized
        positions (closer to a true hash-center loss, à la CSQ, rather
        than a proxy loss) — no loss_optimizer is created since there are
        no trainable parameters to step.
    """

    takes_embeddings = True

    def __init__(self, num_classes=20, embedding_size=64, scale=2.0,
                 proxy_polarization_weight=0.05, freeze_centers=False,
                 label_matrix_path=None, semantic_embeddings_path=None,
                 alpha=0.5, init_seed=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.scale = scale
        self.proxy_polarization_weight = proxy_polarization_weight
        self.freeze_centers = freeze_centers

        label_matrix = None
        if label_matrix_path is not None:
            label_matrix = np.load(label_matrix_path)

        semantic_embeddings = None
        if semantic_embeddings_path is not None:
            # Not implemented yet — build_hybrid_centers raises loudly
            # rather than silently ignoring this. Leave semantic_embeddings_path
            # unset in config until the semantic step lands.
            semantic_embeddings = np.load(semantic_embeddings_path)

        init_centers = center_init.build_hybrid_centers(
            num_classes=num_classes,
            embedding_size=embedding_size,
            label_matrix=label_matrix,
            semantic_embeddings=semantic_embeddings,
            alpha=alpha,
            seed=init_seed,
        )
        self.proxies = nn.Parameter(init_centers, requires_grad=not freeze_centers)

        self.loss_optimizer = None
        if not freeze_centers:
            default_opt = {'name': 'AdamW', 'kwargs': {'lr': 1e-4, 'weight_decay': 1e-4}}
            optim_cfg = kwargs.get('optimizer', default_opt)
            self.loss_optimizer = get_optimizer_for_loss(self, optim_cfg)

    def forward(self, embeddings, labels, **kwargs):
        h = torch.tanh(embeddings)
        proxies_bounded = torch.tanh(self.proxies)

        logits = (h @ proxies_bounded.t()) / self.embedding_size * self.scale  # (N, C)

        bit_bce = F.binary_cross_entropy_with_logits(logits, labels.float())

        proxy_polarization = torch.mean((proxies_bounded.abs() - 1.0) ** 2)

        loss = bit_bce + self.proxy_polarization_weight * proxy_polarization
        return loss

    def step(self):
        if self.loss_optimizer is not None:
            self.loss_optimizer.step()
            self.loss_optimizer.zero_grad()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        sd = super().state_dict(destination, prefix, keep_vars)
        if self.loss_optimizer is not None:
            sd['optimizer_state'] = self.loss_optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        optimizer_state = state_dict.pop('optimizer_state', None)
        super().load_state_dict(state_dict, strict)
        if optimizer_state is not None and self.loss_optimizer is not None:
            self.loss_optimizer.load_state_dict(optimizer_state)


class HashLossV2Quant(HashLossV2):
    """
    HashLossV2 + an embedding-level quantization term (same fixed-weight
    formula as HashLoss V1's quant_loss: mean((|h|-1)^2) on the tanh'd
    embeddings) added back in.

    Why: the mflickr_dino_hashlossv1_vs_v2_bs32 run (V1 vs V2, DINOv2
    ViT-S/14, batch_size=32) showed V1 beating V2 on maphashing_level0
    (82.75 vs 79.07 final) — not because of proxy collapse or the
    normalization mismatch V2 was built to fix, but because V2 has nothing
    penalizing an embedding coordinate for sitting away from saturation.
    bit_balance_level0 (fraction of hash bits that actually split ~50/50
    across the gallery, from main/engine/accuracy_calculator.py) collapsed
    from ~63% to ~41% over V2's training and dragged maphashing down with
    it in lockstep, while V1 (which keeps quant_weight=0.1 active) stayed
    flat at ~50-53% the whole run. V1's quant term was apparently doing
    real work, not just redundant regularization.

    Kept as its own class rather than a new kwarg on HashLossV2, so
    HashLossV2 / hash_loss_v2.yaml / the run already done stay exactly
    reproducible as they are.

    self.last_components (a dict, inherited pattern from HashLoss/HashLossV2)
    is populated after each forward so the training loop can log/plot every
    term — bit_bce, proxy_polarization, quant — as its own curve, separate
    from the combined total. See main/engine/base_update.py, which follows
    the same pattern already used there for fusion_head.last_ortho_loss.

    log_proxy_diagnostics=True (inherited kwarg, off by default) additionally
    merges main.losses.hash_loss.diagnostic_stats()'s output in, prefixed
    "diag_" (intra_class_variance, min_proxy_hamming_distance,
    mean_abs_embedding).
    """

    takes_embeddings = True

    def __init__(self, quant_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.quant_weight = quant_weight

    def forward(self, embeddings, labels, **kwargs):
        h = torch.tanh(embeddings)
        proxies_bounded = torch.tanh(self.proxies)

        logits = (h @ proxies_bounded.t()) / self.embedding_size * self.scale  # (N, C)

        bit_bce = F.binary_cross_entropy_with_logits(logits, labels.float())

        proxy_polarization = torch.mean((proxies_bounded.abs() - 1.0) ** 2)
        quant_loss = torch.mean((h.abs() - 1.0) ** 2)

        self.last_components = {
            "bit_bce": bit_bce.detach(),
            "proxy_polarization": proxy_polarization.detach(),
            "quant": quant_loss.detach(),
        }
        if self.log_proxy_diagnostics:
            diag = diagnostic_stats(embeddings, self.proxies, labels)
            self.last_components.update({f"diag_{k}": v for k, v in diag.items()})

        loss = (
            bit_bce
            + self.proxy_polarization_weight * proxy_polarization
            + self.quant_weight * quant_loss
        )
        return loss


@torch.no_grad()
def diagnostic_stats(embeddings, proxies, labels):
    h = torch.tanh(embeddings)
    proxies_bounded = torch.tanh(proxies)

    # Problème #3 (écrasement) : variance intra-classe des embeddings.
    # Si elle s'effondre vers 0, tous les échantillons d'une classe collapsent
    # sur le même point.
    #
    # Membership-based (each class's own multi-hot column), not argmax-based:
    # argmax(dim=1) would collapse every sample to a single "winning" class
    # (whichever active label has the lowest index, since multi-hot entries
    # tie at 1.0) -- wrong on true multi-label data (VOC/MIRFLICKR), where a
    # sample active in several classes should contribute to every one of
    # them, not just the lowest-indexed. For single-label / one-hot labels
    # this produces exactly the same groupings as the old argmax version.
    if labels.dim() == 1:
        label_matrix = F.one_hot(labels, num_classes=proxies.shape[0]).float()
    else:
        label_matrix = labels

    intra_class_var = 0.0
    n_classes_seen = 0
    for c in range(label_matrix.shape[1]):
        mask = label_matrix[:, c] == 1
        if mask.sum() > 1:
            intra_class_var += h[mask].var(dim=0).mean().item()
            n_classes_seen += 1
    intra_class_var = intra_class_var / max(n_classes_seen, 1)
 
    # Problème #1/fusion : plus petite distance de Hamming entre deux proxies
    # distincts (sur leur version binarisée). Une valeur qui chute vers 0 au
    # fil de l'entraînement signale une fusion en cours.
    binary_proxies = torch.sign(proxies_bounded)
    b = binary_proxies.shape[1]
    pairwise_hamming = (b - binary_proxies @ binary_proxies.t()) / 2
    pairwise_hamming.fill_diagonal_(float('inf'))
    min_proxy_hamming = pairwise_hamming.min().item()
 
    # Qualité de polarisation : à quel point les coordonnées des embeddings
    # sont proches de ±1 (donc peu de risque de bascule au sign() final).
    mean_abs_embedding = h.abs().mean().item()
 
    return {
        'intra_class_variance': intra_class_var,
        'min_proxy_hamming_distance': min_proxy_hamming,
        'mean_abs_embedding': mean_abs_embedding,
    }
 