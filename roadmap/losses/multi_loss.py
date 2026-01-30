import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from .. import utils as lib
# Assure-toi que ces imports fonctionnent dans ton arborescence
from .cross_entropy_loss import CrossEntropy # J'ai harmonisé le nom
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

class MultiLoss(nn.Module):
    takes_embeddings = True

    def __init__(self, criterion: list, weights: list = None):
        """
        criterion: Liste de listes de configurations (une liste par branche).
        weights: Liste des poids globaux pour chaque branche (ex: [1.0, 1.0, 1.0, 1.0]).
        """
        super().__init__()
        
        # Si aucun poids de branche n'est fourni, on met 1.0 partout
        if weights is None:
            weights = [1.0] * len(criterion)
            
        self.branch_weights = weights
        
        # IMPORTANT : Utiliser ModuleList pour que PyTorch enregistre les sous-modules
        self.losses = nn.ModuleList() 
        self.per_loss_weights = [] # Liste de listes pour stocker les poids internes (float)

        for i, branch_config in enumerate(criterion):
            # Récupération des losses et de leurs poids pour cette branche
            modules, internal_weights = self.get_loss_for_branch(branch_config)
            
            self.losses.append(modules)
            self.per_loss_weights.append(internal_weights)

    def get_loss_for_branch(self, config):
        """Instancie les losses pour une branche donnée."""
        modules = nn.ModuleList()
        weights = []
        
        for crit in config:
            # Instanciation dynamique via le nom de la classe
            loss_cls = getattr(sys.modules[__name__], crit.name)
            loss_instance = loss_cls(**crit.kwargs)
            
            weight = crit.weight
            lib.LOGGER.info(f"Adding {crit.name} with weight {weight}")
            
            modules.append(loss_instance)
            weights.append(weight)
            
        return modules, weights
    
    def forward(self, embeddings, labels):
        """
        embeddings: Liste de tenseurs [emb_branche_1, emb_branche_2, ...]
        labels: Tenseur de labels
        """
        total_loss = 0.0
        
        # 1. Boucle sur les branches (ex: LL, LH, HL, HH)
        for i, branch_emb in enumerate(embeddings):
            # Si on a plus de branches que de config, on arrête (sécurité)
            if i >= len(self.losses): break
            
            branch_total = 0.0
            branch_weight = self.branch_weights[i]
            
            # 2. Boucle sur les losses de cette branche (ex: SupAP + Calibration)
            for j, criterion in enumerate(self.losses[i]):
                crit_weight = self.per_loss_weights[i][j]
                
                # Calcul de la Loss individuelle
                if hasattr(criterion, 'takes_embeddings') and criterion.takes_embeddings:
                    loss = criterion(branch_emb, labels.view(-1))
                else:
                    # Cas métrique basée sur matrice de similarité (ex: FastAP parfois)
                    scores = torch.mm(branch_emb, branch_emb.t())
                    # Attention: create_label_matrix peut être coûteux, vérifier s'il faut le faire à chaque fois
                    label_matrix = lib.create_label_matrix(labels, labels)
                    loss = criterion(scores, label_matrix)
                
                # Gestion des losses qui renvoient un tuple (loss, logs) ou dict
                if isinstance(loss, (tuple, list)):
                    loss = loss[0]
                
                # Accumulation pondérée
                # On fait la moyenne si le tenseur n'est pas scalaire
                if loss.dim() > 0:
                    loss = loss.mean()
                    
                branch_total += crit_weight * loss

            # Ajout au total global pondéré par la branche
            total_loss += branch_weight * branch_total
            
        return total_loss