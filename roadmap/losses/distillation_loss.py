import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistillationLoss(nn.Module):
    def __init__(self, teacher_idx=0, student_idx=1, **kwargs):
        super().__init__()
        self.teacher_idx = teacher_idx
        self.student_idx = student_idx

    def forward(self, embeddings, labels=None):
        """
        embeddings: Liste de tenseurs [Batch, Dim] correspondant aux branches.
        """
        # 1. Récupération des embeddings
        # .detach() est CRUCIAL : on ne veut pas que l'erreur de l'élève perturbe le prof.
        # Le prof (LL) ne doit apprendre que de sa propre loss (SupAP), pas de la distillation.
        teacher_emb = embeddings[self.teacher_idx].detach()
        student_emb = embeddings[self.student_idx]

        # 2. Normalisation (Important pour DINOv2 / Retrieval)
        # On compare les vecteurs sur l'hypersphère (Cosine Similarity)
        teacher_norm = F.normalize(teacher_emb, p=2, dim=1)
        student_norm = F.normalize(student_emb, p=2, dim=1)

        # 3. Calcul de la Loss
        # On veut Maximiser la similarité Cosine <=> Minimiser (1 - Cosine)
        # (A * B).sum(1) calcule le produit scalaire par élément du batch
        cosine_sim = (teacher_norm * student_norm).sum(dim=1)
        loss = 1.0 - cosine_sim.mean()
        
        return loss