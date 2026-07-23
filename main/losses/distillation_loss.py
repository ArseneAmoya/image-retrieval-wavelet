import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDistillationLoss(nn.Module):
    """Cosine distillation between a frozen teacher branch and a student branch."""
    requires_all_branches = True

    def __init__(self, teacher_idx=0, student_idx=1, **kwargs):
        super().__init__()
        self.teacher_idx = teacher_idx
        self.student_idx = student_idx

    def forward(self, embeddings, labels=None):
        # detach() is required: the teacher branch must only learn from its own loss (SupAP),
        # not from the distillation gradient.
        teacher_emb = embeddings[self.teacher_idx].detach()
        student_emb = embeddings[self.student_idx]

        teacher_norm = F.normalize(teacher_emb, p=2, dim=1)
        student_norm = F.normalize(student_emb, p=2, dim=1)

        cosine_sim = (teacher_norm * student_norm).sum(dim=1)
        loss = 1.0 - cosine_sim.mean()

        return loss
