import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy(nn.CrossEntropyLoss):
    takes_embeddings = True
    takes_logits = True

    def __str__(self):
        return f"CrossEntropyLoss with label_smoothing={self.label_smoothing}"