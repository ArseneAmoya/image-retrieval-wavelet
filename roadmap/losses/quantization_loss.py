import torch
import torch.nn as nn

class QuantizationLoss(nn.Module):
    def __init__(self, target_value=1.0, **kwargs):
        """
        Calcul la Quantization Loss pour forcer les sorties vers -1 ou 1.
        Formule: mean( (|x| - 1)^2 )
        """
        super().__init__()
        self.target_value = target_value

    def forward(self, embeddings, labels=None, **kwargs):
        """
        Args:
            embeddings: Tensor (Batch, Dim). Doit être la sortie 'tanh' (entre -1 et 1).
            labels: Ignoré (c'est une loss non-supervisée).
        """
        # On calcule l'écart par rapport à 1.0 (en valeur absolue)
        # Si x = 0.99 (bien binarisé) -> |0.99| - 1 = -0.01 -> au carré = 0.0001 (Faible loss)
        # Si x = 0.1 (indécis) -> |0.1| - 1 = -0.9 -> au carré = 0.81 (Forte loss)
        q_loss = torch.mean((torch.abs(embeddings) - self.target_value) ** 2)
        
        return q_loss