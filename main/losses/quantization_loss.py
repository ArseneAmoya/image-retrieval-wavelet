import torch
import torch.nn as nn
from .. import utils as lib

class QuantizationLoss(nn.Module):
    def __init__(self, target_value=1.0, step_type='linear', steps=None, alpha=1.0, starting_weight=0.0001, warmup_step=False, **kwargs):
        """
        Calcul la Quantization Loss pour forcer les sorties vers -1 ou 1.
        Formule: mean( (|x| - 1)^2 )
        """
        super().__init__()
        self.target_value = target_value
        self.actual_step = 0
        self.current_weight = 1.0

        if step_type == 'linear':
            self.epoch_step = self.linear_step
            assert isinstance(warmup_step, int), "Warmup step must be an integer for linear step type."
            self.warmup_step = warmup_step
            assert steps is not None, "Steps must be provided for linear step type."
            self.ep_steps = steps
            self.current_weight = 0.0  # Start with no loss and increase linearly
        elif step_type == 'multi':
            self.epoch_step = self.multi_step
            assert isinstance(warmup_step, bool), "Warmup step must be a boolean for multi step type."
            self.warmup_step = warmup_step
            steps = list(steps) if steps is not None else []
            assert isinstance(steps, list) and all(isinstance(s, int) for s in steps), "Steps must be a list of integers for multi step type."
            self.ep_steps = steps
            self.current_weight = 0.0 if warmup_step else starting_weight  # Start with no loss if warmup, otherwise start with full loss
            self.starting_weight = starting_weight
            assert alpha > 1, "Alpha must be a positive value for multi step type."
            assert len(steps) > 0, "At least one step must be provided for multi step type."
            assert all(s > 0 for s in steps), "All steps must be positive integers."
        else:
            raise ValueError("Invalid step_type. Must be 'linear' or 'multi'.")

        self.ep_steps = steps
        self.alpha = alpha  # Poids de la loss (peut être ajusté dans la config)
        self.warmup_step = warmup_step
        self.weight_step = self.linear_step
    
    def linear_step(self):
        self.actual_step += 1
        self.current_weight = max(0.0, min(1.0, (self.actual_step - self.warmup_step) / self.ep_steps))
        lib.LOGGER.info(f"Quantization Loss weight updated to {self.current_weight:.6f} at step {self.actual_step}")

    def multi_step(self):
        self.actual_step += 1
        if len(self.ep_steps) and self.actual_step >= self.ep_steps[0]:
            self.current_weight = self.starting_weight if self.current_weight == 0.0 else self.current_weight * self.alpha
            del self.ep_steps[0]  # Retire la première étape une fois atteinte
        lib.LOGGER.info(f"Quantization Loss weight updated to {self.current_weight:.6f} at step {self.actual_step}")


    def forward(self, embeddings, labels=None, **kwargs):
        """
        Args:
            embeddings: Tensor (Batch, Dim). Doit être la sortie 'tanh' (entre -1 et 1).
            labels: Ignoré (c'est une loss non-supervisée).
        """
        # On calcule l'écart par rapport à 1.0 (en valeur absolue)
        # Si x = 0.99 (bien binarisé) -> |0.99| - 1 = -0.01 -> au carré = 0.0001 (Faible loss)
        # Si x = 0.1 (indécis) -> |0.1| - 1 = -0.9 -> au carré = 0.81 (Forte loss)
        q_loss = self.current_weight * torch.mean((torch.abs(embeddings) - self.target_value) ** 2)
        
        return q_loss