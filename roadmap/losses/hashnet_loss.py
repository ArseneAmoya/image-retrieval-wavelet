import os
import torch

import torch.nn as nn

import torch
import torch.nn as nn

class HashNetAdapter(nn.Module):
    """
    Adapter of HashNetLoss for the training loop with the current pipeline. The original HashNetLoss is designed to be used in a specific training loop, so this adapter allows us to use it in our more general training loop.
    The main idea is to keep track of the current epoch and update the scale parameter of the HashNetLoss every 'step_continuation' epochs, as described in the original paper. This way, we can benefit from the continuation method proposed by HashNet while using our existing training infrastructure.
    """
    takes_embeddings = True

    def __init__(self, embedding_size=64, alpha=0.1, step_continuation=20, batches_per_epoch=49, **kwargs):
        super().__init__()
        config = {"alpha": alpha, "step_continuation": step_continuation}
        self.criterion = HashNetLoss(config=config, bit=embedding_size)
        
        self.step_continuation = step_continuation
        
        # AJOUT : variables pour le suivi des batches
        self.batches_per_epoch = batches_per_epoch
        self.global_batch_step = 0

    def forward(self, embeddings, labels, **kwargs):
        config_dict = {"alpha": 0.1} # On réinjecte l'alpha pour la perte
        loss = self.criterion(u=embeddings, y=labels, ind=None, config=config_dict)
        return loss

    def step(self):
        # MODIFICATION : Incrémentation par batch plutôt que par epoch
        self.global_batch_step += 1
        
        # Calcul mathématique de l'époque réelle en cours
        current_real_epoch = self.global_batch_step // self.batches_per_epoch
        
        # Mise à jour du scale basée sur l'époque réelle
        self.criterion.scale = (current_real_epoch // self.step_continuation) + 1



# HashNet(ICCV2017)
# paper [HashNet: Deep Learning to Hash by Continuation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)
# code [HashNet caffe and pytorch](https://github.com/thuml/HashNet)

class HashNetLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(HashNetLoss, self).__init__()
        self.scale = 1

    def forward(self, u, y, ind, config):
        u = torch.tanh(self.scale * u)
        S = (y @ y.t() > 0).float()
        sigmoid_alpha = config["alpha"]
        dot_product = sigmoid_alpha * u @ u.t()
        mask_positive = S > 0
        mask_negative = (1 - S).bool()
        
        neg_log_probe = dot_product + torch.log(1 + torch.exp(-dot_product)) -  S * dot_product
        S1 = torch.sum(mask_positive.float())
        S0 = torch.sum(mask_negative.float())
        S = S0 + S1

        neg_log_probe[mask_positive] = neg_log_probe[mask_positive] * S / S1
        neg_log_probe[mask_negative] = neg_log_probe[mask_negative] * S / S0

        loss = torch.sum(neg_log_probe) / S
        return loss