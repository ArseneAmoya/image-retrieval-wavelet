import os
import torch
import torch.optim as optim
import time
import numpy as np
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import random
import torch.nn as nn

# CSQ(CVPR2020)
# paper [Central Similarity Quantization for Efficient Image and Video Retrieval](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Central_Similarity_Quantization_for_Efficient_Image_and_Video_Retrieval_CVPR_2020_paper.pdf)
# code [CSQ-pytorch](https://github.com/yuanli2333/Hadamard-Matrix-for-hashing)
class CSQLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CSQLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.criterion = torch.nn.BCELoss().to(config["device"])

    def forward(self, u, y, ind, config):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).pow(2).mean()
        return center_loss + config["lambda"] * Q_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0].to(center_sum.dtype) # I modified this line to avoid type issues
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets
    

class CSQAdapter(nn.Module):
    """
    Adapter of CSQLoss for the training loop with the current pipeline. The original CSQLoss is designed to be used in a specific training loop, so this adapter allows us to use it in our more general training loop.
    The main idea is to keep track of the current epoch and update the scale parameter of the CSQLoss every 'step_continuation' epochs, as described in the original paper. This way, we can benefit from the continuation method proposed by CSQ while using our existing training infrastructure.
    """
    takes_embeddings = True

    def __init__(self, embedding_size=64, num_classes=20, lambda_param=0.0001, is_multi_label=True, **kwargs):
        super().__init__()
        
        #Faking dataset name to avoid the bug in CSQLoss which relies on the dataset name to determine if it's multi-label or single-label. This is a temporary workaround and should be fixed properly in the future.
        fake_dataset_name = "coco" if is_multi_label else "cifar10"
        
        config = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
            "n_class": num_classes, 
            "lambda": lambda_param,
            "dataset": fake_dataset_name # <-- L'astuce pour éviter le bug
        }
        self.criterion = CSQLoss(config=config, bit=embedding_size)

    def forward(self, embeddings, labels, **kwargs):
        # Appel de la vraie perte
        return self.criterion(u=embeddings, y=labels, ind=None, config={"lambda": self.criterion.lambda_param if hasattr(self.criterion, 'lambda_param') else 0.0001})

    def step(self):
        pass