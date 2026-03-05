import torch
from pytorch_metric_learning import losses, distances
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class CalibrationLoss(losses.ContrastiveLoss):
    takes_embeddings = True

    def get_default_distance(self):
        return distances.DotProductSimilarity()

    def forward(self, embeddings, labels, ref_embeddings=None, ref_labels=None):
        if ref_embeddings is None:
            if labels.ndim > 1:
                matches = torch.matmul(labels.float(), labels.t().float()) > 0
                matches.fill_diagonal_(False)
                a1, p = torch.where(matches)
                a2, n = torch.where(~matches)
                return super().forward(embeddings, labels, (a1, p, a2, n))
            return super().forward(embeddings, labels)

        indices_tuple = self.create_indices_tuple(
            embeddings.size(0),
            embeddings,
            labels,
            ref_embeddings,
            ref_labels,
        )

        combined_embeddings = torch.cat([embeddings, ref_embeddings], dim=0)
        combined_labels = torch.cat([labels, ref_labels], dim=0)
        return super().forward(combined_embeddings, combined_labels, indices_tuple)

    def create_indices_tuple(
        self,
        batch_size,
        embeddings,
        labels,
        E_mem,
        L_mem,
    ):
        if labels.ndim == 1:
            indices_tuple = lmu.get_all_pairs_indices(labels, L_mem)
        else:
            matches = torch.matmul(labels.float(), L_mem.t().float()) > 0
            a1, p = torch.where(matches)
            a2, n = torch.where(~matches)
            indices_tuple = (a1, p, a2, n)
        
        indices_tuple = c_f.shift_indices_tuple(indices_tuple, batch_size)
        return indices_tuple
