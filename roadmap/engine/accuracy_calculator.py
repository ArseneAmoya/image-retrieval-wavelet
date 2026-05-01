import torch
import pytorch_metric_learning.utils.common_functions as c_f
from pytorch_metric_learning.utils.accuracy_calculator import (
    AccuracyCalculator,
    get_label_match_counts,
    get_lone_query_labels,
)
from torchmetrics.retrieval import RetrievalRPrecision, RetrievalMAP, RetrievalPrecisionRecallCurve, RetrievalPrecision
import roadmap.utils as lib
from .get_knn import get_knn
import pandas as pd

EQUALITY = torch.eq


class CustomCalculator(AccuracyCalculator):

    def __init__(
        self,
        *args,
        with_faiss=True,
        distance_metric="l2",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.with_faiss = with_faiss
        self.distance_metric = distance_metric
        self.num_top_k = kwargs.get('k', None)

    def label_comparison_fn(self, query_labels, reference_labels):
        if query_labels.ndim > 1 and reference_labels.ndim > 1:
            if query_labels.dim() == 2 and reference_labels.dim() == 2:
                # Évaluation d'une galerie complète
                return torch.matmul(query_labels.float(), reference_labels.t().float()) > 0
            else:
                # Évaluation locale (ex: query_labels[:, None] contre knn_labels)
                return (query_labels.float() * reference_labels.float()).sum(dim=-1) > 0
        return query_labels.unsqueeze(1) == reference_labels
    
    def n_relevance_at_k(self, knn_labels, query_labels, k):
        r = self.label_comparison_fn(query_labels, knn_labels[:, :k])
        return r.float().sum(1)
    
    # def calculate_rc_from_1_to_10(self, knn_labels, query_labels, label_counts, **kwargs):
    #     rc_values = []
    #     for i in range(1, 11):
    #         rc = self.rc_at_k(knn_labels, query_labels, i)
    #         rc_values.append(rc.mean())
    #     return rc_values
    
    def recall_at_k(self, knn_labels, query_labels, k):
        recall = self.label_comparison_fn(query_labels, knn_labels[:, :k])
        return recall.any(1).float().mean().item()

    def calculate_recall_at_1(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(
            knn_labels,
            query_labels[:, None],
            1,
        )

    def calculate_recall_at_2(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(
            knn_labels,
            query_labels[:, None],
            2,
        )

    def calculate_recall_at_4(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(
            knn_labels,
            query_labels[:, None],
            4,
        )

    def calculate_recall_at_8(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(
            knn_labels,
            query_labels[:, None],
            8,
        )

    def calculate_recall_at_10(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(
            knn_labels,
            query_labels[:, None],
            10,
        )

    def calculate_recall_at_16(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(
            knn_labels,
            query_labels[:, None],
            16,
        )

    def calculate_recall_at_20(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(
            knn_labels,
            query_labels[:, None],
            20,
        )

    def calculate_recall_at_30(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(
            knn_labels,
            query_labels[:, None],
            30,
        )

    def calculate_recall_at_32(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(
            knn_labels,
            query_labels[:, None],
            32,
        )

    def calculate_recall_at_100(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(
            knn_labels,
            query_labels[:, None],
            100,
        )

    def calculate_recall_at_1000(self, knn_labels, query_labels, **kwargs):
        return self.recall_at_k(
            knn_labels,
            query_labels[:, None],
            1000,
        )

    def calculate_rpr(self, query_labels, knn_labels, knn_distances, not_lone_query_mask, **kwargs):
        r_precision = RetrievalRPrecision()
        relevances = self.label_comparison_fn(query_labels[:, None], knn_labels)
        indexes = torch.arange(query_labels.size(0), device=query_labels.device).unsqueeze(1).repeat(1, knn_labels.size(1))
        mask = not_lone_query_mask.unsqueeze(1).expand_as(knn_labels)
        dists = knn_distances[mask] if self.distance_metric in ["hamming", "cosine"] else (1/(knn_distances[mask]+1))  # Avoid division by zero
        dists += torch.arange(dists.size(0), device=dists.device).float() * 1e-8 # Avoid torchmetrics deleting samples with 0 similarity
        return r_precision(
            preds=dists,
            target=relevances[mask],
            indexes= indexes[mask]
        ).item()
    
    def calculate_pr(self, query_labels, knn_labels, knn_distances, not_lone_query_mask, **kwargs):
        r_precision = RetrievalPrecision(top_k=1)
        relevances = self.label_comparison_fn(query_labels[:, None], knn_labels)
        indexes = torch.arange(query_labels.size(0), device=query_labels.device).unsqueeze(1).repeat(1, knn_labels.size(1))
        mask = not_lone_query_mask.unsqueeze(1).expand_as(knn_labels)
        dists = knn_distances[mask] if self.distance_metric in ["hamming", "cosine"] else (1/(knn_distances[mask]+1))  # Avoid division by zero
        return r_precision(
            preds=dists,
            target=relevances[mask],
            indexes= indexes[mask],
        ).item()
    
    def calculate_map(self, query_labels, knn_labels, knn_distances,not_lone_query_mask,  **kwargs):
        r_map = RetrievalMAP()
        relevances = self.label_comparison_fn(query_labels[:, None], knn_labels)
        indexes = torch.arange(query_labels.size(0), device=query_labels.device).unsqueeze(1).repeat(1, knn_labels.size(1))
        mask = not_lone_query_mask.unsqueeze(1).expand_as(knn_labels)
        dists = knn_distances[mask] if self.distance_metric in ["hamming", "cosine"] else (1/(knn_distances[mask]+1))  # Avoid division by zero
        return r_map(
            preds=dists,
            target=relevances[mask],
            indexes= indexes[mask],
        ).item()
    
    def calculate_pr_rc(self, query_labels, knn_labels, knn_distances,not_lone_query_mask,  **kwargs):
        pr_rc = RetrievalPrecisionRecallCurve()
        relevances = self.label_comparison_fn(query_labels[:, None], knn_labels)
        indexes = torch.arange(query_labels.size(0), device=query_labels.device).unsqueeze(1).repeat(1, knn_labels.size(1))
        mask = not_lone_query_mask.unsqueeze(1).expand_as(knn_labels)
        dists = knn_distances[mask] if self.distance_metric in ["hamming", "cosine"] else (1/(knn_distances[mask]+1))  # Avoid division by zero
        pr, rc, _ = pr_rc(preds=dists,  # Avoid division by zero
            target=relevances[mask],
            indexes= indexes[mask]
        )

        pd.DataFrame({"pr": pr.cpu().numpy(), "rc":rc.cpu().numpy()}).to_csv("pr_rc.csv")
        return 0
    
    def calc_hamming_dist(self, qB, rB):
        q = qB.shape[1]
        dist_h = 0.5 * (q - torch.matmul(qB, rB.t()))
        return dist_h


    def calculate_maphashing(self, query, query_labels, reference, reference_labels, topk, **kwargs):
        num_query = query.shape[0]
        topkmap = 0.0

        for i in range(num_query):
            # Ground Truth (gnd) : produit scalaire des labels > 0
            gnd = (torch.matmul(query_labels[i], reference_labels.t()) > 0).float()
            
            # Calcul de distance et tri
            hamm = self.calc_hamming_dist(query[i:i+1], reference).squeeze()
            indices = torch.argsort(hamm)
            gnd = gnd[indices]

            # Sélection du Top-K et calcul de tsum
            tgnd = gnd[0:topk]
            tsum = torch.sum(tgnd).int().item()
            
            if tsum > 0:
                # Rangs des éléments positifs (commençant à 1.0)
                tindex = torch.where(tgnd == 1)[0].float() + 1.0
                # Numérateur basé sur le nombre de positifs trouvés
                count = torch.arange(1, tsum + 1, device=query.device).float()
                topkmap += torch.mean(count / tindex).item()

        return topkmap / num_query
    
    import pandas as pd

    def calculate_pr_rc_hashing(self, query, query_labels, reference, reference_labels, not_lone_query_mask, **kwargs):

        device = query.device
        num_query = query.shape[0]
        num_gallery = reference.shape[0]
        
        all_prec = torch.zeros((num_query, num_gallery), device=device)
        all_recall = torch.zeros((num_query, num_gallery), device=device)

        for i in range(num_query):
            # Ground Truth et Tri
            gnd = (torch.matmul(query_labels[i], reference_labels.t()) > 0).float()
            hamm = self.calc_hamming_dist(query[i:i+1], reference).squeeze()
            indices = torch.argsort(hamm)
            gnd = gnd[indices]

            # Calcul cumulé pour Precision et Recall
            all_sim_num = torch.sum(gnd)
            if all_sim_num > 0:
                prec_sum = torch.cumsum(gnd, dim=0)
                return_images = torch.arange(1, num_gallery + 1, device=device).float()
                
                all_prec[i, :] = prec_sum / return_images
                all_recall[i, :] = prec_sum / all_sim_num

        # Filtrage : not_lone_query_mask ET rappel final == 1.0
        # On identifie les indices qui respectent les deux conditions
        valid_recall = torch.where(all_recall[:, -1] == 1.0)[0]
        valid_queries = torch.where(not_lone_query_mask)[0]
        
        # Intersection des indices valides
        combined_indices = [idx for idx in valid_queries.tolist() if idx in valid_recall.tolist()]

        if len(combined_indices) > 0:
            cum_prec = torch.mean(all_prec[combined_indices], dim=0)
            cum_recall = torch.mean(all_recall[combined_indices], dim=0)

            # Enregistrement dans pr_rc.csv
            pd.DataFrame({
                "pr": cum_prec.cpu().numpy(),
                "rc": cum_recall.cpu().numpy()
            }).to_csv("pr_rc.csv", index=False)
        
        return 0


    def requires_knn(self):
        return super().requires_knn() + ["recall_classic","rpr", "pr", 'pr_rc']

    def get_accuracy(
        self,
        query,
        query_labels,
        reference,
        reference_labels,
        embeddings_come_from_same_source,
        include=(),
        exclude=(),
        return_indices=False,
    ):
        [query, reference, query_labels, reference_labels] = [
            c_f.numpy_to_torch(x)
            for x in [query, reference, query_labels, reference_labels]
        ]

        # Debug: print shapes and dtypes
        # print("query_labels shape:", query_labels.shape, "reference_labels shape:", reference_labels.shape)
        # print("query_labels dtype:", query_labels.dtype, "reference_labels dtype:", reference_labels.dtype)
        # Flatten if not 1D
        if query_labels.ndim == 1 or (query_labels.ndim == 2 and query_labels.size(1) == 1):
            query_labels = query_labels.view(-1)
            reference_labels = reference_labels.view(-1)

        self.curr_function_dict = self.get_function_dict(include, exclude)

        kwargs = {
            "query": query,
            "reference": reference,
            "query_labels": query_labels,
            "reference_labels": reference_labels,
            "embeddings_come_from_same_source": embeddings_come_from_same_source,
            "label_comparison_fn": self.label_comparison_fn,
            "ref_includes_query": embeddings_come_from_same_source,
        }

        if any(x in self.requires_knn() for x in self.get_curr_metrics()):
            label_counts = get_label_match_counts(
                query_labels, reference_labels, self.label_comparison_fn,
            )

            lone_query_labels, not_lone_query_mask = get_lone_query_labels(
                query_labels,
                label_counts,
                embeddings_come_from_same_source,
                self.label_comparison_fn,
            )

            num_k = self.determine_k(
                label_counts[1], len(reference), embeddings_come_from_same_source
            )

            # USE OUR OWN KNN SEARCH
            knn_indices, knn_distances = get_knn(
                reference, query, num_k, embeddings_come_from_same_source,
                with_faiss=self.with_faiss,
                distance_metric=self.distance_metric,
            )
            torch.cuda.empty_cache()

            knn_labels = reference_labels[knn_indices]
            if not any(not_lone_query_mask):
                lib.LOGGER.warning("None of the query labels are in the reference set.")
            kwargs["label_counts"] = label_counts
            kwargs["knn_labels"] = knn_labels
            kwargs["knn_distances"] = knn_distances
            kwargs["lone_query_labels"] = lone_query_labels
            kwargs["not_lone_query_mask"] = not_lone_query_mask
            kwargs["topk"] = self.num_top_k,

        if any(x in self.requires_clustering() for x in self.get_curr_metrics()):
            kwargs["cluster_labels"] = self.get_cluster_labels(**kwargs)

        if return_indices:
            # ADDED
            return knn_indices, self._get_accuracy(self.curr_function_dict, **kwargs)
        return self._get_accuracy(self.curr_function_dict, **kwargs)


def get_accuracy_calculator(
    exclude_ranks=None,
    k=19581,
    with_AP=True,
    **kwargs,
):
    exclude = kwargs.pop('exclude', [])
    if with_AP:
        exclude.extend(['NMI', 'AMI'])
    else:
        exclude.extend(['NMI', 'AMI', 'mean_average_precision', 'mean_average_precision_at_r'])

    if exclude_ranks:
        for r in exclude_ranks:
            exclude.append(f'recall_at_{r}')

    return CustomCalculator(
        exclude=exclude,
        k=k,
        **kwargs,
    )
