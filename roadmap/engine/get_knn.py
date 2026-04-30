import numpy as np
import torch
import faiss
import pytorch_metric_learning.utils.common_functions as c_f

import roadmap.utils as lib


def get_knn(references, queries, num_k, embeddings_come_from_same_source, with_faiss=True, distance_metric="l2"):
    num_k += embeddings_come_from_same_source

    lib.LOGGER.info("running k-nn with k=%d" % num_k)
    lib.LOGGER.info("embedding dimensionality is %d" % references.size(-1))

    if with_faiss:
        distances, indices = get_knn_faiss(references, queries, num_k, distance_metric)
    else:
        distances, indices = get_knn_torch(references, queries, num_k, distance_metric)

    if embeddings_come_from_same_source:
        return indices[:, 1:], distances[:, 1:]

    return indices, distances


def get_knn_faiss(references, queries, num_k, distance_metric="l2"):
    lib.LOGGER.debug(f"Computing k-nn with faiss (metric: {distance_metric})")

    d = references.size(-1)
    device = references.device
    ref_np = c_f.to_numpy(references).astype(np.float32)
    que_np = c_f.to_numpy(queries).astype(np.float32)

    if distance_metric in ["hamming", "cosine"]:
        index = faiss.IndexFlatIP(d) 
    else:
        index = faiss.IndexFlatL2(d)

    try:
        if torch.cuda.device_count() > 1:
            co = faiss.GpuMultipleClonerOptions()
            co.shards = True
            index = faiss.index_cpu_to_all_gpus(index, co)
        else:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
    except AttributeError:
        pass

    index.add(ref_np)
    distances, indices = index.search(que_np, num_k)
    
    distances = c_f.to_device(torch.from_numpy(distances), device=device)
    indices = c_f.to_device(torch.from_numpy(indices), device=device)
    index.reset()
    return distances, indices


def get_knn_torch(references, queries, num_k, distance_metric="l2"):
    lib.LOGGER.debug(f"Computing k-nn with torch (metric: {distance_metric})")

    if distance_metric in ["hamming", "cosine"]:
        # Les vecteurs étant déjà binarisés ou normalisés, 
        # le produit scalaire donne directement la similarité[cite: 1]
        scores = queries @ references.t()
        distances, indices = torch.topk(scores, num_k, largest=True)
    else:
        # Distance Euclidienne (L2) : on minimise la distance[cite: 1]
        dist_matrix = torch.cdist(queries, references, p=2)
        distances, indices = torch.topk(dist_matrix, num_k, largest=False)

    return distances, indices
