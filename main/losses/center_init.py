"""
Hybrid center initialization for proxy-based hashing losses.

Context (see HashLossV3 in hash_loss.py): the original HashLoss / HashLossV2
initialize their class proxies with random Xavier init and let gradient
descent place them freely. With no geometric constraint on the initial
layout, nothing stops two classes' proxies from drifting towards each other
early in training (proxy collapse: feature crushing / center merging).

This module builds a *structured* initialization instead: it derives an
inter-class affinity matrix from signals we actually know about the classes,
then embeds that affinity into `embedding_size`-dimensional space via
classical (spectral) MDS, so proxies start out already spread apart in a way
that reflects real inter-class relationships rather than random chance.

Two affinity signals are supported and can be combined:
  - statistical: normalized pointwise mutual information (NPMI) between
    classes, estimated from label co-occurrence in the training set. Two
    classes that rarely/never co-occur get pushed apart; classes that
    co-occur a lot get pulled together. This is the piece implemented and
    wired up in this step.
  - semantic: cosine similarity between LLM/text embeddings of the class
    names, projected down to embedding_size via a random orthogonal
    projection. This is the natural next step but is NOT implemented yet
    (`semantic_embeddings=None` below is the intended default until then) —
    `build_hybrid_centers` already accepts it so wiring it in later doesn't
    require touching HashLossV3 again.

Everything here is pure numpy/torch, no dataset-specific code — the label
matrix is expected to already be extracted (see scripts/compute_class_cooccurrence.py
for a one-off script that dumps it from a dataset object).
"""
import numpy as np
import torch


def compute_npmi_matrix(label_matrix, eps=1e-8):
    """
    Normalized pointwise mutual information between classes, from a
    multi-hot label matrix.

    Args:
        label_matrix: (N, C) array-like, multi-hot (0/1) or hard-binarized
            (a threshold is not applied here — binarize before calling if
            your labels are soft/probabilistic).
        eps: numerical floor to avoid log(0) / div-by-0 for class pairs
            that never co-occur or never appear.

    Returns:
        npmi: (C, C) numpy array in [-1, 1]. npmi[i, i] = 1 by convention.
            npmi[i, j] close to 1 means i and j co-occur far more than
            chance; close to -1 means they co-occur far less than chance
            (in the limit, never together); close to 0 means ~independent.

    Formula: for classes i, j with marginals p(i), p(j) and joint p(i, j)
    estimated from frequencies over the N samples,
        pmi(i, j)  = log( p(i, j) / (p(i) * p(j)) )
        npmi(i, j) = pmi(i, j) / -log( p(i, j) )
    which rescales PMI into [-1, 1] so it's comparable across class pairs
    with very different base rates (important for VOC-style long-tailed
    multi-label data).
    """
    Y = np.asarray(label_matrix, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"label_matrix must be (N, C), got shape {Y.shape}")
    n, c = Y.shape

    co_occurrence = Y.T @ Y  # (C, C), co_occurrence[i, j] = #samples where both i and j are present
    marginal = np.diag(co_occurrence).copy()  # (C,), #samples where class i is present

    p_i = marginal / n
    p_ij = co_occurrence / n

    npmi = np.zeros((c, c), dtype=np.float64)
    for i in range(c):
        for j in range(c):
            if i == j:
                npmi[i, j] = 1.0
                continue
            joint = p_ij[i, j]
            denom = p_i[i] * p_i[j]
            if joint <= eps or denom <= eps:
                # never co-occur (or one/both classes unseen): treat as
                # maximally dissimilar rather than undefined.
                npmi[i, j] = -1.0
                continue
            pmi = np.log(joint / denom)
            npmi[i, j] = pmi / (-np.log(joint) + eps)

    return np.clip(npmi, -1.0, 1.0)


def classical_mds(affinity, dim, eps=1e-8):
    """
    Classical (spectral) MDS: embed C points into `dim`-dimensional
    Euclidean space such that their Gram matrix approximates `affinity`,
    i.e. find V (C, dim) with V @ V.T ~= affinity.

    Args:
        affinity: (C, C) symmetric similarity matrix, values roughly in
            [-1, 1] (an NPMI matrix, a semantic cosine-similarity matrix, or
            a convex combination of both — see build_hybrid_centers).
        dim: target embedding dimensionality (== embedding_size / bit count
            of the hash).
        eps: negative-eigenvalue floor; real-world affinity matrices are
            rarely exactly PSD, so tiny negative eigenvalues (numerical
            noise) are clipped to 0 rather than propagated as NaN/complex.

    Returns:
        V: (C, dim) numpy array. If C - 1 < dim (fewer classes than target
            dimensions, e.g. VOC's 20 classes vs. a 64-bit hash), the
            remaining coordinates are filled with small random Gaussian
            noise (not zeros) so proxies don't collapse onto a
            lower-dimensional subspace and lose full use of the hash code.

    Note on the "VᵀV = S structure" framing: what's produced here is V with
    V @ V.T ~= S (Gram/affinity), i.e. V's *rows* are the class centers.
    (V.T @ V is the dim x dim second-moment matrix of those centers, not
    the object being matched to S — worth being precise about which
    product is meant.)
    """
    S = np.asarray(affinity, dtype=np.float64)
    c = S.shape[0]
    if S.shape != (c, c):
        raise ValueError(f"affinity must be square, got shape {S.shape}")

    eigvals, eigvecs = np.linalg.eigh(S)  # ascending order, S is symmetric
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals_clipped = np.clip(eigvals, eps, None)

    n_usable = min(dim, c)
    V = eigvecs[:, :n_usable] * np.sqrt(eigvals_clipped[:n_usable])[None, :]

    if n_usable < dim:
        # More hash bits than classes: pad remaining dims with small noise
        # so every coordinate still carries some signal instead of being
        # identically 0 for every class (which the tanh/quantization loss
        # would then have no gradient to move away from).
        rng = np.random.default_rng(0)
        pad = rng.normal(scale=0.01, size=(c, dim - n_usable))
        V = np.concatenate([V, pad], axis=1)

    # Rescale so the embedding sits at roughly unit norm per coordinate on
    # average, matching the scale HashLossV3's tanh-bounded proxies expect
    # (proxies get passed through tanh(), so wildly large or tiny initial
    # magnitudes both waste the useful part of tanh's range).
    scale = np.sqrt(dim) / (np.linalg.norm(V, axis=1).mean() + eps)
    V = V * scale

    return V


def build_hybrid_centers(
    num_classes,
    embedding_size,
    label_matrix=None,
    semantic_embeddings=None,
    alpha=0.5,
    seed=None,
):
    """
    Build the initial (num_classes, embedding_size) proxy/center tensor by
    combining statistical and (eventually) semantic affinity signals and
    spectrally embedding the result — replacing HashLoss/V2's random Xavier
    init.

    Args:
        num_classes, embedding_size: same meaning as elsewhere in this
            codebase (C classes, D-bit hash).
        label_matrix: (N, C) multi-hot array, or None. When given, the
            statistical (NPMI) affinity is computed from it. When None, the
            statistical component is skipped.
        semantic_embeddings: (C, D_sem) array of per-class semantic vectors
            (e.g. from an LLM/text encoder over class names), or None.
            NOT YET WIRED UP — passing this currently raises NotImplementedError
            so it fails loudly instead of silently ignoring the argument;
            the projection + combination logic is the next step.
        alpha: weight in [0, 1] for the statistical affinity when both
            signals are eventually present (semantic gets 1 - alpha).
            Currently unused except as a guard (see below), since only the
            statistical signal is implemented.
        seed: if neither label_matrix nor semantic_embeddings is provided,
            centers fall back to the original random Xavier scheme, seeded
            for reproducibility.

    Returns:
        torch.FloatTensor of shape (num_classes, embedding_size).
    """
    if semantic_embeddings is not None:
        raise NotImplementedError(
            "Semantic (LLM-embedding) affinity is not wired up yet — "
            "this is the next step, not this one. Pass semantic_embeddings=None "
            "and use label_matrix for the statistical-only initialization."
        )

    if label_matrix is None:
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        centers = torch.empty(num_classes, embedding_size)
        torch.nn.init.xavier_uniform_(centers)
        return centers

    affinity = compute_npmi_matrix(label_matrix)
    if affinity.shape[0] != num_classes:
        raise ValueError(
            f"label_matrix implies {affinity.shape[0]} classes but num_classes={num_classes}"
        )

    V = classical_mds(affinity, dim=embedding_size)
    return torch.from_numpy(V).float()
