"""Embedding-space dispersion of beam centroids (spec §13.5).

Tree divergence measures hierarchy separation; centroid dispersion measures *continuous
embedding-space* separation. The two must both be measured: a hierarchical id can diverge
early without the corresponding regions being far apart in the dense space (spec §13.5
"Important"). Each beam maps (via a prefix) to a centroid c(z) = mean dense embedding of
the items under that prefix; we measure how spread out those centroids are.
"""

from __future__ import annotations

import numpy as np


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_distance_matrix(centroids: np.ndarray) -> np.ndarray:
    """Pairwise cosine distances (1 - cos) among centroids. Shape (n, n)."""
    c = _l2_normalize(np.asarray(centroids, dtype=np.float64))
    sim = c @ c.T
    return 1.0 - sim


def mean_pairwise_cosine_dispersion(centroids: np.ndarray) -> float:
    """Mean off-diagonal cosine distance among beam centroids. <2 centroids -> 0.0."""
    c = np.asarray(centroids, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 2:
        return 0.0
    d = cosine_distance_matrix(c)
    n = c.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(d[iu].mean())


def max_pairwise_cosine_dispersion(centroids: np.ndarray) -> float:
    c = np.asarray(centroids, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] < 2:
        return 0.0
    d = cosine_distance_matrix(c)
    n = c.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(d[iu].max())


def weighted_centroid_dispersion(centroids: np.ndarray, weights: np.ndarray) -> float:
    """Probability-weighted dispersion: sum_i w_i * cos_dist(c_i, weighted_mean_centroid).

    Uses the L2-normalized weighted mean as the reference direction, so it captures how far
    the (probability-weighted) beam centroids spread around their consensus.
    """
    c = _l2_normalize(np.asarray(centroids, dtype=np.float64))
    w = np.asarray(weights, dtype=np.float64)
    if c.shape[0] < 2 or w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    mean_dir = _l2_normalize((w[:, None] * c).sum(axis=0, keepdims=True))[0]
    cos_dist = 1.0 - c @ mean_dir
    return float((w * cos_dist).sum())


def query_to_centroid_distances(query_vec: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Cosine distance from the base dense query vector to each beam centroid (spec §13.5)."""
    q = _l2_normalize(np.asarray(query_vec, dtype=np.float64).reshape(1, -1))
    c = _l2_normalize(np.asarray(centroids, dtype=np.float64))
    return (1.0 - (c @ q.T).ravel())
