"""Transparent, training-free hierarchical k-means semantic IDs (spec §12.2).

An alternative to RQ-VAE ids that needs no training: recursively k-means the item
embeddings, forming a balanced-ish tree of branching factor B and depth D. Each item gets
an id = the path of cluster indices from root to leaf. For the planner use case a leaf need
not be unique — a prefix may cover many items, since ANN does final retrieval (spec §12.2).

This module is the reference id system for early diagnostics and as an ablation against the
RQ-VAE ids shipped by LIGER. It also exposes the prefix -> descendants / centroid tables
that the diagnostics (centroid dispersion, prefix survival) consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class HierarchicalKMeansIDs:
    branching: int
    depth: int
    ids: dict[str, tuple[int, ...]]  # item_id -> semantic id path
    centroids: dict[tuple[int, ...], np.ndarray]  # prefix -> mean embedding of descendants
    members: dict[tuple[int, ...], list[str]] = field(default_factory=dict)  # prefix -> item ids

    def descendants(self, prefix: tuple[int, ...]) -> list[str]:
        return self.members.get(tuple(prefix), [])

    def descendant_count(self, prefix: tuple[int, ...]) -> int:
        return len(self.members.get(tuple(prefix), []))

    def centroid(self, prefix: tuple[int, ...]) -> np.ndarray | None:
        return self.centroids.get(tuple(prefix))


def _kmeans(x: np.ndarray, k: int, seed: int, n_init: int = 4) -> np.ndarray:
    """Return integer cluster labels for rows of x. Falls back gracefully when k>=n."""
    n = x.shape[0]
    if n <= k:
        return np.arange(n) % k
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    return km.fit_predict(x)


def build_hierarchical_kmeans(
    item_ids: list[str],
    embeddings: np.ndarray,
    branching: int = 32,
    depth: int = 3,
    min_leaf_size: int = 1,
    seed: int = 0,
) -> HierarchicalKMeansIDs:
    """Build hierarchical k-means semantic ids over item embeddings.

    Recursively clusters into ``branching`` groups per node down to ``depth`` levels, or
    until a node has <= ``min_leaf_size`` items. Returns the id map plus prefix -> centroid
    and prefix -> members tables. Deterministic given ``seed``.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape[0] != len(item_ids):
        raise ValueError("item_ids and embeddings must align")

    ids: dict[str, list[int]] = {iid: [] for iid in item_ids}
    centroids: dict[tuple[int, ...], np.ndarray] = {}
    members: dict[tuple[int, ...], list[str]] = {}

    # Root centroid / membership.
    all_idx = np.arange(len(item_ids))
    centroids[()] = embeddings.mean(axis=0)
    members[()] = list(item_ids)

    def recurse(idx: np.ndarray, prefix: tuple[int, ...], level: int) -> None:
        if level >= depth or len(idx) <= max(min_leaf_size, 1) or len(idx) <= 1:
            return
        # Deterministic seed per node from the prefix.
        node_seed = (seed * 1000003 + hash(prefix)) % (2**31 - 1)
        labels = _kmeans(embeddings[idx], branching, node_seed)
        for c in range(branching):
            child_local = np.where(labels == c)[0]
            if child_local.size == 0:
                continue
            child_idx = idx[child_local]
            child_prefix = prefix + (c,)
            child_items = [item_ids[i] for i in child_idx]
            centroids[child_prefix] = embeddings[child_idx].mean(axis=0)
            members[child_prefix] = child_items
            for i in child_idx:
                ids[item_ids[i]].append(c)
            recurse(child_idx, child_prefix, level + 1)

    recurse(all_idx, (), 0)

    return HierarchicalKMeansIDs(
        branching=branching,
        depth=depth,
        ids={k: tuple(v) for k, v in ids.items()},
        centroids=centroids,
        members=members,
    )
