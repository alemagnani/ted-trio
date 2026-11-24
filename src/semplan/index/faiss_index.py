"""FAISS wrapper for dense item indices (spec §9, §16, §17).

Cosine similarity is realized as inner-product over L2-normalized vectors (the standard
Contriever convention). This class builds a flat exact index by default (correct for
Beauty-scale corpora and for computing exact recall), with a hook for IVF/HNSW later. It
supports single-vector search and batched multi-hypothesis search (one query vector per
planner hypothesis), returning item ids so results flow straight into RunRecords.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


class DenseIndex:
    def __init__(self, dim: int, metric: str = "cosine"):
        import faiss

        self.dim = dim
        self.metric = metric
        # Cosine and dot both use inner product; cosine normalizes on add/search.
        self._index = faiss.IndexFlatIP(dim)
        self.item_ids: list[str] = []

    @property
    def size(self) -> int:
        return len(self.item_ids)

    def _prep(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.metric == "cosine":
            x = l2_normalize(x)
        return np.ascontiguousarray(x)

    def add(self, item_ids: list[str], embeddings: np.ndarray) -> None:
        if len(item_ids) != embeddings.shape[0]:
            raise ValueError("item_ids and embeddings length mismatch")
        self._index.add(self._prep(embeddings))
        self.item_ids.extend(item_ids)

    def search(self, query: np.ndarray, k: int) -> tuple[list[list[str]], np.ndarray]:
        """Search one or many query vectors. Returns (ids per query, scores array [n, k])."""
        q = self._prep(query)
        k = min(k, self.size)
        scores, idx = self._index.search(q, k)
        ids = [[self.item_ids[i] for i in row if i != -1] for row in idx]
        return ids, scores

    def search_one(self, query: np.ndarray, k: int) -> tuple[list[str], list[float]]:
        ids, scores = self.search(query, k)
        return ids[0], scores[0].tolist()

    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        np.save(path.with_suffix(".ids.npy"), np.array(self.item_ids, dtype=object), allow_pickle=True)

    @classmethod
    def load(cls, path: str | Path, metric: str = "cosine") -> "DenseIndex":
        import faiss

        path = Path(path)
        index = faiss.read_index(str(path.with_suffix(".faiss")))
        obj = cls.__new__(cls)
        obj._index = index
        obj.dim = index.d
        obj.metric = metric
        obj.item_ids = list(np.load(path.with_suffix(".ids.npy"), allow_pickle=True))
        return obj
