"""Reusable item-embedding spaces + blend helpers (extracted from E3c, I7).

E3c (research/experiments/e3c_collab_space.py) showed the training-free winner on Beauty is
``blend_recency``: z-score-sum of cosine scores from the CONTENT space (sentence-t5) and a
COLLABORATIVE space (windowed co-occurrence over train-only sequences -> PPMI -> truncated SVD),
with a recency-weighted user vector (RECENCY=0.8): diagnostic dense HR@10 0.0472 -> 0.0751.

This module makes that recipe importable by the J2/J3 planner inference code:
  * ``build_collab_embeddings`` — the exact E3c PPMI+SVD construction, parameterised and with a
    small npz disk cache (default: outputs/artifacts/collab_space_beauty.npz).
  * ``user_vec`` — E3c's recency-weighted, L2-normalised user vector.
  * ``zscore`` / ``blended_scores`` — the z-score-sum score merge across spaces.

Protocol safety: callers must pass TRAIN-ONLY sequences (s[:-2] per user) to
``build_collab_embeddings``; nothing here reads the split itself.
"""
from __future__ import annotations

import os

import numpy as np

RECENCY = 0.8          # E3c blend_recency decay per step back from the end of history
WHITEN_SHRINKAGE = 0.1  # P13 ZCA covariance shrinkage alpha (val-selected, item-side stats)
WHITEN_RECENCY = 0.3    # P13b val-selected recency for the whitened z-sum blend
DEFAULT_CACHE = "outputs/artifacts/collab_space_beauty.npz"


def build_collab_embeddings(train_seqs: list[list[int]], n_items: int, d: int = 128,
                            window: int = 5,
                            cache_path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Windowed co-occurrence (weight 1/dist) over TRAIN-ONLY sequences -> PPMI -> SVD.

    Returns ``(vec [n_items, d] L2-normalised (zero rows for uncovered items), covered [n_items]
    bool)``. If ``cache_path`` is given and exists, the cached SVD result is loaded instead of
    recomputed; otherwise the result is written there (npz with keys vec/covered).
    """
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        vec, covered = z["vec"], z["covered"]
        if vec.shape == (n_items, d):
            return vec.astype(np.float32), covered.astype(bool)
        # stale cache (different n_items/d) — fall through and rebuild

    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.linalg import svds

    rows, cols, vals = [], [], []
    for seq in train_seqs:
        L = len(seq)
        for i in range(L):
            for j in range(i + 1, min(i + 1 + window, L)):
                w = 1.0 / (j - i)
                rows.append(seq[i]); cols.append(seq[j]); vals.append(w)
                rows.append(seq[j]); cols.append(seq[i]); vals.append(w)
    C = coo_matrix((vals, (rows, cols)), shape=(n_items, n_items)).tocsr()
    total = C.sum()
    rs = np.asarray(C.sum(axis=1)).ravel()
    cs = np.asarray(C.sum(axis=0)).ravel()
    C = C.tocoo()
    pmi = np.log(np.maximum(C.data * total / np.maximum(rs[C.row] * cs[C.col], 1e-9), 1e-9))
    keep = pmi > 0
    P = csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=(n_items, n_items))
    u, s, _vt = svds(P, k=d)
    vec = u * np.sqrt(np.maximum(s, 0))[None, :]
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    covered = norm.ravel() > 1e-9
    vec = (vec / np.maximum(norm, 1e-9)).astype(np.float32)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, vec=vec, covered=covered)
    return vec, covered


def user_vec(space: np.ndarray, hist: list[int], recency: float | None = RECENCY) -> np.ndarray:
    """E3c user vector: recency-weighted sum of the history rows, L2-normalised.

    ``recency=None`` gives the unweighted mean-direction vector.
    """
    h = np.asarray(hist, dtype=np.int64)
    w = (recency ** np.arange(len(h) - 1, -1, -1)) if recency is not None \
        else np.ones(len(h))
    v = (w[:, None] * space[h]).sum(0)
    nv = np.linalg.norm(v)
    return (v / nv if nv > 0 else v).astype(np.float32)


# --------------------------------------------------------------------- P13/P13b whitening
def zca_whiten(X: np.ndarray, shrinkage: float = WHITEN_SHRINKAGE,
               mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Fit ITEM-SIDE ZCA whitening stats (P13 recipe). Returns ``(mu, W)``.

    Two-tower-preserving: the transform ``x -> (x - mu) @ W`` (W symmetric d x d) is applied
    identically to item vectors offline and to query vectors (or, equivalently, queries are
    built as recency-weighted means of the ALREADY-transformed item vectors) — the score
    stays a dot product of independently computed vectors, ANN-servable.

    ``mask`` restricts the fit to covered rows (collab space: fit on covered items only).
    Stats are fit on the item matrix only — never on val/test labels.
    """
    sel = X if mask is None else X[mask]
    mu = sel.mean(axis=0)
    Xc = sel - mu
    S = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    d = S.shape[0]
    S = (1 - shrinkage) * S + shrinkage * (np.trace(S) / d) * np.eye(d, dtype=S.dtype)
    lam, V = np.linalg.eigh(S.astype(np.float64))
    W = (V * (1.0 / np.sqrt(np.maximum(lam, 1e-12)))) @ V.T
    return mu.astype(np.float32), W.astype(np.float32)


def apply_zca(X: np.ndarray, mu: np.ndarray, W: np.ndarray,
              mask: np.ndarray | None = None) -> np.ndarray:
    """Apply fitted ZCA stats to item OR query vectors; rows are L2-renormalised.

    ``mask`` (item side, collab space): uncovered rows are reset to exact zero vectors
    (the P1/P13 convention — cold-collab items carry no collab signal of their own).
    """
    n = np.linalg.norm
    out = (np.atleast_2d(X) - mu) @ W
    out = (out / np.maximum(n(out, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    if mask is not None:
        out[~np.asarray(mask, dtype=bool)] = 0.0
    return out.reshape(np.shape(X))


def whiten_blend_spaces(content: np.ndarray, collab: np.ndarray, covered: np.ndarray,
                        shrinkage: float = WHITEN_SHRINKAGE
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Fit + apply the P13-winner transforms: ZCA-whitened content (full-catalog stats) and
    ZCA-whitened collab (covered-item stats; uncovered rows stay zero). Returns the two
    L2-row-normalised item matrices ready for ``blended_scores_whitened``."""
    mu_c, W_c = zca_whiten(content, shrinkage=shrinkage)
    mu_l, W_l = zca_whiten(collab, shrinkage=shrinkage, mask=covered)
    return apply_zca(content, mu_c, W_c), apply_zca(collab, mu_l, W_l, mask=covered)


def blended_scores_whitened(content_w: np.ndarray, collab_w: np.ndarray,
                            covered: np.ndarray, hist: list[int],
                            recency: float | None = WHITEN_RECENCY,
                            cold_fallback: bool = True) -> np.ndarray:
    """P13b deployable blend: z(whitened-content cos) + z(whitened-collab cos), with the P2
    content-fallback fixing the cold-item pathology.

    Inputs are the matrices from ``whiten_blend_spaces``. ``cold_fallback=True`` (default):
    items with no collab vector get the whitened-CONTENT z in the collab slot, and a user
    whose history carries no collab signal at all scores content-only — P2's convention, so
    collab-less items get nonzero scores instead of the old blend cold-zero collapse.
    ``cold_fallback=False`` reproduces the raw P13 ``blend_zca`` arm exactly.

    Two-tower note: each arm's score is a dot product of an independently computed query
    vector with the transformed item matrix; the per-query z-scoring is a monotone per-arm
    calibration (rank-preserving within an arm), applied at merge time.
    """
    zc = zscore(content_w @ user_vec(content_w, hist, recency))
    sl = collab_w @ user_vec(collab_w, hist, recency)
    zl = zscore(sl)
    if cold_fallback:
        cov = np.asarray(covered, dtype=bool)
        zl = np.where(cov, zl, zc)
        if float(np.abs(sl).max(initial=0.0)) < 1e-12:   # no collab signal in the history
            zl = zc
    return zc + zl


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else x - x.mean()


def blended_scores(spaces: list[np.ndarray], hist: list[int],
                   recency: float | None = RECENCY) -> np.ndarray:
    """E3c ``blend_recency`` scoring: sum over spaces of z-scored cosine to the user vector.

    Each entry of ``spaces`` is an L2-row-normalised [n_items, d_i] matrix (dims may differ).
    """
    total: np.ndarray | None = None
    for sp in spaces:
        s = zscore(sp @ user_vec(sp, hist, recency))
        total = s if total is None else total + s
    assert total is not None, "blended_scores needs at least one space"
    return total
