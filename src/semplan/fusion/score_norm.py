"""Per-hypothesis score normalization for score-based fusion (spec §19.2).

ANN scores from different query vectors are not automatically comparable, so before summing
scores across hypotheses we normalize each hypothesis's scores. Provided schemes: z-score,
min-max, and softmax (with temperature). ``fuse_by_score`` then aggregates normalized scores
per item (max or sum). RRF (rank-only) remains the default; these are the score-based
alternatives the ablation compares against.
"""

from __future__ import annotations

import math
from typing import Sequence


def zscore(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    std = math.sqrt(var)
    if std == 0:
        return [0.0] * n
    return [(s - mean) / std for s in scores]


def minmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
    if not scores:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


_NORMALIZERS = {"zscore": zscore, "minmax": minmax, "softmax": softmax}


def fuse_by_score(
    ranked_lists: Sequence[Sequence[str]],
    score_lists: Sequence[Sequence[float]],
    scheme: str = "zscore",
    aggregate: str = "max",
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Normalize each hypothesis's scores then aggregate per item (max or sum)."""
    if scheme not in _NORMALIZERS:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {sorted(_NORMALIZERS)}")
    if aggregate not in ("max", "sum"):
        raise ValueError("aggregate must be 'max' or 'sum'")
    normalizer = _NORMALIZERS[scheme]
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    agg: dict[str, float] = {}
    for ids, scores, w in zip(ranked_lists, score_lists, weights):
        norm = normalizer(scores)
        for item, s in zip(ids, norm):
            val = w * s
            if item not in agg:
                agg[item] = val
            elif aggregate == "max":
                agg[item] = max(agg[item], val)
            else:
                agg[item] += val
    return sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
