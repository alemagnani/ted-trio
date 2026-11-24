"""Reciprocal rank fusion for merging multi-hypothesis candidate lists (spec §19.1).

RRF is the parameter-light merger for the MVP planner: given several ranked candidate
lists (base dense + one per planner hypothesis), an item's fused score is the sum over
lists of w_h / (k + rank_h(i)). Rank-only fusion sidesteps the ANN-score calibration
problem (spec §19.2) — scores from different query vectors are not comparable.
"""

from __future__ import annotations

from typing import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one list sorted by descending RRF score.

    ``ranked_lists[h]`` is the ordered top candidates from hypothesis h (rank 0 = best).
    ``weights[h]`` scales that list's contribution (defaults to all 1.0). ``k`` is the RRF
    smoothing constant. Returns [(item_id, score)] sorted high-to-low, ties broken by id.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must align with ranked_lists")

    scores: dict[str, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + w / (k + rank + 1)

    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def fuse_to_ids(
    ranked_lists: Sequence[Sequence[str]],
    weights: Sequence[float] | None = None,
    k: int = 60,
    top_n: int | None = None,
) -> tuple[list[str], list[float]]:
    """Convenience: RRF then split into (ids, scores), optionally truncated to top_n."""
    fused = reciprocal_rank_fusion(ranked_lists, weights, k)
    if top_n is not None:
        fused = fused[:top_n]
    ids = [i for i, _ in fused]
    scores = [s for _, s in fused]
    return ids, scores
