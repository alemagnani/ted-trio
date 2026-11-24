"""Retrieval metrics computed directly on ``RunRecord`` objects (spec §11.3, §22).

All functions operate on a single ranked list plus the relevant set / grades, and are
deduplicated defensively (a merged candidate list may repeat ids). Aggregate helpers
average a per-query metric over a collection of records. These are the single source of
truth for Recall@K / NDCG@K / MRR@10 so every system is scored identically.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

from semplan.common.run_schema import RunRecord


def _dedup_keep_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def recall_at_k(rec: RunRecord, k: int) -> float:
    """Fraction of relevant items appearing in the top-k retrieved (0 if no relevants)."""
    rel = rec._relevant_set
    if not rel:
        return 0.0
    topk = _dedup_keep_order(rec.retrieved_item_ids)[:k]
    hits = sum(1 for it in topk if it in rel)
    return hits / len(rel)


def hit_at_k(rec: RunRecord, k: int) -> float:
    """1.0 if any relevant item is in the top-k (a.k.a. Recall@K in single-target rec setups)."""
    rel = rec._relevant_set
    if not rel:
        return 0.0
    topk = _dedup_keep_order(rec.retrieved_item_ids)[:k]
    return 1.0 if any(it in rel for it in topk) else 0.0


def dcg_at_k(rec: RunRecord, k: int) -> float:
    topk = _dedup_keep_order(rec.retrieved_item_ids)[:k]
    return sum(rec.grade(it) / math.log2(i + 2) for i, it in enumerate(topk))


def ndcg_at_k(rec: RunRecord, k: int) -> float:
    """Standard NDCG@K using graded relevance (binary grade=1 if no explicit grades)."""
    ideal_grades = sorted((rec.grade(it) for it in rec.relevant_item_ids), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_grades))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(rec, k) / idcg


def mrr_at_k(rec: RunRecord, k: int = 10) -> float:
    """Reciprocal rank of the first relevant item within top-k (0 if none)."""
    rel = rec._relevant_set
    topk = _dedup_keep_order(rec.retrieved_item_ids)[:k]
    for i, it in enumerate(topk):
        if it in rel:
            return 1.0 / (i + 1)
    return 0.0


def mean_metric(records: Sequence[RunRecord], fn: Callable[[RunRecord], float]) -> float:
    if not records:
        return 0.0
    return sum(fn(r) for r in records) / len(records)


def evaluate(records: Sequence[RunRecord], ks: Sequence[int] = (5, 10, 20, 100)) -> dict[str, float]:
    """Aggregate the standard metric bundle over a set of run records."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"Recall@{k}"] = mean_metric(records, lambda r, k=k: recall_at_k(r, k))
        out[f"NDCG@{k}"] = mean_metric(records, lambda r, k=k: ndcg_at_k(r, k))
        out[f"Hit@{k}"] = mean_metric(records, lambda r, k=k: hit_at_k(r, k))
    out["MRR@10"] = mean_metric(records, lambda r: mrr_at_k(r, 10))
    out["num_queries"] = float(len(records))
    return out
