"""Oracle-union recall across systems (spec §13.7, §31 step 6).

Given two (or more) systems' run records for the same queries, the oracle union is the
recall achievable by taking the *union* of their top-k candidate sets. This upper-bounds
what a perfect merger of the systems could achieve and is the central quantity for
deciding whether a dense + generative hybrid has headroom over dense alone.
"""

from __future__ import annotations

from typing import Sequence

from semplan.common.run_schema import RunRecord
from semplan.metrics.core import _dedup_keep_order


def _by_qid(records: Sequence[RunRecord]) -> dict[str, RunRecord]:
    return {r.query_id: r for r in records}


def union_recall_at_k(system_records: Sequence[Sequence[RunRecord]], k: int) -> float:
    """Mean Recall@K of the per-query union of top-k candidates across all systems.

    Each element of ``system_records`` is one system's list of RunRecords. Systems are
    aligned by ``query_id``; queries missing from a system contribute that system's empty
    candidate set. Relevant set is taken from the first system that has the query.
    """
    maps = [_by_qid(recs) for recs in system_records]
    all_qids = set().union(*[set(m) for m in maps]) if maps else set()
    if not all_qids:
        return 0.0

    total = 0.0
    for qid in all_qids:
        rel: set[str] = set()
        union_topk: list[str] = []
        for m in maps:
            r = m.get(qid)
            if r is None:
                continue
            if not rel:
                rel = r._relevant_set
            union_topk.extend(_dedup_keep_order(r.retrieved_item_ids)[:k])
        if not rel:
            continue
        union_set = set(union_topk)
        hits = sum(1 for it in rel if it in union_set)
        total += hits / len(rel)
    return total / len(all_qids)
