"""Branch/prefix survival of relevant items (spec §13.6).

At each decoding depth d, does at least one surviving beam prefix still cover a relevant
item's semantic id? If a relevant item's id shares its length-d prefix with some beam, it
is still reachable under that branch. The survival curve S_d shows how early the generative
hierarchy prunes away relevant items — critical because a generally-accurate generator can
still lose a relevant target high in the tree (spec §13.6).
"""

from __future__ import annotations

from typing import Mapping, Sequence

from semplan.common.beam_schema import BeamRecord


def prefix_survival_at_depth(
    beams: Sequence[BeamRecord],
    relevant_semantic_ids: Sequence[Sequence[int]],
    depth: int,
) -> int:
    """1 if some relevant id shares its length-``depth`` prefix with a valid beam, else 0.

    ``depth`` is 1-indexed (depth=1 compares the first token). Beams shorter than ``depth``
    contribute their full-length prefix.
    """
    if depth <= 0:
        raise ValueError("depth must be >= 1")
    beam_prefixes = {
        tuple(b.semantic_tokens[:depth]) for b in beams if b.valid and b.semantic_tokens
    }
    for rid in relevant_semantic_ids:
        if tuple(rid[:depth]) in beam_prefixes:
            return 1
    return 0


def survival_curve(
    beams: Sequence[BeamRecord],
    relevant_semantic_ids: Sequence[Sequence[int]],
    max_depth: int,
) -> list[int]:
    """Per-depth survival indicators S_1..S_max_depth for one query."""
    return [
        prefix_survival_at_depth(beams, relevant_semantic_ids, d)
        for d in range(1, max_depth + 1)
    ]


def dataset_survival_curve(
    per_query: Sequence[tuple[Sequence[BeamRecord], Sequence[Sequence[int]]]],
    max_depth: int,
) -> list[float]:
    """Mean survival at each depth across queries (spec §13.6 dataset-level curve).

    ``per_query`` is a sequence of (beams, relevant_semantic_ids) tuples.
    """
    if not per_query:
        return [0.0] * max_depth
    sums = [0 for _ in range(max_depth)]
    for beams, rids in per_query:
        curve = survival_curve(beams, rids, max_depth)
        for i, s in enumerate(curve):
            sums[i] += s
    return [s / len(per_query) for s in sums]
