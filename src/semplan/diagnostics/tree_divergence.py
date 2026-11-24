"""Semantic-ID tree divergence signals (spec §13.4).

Semantic ids are hierarchical token paths. Two beams that share a long common prefix live
in the same coarse region; beams that diverge at the first token point to distant regions
of the hierarchy. We quantify this with longest-common-prefix based tree divergence and
its probability-weighted / min / max aggregates over a beam set.
"""

from __future__ import annotations

from typing import Sequence

from semplan.common.beam_schema import BeamRecord
from semplan.diagnostics.beam_entropy import normalized_probs


def longest_common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common prefix of two token sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def tree_divergence(a: Sequence[int], b: Sequence[int], depth: int | None = None) -> float:
    """Normalized divergence ``1 - LCP(a,b)/D`` in [0, 1] (spec §13.4).

    ``depth`` D defaults to the max length of the two sequences. Divergence is 0 when one
    is a prefix of the other along its full length, 1 when they differ at the first token.
    """
    lcp = longest_common_prefix(a, b)
    D = depth if depth is not None else max(len(a), len(b))
    if D == 0:
        return 0.0
    return 1.0 - lcp / D


def _valid_token_lists(beams: Sequence[BeamRecord]) -> list[list[int]]:
    return [list(b.semantic_tokens) for b in beams if b.valid]


def min_pairwise_lcp(beams: Sequence[BeamRecord]) -> int:
    """Smallest LCP among all beam pairs (earliest divergence depth). 0 or 1 valid beam -> a large/degenerate value handled by callers."""
    toks = _valid_token_lists(beams)
    if len(toks) < 2:
        return 0
    return min(
        longest_common_prefix(toks[i], toks[j])
        for i in range(len(toks))
        for j in range(i + 1, len(toks))
    )


def max_pairwise_divergence(beams: Sequence[BeamRecord], depth: int | None = None) -> float:
    toks = _valid_token_lists(beams)
    if len(toks) < 2:
        return 0.0
    return max(
        tree_divergence(toks[i], toks[j], depth)
        for i in range(len(toks))
        for j in range(i + 1, len(toks))
    )


def weighted_pairwise_divergence(
    beams: Sequence[BeamRecord], temperature: float = 1.0, depth: int | None = None
) -> float:
    """Probability-weighted mean pairwise tree divergence (spec §13.4).

    Weights each pair (i, j) by p_i * p_j (normalized over valid beams) and averages the
    normalized tree divergence. Emphasizes divergence among *high-probability* hypotheses,
    so a diverse but low-probability tail does not dominate.
    """
    valid = [b for b in beams if b.valid]
    toks = [list(b.semantic_tokens) for b in valid]
    probs = normalized_probs(valid, temperature)
    if len(toks) < 2:
        return 0.0
    num = 0.0
    den = 0.0
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            w = probs[i] * probs[j]
            num += w * tree_divergence(toks[i], toks[j], depth)
            den += w
    return num / den if den > 0 else 0.0


def distinct_tokens_per_level(beams: Sequence[BeamRecord]) -> list[int]:
    """Number of distinct token values at each decoding level across valid beams (spec §13.4)."""
    toks = _valid_token_lists(beams)
    if not toks:
        return []
    max_len = max(len(t) for t in toks)
    out = []
    for d in range(max_len):
        vals = {t[d] for t in toks if d < len(t)}
        out.append(len(vals))
    return out
