"""Beam-distribution uncertainty signals (spec §13.2, §13.3).

Given a query's beams (each with a ``sequence_logprob``), we form a normalized
distribution over the valid beams with temperature T and compute entropy and the
probability margin between the top two hypotheses. These are the primary "how uncertain
is the planner" signals that the adaptive-budget policy (spec §18) consumes.
"""

from __future__ import annotations

import math
from typing import Sequence

from semplan.common.beam_schema import BeamRecord


def normalized_probs(beams: Sequence[BeamRecord], temperature: float = 1.0) -> list[float]:
    """Softmax over valid beams' sequence log-probs at temperature T. Empty -> []."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logps = [b.sequence_logprob for b in beams if b.valid]
    if not logps:
        return []
    m = max(logps)
    exps = [math.exp((lp - m) / temperature) for lp in logps]
    z = sum(exps)
    return [e / z for e in exps]


def beam_entropy(beams: Sequence[BeamRecord], temperature: float = 1.0) -> float:
    """Shannon entropy (nats) of the temperature-normalized valid-beam distribution."""
    ps = normalized_probs(beams, temperature)
    return -sum(p * math.log(p) for p in ps if p > 0.0)


def normalized_beam_entropy(beams: Sequence[BeamRecord], temperature: float = 1.0) -> float:
    """Entropy divided by log(B) so it lies in [0, 1] regardless of beam count (spec §13.2)."""
    ps = normalized_probs(beams, temperature)
    b = len(ps)
    if b <= 1:
        return 0.0
    return (-sum(p * math.log(p) for p in ps if p > 0.0)) / math.log(b)


def probability_margin(beams: Sequence[BeamRecord], temperature: float = 1.0) -> float:
    """p1 - p2 over the normalized distribution (spec §13.3). 1.0 if a single valid beam."""
    ps = sorted(normalized_probs(beams, temperature), reverse=True)
    if not ps:
        return 0.0
    if len(ps) == 1:
        return 1.0
    return ps[0] - ps[1]


def logprob_gap(beams: Sequence[BeamRecord]) -> float:
    """Raw log-prob gap l1 - l2 between the two highest-scoring valid beams (spec §13.3)."""
    logps = sorted((b.sequence_logprob for b in beams if b.valid), reverse=True)
    if len(logps) < 2:
        return math.inf
    return logps[0] - logps[1]
