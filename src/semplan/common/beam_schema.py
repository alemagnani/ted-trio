"""Canonical semantic-ID beam record (spec §26, Task 4; §10.1).

One ``BeamRecord`` is a single decoded semantic-ID hypothesis for a query: the token
sequence, per-token and sequence log-probabilities, the list of partial prefixes emitted
during decoding, and (when the id resolves to a catalogue item) the mapped item id.

A query's full beam set is a ``list[BeamRecord]`` ordered by ``beam_rank`` (0 = top beam).
``prefixes[d]`` is the length-(d+1) prefix tuple, so the whole decoding path is recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class BeamRecord:
    query_id: str
    beam_rank: int
    semantic_tokens: list[int]
    token_logprobs: list[float]
    sequence_logprob: float
    probability: float  # normalized over the query's valid beams; see diagnostics.beam_entropy
    prefixes: list[list[int]]
    mapped_item_id: str | None = None
    valid: bool = True

    def __post_init__(self) -> None:
        if len(self.semantic_tokens) != len(self.token_logprobs):
            raise ValueError(
                f"semantic_tokens ({len(self.semantic_tokens)}) and token_logprobs "
                f"({len(self.token_logprobs)}) must match for query {self.query_id} "
                f"beam {self.beam_rank}"
            )

    @classmethod
    def with_prefixes(cls, **kwargs: Any) -> "BeamRecord":
        """Construct filling ``prefixes`` from ``semantic_tokens`` if not provided."""
        if "prefixes" not in kwargs or kwargs["prefixes"] is None:
            toks = kwargs["semantic_tokens"]
            kwargs["prefixes"] = [list(toks[: d + 1]) for d in range(len(toks))]
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BeamRecord":
        fields = {
            "query_id", "beam_rank", "semantic_tokens", "token_logprobs",
            "sequence_logprob", "probability", "prefixes", "mapped_item_id", "valid",
        }
        return cls(**{k: v for k, v in d.items() if k in fields})
