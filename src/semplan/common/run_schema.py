"""Canonical per-query run record (spec §26, Task 3).

A ``RunRecord`` captures the output of *one retrieval system* for *one query*: the
ranked list of retrieved item ids with their scores, plus the ground-truth relevant
items and provenance (system, dataset, checkpoint, seed, latency). Every baseline and
proposed system serializes to this schema so metrics and diagnostics are computed
identically regardless of the underlying model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RunRecord:
    query_id: str
    query_text: str
    relevant_item_ids: list[str]
    retrieved_item_ids: list[str]
    retrieved_scores: list[float]
    system_name: str
    dataset: str
    checkpoint: str = ""
    seed: int = 0
    latency_ms: float = 0.0
    # Optional per-query relevance grades, keyed by item id. Absent => binary (grade 1).
    relevance_grades: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.retrieved_item_ids) != len(self.retrieved_scores):
            raise ValueError(
                f"retrieved_item_ids ({len(self.retrieved_item_ids)}) and retrieved_scores "
                f"({len(self.retrieved_scores)}) must have equal length for query {self.query_id}"
            )

    def grade(self, item_id: str) -> float:
        """Relevance grade for ``item_id``: explicit grade, else 1.0 if relevant, else 0.0."""
        if item_id in self.relevance_grades:
            return self.relevance_grades[item_id]
        return 1.0 if item_id in self._relevant_set else 0.0

    @property
    def _relevant_set(self) -> set[str]:
        # Cached lazily; recomputed if the list identity changes.
        cached = getattr(self, "_relevant_set_cache", None)
        if cached is None or cached[0] is not self.relevant_item_ids:
            s = set(self.relevant_item_ids)
            object.__setattr__(self, "_relevant_set_cache", (self.relevant_item_ids, s))
            return s
        return cached[1]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_relevant_set_cache", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunRecord":
        fields = {
            "query_id", "query_text", "relevant_item_ids", "retrieved_item_ids",
            "retrieved_scores", "system_name", "dataset", "checkpoint", "seed",
            "latency_ms", "relevance_grades",
        }
        return cls(**{k: v for k, v in d.items() if k in fields})
