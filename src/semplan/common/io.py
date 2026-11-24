"""JSONL / Parquet IO for run and beam records.

JSONL is the default (human-inspectable, streamable, git-diffable for small fixtures).
Parquet is offered for large dumps (MS MARCO / full beam sets). Both round-trip through
the ``to_dict`` / ``from_dict`` of the dataclasses so the on-disk schema is stable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

from semplan.common.run_schema import RunRecord
from semplan.common.beam_schema import BeamRecord

T = TypeVar("T", RunRecord, BeamRecord)


def write_jsonl(records: Iterable, path: str | Path) -> int:
    """Write dataclass records (with ``to_dict``) to a JSONL file. Returns count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")
            n += 1
    return n


def _read_jsonl_dicts(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_runs(path: str | Path) -> list[RunRecord]:
    return [RunRecord.from_dict(d) for d in _read_jsonl_dicts(path)]


def read_beams(path: str | Path) -> list[BeamRecord]:
    return [BeamRecord.from_dict(d) for d in _read_jsonl_dicts(path)]


def group_beams_by_query(beams: Sequence[BeamRecord]) -> dict[str, list[BeamRecord]]:
    """Group a flat beam list into ``query_id -> beams sorted by beam_rank``."""
    out: dict[str, list[BeamRecord]] = {}
    for b in beams:
        out.setdefault(b.query_id, []).append(b)
    for qid in out:
        out[qid].sort(key=lambda b: b.beam_rank)
    return out


def write_parquet(records: Sequence, path: str | Path) -> int:
    """Write records to Parquet via pandas.

    Dict-valued columns (e.g. ``relevance_grades``) are JSON-encoded because Arrow cannot
    represent an empty/variable struct; nested list columns are preserved natively. Use
    ``read_*_parquet`` counterparts (not yet needed) to decode. Large beam/run dumps use this.
    """
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in records]
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, dict)).any():
            df[col] = df[col].map(json.dumps)
    df.to_parquet(path, index=False)
    return len(df)
