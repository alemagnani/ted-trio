"""Artifact / dataset manifests with checksums and counts (spec §5.4, §26).

Every dataset build and experiment run should emit a manifest so that a reported metric
is traceable to an exact dataset fingerprint (release, k-core, split, checksums) and an
exact environment. ``dataset_fingerprint`` produces the short id shown in result tables.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dataset_fingerprint(fields: dict[str, Any], length: int = 12) -> str:
    """Deterministic short fingerprint over sorted key/values (spec: every table shows one)."""
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)[:length]


def environment_info() -> dict[str, str]:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("numpy", "torch", "faiss", "transformers"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            info[mod] = "absent"
    return info


@dataclass
class Manifest:
    name: str
    kind: str  # "dataset" | "run" | "index" | ...
    fields: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    checksums: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=environment_info)

    @property
    def fingerprint(self) -> str:
        return dataset_fingerprint(self.fields)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["fingerprint"] = self.fingerprint
        path.write_text(json.dumps(d, indent=2, sort_keys=True))
        return path

    @classmethod
    def read(cls, path: str | Path) -> "Manifest":
        d = json.loads(Path(path).read_text())
        d.pop("fingerprint", None)
        return cls(**d)
