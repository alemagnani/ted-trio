"""Item-text embedders behind every embedding script's `--embedder` option (release addition).

  openai                  OpenAI text-embedding-3-large via the REST API, exactly the request
                          loop the paper used (batches of 256, retry with back-off, 1.5 s pause
                          between batches). Key read ONLY from $OPENAI_API_KEY.
  fake                    deterministic, offline hashed bag-of-words embedder (no network,
                          no model download): each lower-cased token maps to a fixed random
                          Gaussian vector seeded by its md5; a text is the L2-normalised sum.
                          Texts sharing words are similar, so the pipeline behaves sensibly.
                          For smoke tests / users without an API key ONLY: numbers produced
                          with it are not the paper's.
  st:<model>              any sentence-transformers model, e.g. st:<org>/<model-name>
                          (L2-normalised outputs).

The paper's numbers require `openai` (3072-d text-embedding-3-large, queried Aug 2026).
"""
from __future__ import annotations

import hashlib
import os
import re
import time

import numpy as np
import torch

OPENAI_MODEL = "text-embedding-3-large"
OPENAI_DIM = 3072
EMBEDDER_HELP = ("item-text embedder: 'openai' (paper; needs $OPENAI_API_KEY), 'fake' "
                 "(offline deterministic hashed bag-of-words, smoke tests only), or "
                 "'st:<sentence-transformers model>'")


def openai_embed(texts: list[str], batch: int = 256, model: str = OPENAI_MODEL,
                 log_every: int = 10) -> torch.Tensor:
    import requests

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("--embedder openai needs the OPENAI_API_KEY environment variable "
                         "(or use --embedder fake / st:<model>)")
    n = len(texts)
    rows: list = [None] * n
    t0 = time.time()
    for s in range(0, n, batch):
        chunk = texts[s: s + batch]
        for attempt in range(8):
            r = requests.post("https://api.openai.com/v1/embeddings",
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": model, "input": chunk}, timeout=120)
            if r.status_code == 200:
                break
            print(f"  batch {s}: HTTP {r.status_code} ({r.text[:120]}), retry {attempt+1}",
                  flush=True)
            time.sleep(20 * (attempt + 1))
        r.raise_for_status()
        data = r.json()["data"]
        assert len(data) == len(chunk)
        for d in data:
            rows[s + d["index"]] = d["embedding"]
        time.sleep(1.5)  # rate-limit headroom
        if (s // batch) % log_every == 0:
            print(f"  {s + len(chunk)}/{n} [{time.time()-t0:.0f}s]", flush=True)
    assert all(r is not None for r in rows)
    return torch.tensor(rows, dtype=torch.float32)


_TOKEN = re.compile(r"[a-z0-9]+")


def _token_vec(tok: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def fake_embed(texts: list[str], dim: int = OPENAI_DIM) -> torch.Tensor:
    cache: dict[str, np.ndarray] = {}
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = _TOKEN.findall(t.lower()) or ["<empty>"]
        v = np.zeros(dim, dtype=np.float32)
        for tok in toks:
            if tok not in cache:
                cache[tok] = _token_vec(tok, dim)
            v += cache[tok]
        out[i] = v / max(float(np.linalg.norm(v)), 1e-9)
    return torch.from_numpy(out)


def st_embed(texts: list[str], model_name: str) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=32, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    return torch.from_numpy(np.asarray(emb, dtype=np.float32))


def embed_texts(texts: list[str], embedder: str = "openai", dim: int = OPENAI_DIM
                ) -> torch.Tensor:
    """Embed `texts` (empty strings replaced by a single space, as the API requires)."""
    texts = [t if t else " " for t in texts]
    if embedder == "openai":
        return openai_embed(texts)
    if embedder == "fake":
        return fake_embed(texts, dim=dim)
    if embedder.startswith("st:"):
        return st_embed(texts, embedder[3:])
    raise ValueError(f"unknown embedder {embedder!r} ({EMBEDDER_HELP})")


def add_embedder_arg(ap) -> None:
    ap.add_argument("--embedder", default="openai", help=EMBEDDER_HELP)
