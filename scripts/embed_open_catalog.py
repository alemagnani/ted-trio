"""Embed the Beauty catalog with an open-source embedding model (set $TT_OPEN_EMBEDDER; open
stand-in for the I16 commercial item side; the commercial matrix will drop into the same .pt slot).
Output: data/amazon_beauty_backbone/Beauty_openemb-large_embeddings.pt  [12101, 1024] float32
"""
import json, os, sys, time
from pathlib import Path

import torch

MODEL = os.environ.get("TT_OPEN_EMBEDDER")
if not MODEL:
    sys.exit("set TT_OPEN_EMBEDDER to the Hugging Face id of a sentence-transformers model")
DEVICE = os.environ.get("TT_DEVICE", "cpu")

ROOT = Path(__file__).resolve().parents[1]
meta = json.load(open(ROOT / "data/amazon_meta/beauty_id2meta.json"))
N = 12101
texts = [""] * N
for k, t in meta.items():
    texts[int(k) - 1] = t  # id2meta keys are 1-indexed; embedding rows are id-1 (e0 convention)
missing = sum(1 for t in texts if not t)
print(f"{N} items, {missing} without meta text", flush=True)

from sentence_transformers import SentenceTransformer  # noqa: E402

t0 = time.time()
model = SentenceTransformer(MODEL, device=DEVICE)
emb = model.encode(texts, batch_size=32, show_progress_bar=False,
                   convert_to_numpy=True, normalize_embeddings=True)
out = ROOT / "data/amazon_beauty_backbone/Beauty_openemb-large_embeddings.pt"
torch.save(torch.from_numpy(emb).float(), out)
print(f"wrote {out} shape {emb.shape} in {(time.time()-t0)/60:.1f} min", flush=True)
