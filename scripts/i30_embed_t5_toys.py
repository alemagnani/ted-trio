"""I30 — Toys sentence-t5-xl content embeddings (free-arm parity with Beauty).

Encodes the PLAIN toys_id2meta texts (no enrichment — mirrors
Beauty_sentence-t5-xl_embeddings.pt, which predates I22) with
sentence-transformers/sentence-t5-xl -> Toys_sentence-t5-xl_embeddings.pt
[11924, 768]. GPU-friendly; run on a GPU box.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
N = 11924
OUT = ROOT / "data/amazon_toys_backbone/Toys_sentence-t5-xl_embeddings.pt"

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--model", default="sentence-transformers/sentence-t5-xl",
                help="paper: sentence-t5-xl; a small model only for smoke/quick runs")
args = ap.parse_args()
if OUT.exists():
    print(f"skip: {OUT} exists", flush=True)
    sys.exit(0)

meta = json.load(open(ROOT / "data/amazon_meta/toys_id2meta.json"))
texts = [" "] * N
for k, t in meta.items():
    texts[int(k) - 1] = t or " "

t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(args.model, device=device)
emb = model.encode(texts, batch_size=64, convert_to_tensor=True,
                   show_progress_bar=True, device=device).cpu().float()
assert emb.shape[0] == N, emb.shape
torch.save(emb, OUT)
print(f"wrote {OUT} shape {tuple(emb.shape)} in {(time.time()-t0)/60:.1f} min",
      flush=True)
sys.exit(0)
