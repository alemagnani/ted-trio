"""I50/I51 — sentence-t5-xl content embeddings for the EXPAND backbones
(free-arm parity with Beauty/Toys/Sports). Mirror of i35_embed_t5_sports.py,
generalized over --dataset {instruments,yelp}.

Encodes the PLAIN id2meta texts (no enrichment — mirrors
Beauty_sentence-t5-xl_embeddings.pt, which predates I22) with
sentence-transformers/sentence-t5-xl -> {Prefix}_sentence-t5-xl_embeddings.pt
[N, 768]. GPU-friendly (CPU works, slowly).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "instruments": dict(
        meta=ROOT / "data/amazon_meta/instruments_id2meta.json",
        out=ROOT / "data/amazon_instruments_backbone/Instruments_sentence-t5-xl_embeddings.pt",
        n=9922,
    ),
    "yelp": dict(
        meta=ROOT / "data/yelp_meta/yelp_id2meta.json",
        out=ROOT / "data/yelp_backbone/Yelp_sentence-t5-xl_embeddings.pt",
        n=20033,
    ),
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
ap.add_argument("--model", default="sentence-transformers/sentence-t5-xl",
                help="paper: sentence-t5-xl; a small model only for smoke/quick runs")
args = ap.parse_args()
cfg = DATASETS[args.dataset]
N = cfg["n"]
if cfg["out"].exists():
    print(f"skip: {cfg['out']} exists", flush=True)
    sys.exit(0)

meta = json.load(open(cfg["meta"]))
assert len(meta) == N, (len(meta), N)
texts = [" "] * N
for k, t in meta.items():
    texts[int(k) - 1] = t or " "

t0 = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(args.model, device=device)
emb = model.encode(texts, batch_size=64, convert_to_tensor=True,
                   show_progress_bar=True, device=device).cpu().float()
assert emb.shape[0] == N, emb.shape
torch.save(emb, cfg["out"])
print(f"wrote {cfg['out']} shape {tuple(emb.shape)} in {(time.time()-t0)/60:.1f} min",
      flush=True)
