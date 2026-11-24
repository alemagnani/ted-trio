"""Embed the plain Beauty catalog (I16 commercial item side) -> the 'p' (plain) item index.

Texts = data/amazon_meta/beauty_id2meta.json (title + price + brand + categories), row =
int(id) - 1. Paper: OpenAI text-embedding-3-large (key from $OPENAI_API_KEY only).
Toys/Sports/Instruments/Yelp plain catalogs are built by i44_ablation_prep.py.

Output: data/amazon_beauty_backbone/Beauty_openai-3-large_embeddings.pt  [12101, 3072] float32
Usage:  python scripts/embed_openai_catalog.py [--embedder openai|fake|st:<model>] [--force]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.embedders import add_embedder_arg, embed_texts  # noqa: E402
from semplan.enrich import base_texts_from_meta  # noqa: E402

OUT = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_embeddings.pt"
N = 12101


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_embedder_arg(ap)
    ap.add_argument("--force", action="store_true", help="re-embed even if the output exists")
    args = ap.parse_args()
    if OUT.exists() and not args.force:
        print(f"skip: {OUT} exists", flush=True)
        return
    texts = base_texts_from_meta(json.load(open(ROOT / "data/amazon_meta/beauty_id2meta.json")), N)
    print(f"{N} items, {sum(1 for t in texts if not t)} without meta text", flush=True)
    t0 = time.time()
    emb = embed_texts(texts, args.embedder)
    torch.save(emb, OUT)
    print(f"wrote {OUT} shape {tuple(emb.shape)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
