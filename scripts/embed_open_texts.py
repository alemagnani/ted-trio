"""Embed a JSON list of item texts with an open-source embedding model (open-stack item side).

The model is given by --model or $TT_OPEN_EMBEDDER (a sentence-transformers Hugging Face id).

Used for the I25 open-embedder check (paper §2.1: with one leading open-source embedding model as the frozen
item side, enrichment lifts Beauty inner pseudo-test HR@5 by +0.0075..+0.0106 on all three
seeds). The plain open-embedder matrix comes from scripts/embed_open_catalog.py; the pred3-enriched
inner-world texts come from scripts/i25_dump_enriched_texts.py and are embedded here with
the SAME encode settings as embed_open_catalog.py (normalize_embeddings=True, batch 32).
The original enriched embedding was produced ad hoc on a GPU box; this script is the
reconstruction of that step (row i = item id i+1).

Usage:
  python scripts/i25_dump_enriched_texts.py
  python scripts/embed_open_texts.py \
      --texts data/amazon_beauty_backbone/i25_texts_inner_pred3.json \
      --out data/amazon_beauty_backbone/Beauty_openemb-large_i25_pred3_inner_embeddings.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from sentence_transformers import SentenceTransformer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--texts", required=True, help="JSON list[str], row i = item id i+1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("TT_OPEN_EMBEDDER"),
                    help="Hugging Face id of a sentence-transformers model (or $TT_OPEN_EMBEDDER)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not args.model:
        ap.error("pass --model or set TT_OPEN_EMBEDDER")
    texts = json.load(open(args.texts))
    t0 = time.time()
    model = SentenceTransformer(args.model, device=args.device)
    emb = model.encode(texts, batch_size=32, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True)
    torch.save(torch.from_numpy(emb).float(), args.out)
    print(f"wrote {args.out} shape {emb.shape} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
