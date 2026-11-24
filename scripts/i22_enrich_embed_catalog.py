"""I22 — Item-side collab-text enrichment (E) for Beauty (ledger J6-KILL follow-on).

Per item i, find its top predecessor items p by directed consecutive co-occurrence
count(p -> i) in the training transitions, and append
"Often bought after: '<title(p1)>'; '<title(p2)>'; '<title(p3)>'."
to the frozen I16 meta text, then re-embed the catalog (paper: openai-3-large).
The collab signal moves into the item geometry; the index stays frozen/ANN-servable.

Two worlds, each excluding its final (held-out) event per user, mirroring the
include-val standing rule (towers train on everything but the last event):
  real : Beauty.txt        -> Beauty_openai-3-large_i22real_embeddings.pt
  inner: Beauty_inner.txt  -> Beauty_openai-3-large_i22inner_embeddings.pt
Mining / verbalization: semplan.enrich (shared by all datasets).

Usage: python scripts/i22_enrich_embed_catalog.py [--embedder openai|fake|st:<model>]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan import enrich as E  # noqa: E402
from semplan.embedders import add_embedder_arg, embed_texts  # noqa: E402

N = 12101
DATA = ROOT / "data/amazon_beauty_backbone"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_embedder_arg(ap)
    args = ap.parse_args()
    base = E.base_texts_from_meta(json.load(open(ROOT / "data/amazon_meta/beauty_id2meta.json")), N)
    titles = E.titles_from_texts(base)
    for world, seq, out in [
        ("inner", "Beauty_inner.txt", "Beauty_openai-3-large_i22inner_embeddings.pt"),
        ("real", "Beauty.txt", "Beauty_openai-3-large_i22real_embeddings.pt"),
    ]:
        out_path = DATA / out
        if out_path.exists():
            print(f"skip {world}: {out_path} exists", flush=True)
            continue
        texts, n = E.enriched_texts(base, titles, E.pred_counts(DATA / seq))
        print(f"{seq}: {n}/{N} items enriched", flush=True)
        emb = embed_texts(texts, args.embedder)
        torch.save(emb, out_path)
        print(f"wrote {out_path} shape {tuple(emb.shape)}", flush=True)


if __name__ == "__main__":
    main()
