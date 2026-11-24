"""Generate a tiny synthetic dataset in the real data layout (release addition, smoke tests).

Writes, for the requested dataset name, the files the pipeline reads -- the sequence file,
its inner split, the item text (id2meta), item attributes (Amazon datasets) and a FAKE
768-d "sentence-t5-xl" content matrix -- under --out, with the dataset's real catalog size
(the runners hard-code it) but only --users synthetic users. Sequences are random walks
inside 50 latent categories, so items have real predecessors, co-occurrence and
transition structure for every stage to work on. Item embeddings are NOT written: the
smoke pipeline builds them with `--embedder fake` (scripts/smoke_e2e.sh).

Usage: python scripts/make_synthetic_dataset.py --dataset beauty --users 300 --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.embedders import fake_embed  # noqa: E402

DATASETS = {  # name: (backbone dir, sequence file, inner file, prefix, meta path, n_items, attrs)
    "beauty": ("amazon_beauty_backbone", "Beauty.txt", "Beauty_inner.txt", "Beauty",
               "amazon_meta/beauty_id2meta.json", 12101, "Beauty_item2attributes.json"),
    "toys": ("amazon_toys_backbone", "Toys_and_Games.txt", "Toys_inner.txt", "Toys",
             "amazon_meta/toys_id2meta.json", 11924, "Toys_and_Games_item2attributes.json"),
    "sports": ("amazon_sports_backbone", "Sports_and_Outdoors.txt", "Sports_inner.txt", "Sports",
               "amazon_meta/sports_id2meta.json", 18357,
               "Sports_and_Outdoors_item2attributes.json"),
    "instruments": ("amazon_instruments_backbone", "Musical_Instruments.txt",
                    "Instruments_inner.txt", "Instruments",
                    "amazon_meta/instruments_id2meta.json", 9922, None),
    "yelp": ("yelp_backbone", "Yelp.txt", "Yelp_inner.txt", "Yelp",
             "yelp_meta/yelp_id2meta.json", 20033, None),
}
WORDS = ["red", "blue", "matte", "gloss", "travel", "pro", "mini", "classic", "organic",
         "deluxe", "hydrating", "vintage", "smart", "ultra", "soft", "bold"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", nargs="+", choices=sorted(DATASETS), default=["beauty"])
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    for ds in args.dataset:
        bb, seq, inner, prefix, meta_rel, n, attrs = DATASETS[ds]
        rng = np.random.default_rng(args.seed)
        n_cat = 50
        cat = rng.integers(0, n_cat, size=n)
        by_cat = [np.flatnonzero(cat == c) for c in range(n_cat)]
        hot = [np.sort(rng.choice(b, size=min(len(b), 40), replace=False)) for b in by_cat]
        lines = []
        for u in range(args.users):
            c = int(rng.integers(0, n_cat))
            length = int(rng.integers(5, 16))
            items, cur = [], int(rng.choice(hot[c]))
            for _ in range(length):
                items.append(cur + 1)
                if rng.random() < 0.15:
                    c = int(rng.integers(0, n_cat))
                nxt_pool = hot[c]
                cur = int(nxt_pool[(np.searchsorted(nxt_pool, cur) + 1) % len(nxt_pool)]
                          if rng.random() < 0.6 else rng.choice(nxt_pool))
            lines.append(f"{u + 1} " + " ".join(map(str, items)))
        d = out / bb
        d.mkdir(parents=True, exist_ok=True)
        (d / seq).write_text("\n".join(lines) + "\n")
        (d / inner).write_text("\n".join(" ".join(l.split()[:-1]) for l in lines) + "\n")
        meta = {}
        for i in range(n):
            w = rng.choice(WORDS, size=2, replace=False)
            meta[str(i + 1)] = (f"This item is called '{w[0].title()} {w[1]} item {i + 1}'. "
                                f"It belongs to the categories of Category{cat[i]}. ")
        (out / meta_rel).parent.mkdir(parents=True, exist_ok=True)
        (out / meta_rel).write_text(json.dumps(meta))
        if attrs:
            (d / attrs).write_text(json.dumps({str(i + 1): [int(cat[i])] for i in range(n)}))
        texts = [meta[str(i + 1)] for i in range(n)]
        torch.save(fake_embed(texts, dim=768), d / f"{prefix}_sentence-t5-xl_embeddings.pt")
        print(f"[synthetic] {ds}: {len(lines)} users, {n} items -> {d}")


if __name__ == "__main__":
    main()
