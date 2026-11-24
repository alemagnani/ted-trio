"""I22b — enrichment-variant catalogs for the inner world (gate-first, cents each).

Variants over the I22 winner (top-3 predecessor titles):
  succ : top-3 SUCCESSOR titles ("Often bought before: ...")
  both : top-3 predecessors + top-3 successors
  pred5: top-5 predecessors
Inner world only (Beauty_inner.txt, exclude last event) — real-world embeds are built
only for a variant that passes the 3-seed inner gate.

Output: data/amazon_beauty_backbone/Beauty_openai-3-large_i22b_{succ,both,pred5}_inner_embeddings.pt
"""
import argparse
import json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.embedders import add_embedder_arg, embed_texts  # noqa: E402
MODEL = "text-embedding-3-large"
N = 12101
BATCH = 256

meta = json.load(open(ROOT / "data/amazon_meta/beauty_id2meta.json"))
base_texts = [""] * N
for k, t in meta.items():
    base_texts[int(k) - 1] = t
titles = [""] * N
for i, t in enumerate(base_texts):
    m = re.search(r"called '(.*?)'\.", t)
    titles[i] = m.group(1) if m else ""

pred = defaultdict(Counter)
succ = defaultdict(Counter)
for line in open(ROOT / "data/amazon_beauty_backbone/Beauty_inner.txt"):
    ids = [int(x) for x in line.split()]
    ids = ids[1:] if len(ids) > 1 else ids
    ids = ids[:-1]  # exclude held-out final event (include-val convention)
    for p, i in zip(ids, ids[1:]):
        pred[i][p] += 1
        succ[p][i] += 1


def tops(counter, item_id, k):
    return [p for p, _ in counter.get(item_id, Counter()).most_common(k)
            if titles[p - 1]]


def build(variant):
    texts = []
    for i in range(N):
        t = base_texts[i] if base_texts[i] else " "
        if variant in ("pred5",):
            ps = tops(pred, i + 1, 5)
            if ps:
                t += " Often bought after: " + "; ".join(f"'{titles[p-1]}'" for p in ps) + "."
        if variant in ("succ", "both"):
            ss = tops(succ, i + 1, 3)
            if ss:
                t += " Often bought before: " + "; ".join(f"'{titles[s-1]}'" for s in ss) + "."
        if variant == "both":
            ps = tops(pred, i + 1, 3)
            if ps:
                t += " Often bought after: " + "; ".join(f"'{titles[p-1]}'" for p in ps) + "."
        texts.append(t)
    return texts


def embed(texts, out_path, embedder="openai"):
    t0 = time.time()
    torch.save(embed_texts(texts, embedder), out_path)
    print(f"wrote {out_path} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_embedder_arg(ap)
    args = ap.parse_args()
    for v in ("succ", "both", "pred5"):
        out = ROOT / f"data/amazon_beauty_backbone/Beauty_openai-3-large_i22b_{v}_inner_embeddings.pt"
        if out.exists():
            print(f"skip {v}", flush=True)
            continue
        embed(build(v), out, args.embedder)
