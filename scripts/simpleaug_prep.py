"""SIMPLE-AUG real prep (task #87) — build enriched E embeddings + [own;centroid] C input
for one augmentation selector, matching the I22/I24/I33 recipe EXACTLY (only the selector
that chooses the 3 neighbours changes). Inner world (Beauty_inner.txt).

For selector NAME, writes (openai text-embedding-3-large, 3072-d):
  data/amazon_beauty_backbone/Beauty_openai-3-large_aug_{NAME}_inner_embeddings.pt   (E)
  data/amazon_beauty_backbone/Beauty_openai-3-large_aug_{NAME}_inner_input.pt        (C = [l2(e); l2(centroid)])

Recipe (identical to i33_prep build_i22_texts + input rebuild):
  E: item text + " Often bought after: 'A'; 'B'; 'C'."  (selector's 3 neighbour titles)
  C: concat(l2(e), l2(mean(e[selector(i)])))   using the SAME selector's neighbours
Then towers train on the box with scripts/j5 (item-emb = E file, input-emb = C file).

'pred3' and 'none' are anchors we already have (i22inner / base) — this script skips them.
Embedder: --embedder openai (default; $OPENAI_API_KEY) | fake | st:<model>. Resumable via per-selector emb cache.

Usage: python scripts/simpleaug_prep.py [--embedder fake] contentnn3 ppmi3 translift3 ...
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import simpleaug_selectors as sel  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from semplan.embedders import embed_texts  # noqa: E402

DATA = ROOT / "data/amazon_beauty_backbone"
SEQ = DATA / "Beauty_inner.txt"
META = ROOT / "data/amazon_meta/beauty_id2meta.json"
BASE_EMB = DATA / "Beauty_openai-3-large_embeddings.pt"
EMB_MODEL = "text-embedding-3-large"
N = 12101
BATCH = 256


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def load_seqs():
    return [[int(x) for x in line.split()][1:] for line in open(SEQ)]


def embed(texts, cache, embedder="openai"):
    if cache.exists():
        return torch.load(cache, map_location="cpu", weights_only=True).float().numpy()
    arr = embed_texts(texts, embedder).numpy().astype(np.float32)
    torch.save(torch.from_numpy(arr), cache)
    return arr


def _pop_embedder(argv):
    """`--embedder X` (openai | fake | st:<model>) may appear anywhere in argv."""
    argv = list(argv)
    if "--embedder" in argv:
        i = argv.index("--embedder")
        emb = argv[i + 1]
        del argv[i:i + 2]
        return emb, argv
    return "openai", argv


def main():
    embedder, argv = _pop_embedder(sys.argv[1:])
    names = argv
    if not names:
        print("usage: simpleaug_prep.py <selector> [<selector> ...]"); return
    meta = json.load(open(META))
    base_texts = [""] * N
    for k, t in meta.items():
        base_texts[int(k) - 1] = t
    titles = sel.load_titles(base_texts)
    seqs = load_seqs()
    base_emb = torch.load(BASE_EMB, map_location="cpu", weights_only=True).float().numpy()
    selmap = sel.build_all(seqs, base_emb, titles, N)

    for name in names:
        if name in ("none", "pred3"):
            print(f"[{name}] anchor (already have base/i22inner) — skip"); continue
        if name == "bm25_3":
            selector = sel.build_bm25(base_texts, N)
        elif name == "catbrand3":
            selector = sel.build_catbrand(
                ROOT / "data/amazon_beauty_backbone/Beauty_item2attributes.json",
                titles, N)
        else:
            selector = selmap.get(name)
        if selector is None:
            print(f"[{name}] unknown selector — skip"); continue
        t0 = time.time()
        # E: enriched text
        texts, cov = [], 0
        for i in range(1, N + 1):
            t = base_texts[i - 1] if base_texts[i - 1] else " "
            tops = [p for p in selector(i) if 1 <= p <= N and titles[p - 1]][:3]
            if tops:
                t = t + " Often bought after: " + "; ".join(
                    f"'{titles[p - 1]}'" for p in tops) + "."
                cov += 1
            texts.append(t)
        print(f"[{name}] {cov}/{N} augmented; embedding E ...", flush=True)
        emb = embed(texts, DATA / f"Beauty_openai-3-large_aug_{name}_inner_embeddings.pt", embedder)
        # C: [own ; centroid of selector neighbours] on the ENRICHED embeddings (i30 recipe)
        e = l2(emb)
        aug = e.copy()
        for i in range(1, N + 1):
            ps = [p - 1 for p in selector(i) if 1 <= p <= N and titles[p - 1]][:3]
            if ps:
                aug[i - 1] = e[ps].mean(axis=0)
        out = np.concatenate([e, l2(aug)], axis=1).astype(np.float32)
        torch.save(torch.from_numpy(out),
                   DATA / f"Beauty_openai-3-large_aug_{name}_inner_input.pt")
        print(f"[{name}] E+C written, input shape {out.shape} [{time.time()-t0:.0f}s]",
              flush=True)


if __name__ == "__main__":
    main()
