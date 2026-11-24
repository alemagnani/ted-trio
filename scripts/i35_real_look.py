"""I35 — Sports REAL test look (ONE look; ledger I35-REAL-LOOK-PREREG 2026-08-06T12:20:38Z).

PRIMARY: real trio (i35_sports_real_openai_k4_s{42,43,44}_iv; mean per-query z of K4
max-IP) test HR@5 at w=1.0 (carried from inner world — ALL combos selected w=1.0;
real val is include-val contaminated so no real-val selection) STRICTLY > w=0
free-arm real test HR@5.
SECONDARY (same scores, no extra look): prospective cold-routing — users whose
TARGET test item has zero occurrences in real train histories are routed to the
free arm; report routed-policy vs pure-trio HR@5/HR@10. Policy fixed BEFORE
seeing numbers (CODEX-CHECKPOINT-2 rank 3).
Context: published TIGER Sports real test HR@5 0.0189; inner trio pseudo 0.0439.
Output: results/i35_real_look.json + .i35_real_look_taken marker.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan import spaces as sp  # noqa: E402
from semplan.dense.frozen_query import encode_split_queries, load_checkpoint  # noqa: E402
from semplan.eval import protocol, significance  # noqa: E402

DATA = ROOT / "data/amazon_sports_backbone"
EMB_T5 = DATA / "Sports_sentence-t5-xl_embeddings.pt"
SEQ = DATA / "Sports_and_Outdoors.txt"
COLLAB_CACHE = ROOT / "outputs/artifacts/collab_space_sports_real.npz"
ITEM_EMB = DATA / "Sports_openai-3-large_i22real_embeddings.pt"
INPUT_EMB = DATA / "Sports_openai-3-large_i24real_input.pt"
SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/i35_sports_real_openai_k4_s{s}_iv/j5.pt"
         for s in SEEDS}
OUT_JSON = ROOT / "results/i35_real_look.json"
MARKER = ROOT / "results/.i35_real_look_taken"

N_ITEMS = 18357
KS = (5, 10)
W = 1.0  # carried from inner world; no selection here


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s):
    return (s - s.mean()) / max(float(s.std()), 1e-9)


def main():
    if MARKER.exists():
        sys.exit("REFUSING: real look already taken (marker present)")
    t0 = time.time()
    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)

    item = l2(torch.load(ITEM_EMB, map_location="cpu", weights_only=True).float().numpy())
    item_t = torch.from_numpy(item)
    inp = torch.load(INPUT_EMB, map_location="cpu", weights_only=True).float().numpy()
    models = {str(s): load_checkpoint(str(CKPTS[s]), item, input_emb=inp) for s in SEEDS}

    content = l2(torch.load(EMB_T5, map_location="cpu").float().numpy().astype(np.float32))
    train_seqs = [list(h) for h, _t in split.val]  # real train histories (up to -2)
    collab, covered = sp.build_collab_embeddings(train_seqs, N_ITEMS,
                                                 cache_path=str(COLLAB_CACHE))
    content_w, collab_w = sp.whiten_blend_spaces(content, collab, covered)

    # cold set: target test item never appears in any train history (prospective)
    train_count = np.zeros(N_ITEMS, dtype=np.int64)
    for h in train_seqs:
        for it in h:
            train_count[it] += 1
    print(f"[i35r] spaces ready; cold items in catalog: "
          f"{int((train_count == 0).sum())} [{time.time()-t0:.0f}s]", flush=True)

    q_test = {nm: encode_split_queries(m, split, which="test", device="cpu")
              for nm, m in models.items()}
    events = [(list(h), t) for h, t in split.test]
    n = len(events)
    arms = ("trio", "free", "routed") + tuple(str(s) for s in SEEDS)
    hits = {(a, k): np.zeros(n, dtype=np.float32) for a in arms for k in KS}
    dcgs = {(a, k): np.zeros(n, dtype=np.float32) for a in arms for k in KS}
    cold_mask = np.zeros(n, dtype=bool)
    for qi, (hist, tgt) in enumerate(events):
        cold_mask[qi] = train_count[tgt] == 0
        b03 = sp.blended_scores_whitened(content_w, collab_w, covered, hist,
                                         recency=sp.WHITEN_RECENCY)
        b01 = sp.blended_scores_whitened(content_w, collab_w, covered, hist, recency=0.1)
        z_free = zscore(0.9 * b03 + 0.1 * b01)
        z_tow = {}
        for nm in (str(s) for s in SEEDS):
            ip = torch.einsum("kd,nd->kn", q_test[nm][qi], item_t).numpy()
            z_tow[nm] = zscore(ip.max(axis=0))
        z_trio = np.mean(list(z_tow.values()), axis=0)
        scores = {"trio": W * z_trio + (1.0 - W) * z_free, "free": z_free}
        scores["routed"] = scores["free"] if cold_mask[qi] else scores["trio"]
        for s in SEEDS:
            scores[str(s)] = z_tow[str(s)]
        for a in arms:
            sc = scores[a]
            rank = int((sc > sc[tgt]).sum())
            for k in KS:
                if rank < k:
                    hits[(a, k)][qi] = 1.0
                    dcgs[(a, k)][qi] = 1.0 / np.log2(rank + 2)
        if (qi + 1) % 5000 == 0:
            print(f"  [i35r] test: {qi+1}/{n} [{time.time()-t0:.0f}s]", flush=True)

    def rep(a, mask=None):
        m = np.ones(n, dtype=bool) if mask is None else mask
        nn = max(int(m.sum()), 1)
        return {f"HR@{k}": round(float(hits[(a, k)][m].sum()) / nn, 4) for k in KS} | \
               {f"NDCG@{k}": round(float(dcgs[(a, k)][m].sum()) / nn, 4) for k in KS}

    out = {"experiment": "I35_sports_real_look",
           "prereg": "I35-REAL-LOOK-PREREG 2026-08-06T12:20:38Z", "w": W,
           "context": {"TIGER_Sports_HR@5": 0.0189, "inner_trio_pseudo_HR@5": 0.0439},
           "n_test": n, "n_cold": int(cold_mask.sum()),
           "arms": {a: rep(a) for a in arms},
           "cold_slice": {a: rep(a, cold_mask) for a in ("trio", "free", "routed")},
           "noncold_slice": {a: rep(a, ~cold_mask) for a in ("trio", "free")}}
    trio_hr5, free_hr5 = out["arms"]["trio"]["HR@5"], out["arms"]["free"]["HR@5"]
    out["primary_pass"] = bool(trio_hr5 > free_hr5)
    def vec(a):
        return {f"HR@{k}": hits[(a, k)] for k in KS} | \
               {f"NDCG@{k}": dcgs[(a, k)] for k in KS}
    out["significance_trio_vs_free"] = significance.compare(
        vec("trio"), vec("free"), cold_mask=cold_mask)
    out["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    MARKER.touch()
    for a in arms:
        print(f"[i35r] {a}: {out['arms'][a]}", flush=True)
    print(f"[i35r] cold (n={out['n_cold']}): {out['cold_slice']}", flush=True)
    print(f"[i35r] PRIMARY {'PASS' if out['primary_pass'] else 'FAIL'} "
          f"(trio {trio_hr5} vs free {free_hr5}); wrote {OUT_JSON} "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
