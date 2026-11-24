"""I50/I51 — EXPAND REAL test looks (ONE look per dataset; ledger
I50-REAL-LOOK-PREREG / I51-REAL-LOOK-PREREG). Exact i35_real_look.py mirror
generalized over --dataset {instruments,yelp}.

PRIMARY: real trio (i5x_{name}_real_openai_k4_s{42,43,44}_iv; mean per-query z of
K4 max-IP) test HR@5 at the inner-world-carried w (ALL combos selected the same
w per dataset: instruments 1.0, yelp 0.9; real val is include-val contaminated
so no real-val selection) STRICTLY > w=0 free-arm real test HR@5.
SECONDARY (same scores, no extra look): prospective cold-routing — users whose
TARGET test item has zero occurrences in real train histories are routed to the
free arm; report routed-policy vs pure-trio HR@5/HR@10. Policy fixed BEFORE
seeing numbers (same as I35).
Context bars (published real test HR@5): Instruments SASRec 0.0751 / TIGER repro
0.0863-0.0870 / LLM-scale 0.099-0.102; Yelp BSARec 0.0275 / TIGER repro 0.0253.
Output: results/{tag}_real_look_{name}.json + marker (one-look guard).
"""
from __future__ import annotations

import argparse
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

DATASETS = {
    "instruments": dict(
        data=ROOT / "data/amazon_instruments_backbone", prefix="Instruments",
        seq="Musical_Instruments.txt", n=9922, tag="i50", w=1.0,
        context={"SASRec_HR@5": 0.0751, "TIGER_repro_HR@5": "0.0863-0.0870",
                 "LLM_scale_band": "0.099-0.102", "inner_trio_pseudo_HR@5": 0.1109},
    ),
    "yelp": dict(
        data=ROOT / "data/yelp_backbone", prefix="Yelp",
        seq="Yelp.txt", n=20033, tag="i51", w=0.9,
        context={"BSARec_HR@5": 0.0275, "TIGER_repro_HR@5": 0.0253},
    ),
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
NAME = ap.parse_args().dataset
cfg = DATASETS[NAME]
DATA, P, TAG = cfg["data"], cfg["prefix"], cfg["tag"]
EMB_T5 = DATA / f"{P}_sentence-t5-xl_embeddings.pt"
SEQ = DATA / cfg["seq"]
COLLAB_CACHE = ROOT / f"outputs/artifacts/collab_space_{NAME}_real.npz"
ITEM_EMB = DATA / f"{P}_openai-3-large_i22real_embeddings.pt"
INPUT_EMB = DATA / f"{P}_openai-3-large_i24real_input.pt"
SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/{TAG}_{NAME}_real_openai_k4_s{s}_iv/j5.pt"
         for s in SEEDS}
OUT_JSON = ROOT / f"results/{TAG}_real_look_{NAME}.json"
MARKER = ROOT / f"results/.{TAG}_real_look_{NAME}_taken"

N_ITEMS = cfg["n"]
KS = (5, 10)
W = cfg["w"]  # carried from inner world (all combos: instruments 1.0, yelp 0.9); no selection here


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
    print(f"[{TAG}r] spaces ready; cold items in catalog: "
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
            print(f"  [{TAG}r] test: {qi+1}/{n} [{time.time()-t0:.0f}s]", flush=True)

    def rep(a, mask=None):
        m = np.ones(n, dtype=bool) if mask is None else mask
        nn = max(int(m.sum()), 1)
        return {f"HR@{k}": round(float(hits[(a, k)][m].sum()) / nn, 4) for k in KS} | \
               {f"NDCG@{k}": round(float(dcgs[(a, k)][m].sum()) / nn, 4) for k in KS}

    out = {"experiment": f"{TAG.upper()}_{NAME}_real_look",
           "prereg": f"{TAG.upper()}-REAL-LOOK-PREREG", "w": W,
           "context": cfg["context"],
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
        print(f"[{TAG}r] {a}: {out['arms'][a]}", flush=True)
    print(f"[{TAG}r] cold (n={out['n_cold']}): {out['cold_slice']}", flush=True)
    print(f"[{TAG}r] PRIMARY {'PASS' if out['primary_pass'] else 'FAIL'} "
          f"(trio {trio_hr5} vs free {free_hr5}); wrote {OUT_JSON} "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
