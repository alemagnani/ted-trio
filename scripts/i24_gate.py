"""I24 — collab-augmented query-input gate (inner world only).

PREREG (UTC 2026-08-05T05:35:00Z, REV-7 prereg discipline):
  PRIMARY criterion (single, numeric): I24 trio 42+43+44 pseudo-test HR@5 at its
  inner-val-selected w must EXCEED the I23 trio's pseudo-test HR@5 = 0.0776.
  Pass => preregister one real-world look for the I24 real trio (towers already
  building speculatively), significance vs results/i23_test_per_user.npz.
  Fail => book "input-side pred3-centroid augmentation adds nothing over item-side
  enrichment + seed ensembling"; no look, no further spend on this axis.
  Secondary (descriptive, no veto): single-seed reads vs 0.0756/0.0740/0.0732.

Towers: i24_inner_openai_k4_s{42,43,44}_iv — query input = [own emb ; pred3 centroid]
(6144-d), scoring side = i22inner enriched matrix (unchanged). Same mean-z ensemble
machinery as i23_ensemble_gate.py.
Output: results/i24_gate.json
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan import spaces as sp  # noqa: E402
from semplan.dense.frozen_query import encode_split_queries, load_checkpoint  # noqa: E402
from semplan.eval import protocol  # noqa: E402

EMB_T5 = ROOT / "data/amazon_beauty_backbone/Beauty_sentence-t5-xl_embeddings.pt"
SEQ = ROOT / "data/amazon_beauty_backbone/Beauty_inner.txt"
COLLAB_CACHE = ROOT / "outputs/artifacts/collab_space_beauty_inner.npz"
ITEM_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i22inner_embeddings.pt"
INPUT_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i24inner_input.pt"
SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/i24_inner_openai_k4_s{s}_iv/j5.pt" for s in SEEDS}
OUT_JSON = ROOT / "results/i24_gate.json"

N_ITEMS = 12101
KS = (5, 10)
WEIGHTS = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
COMBOS = [(s,) for s in SEEDS] + [SEEDS]
BARS = {"i23_trio_pseudo_w1.0": {"HR@5": 0.0776, "HR@10": 0.1176, "NDCG@5": 0.0507},
        "i22_single_seed_pseudo_w1.0_HR@5": {"42": 0.0756, "43": 0.0740, "44": 0.0732},
        "gate": "PRIMARY: I24 trio pseudo-test HR@5 at selected w must exceed 0.0776"}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s):
    return (s - s.mean()) / max(float(s.std()), 1e-9)


def sweep_split(split, which, qs, item_t, content_w, collab_w, covered, t0):
    events = [(list(h), t) for h, t in getattr(split, which)]
    n = len(events)
    hits = {(c, w, k): np.zeros(n, dtype=np.float32)
            for c in COMBOS for w in WEIGHTS for k in KS}
    dcgs = {(c, w, k): np.zeros(n, dtype=np.float32)
            for c in COMBOS for w in WEIGHTS for k in KS}
    for qi, (hist, tgt) in enumerate(events):
        b03 = sp.blended_scores_whitened(content_w, collab_w, covered, hist,
                                         recency=sp.WHITEN_RECENCY)
        b01 = sp.blended_scores_whitened(content_w, collab_w, covered, hist, recency=0.1)
        z_free = zscore(0.9 * b03 + 0.1 * b01)
        z_tow = {s: zscore(torch.einsum("kd,nd->kn", qs[s][qi], item_t)
                           .max(dim=0).values.numpy()) for s in SEEDS}
        for c in COMBOS:
            z_ens = np.mean([z_tow[s] for s in c], axis=0)
            for w in WEIGHTS:
                sc = (1.0 - w) * z_free + w * z_ens
                rank = int((sc > sc[tgt]).sum())
                for k in KS:
                    if rank < k:
                        hits[(c, w, k)][qi] = 1.0
                        dcgs[(c, w, k)][qi] = 1.0 / np.log2(rank + 2)
        if (qi + 1) % 5000 == 0:
            print(f"  [i24] {which}: {qi+1}/{n} [{time.time()-t0:.0f}s]", flush=True)
    return {"+".join(map(str, c)): {
        str(w): {f"HR@{k}": round(float(hits[(c, w, k)].mean()), 4) for k in KS} |
                {f"NDCG@{k}": round(float(dcgs[(c, w, k)].mean()), 4) for k in KS}
        for w in WEIGHTS} for c in COMBOS}


def main():
    t0 = time.time()
    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)

    item = l2(torch.load(ITEM_EMB, map_location="cpu", weights_only=True).float().numpy())
    item_t = torch.from_numpy(item)
    inp = torch.load(INPUT_EMB, map_location="cpu", weights_only=True).float().numpy()
    models = {s: load_checkpoint(str(CKPTS[s]), item, input_emb=inp) for s in SEEDS}

    content = l2(torch.load(EMB_T5, map_location="cpu").float().numpy().astype(np.float32))
    train_seqs = [list(h) for h, _t in split.val]
    collab, covered = sp.build_collab_embeddings(train_seqs, N_ITEMS,
                                                 cache_path=str(COLLAB_CACHE))
    content_w, collab_w = sp.whiten_blend_spaces(content, collab, covered)
    print(f"[i24] inner spaces ready [{time.time()-t0:.0f}s]", flush=True)

    q_val = {s: encode_split_queries(models[s], split, which="val", device="cpu")
             for s in SEEDS}
    val_sweep = sweep_split(split, "val", q_val, item_t, content_w, collab_w, covered, t0)
    selected = {}
    for c_key, sweep in val_sweep.items():
        best = max(WEIGHTS, key=lambda w: (sweep[str(w)]["HR@10"], sweep[str(w)]["NDCG@5"]))
        selected[c_key] = best
        print(f"[i24] {c_key} inner-val winner w={best}: {sweep[str(best)]}", flush=True)

    q_test = {s: encode_split_queries(models[s], split, which="test", device="cpu")
              for s in SEEDS}
    test_sweep = sweep_split(split, "test", q_test, item_t, content_w, collab_w, covered, t0)
    for c_key in test_sweep:
        w = selected[c_key]
        print(f"[i24] {c_key} pseudo-test w={w}: {test_sweep[c_key][str(w)]}  "
              f"(w=1.0: {test_sweep[c_key]['1.0']})", flush=True)

    trio = "+".join(map(str, SEEDS))
    verdict = test_sweep[trio][str(selected[trio])]["HR@5"] > BARS["i23_trio_pseudo_w1.0"]["HR@5"]
    json.dump({"experiment": "I24_collab_input_gate", "prereg_utc": "2026-08-05T05:35:00Z",
               "bars": BARS, "primary_pass": bool(verdict),
               "weights": WEIGHTS, "selected_w": selected,
               "inner_val": val_sweep, "pseudo_test": test_sweep,
               "runtime_sec": round(time.time() - t0, 1)},
              open(OUT_JSON, "w"), indent=2)
    print(f"[i24] PRIMARY {'PASS' if verdict else 'FAIL'}; wrote {OUT_JSON} "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
