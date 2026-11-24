"""I30 — THE one sanctioned REAL-test look for the Toys transfer trio.

Prereg ledger I30-REAL-PREREG (UTC 2026-08-06T00:38:45Z), licensed by inner gate PASS
(i30_gate.json: trio pseudo HR@5 0.0823 > free 0.0659; every combo selected w=1.0).
Candidate: trio = mean per-query z of K4 max-IP scores of
i30_toys_real_openai_k4_s{42,43,44}_iv (include-val towers on Toys_and_Games.txt,
i22real-enriched openai catalog scoring side, [own;pred3] i24real input side).
Per prereg: blend w over [0.0,0.5,0.7,0.8,0.9,1.0] with t5-xl+collab whitened free arm,
w selected on REAL val by (HR@10, NDCG@5) — CAVEAT: include-val towers make real val
contaminated for tower-heavy w; the uncontaminated inner world independently selected
w=1.0 for every combo, recorded here for transparency. ONE test look at the selected w
(w=0 free arm also scored for the paired-significance reference). Refuses to run twice.
Env guard: I30_REAL_LOOK=1.

Context published bars (Toys, LOO full-corpus): TIGER HR@5 0.0521 (NDCG@5 0.0371).
Output: results/i30_real_look.json + i30_real_per_user.npz
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
sys.path.insert(0, str(ROOT / "src"))
from semplan import spaces as sp  # noqa: E402
from semplan.dense.frozen_query import encode_split_queries, load_checkpoint  # noqa: E402
from semplan.eval import protocol, significance  # noqa: E402

DATA = ROOT / "data/amazon_toys_backbone"
EMB_T5 = DATA / "Toys_sentence-t5-xl_embeddings.pt"
SEQ = DATA / "Toys_and_Games.txt"
ITEM_EMB = DATA / "Toys_openai-3-large_i22real_embeddings.pt"
INPUT_EMB = DATA / "Toys_openai-3-large_i24real_input.pt"
COLLAB_CACHE = ROOT / "outputs/artifacts/collab_space_toys_real.npz"
SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/i30_toys_real_openai_k4_s{s}_iv/j5.pt"
         for s in SEEDS}
OUT_JSON = ROOT / "results/i30_real_look.json"
OUT_NPZ = ROOT / "results/i30_real_per_user.npz"
MARKER = ROOT / "results/.i30_real_look_taken"

N_ITEMS = 11924
KS = (5, 10)
WEIGHTS = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
BARS = {"TIGER_Toys": {"HR@5": 0.0521, "NDCG@5": 0.0371},
        "inner_gate_trio_pseudo_w1.0": {"HR@5": 0.0823, "HR@10": 0.1229,
                                        "NDCG@5": 0.0536, "NDCG@10": 0.0668},
        "inner_world_selected_w": 1.0}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s):
    return (s - s.mean()) / max(float(s.std()), 1e-9)


def trio_sweep(split, which, qs, item_t, content_w, collab_w, covered, t0):
    events = [(list(h), t) for h, t in getattr(split, which)]
    n = len(events)
    hits = {(w, k): np.zeros(n, dtype=np.float32) for w in WEIGHTS for k in KS}
    dcgs = {(w, k): np.zeros(n, dtype=np.float32) for w in WEIGHTS for k in KS}
    for qi, (hist, tgt) in enumerate(events):
        b03 = sp.blended_scores_whitened(content_w, collab_w, covered, hist,
                                         recency=sp.WHITEN_RECENCY)
        b01 = sp.blended_scores_whitened(content_w, collab_w, covered, hist, recency=0.1)
        z_free = zscore(0.9 * b03 + 0.1 * b01)
        z_ens = np.mean([zscore(torch.einsum("kd,nd->kn", qs[s][qi], item_t)
                                .max(dim=0).values.numpy()) for s in SEEDS], axis=0)
        for w in WEIGHTS:
            sc = (1.0 - w) * z_free + w * z_ens
            rank = int((sc > sc[tgt]).sum())
            for k in KS:
                if rank < k:
                    hits[(w, k)][qi] = 1.0
                    dcgs[(w, k)][qi] = 1.0 / np.log2(rank + 2)
        if (qi + 1) % 5000 == 0:
            print(f"  [i30look] {which}: {qi+1}/{n} [{time.time()-t0:.0f}s]", flush=True)
    return hits, dcgs


def main():
    assert os.environ.get("I30_REAL_LOOK") == "1", "set I30_REAL_LOOK=1 to take the look"
    assert not MARKER.exists(), "REFUSED: the I30 real-test look was already taken"
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
    print(f"[i30look] real spaces ready [{time.time()-t0:.0f}s]", flush=True)

    q_val = {s: encode_split_queries(m, split, which="val", device="cpu")
             for s, m in models.items()}
    vh, vd = trio_sweep(split, "val", q_val, item_t, content_w, collab_w, covered, t0)
    val_rep = {str(w): {f"HR@{k}": round(float(vh[(w, k)].mean()), 4) for k in KS} |
                       {f"NDCG@{k}": round(float(vd[(w, k)].mean()), 4) for k in KS}
               for w in WEIGHTS}
    w_sel = max(WEIGHTS, key=lambda w: (val_rep[str(w)]["HR@10"],
                                        val_rep[str(w)]["NDCG@5"]))
    print(f"[i30look] real-val selected w={w_sel} (contaminated-val caveat; inner-world "
          f"pick was 1.0): {val_rep[str(w_sel)]}", flush=True)

    q_test = {s: encode_split_queries(m, split, which="test", device="cpu")
              for s, m in models.items()}
    th, td = trio_sweep(split, "test", q_test, item_t, content_w, collab_w, covered, t0)

    n = len(split.test)
    cold = np.zeros(n, dtype=bool)
    cold[list(split.cold_test_idx)] = True

    def summarize(w):
        rep = {f"HR@{k}": round(float(th[(w, k)].mean()), 4) for k in KS}
        rep |= {f"NDCG@{k}": round(float(td[(w, k)].mean()), 4) for k in KS}
        rep["cold"] = {f"HR@{k}": round(float(th[(w, k)][cold].mean()), 4) for k in KS}
        return rep

    results = {str(w): summarize(w) for w in (0.0, w_sel)}
    for w in (0.0, w_sel):
        print(f"[i30look] test w={w}: {json.dumps(results[str(w)])}", flush=True)

    metric_keys = [("HR@5", 5, "h"), ("HR@10", 10, "h"),
                   ("NDCG@5", 5, "d"), ("NDCG@10", 10, "d")]
    cand = {m: (th if t == "h" else td)[(w_sel, k)] for m, k, t in metric_keys}
    free = {m: (th if t == "h" else td)[(0.0, k)] for m, k, t in metric_keys}
    sig_vs_free = significance.compare(cand, free)
    print(f"[i30look] significance vs w=0: {json.dumps(sig_vs_free, default=str)}",
          flush=True)

    report = {"experiment": "I30_real_look", "test_looks": 1,
              "candidate": {"w_selected": w_sel,
                            "selection": "real val (contaminated caveat; inner-world "
                                         "independently selected 1.0)",
                            "checkpoints": {s: str(CKPTS[s]) for s in SEEDS},
                            "item_emb": str(ITEM_EMB), "input_emb": str(INPUT_EMB),
                            "ensemble": "mean of per-query z-scored K4 max-IP tower "
                                        "scores (3 seeds, collab-augmented inputs)"},
              "bars": BARS, "real_val_sweep": val_rep, "test": results,
              "significance_vs_w0": sig_vs_free,
              "runtime_sec": round(time.time() - t0, 1)}
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    pu = {f"w{w}_hit{k}": th[(w, k)] for w in (0.0, w_sel) for k in KS}
    pu |= {f"w{w}_dcg{k}": td[(w, k)] for w in (0.0, w_sel) for k in KS}
    np.savez_compressed(OUT_NPZ, cold_mask=cold, **pu)
    MARKER.write_text("taken " + time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"[i30look] wrote {OUT_JSON} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
