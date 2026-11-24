"""I22 — THE one sanctioned REAL-test look for the collab-text-enriched-catalog candidate.

Candidate (fixed BEFORE this script runs; selection 100% inner-world, real val is
contaminated for this arm by construction; prereg ledger I22-LOOK-PREREG):
  w=1.0 pure tower score: include-val openai_k4 tower over the I22-enriched catalog
  (items re-embedded with top-3 co-occurrence-predecessor titles appended), K=4 max-IP.
  Tower = i22_real_openai_k4_s42_iv (canonical seed 42; trained on Beauty.txt WITH the
  val transition; item emb Beauty_openai-3-large_i22real_embeddings.pt whose enrichment
  co-occurrence uses transitions up to -2 only — real test target never seen anywhere).
  Inner gate 3/3: all inner worlds select w=1.0 — pseudo-test HR@5 0.0756/0.0740/0.0732,
  HR@10 0.1122/0.1130/0.1105, NDCG@5 0.0492/0.0488/0.0478; blend w=0.8 also beats the
  K4-iv bars 0.1145/0.1151/0.1131 vs 0.1107/0.1095/0.1106.

Scores real test at w in {0.0, 1.0}: w=0 reproduces the P21 mix (parity check vs
0.0508/0.0821); w=1.0 is the look. Paired significance vs w=0 AND vs the frozen J7
per-user records (results/j7_test_per_user.npz). Refuses to run twice.
Env guard: I22_TEST_LOOK=1 required.

Output: results/i22_test_look.json + i22_test_per_user.npz
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

EMB_T5 = ROOT / "data/amazon_beauty_backbone/Beauty_sentence-t5-xl_embeddings.pt"
SEQ = ROOT / "data/amazon_beauty_backbone/Beauty.txt"
CKPT = ROOT / "outputs/artifacts/i22_real_openai_k4_s42_iv/j5.pt"
ITEM_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i22real_embeddings.pt"
J7_NPZ = ROOT / "results/j7_test_per_user.npz"
OUT_JSON = ROOT / "results/i22_test_look.json"
OUT_NPZ = ROOT / "results/i22_test_per_user.npz"
MARKER = ROOT / "results/.i22_test_look_taken"

N_ITEMS = 12101
KS = (5, 10)
W_SELECTED = 1.0
BARS = {"J7_test": {"HR@5": 0.0631, "HR@10": 0.1042, "NDCG@5": 0.0332, "NDCG@10": 0.0465},
        "P21_test": {"HR@5": 0.0508, "HR@10": 0.0821, "NDCG@5": 0.0267, "NDCG@10": 0.0368},
        "CAR_HR@5": 0.0667, "GLoSS_8B_HR@5": 0.0681,
        "inner_gate_pseudo_test_w1.0": {
            "s42": {"HR@5": 0.0756, "HR@10": 0.1122, "NDCG@5": 0.0492},
            "s43": {"HR@5": 0.0740, "HR@10": 0.1130, "NDCG@5": 0.0488},
            "s44": {"HR@5": 0.0732, "HR@10": 0.1105, "NDCG@5": 0.0478}}}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s):
    return (s - s.mean()) / max(float(s.std()), 1e-9)


def main():
    assert os.environ.get("I22_TEST_LOOK") == "1", "set I22_TEST_LOOK=1 to take the look"
    assert not MARKER.exists(), "REFUSED: the I22 real-test look was already taken"
    t0 = time.time()
    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)

    item = l2(torch.load(ITEM_EMB, map_location="cpu", weights_only=True).float().numpy())
    model = load_checkpoint(str(CKPT), item)
    item_t = torch.from_numpy(item)
    q = encode_split_queries(model, split, which="test", device="cpu")

    content = l2(torch.load(EMB_T5, map_location="cpu").float().numpy().astype(np.float32))
    train_seqs = [list(h) for h, _t in split.val]
    collab, covered = sp.build_collab_embeddings(train_seqs, N_ITEMS,
                                                 cache_path=str(ROOT / sp.DEFAULT_CACHE))
    content_w, collab_w = sp.whiten_blend_spaces(content, collab, covered)
    print(f"[look] spaces + queries ready [{time.time()-t0:.0f}s]", flush=True)

    events = [(list(h), t) for h, t in split.test]
    n = len(events)
    ws = (0.0, W_SELECTED)
    pu = {f"w{w}_{m}{k}": np.zeros(n, dtype=np.float32)
          for w in ws for k in KS for m in ("hit", "dcg")}
    cold = np.zeros(n, dtype=bool)
    cold[list(split.cold_test_idx)] = True
    for qi, (hist, tgt) in enumerate(events):
        b03 = sp.blended_scores_whitened(content_w, collab_w, covered, hist,
                                         recency=sp.WHITEN_RECENCY)
        b01 = sp.blended_scores_whitened(content_w, collab_w, covered, hist, recency=0.1)
        z_free = zscore(0.9 * b03 + 0.1 * b01)
        s_tr = torch.einsum("kd,nd->kn", q[qi], item_t).max(dim=0).values.numpy()
        z_tr = zscore(s_tr)
        for w in ws:
            s = (1.0 - w) * z_free + w * z_tr
            rank = int((s > s[tgt]).sum())
            for k in KS:
                if rank < k:
                    pu[f"w{w}_hit{k}"][qi] = 1.0
                    pu[f"w{w}_dcg{k}"][qi] = 1.0 / np.log2(rank + 2)
        if (qi + 1) % 5000 == 0:
            print(f"  [look] test: {qi+1}/{n} [{time.time()-t0:.0f}s]", flush=True)

    def summarize(w):
        rep = {}
        for k in KS:
            rep[f"HR@{k}"] = round(float(pu[f"w{w}_hit{k}"].mean()), 4)
            rep[f"NDCG@{k}"] = round(float(pu[f"w{w}_dcg{k}"].mean()), 4)
        rep["cold"] = {f"HR@{k}": round(float(pu[f"w{w}_hit{k}"][cold].mean()), 4)
                       for k in KS}
        return rep

    results = {str(w): summarize(w) for w in ws}
    for w in ws:
        print(f"[look] test w={w}: {json.dumps(results[str(w)])}", flush=True)

    metric_keys = [("HR@5", "hit5"), ("HR@10", "hit10"),
                   ("NDCG@5", "dcg5"), ("NDCG@10", "dcg10")]
    cand = {m: pu[f"w{W_SELECTED}_{t}"] for m, t in metric_keys}
    sig_vs_w0 = significance.compare(
        cand, {m: pu[f"w0.0_{t}"] for m, t in metric_keys})
    print(f"[look] significance vs w=0 (P21 mix): {json.dumps(sig_vs_w0, default=str)}",
          flush=True)

    j7 = np.load(J7_NPZ)
    sig_vs_j7 = significance.compare(
        cand, {m: j7[f"w0.8_{t}"] for m, t in metric_keys})
    print(f"[look] significance vs J7 w=0.8: {json.dumps(sig_vs_j7, default=str)}",
          flush=True)

    report = {"experiment": "I22_test_look", "test_looks": 1,
              "candidate": {"w": W_SELECTED, "checkpoint": str(CKPT),
                            "item_emb": str(ITEM_EMB), "seed": 42,
                            "selection": "inner-world only (real val contaminated)"},
              "bars": BARS, "test": results,
              "significance_vs_w0": sig_vs_w0,
              "significance_vs_j7": sig_vs_j7,
              "runtime_sec": round(time.time() - t0, 1)}
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    np.savez_compressed(OUT_NPZ, cold_mask=cold, **pu)
    MARKER.write_text("taken " + time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"wrote {OUT_JSON} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
