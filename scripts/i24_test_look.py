"""I24 — THE one sanctioned REAL-test look for the collab-input trio candidate.

Candidate (fixed BEFORE this script runs; selection 100% inner-world; prereg ledger
I24-LOOK-PREREG, UTC 2026-08-05T17:19:00Z): w=1.0 ensemble = mean of per-query z-scored
K4 max-IP tower scores of i24_real_openai_k4_s{42,43,44}_iv, where the query tower INPUT
is [own emb ; pred3 centroid] (Beauty_openai-3-large_i24real_input.pt) and the SCORING
side is the unchanged I22-enriched real catalog. Inner gate (i24_gate.json): all combos
selected w=1.0; trio pseudo-test 0.0802/0.1173/0.0523/0.0642 beat the I23 trio bar
0.0776 HR@5 (PRIMARY PASS). Serving class unchanged: 12 ANN probes + fixed merge
(I23-ANN replay validated the class).

Scores real test at w in {0.0, 1.0}: w=0 reproduces the P21 mix (parity check vs
0.0508/0.0821); w=1.0 is the look. Paired significance vs w=0 and vs the I23 frontier
per-user records. Refuses to run twice. Env guard: I24_TEST_LOOK=1.

Output: results/i24_test_look.json + i24_test_per_user.npz
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
SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/i24_real_openai_k4_s{s}_iv/j5.pt" for s in SEEDS}
ITEM_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i22real_embeddings.pt"
INPUT_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i24real_input.pt"
I23_NPZ = ROOT / "results/i23_test_per_user.npz"
OUT_JSON = ROOT / "results/i24_test_look.json"
OUT_NPZ = ROOT / "results/i24_test_per_user.npz"
MARKER = ROOT / "results/.i24_test_look_taken"

N_ITEMS = 12101
KS = (5, 10)
W_SELECTED = 1.0
BARS = {"I23_test": {"HR@5": 0.0690, "HR@10": 0.1054, "NDCG@5": 0.0443, "NDCG@10": 0.0560},
        "P21_test": {"HR@5": 0.0508, "HR@10": 0.0821, "NDCG@5": 0.0267, "NDCG@10": 0.0368},
        "CAR_HR@5": 0.0667, "GLoSS_8B_HR@5": 0.0681, "URM_7B_HR@5_unreplicated": 0.0929,
        "inner_gate_trio_pseudo_w1.0": {"HR@5": 0.0802, "HR@10": 0.1173,
                                        "NDCG@5": 0.0523, "NDCG@10": 0.0642}}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s):
    return (s - s.mean()) / max(float(s.std()), 1e-9)


def main():
    assert os.environ.get("I24_TEST_LOOK") == "1", "set I24_TEST_LOOK=1 to take the look"
    assert not MARKER.exists(), "REFUSED: the I24 real-test look was already taken"
    t0 = time.time()
    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)

    item = l2(torch.load(ITEM_EMB, map_location="cpu", weights_only=True).float().numpy())
    item_t = torch.from_numpy(item)
    inp = torch.load(INPUT_EMB, map_location="cpu", weights_only=True).float().numpy()
    qs = {}
    for s in SEEDS:
        model = load_checkpoint(str(CKPTS[s]), item, input_emb=inp)
        qs[s] = encode_split_queries(model, split, which="test", device="cpu")

    content = l2(torch.load(EMB_T5, map_location="cpu").float().numpy().astype(np.float32))
    train_seqs = [list(h) for h, _t in split.val]
    collab, covered = sp.build_collab_embeddings(train_seqs, N_ITEMS,
                                                 cache_path=str(ROOT / sp.DEFAULT_CACHE))
    content_w, collab_w = sp.whiten_blend_spaces(content, collab, covered)
    print(f"[look] spaces + 3x queries ready [{time.time()-t0:.0f}s]", flush=True)

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
        z_ens = np.mean([zscore(torch.einsum("kd,nd->kn", qs[s][qi], item_t)
                                .max(dim=0).values.numpy()) for s in SEEDS], axis=0)
        for w in ws:
            sc = (1.0 - w) * z_free + w * z_ens
            rank = int((sc > sc[tgt]).sum())
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

    if I23_NPZ.exists():
        i23 = np.load(I23_NPZ)
        sig_vs_i23 = significance.compare(
            cand, {m: i23[f"w1.0_{t}"] for m, t in metric_keys})
    else:  # release: the I23 per-user vectors are an input from an earlier look
        sig_vs_i23 = {"skipped": f"{I23_NPZ.relative_to(ROOT)} not present"}
    print(f"[look] significance vs I23 w=1.0: {json.dumps(sig_vs_i23, default=str)}",
          flush=True)

    report = {"experiment": "I24_test_look", "test_looks": 1,
              "candidate": {"w": W_SELECTED, "checkpoints": {s: str(CKPTS[s]) for s in SEEDS},
                            "item_emb": str(ITEM_EMB), "input_emb": str(INPUT_EMB),
                            "ensemble": "mean of per-query z-scored K4 max-IP tower "
                            "scores (3 seeds, collab-augmented inputs)",
                            "selection": "inner-world only (real val contaminated)"},
              "bars": BARS, "test": results,
              "significance_vs_w0": sig_vs_w0,
              "significance_vs_i23": sig_vs_i23,
              "runtime_sec": round(time.time() - t0, 1)}
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    np.savez_compressed(OUT_NPZ, cold_mask=cold, **pu)
    MARKER.write_text("taken " + time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"wrote {OUT_JSON} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
