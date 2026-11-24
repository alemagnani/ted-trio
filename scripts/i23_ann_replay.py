"""I23-ANN — serving-fidelity replay of the I23 ensemble (REV-7 blocker 1).

NOT a new look: same already-looked config (i23_test_look.py), re-scored under a
realistic ANN serving pipeline, parameters FIXED a priori (ledger REV-7-POST-FM6):
  - candidate generation: per query vector (3 seeds x K=4 = 12 probes), exact top-500
    by inner product (upper bound for a well-tuned ANN index at that depth);
  - union of the 12 candidate lists (plus nothing else — items outside the union get
    -inf, i.e. are unreachable, exactly as in serving);
  - per-seed z-stats (mean/std of that seed's max-IP score) estimated from a FIXED
    random sample of 1024 items (numpy seed 0), not the full corpus;
  - ensemble = mean over seeds of sampled-z scores, evaluated on the union only.
Reports served HR/NDCG vs the exact full-corpus look (0.0690/0.1054/0.0443/0.0560),
plus union-candidate recall of the test target. No tuning, no selection.

Output: results/i23_ann_replay.json
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
from semplan.dense.frozen_query import encode_split_queries, load_checkpoint  # noqa: E402
from semplan.eval import protocol  # noqa: E402

SEQ = ROOT / "data/amazon_beauty_backbone/Beauty.txt"
SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/i22_real_openai_k4_s{s}_iv/j5.pt" for s in SEEDS}
ITEM_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i22real_embeddings.pt"
OUT_JSON = ROOT / "results/i23_ann_replay.json"

N_ITEMS = 12101
KS = (5, 10)
DEPTH = 500          # fixed a priori (REV-7)
N_STAT_SAMPLE = 1024  # fixed a priori (REV-7), numpy seed 0
EXACT = {"HR@5": 0.0690, "HR@10": 0.1054, "NDCG@5": 0.0443, "NDCG@10": 0.0560}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def main():
    t0 = time.time()
    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)

    item = l2(torch.load(ITEM_EMB, map_location="cpu", weights_only=True).float().numpy())
    item_t = torch.from_numpy(item)
    qs = {}
    for s in SEEDS:
        model = load_checkpoint(str(CKPTS[s]), item)
        qs[s] = encode_split_queries(model, split, which="test", device="cpu")
    print(f"[replay] 3x queries ready [{time.time()-t0:.0f}s]", flush=True)

    rng = np.random.default_rng(0)
    stat_idx = torch.from_numpy(rng.choice(N_ITEMS, size=N_STAT_SAMPLE, replace=False))
    item_stat = item_t[stat_idx]

    events = [(list(h), t) for h, t in split.test]
    n = len(events)
    hits = {k: np.zeros(n, dtype=np.float32) for k in KS}
    dcgs = {k: np.zeros(n, dtype=np.float32) for k in KS}
    target_in_union = np.zeros(n, dtype=np.float32)
    union_sizes = np.zeros(n, dtype=np.float32)
    for qi, (_hist, tgt) in enumerate(events):
        cand = set()
        seed_scores = {}
        for s in SEEDS:
            per_probe = torch.einsum("kd,nd->kn", qs[s][qi], item_t)  # [K, N]
            for k in range(per_probe.shape[0]):
                cand.update(torch.topk(per_probe[k], DEPTH).indices.tolist())
            seed_scores[s] = per_probe.max(dim=0).values.numpy()
        cand = np.fromiter(cand, dtype=np.int64)
        union_sizes[qi] = len(cand)
        in_union = bool((cand == tgt).any())
        target_in_union[qi] = float(in_union)
        # sampled z-stats per seed (servable: 1024 fixed IPs per seed per query)
        z_sum = np.zeros(len(cand), dtype=np.float32)
        for s in SEEDS:
            samp = torch.einsum("kd,nd->kn", qs[s][qi], item_stat).max(dim=0).values.numpy()
            mu, sd = float(samp.mean()), max(float(samp.std()), 1e-9)
            z_sum += (seed_scores[s][cand] - mu) / sd
        if in_union:
            tgt_pos = int(np.nonzero(cand == tgt)[0][0])
            rank = int((z_sum > z_sum[tgt_pos]).sum())
            for k in KS:
                if rank < k:
                    hits[k][qi] = 1.0
                    dcgs[k][qi] = 1.0 / np.log2(rank + 2)
        if (qi + 1) % 2000 == 0:
            print(f"  [replay] {qi+1}/{n} [{time.time()-t0:.0f}s]", flush=True)

    served = {f"HR@{k}": round(float(hits[k].mean()), 4) for k in KS} | \
             {f"NDCG@{k}": round(float(dcgs[k].mean()), 4) for k in KS}
    report = {"experiment": "I23_ANN_serving_replay", "config": {
                  "depth_per_probe": DEPTH, "n_probes": 12,
                  "z_stat_sample": N_STAT_SAMPLE, "sample_rng_seed": 0},
              "exact_reference": EXACT, "served": served,
              "delta_vs_exact": {m: round(served[m] - EXACT[m], 4) for m in EXACT},
              "target_in_union_recall": round(float(target_in_union.mean()), 4),
              "mean_union_size": round(float(union_sizes.mean()), 1),
              "runtime_sec": round(time.time() - t0, 1)}
    print(json.dumps(report, indent=2), flush=True)
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
