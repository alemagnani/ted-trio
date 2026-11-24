"""P26 — inner-validation derisk for the P25c frontier candidate (J4a gap response).

World shifted one event back: corpus = Beauty_inner.txt (each user's LAST item dropped),
so the canonical split gives inner-val = original −3 and pseudo-test = original −2. The
REAL test target (−1) appears in no P26 file, so scoring pseudo-test here costs nothing.

Procedure (mirrors the exactly-once discipline we would apply to the real look):
  1. sweep blend weight w on INNER-VAL only, select the winner;
  2. score PSEUDO-TEST at w ∈ {0 (= P21-mix reference), selected, 1 (= tower only)};
  3. report the transfer ratio: (pseudo-test gain at selected w) / (inner-val gain).
A ratio near 1 says the P25c val edge is real; a J4a-style collapse says HOLD the look.

Env: P26_CKPT (j5.pt trained with --seq Beauty_inner.txt), P26_ITEM_EMB, P26_TAG
Argv: weight grid (default 0.0 0.3 0.5 0.6 0.7 0.8 0.9 1.0)
Output: results/p26_inner_validation.json (merges per tag)
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
from semplan.eval import protocol  # noqa: E402

EMB_T5 = ROOT / "data/amazon_beauty_backbone/Beauty_sentence-t5-xl_embeddings.pt"
SEQ = ROOT / "data/amazon_beauty_backbone/Beauty_inner.txt"
COLLAB_CACHE = ROOT / "outputs/artifacts/collab_space_beauty_inner.npz"  # NOT the real-world cache
CKPT = Path(os.environ["P26_CKPT"])
ITEM_EMB = Path(os.environ["P26_ITEM_EMB"])
TAG = os.environ["P26_TAG"]
OUT_JSON = ROOT / "results/p26_inner_validation.json"

N_ITEMS = 12101
KS = (5, 10)
WEIGHTS = [float(w) for w in sys.argv[1:]] or [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BARS = {"real_world_P25c_openai_k4": {"selected_w": 0.8, "val_HR@10": 0.1109,
                                      "val_NDCG@5": 0.0364},
        "real_world_P21_val_HR@10": 0.0882,
        "J4a_gap_warning": {"val_HR@10": 0.0879, "test_HR@10": 0.0571}}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s):
    return (s - s.mean()) / max(float(s.std()), 1e-9)


def sweep_split(split, which, weights, q, item_t, content_w, collab_w, covered, t0):
    """Score every event of split.<which> at every w; returns {w: metrics}."""
    events = [(list(h), t) for h, t in getattr(split, which)]
    n = len(events)
    hits = {(w, k): np.zeros(n, dtype=np.float32) for w in weights for k in KS}
    dcgs = {(w, k): np.zeros(n, dtype=np.float32) for w in weights for k in KS}
    for qi, (hist, tgt) in enumerate(events):
        b03 = sp.blended_scores_whitened(content_w, collab_w, covered, hist,
                                         recency=sp.WHITEN_RECENCY)
        b01 = sp.blended_scores_whitened(content_w, collab_w, covered, hist, recency=0.1)
        z_free = zscore(0.9 * b03 + 0.1 * b01)
        s_tr = torch.einsum("kd,nd->kn", q[qi], item_t).max(dim=0).values.numpy()
        z_tr = zscore(s_tr)
        for w in weights:
            s = (1.0 - w) * z_free + w * z_tr
            rank = int((s > s[tgt]).sum())
            for k in KS:
                if rank < k:
                    hits[(w, k)][qi] = 1.0
                    dcgs[(w, k)][qi] = 1.0 / np.log2(rank + 2)
        if (qi + 1) % 5000 == 0:
            print(f"  [{TAG}] {which}: {qi+1}/{n} [{time.time()-t0:.0f}s]", flush=True)
    return {str(w): {f"HR@{k}": round(float(hits[(w, k)].mean()), 4) for k in KS} |
                    {f"NDCG@{k}": round(float(dcgs[(w, k)].mean()), 4) for k in KS}
            for w in weights}


def main():
    t0 = time.time()
    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)

    item = l2(torch.load(ITEM_EMB, map_location="cpu", weights_only=True).float().numpy())
    model = load_checkpoint(str(CKPT), item)
    item_t = torch.from_numpy(item)

    content = l2(torch.load(EMB_T5, map_location="cpu").float().numpy().astype(np.float32))
    train_seqs = [list(h) for h, _t in split.val]          # inner-train = orig s[:-3]
    collab, covered = sp.build_collab_embeddings(train_seqs, N_ITEMS,
                                                 cache_path=str(COLLAB_CACHE))
    content_w, collab_w = sp.whiten_blend_spaces(content, collab, covered)
    print(f"[{TAG}] inner spaces ready [{time.time()-t0:.0f}s]", flush=True)

    # 1. inner-val sweep (selection happens HERE and only here).
    # P26_SKIP_VAL=1 reuses a prior run's sweep for this TAG (e.g. to add pseudo-test
    # weights for an include-val tower whose inner-val is contaminated anyway).
    if os.environ.get("P26_SKIP_VAL") and OUT_JSON.exists() and \
            TAG in json.load(open(OUT_JSON)).get("towers", {}):
        prior = json.load(open(OUT_JSON))["towers"][TAG]
        val_sweep, best = prior["inner_val_sweep"], prior["selected_w"]
        print(f"[{TAG}] inner-val sweep reused from prior run (selected w={best})", flush=True)
    else:
        q_val = encode_split_queries(model, split, which="val", device="cpu")
        val_sweep = sweep_split(split, "val", WEIGHTS, q_val, item_t,
                                content_w, collab_w, covered, t0)
        for w in WEIGHTS:
            print(f"  [{TAG}] inner-val w={w}: {val_sweep[str(w)]}", flush=True)
        best = max(WEIGHTS, key=lambda w: (val_sweep[str(w)]["HR@10"],
                                           val_sweep[str(w)]["NDCG@5"]))
        print(f"[{TAG}] inner-val winner: w={best}  {val_sweep[str(best)]}", flush=True)

    # 2. pseudo-test at {0 = P21-mix ref, selected, 1 = tower-only} — mimics the one look.
    # P26_EXTRA_WS adds fixed weights (comma-separated) for matched-w comparisons.
    extra = [float(x) for x in os.environ.get("P26_EXTRA_WS", "").split(",") if x]
    look_ws = sorted({0.0, best, 1.0, *extra})
    q_test = encode_split_queries(model, split, which="test", device="cpu")
    test_scores = sweep_split(split, "test", look_ws, q_test, item_t,
                              content_w, collab_w, covered, t0)
    for w in look_ws:
        print(f"  [{TAG}] pseudo-test w={w}: {test_scores[str(w)]}", flush=True)

    # 3. transfer diagnostics
    vg = val_sweep[str(best)]["HR@10"] - val_sweep["0.0"]["HR@10"]
    tg = test_scores[str(best)]["HR@10"] - test_scores["0.0"]["HR@10"]
    transfer = round(tg / vg, 3) if vg > 0 else None
    print(f"[{TAG}] gain inner-val {vg:+.4f} -> pseudo-test {tg:+.4f}  "
          f"transfer={transfer}", flush=True)

    report = json.load(open(OUT_JSON)) if OUT_JSON.exists() else {
        "experiment": "P26_inner_validation",
        "protocol": "canonical LOO on Beauty_inner.txt (last event dropped per user): "
                    "inner-val = orig -3, pseudo-test = orig -2; REAL test never read",
        "config": "P25c blend (1-w)*z(P21 mix) + w*z(tower maxIP), per-arm per-query z-cal",
        "bars": BARS, "towers": {}}
    report["towers"][TAG] = {
        "checkpoint": str(CKPT), "item_emb": str(ITEM_EMB), "weights": WEIGHTS,
        "inner_val_sweep": val_sweep, "selected_w": best,
        "pseudo_test": test_scores,
        "gain_val_HR@10": round(vg, 4), "gain_pseudo_test_HR@10": round(tg, 4),
        "transfer_ratio": transfer,
        "runtime_sec": round(time.time() - t0, 1)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT_JSON, "w"), indent=2)
    print(f"wrote {OUT_JSON} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
