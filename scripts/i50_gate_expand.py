"""I50/I51 — EXPAND recipe-transfer gates (inner world only), exact i35_gate.py
mirror generalized over --dataset {instruments,yelp}.

PREREG (ledger I50-GATE-PREREG / I51-GATE-PREREG, utc 2026-08-10T12:41:02Z):
  PRIMARY criterion (single, numeric): the trio of inner towers
  (i5x_{name}_inner_openai_k4_s{42,43,44}_iv; mean per-query z of K4 max-IP scores,
  blend w selected on inner val over the standard grid) must have pseudo-test HR@5
  STRICTLY GREATER than the training-free w=0 arm's pseudo-test HR@5 (same sweep).
  Pass => build 3 REAL towers + prereg ONE real test look vs published bars
  (Instruments: SASRec 0.0751 / TIGER repro 0.0863-0.0870; Yelp: BSARec 0.0275 /
  TIGER repro 0.0253). Fail => book clean non-transfer; artifacts only.
  Secondary (descriptive, NO veto): per-seed singles, trio union recall@500.

Data per dataset: {Prefix}_inner.txt, i22inner openai catalog scoring side,
i24inner [own;pred3] input side, sentence-t5-xl free content arm.
Output: results/i5x_gate_{name}.json
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
from semplan.eval import protocol  # noqa: E402

DATASETS = {
    "instruments": dict(
        data=ROOT / "data/amazon_instruments_backbone", prefix="Instruments",
        n=9922, tag="i50", trio="instruments_trio",
        bars={"SASRec_published_HR@5": 0.0751, "TIGER_repro_HR@5": "0.0863-0.0870"},
    ),
    "yelp": dict(
        data=ROOT / "data/yelp_backbone", prefix="Yelp",
        n=20033, tag="i51", trio="yelp_trio",
        bars={"BSARec_published_HR@5": 0.0275, "TIGER_repro_HR@5": 0.0253},
    ),
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
ap.add_argument("--ckpt-prefix", default=None,
                help="I47: tower dir prefix (seed appended, /tower.pt inside) "
                     "loaded via load_tricks_checkpoint; default = i50/i51 towers")
ap.add_argument("--out", default=None, help="output json path override")
_args = ap.parse_args()
NAME = _args.dataset
cfg = DATASETS[NAME]
DATA, P, TAG = cfg["data"], cfg["prefix"], cfg["tag"]
EMB_T5 = DATA / f"{P}_sentence-t5-xl_embeddings.pt"
SEQ = DATA / f"{P}_inner.txt"
COLLAB_CACHE = ROOT / f"outputs/artifacts/collab_space_{NAME}_inner.npz"
ITEM_EMB = DATA / f"{P}_openai-3-large_i22inner_embeddings.pt"
INPUT_EMB = DATA / f"{P}_openai-3-large_i24inner_input.pt"
SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/{TAG}_{NAME}_inner_openai_k4_s{s}_iv/j5.pt"
         for s in SEEDS}
TRICKS_LOADER = False
if _args.ckpt_prefix:                                       # I47 TED towers
    CKPTS = {s: ROOT / f"{_args.ckpt_prefix}{s}/tower.pt" for s in SEEDS}
    TRICKS_LOADER = True
OUT_JSON = ROOT / (_args.out if _args.out
                   else f"results/{TAG}_gate_{NAME}.json")

N_ITEMS = cfg["n"]
KS = (5, 10)
WEIGHTS = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
BARS = {"primary": "trio pseudo-test HR@5 at inner-val-selected w > w=0 free-arm "
                   "pseudo-test HR@5",
        "context_published_real_test": cfg["bars"],
        "sports_reference_inner": {"trio_pseudo_HR@5": 0.0439}}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s):
    return (s - s.mean()) / max(float(s.std()), 1e-9)


def sweep_split(split, which, qs, combos, item_t, content_w, collab_w, covered, t0,
                probe_rank_out=None):
    events = [(list(h), t) for h, t in getattr(split, which)]
    n = len(events)
    names = list(qs)
    hits = {(c, w, k): np.zeros(n, dtype=np.float32)
            for c in combos for w in WEIGHTS for k in KS}
    dcgs = {(c, w, k): np.zeros(n, dtype=np.float32)
            for c in combos for w in WEIGHTS for k in KS}
    for qi, (hist, tgt) in enumerate(events):
        b03 = sp.blended_scores_whitened(content_w, collab_w, covered, hist,
                                         recency=sp.WHITEN_RECENCY)
        b01 = sp.blended_scores_whitened(content_w, collab_w, covered, hist, recency=0.1)
        z_free = zscore(0.9 * b03 + 0.1 * b01)
        z_tow = {}
        for nm in names:
            ip = torch.einsum("kd,nd->kn", qs[nm][qi], item_t).numpy()
            z_tow[nm] = zscore(ip.max(axis=0))
            if probe_rank_out is not None:
                best = int((ip > ip[:, tgt:tgt + 1]).sum(axis=1).min())
                probe_rank_out.setdefault(nm, np.full(n, N_ITEMS, np.int64))
                probe_rank_out[nm][qi] = best
        for c, members in combos.items():
            z_ens = np.mean([z_tow[m] for m in members], axis=0)
            for w in WEIGHTS:
                sc = (1.0 - w) * z_free + w * z_ens
                rank = int((sc > sc[tgt]).sum())
                for k in KS:
                    if rank < k:
                        hits[(c, w, k)][qi] = 1.0
                        dcgs[(c, w, k)][qi] = 1.0 / np.log2(rank + 2)
        if (qi + 1) % 5000 == 0:
            print(f"  [{TAG}] {which}: {qi+1}/{n} [{time.time()-t0:.0f}s]", flush=True)
    return {c: {str(w): {f"HR@{k}": round(float(hits[(c, w, k)].mean()), 4) for k in KS} |
                        {f"NDCG@{k}": round(float(dcgs[(c, w, k)].mean()), 4) for k in KS}
                for w in WEIGHTS} for c in combos}


def main():
    t0 = time.time()
    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)

    item = l2(torch.load(ITEM_EMB, map_location="cpu", weights_only=True).float().numpy())
    item_t = torch.from_numpy(item)
    inp = torch.load(INPUT_EMB, map_location="cpu", weights_only=True).float().numpy()
    if TRICKS_LOADER:
        from semplan.dense.tricks_query import load_tricks_checkpoint
        models = {str(s): load_tricks_checkpoint(str(CKPTS[s]), item, input_emb=inp)
                  for s in SEEDS}
    else:
        models = {str(s): load_checkpoint(str(CKPTS[s]), item, input_emb=inp)
                  for s in SEEDS}
    combos = {cfg["trio"]: [str(s) for s in SEEDS]} | {str(s): [str(s)] for s in SEEDS}

    content = l2(torch.load(EMB_T5, map_location="cpu").float().numpy().astype(np.float32))
    train_seqs = [list(h) for h, _t in split.val]
    collab, covered = sp.build_collab_embeddings(train_seqs, N_ITEMS,
                                                 cache_path=str(COLLAB_CACHE))
    content_w, collab_w = sp.whiten_blend_spaces(content, collab, covered)
    print(f"[{TAG}] {NAME} inner spaces ready [{time.time()-t0:.0f}s]", flush=True)

    q_val = {nm: encode_split_queries(m, split, which="val", device="cpu")
             for nm, m in models.items()}
    val_sweep = sweep_split(split, "val", q_val, combos, item_t, content_w, collab_w,
                            covered, t0)
    sel = {}
    for c, sweep in val_sweep.items():
        best = max(WEIGHTS, key=lambda w: (sweep[str(w)]["HR@10"], sweep[str(w)]["NDCG@5"]))
        sel[c] = best
        print(f"[{TAG}] {c} inner-val winner w={best}: {sweep[str(best)]}", flush=True)

    probe_rank = {}
    q_test = {nm: encode_split_queries(m, split, which="test", device="cpu")
              for nm, m in models.items()}
    test_sweep = sweep_split(split, "test", q_test, combos, item_t, content_w, collab_w,
                             covered, t0, probe_rank_out=probe_rank)
    for c in combos:
        print(f"[{TAG}] {c} pseudo-test w={sel[c]}: {test_sweep[c][str(sel[c])]}",
              flush=True)

    t_rank = np.minimum.reduce([probe_rank[str(s)] for s in SEEDS])
    union_recall_500 = round(float((t_rank < 500).mean()), 4)
    print(f"[{TAG}] {cfg['trio']} union recall@500 = {union_recall_500}", flush=True)

    trio_hr5 = test_sweep[cfg["trio"]][str(sel[cfg["trio"]])]["HR@5"]
    free_hr5 = test_sweep[cfg["trio"]]["0.0"]["HR@5"]
    verdict = trio_hr5 > free_hr5
    json.dump({"experiment": f"{TAG.upper()}_{NAME}_transfer_gate", "bars": BARS,
               "primary_pass": bool(verdict),
               "trio_pseudo_HR@5": trio_hr5, "free_arm_pseudo_HR@5": free_hr5,
               "union_recall_500_trio": union_recall_500,
               "weights": WEIGHTS, "selected_w": sel,
               "inner_val": val_sweep, "pseudo_test": test_sweep,
               "runtime_sec": round(time.time() - t0, 1)},
              open(OUT_JSON, "w"), indent=2)
    print(f"[{TAG}] PRIMARY {'PASS' if verdict else 'FAIL'} (trio {trio_hr5} vs free "
          f"{free_hr5}); wrote {OUT_JSON} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
