"""I35 — Sports recipe-transfer gate (Sports inner world only).

PREREG (ledger I35-GATE-PREREG; utc stamped there):
  PRIMARY criterion (single, numeric): the trio of Sports inner towers
  (i35_sports_inner_openai_k4_s{42,43,44}_iv; mean per-query z of K4 max-IP scores,
  blend w selected on inner val over the standard grid) must have pseudo-test HR@5
  STRICTLY GREATER than the training-free w=0 arm's pseudo-test HR@5 (same sweep) —
  i.e. the trained recipe transfers beyond the free whitened blend on a second dataset.
  Pass => build 3 REAL Sports towers (Sports_and_Outdoors.txt, i22real item side, i24real
  input) + prereg ONE real test look (published context bars: TIGER Sports HR@5 0.0189).
  Fail => book "recipe does not transfer to Sports"; artifacts only.
  Secondary (descriptive, NO veto): per-seed singles, trio union recall@500,
  free-arm absolute level vs Beauty's (0.1173 inner HR@10 at w=1.0 trio).

Data: Sports_inner.txt (last item dropped, 35,598 users / 18,357 items), i22inner openai
catalog scoring side, i24inner [own;pred3] input side, sentence-t5-xl free content arm.
Output: results/i35_gate.json
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
from semplan.eval import protocol  # noqa: E402

DATA = ROOT / "data/amazon_sports_backbone"
EMB_T5 = DATA / "Sports_sentence-t5-xl_embeddings.pt"
SEQ = DATA / "Sports_inner.txt"
COLLAB_CACHE = ROOT / "outputs/artifacts/collab_space_sports_inner.npz"
ITEM_EMB = DATA / "Sports_openai-3-large_i22inner_embeddings.pt"
INPUT_EMB = DATA / "Sports_openai-3-large_i24inner_input.pt"
SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/i35_sports_inner_openai_k4_s{s}_iv/j5.pt"
         for s in SEEDS}
OUT_JSON = ROOT / "results/i35_gate.json"

N_ITEMS = 18357
KS = (5, 10)
WEIGHTS = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
BARS = {"primary": "trio pseudo-test HR@5 at inner-val-selected w > w=0 free-arm "
                   "pseudo-test HR@5",
        "context_published_real_test": {"TIGER_Sports_HR@5": 0.0189},
        "beauty_reference_inner": {"trio_w1.0_HR@10": 0.1173}}


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
            print(f"  [i35] {which}: {qi+1}/{n} [{time.time()-t0:.0f}s]", flush=True)
    return {c: {str(w): {f"HR@{k}": round(float(hits[(c, w, k)].mean()), 4) for k in KS} |
                        {f"NDCG@{k}": round(float(dcgs[(c, w, k)].mean()), 4) for k in KS}
                for w in WEIGHTS} for c in combos}


def main():
    t0 = time.time()
    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)

    item = l2(torch.load(ITEM_EMB, map_location="cpu", weights_only=True).float().numpy())
    item_t = torch.from_numpy(item)
    inp = torch.load(INPUT_EMB, map_location="cpu", weights_only=True).float().numpy()
    models = {str(s): load_checkpoint(str(CKPTS[s]), item, input_emb=inp) for s in SEEDS}
    combos = {"sports_trio": [str(s) for s in SEEDS]} | {str(s): [str(s)] for s in SEEDS}

    content = l2(torch.load(EMB_T5, map_location="cpu").float().numpy().astype(np.float32))
    train_seqs = [list(h) for h, _t in split.val]
    collab, covered = sp.build_collab_embeddings(train_seqs, N_ITEMS,
                                                 cache_path=str(COLLAB_CACHE))
    content_w, collab_w = sp.whiten_blend_spaces(content, collab, covered)
    print(f"[i35] sports inner spaces ready [{time.time()-t0:.0f}s]", flush=True)

    q_val = {nm: encode_split_queries(m, split, which="val", device="cpu")
             for nm, m in models.items()}
    val_sweep = sweep_split(split, "val", q_val, combos, item_t, content_w, collab_w,
                            covered, t0)
    sel = {}
    for c, sweep in val_sweep.items():
        best = max(WEIGHTS, key=lambda w: (sweep[str(w)]["HR@10"], sweep[str(w)]["NDCG@5"]))
        sel[c] = best
        print(f"[i35] {c} inner-val winner w={best}: {sweep[str(best)]}", flush=True)

    probe_rank = {}
    q_test = {nm: encode_split_queries(m, split, which="test", device="cpu")
              for nm, m in models.items()}
    test_sweep = sweep_split(split, "test", q_test, combos, item_t, content_w, collab_w,
                             covered, t0, probe_rank_out=probe_rank)
    for c in combos:
        print(f"[i35] {c} pseudo-test w={sel[c]}: {test_sweep[c][str(sel[c])]}",
              flush=True)

    t_rank = np.minimum.reduce([probe_rank[str(s)] for s in SEEDS])
    union_recall_500 = round(float((t_rank < 500).mean()), 4)
    print(f"[i35] sports_trio union recall@500 = {union_recall_500}", flush=True)

    trio_hr5 = test_sweep["sports_trio"][str(sel["sports_trio"])]["HR@5"]
    free_hr5 = test_sweep["sports_trio"]["0.0"]["HR@5"]
    verdict = trio_hr5 > free_hr5
    json.dump({"experiment": "I35_sports_transfer_gate", "bars": BARS,
               "primary_pass": bool(verdict),
               "trio_pseudo_HR@5": trio_hr5, "free_arm_pseudo_HR@5": free_hr5,
               "union_recall_500_trio": union_recall_500,
               "weights": WEIGHTS, "selected_w": sel,
               "inner_val": val_sweep, "pseudo_test": test_sweep,
               "runtime_sec": round(time.time() - t0, 1)},
              open(OUT_JSON, "w"), indent=2)
    print(f"[i35] PRIMARY {'PASS' if verdict else 'FAIL'} (trio {trio_hr5} vs free "
          f"{free_hr5}); wrote {OUT_JSON} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
