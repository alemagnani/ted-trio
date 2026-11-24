"""CLOSE-2 Sports zero-look REPLAY of the spent I35 real-test look (prereg §3).

Deterministic replay of scripts/i35_real_look.py scoring path at
w=1.0 (frozen; no selection, no sweeps, zero new looks). Rebuilds the Sports
real split, loads the 3 frozen towers, scores the real TEST split, and
HARD-ASSERTS the trio aggregates reproduce the frozen i35_real_look.json row
EXACTLY (HR@5 0.0401 / HR@10 0.0622 / NDCG@5 0.0259 / NDCG@10 0.0330) before
writing results/i35_real_per_user.npz (same key convention as
i24_test_per_user.npz: cold_mask + w{0.0,1.0}_{hit,dcg}{5,10}, split.test
order). Also checks the routed secondary (HR@5 0.0405) since the free arm is
computed inline in the verbatim path. Does NOT touch ledger/RESULTS.md.
"""
from __future__ import annotations

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
from semplan.quick import is_quick  # noqa: E402

DATA = ROOT / "data/amazon_sports_backbone"
EMB_T5 = DATA / "Sports_sentence-t5-xl_embeddings.pt"
SEQ = DATA / "Sports_and_Outdoors.txt"
COLLAB_CACHE = ROOT / "outputs/artifacts/collab_space_sports_real.npz"
ITEM_EMB = DATA / "Sports_openai-3-large_i22real_embeddings.pt"
INPUT_EMB = DATA / "Sports_openai-3-large_i24real_input.pt"
SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/i35_sports_real_openai_k4_s{s}_iv/j5.pt"
         for s in SEEDS}
OUT_NPZ = ROOT / "results/i35_real_per_user.npz"

N_ITEMS = 18357
KS = (5, 10)
W = 1.0  # frozen; carried from inner world (i35_real_look.py)

# Frozen i35_real_look.json aggregates (rounded to 4 dp, as published).
FROZEN_TRIO = {"HR@5": 0.0401, "HR@10": 0.0622, "NDCG@5": 0.0259, "NDCG@10": 0.0330}
FROZEN_ROUTED_HR5 = 0.0405


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s):
    return (s - s.mean()) / max(float(s.std()), 1e-9)


def main():
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

    train_count = np.zeros(N_ITEMS, dtype=np.int64)
    for h in train_seqs:
        for it in h:
            train_count[it] += 1
    print(f"[replay] spaces ready; cold items in catalog: "
          f"{int((train_count == 0).sum())} [{time.time()-t0:.0f}s]", flush=True)

    q_test = {nm: encode_split_queries(m, split, which="test", device="cpu")
              for nm, m in models.items()}
    events = [(list(h), t) for h, t in split.test]
    n = len(events)
    arms = ("trio", "free", "routed")
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
        for a in arms:
            sc = scores[a]
            rank = int((sc > sc[tgt]).sum())
            for k in KS:
                if rank < k:
                    hits[(a, k)][qi] = 1.0
                    dcgs[(a, k)][qi] = 1.0 / np.log2(rank + 2)
        if (qi + 1) % 5000 == 0:
            print(f"  [replay] test: {qi+1}/{n} [{time.time()-t0:.0f}s]", flush=True)

    def rep(a):
        return {f"HR@{k}": round(float(hits[(a, k)].sum()) / n, 4) for k in KS} | \
               {f"NDCG@{k}": round(float(dcgs[(a, k)].sum()) / n, 4) for k in KS}

    trio = rep("trio")
    print(f"[replay] n={n} n_cold={int(cold_mask.sum())} trio={trio}", flush=True)

    # HARD ASSERT: frozen aggregates must reproduce EXACTLY, else abort (no npz).
    mism = {k: (trio[k], FROZEN_TRIO[k]) for k in FROZEN_TRIO if trio[k] != FROZEN_TRIO[k]}
    if mism:
        print(f"[replay] ASSERT FAIL — trio mismatch vs frozen i35_real_look.json: "
              f"{mism}; NOT writing {OUT_NPZ}", flush=True)
        if not is_quick():  # TT_QUICK=1 (user subset) continues so the chain can be tested
            sys.exit(1)
    else:
        print("[replay] ASSERT PASS — trio aggregates reproduce frozen i35_real_look.json "
              "exactly", flush=True)

    routed_hr5 = rep("routed")["HR@5"]
    print(f"[replay] routed secondary HR@5 {routed_hr5} "
          f"({'PASS' if routed_hr5 == FROZEN_ROUTED_HR5 else 'MISMATCH'} vs frozen "
          f"{FROZEN_ROUTED_HR5})", flush=True)

    np.savez(OUT_NPZ, cold_mask=cold_mask,
             **{f"w0.0_hit{k}": hits[("free", k)] for k in KS},
             **{f"w0.0_dcg{k}": dcgs[("free", k)] for k in KS},
             **{f"w1.0_hit{k}": hits[("trio", k)] for k in KS},
             **{f"w1.0_dcg{k}": dcgs[("trio", k)] for k in KS})
    print(f"[replay] wrote {OUT_NPZ} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
