"""I29 — seed-disagreement as the adaptive-K uncertainty signal (inner, diagnostic).

Paper link: the program's original claim was "generative beam divergence predicts dense
failure -> plan variable ANN probes". The planner family died, but the I24 trio gives a
DENSE incarnation of the same signal: disagreement between independently-seeded query
towers. This script asks, on the inner world (no look, no selection):

  1. Does per-query seed disagreement predict ensemble failure (target outside top-10)?
     Metrics (all serve-computable):
       d_jac   = 1 - mean pairwise Jaccard of the seeds' top-10 item sets (post-score)
       d_qdisp = 1 - mean pairwise cosine of the seeds' mean query vectors (PRE-ANN:
                 available before any search - the true adaptive-K signal)
  2. Adaptive-K serving simulation: agreement-gated policy serves ONE seed (4 probes)
     when disagreement < t, the full trio (12 probes) otherwise. Sweep t over quantiles;
     report the HR@10-vs-mean-probes frontier vs the fixed 4- and 12-probe policies.

Output: results/i29_adaptive_k.json + i29_arrays.npz
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

EMB_T5 = ROOT / "data/amazon_beauty_backbone/Beauty_sentence-t5-xl_embeddings.pt"
SEQ = ROOT / "data/amazon_beauty_backbone/Beauty_inner.txt"
ITEM_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i22inner_embeddings.pt"
INPUT_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i24inner_input.pt"
SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/i24_inner_openai_k4_s{s}_iv/j5.pt" for s in SEEDS}
OUT_JSON = ROOT / "results/i29_adaptive_k.json"
OUT_NPZ = ROOT / "results/i29_arrays.npz"

N_ITEMS = 12101
SINGLE = 42          # the seed served in the cheap tier
TOPSET = 10


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s):
    return (s - s.mean()) / max(float(s.std()), 1e-9)


def auc(pos_scores, neg_scores):
    """Mann-Whitney AUC: P(score(pos) > score(neg))."""
    a = np.concatenate([pos_scores, neg_scores])
    r = a.argsort().argsort().astype(np.float64) + 1
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    return float((r[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    t0 = time.time()
    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)
    item = l2(torch.load(ITEM_EMB, map_location="cpu", weights_only=True).float().numpy())
    item_t = torch.from_numpy(item)
    inp = torch.load(INPUT_EMB, map_location="cpu", weights_only=True).float().numpy()
    qs = {}
    for s in SEEDS:
        model = load_checkpoint(str(CKPTS[s]), item, input_emb=inp)
        qs[s] = encode_split_queries(model, split, which="test", device="cpu")
    print(f"[i29] queries ready [{time.time()-t0:.0f}s]", flush=True)

    events = [(list(h), t) for h, t in split.test]
    n = len(events)
    d_jac = np.zeros(n, dtype=np.float32)
    d_qdisp = np.zeros(n, dtype=np.float32)
    ens_rank = np.zeros(n, dtype=np.int32)
    single_rank = np.zeros(n, dtype=np.int32)
    pairs = [(a, b) for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]]
    for qi, (hist, tgt) in enumerate(events):
        z_acc = np.zeros(N_ITEMS, dtype=np.float32)
        tops, cents = {}, {}
        for s in SEEDS:
            ip = torch.einsum("kd,nd->kn", qs[s][qi], item_t).numpy()
            mx = ip.max(axis=0)
            z = zscore(mx)
            z_acc += z
            tops[s] = set(np.argpartition(-mx, TOPSET)[:TOPSET].tolist())
            c = qs[s][qi].mean(dim=0).numpy()
            cents[s] = c / max(np.linalg.norm(c), 1e-9)
            if s == SINGLE:
                single_rank[qi] = int((mx > mx[tgt]).sum())
        ens_rank[qi] = int((z_acc > z_acc[tgt]).sum())
        d_jac[qi] = 1.0 - float(np.mean(
            [len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for a, b in pairs]))
        d_qdisp[qi] = 1.0 - float(np.mean(
            [cents[a] @ cents[b] for a, b in pairs]))
        if (qi + 1) % 5000 == 0:
            print(f"  [i29] {qi+1}/{n} [{time.time()-t0:.0f}s]", flush=True)

    fail = ens_rank >= 10
    rep = {"experiment": "I29_adaptive_k", "world": "inner", "trio": "i24_s42+43+44",
           "n": n, "ens_HR@10": round(float((~fail).mean()), 4),
           "single_HR@10": round(float((single_rank < 10).mean()), 4),
           "fail_rate": round(float(fail.mean()), 4)}

    # 1. failure prediction
    for name, d in (("d_jac", d_jac), ("d_qdisp", d_qdisp)):
        rep[f"auc_{name}"] = round(auc(d[fail], d[~fail]), 4)
        qcuts = np.quantile(d, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(d, qcuts)
        rep[f"fail_rate_by_{name}_quintile"] = [
            round(float(fail[bins == b].mean()), 4) for b in range(5)]

    # 2. adaptive-K simulation (both signals; d_qdisp is the PRE-ANN one)
    hit_full = (ens_rank < 10).astype(np.float32)
    hit_single = (single_rank < 10).astype(np.float32)
    for name, d in (("d_jac", d_jac), ("d_qdisp", d_qdisp)):
        frontier = []
        for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            t = float(np.quantile(d, q))
            cheap = d <= t
            hr = float(np.where(cheap, hit_single, hit_full).mean())
            probes = float(np.where(cheap, 4.0, 12.0).mean())
            frontier.append({"q": q, "thresh": round(t, 4), "HR@10": round(hr, 4),
                             "mean_probes": round(probes, 2)})
        rep[f"adaptive_frontier_{name}"] = frontier
    rep["fixed_policies"] = {"single_4probe_HR@10": rep["single_HR@10"],
                             "full_12probe_HR@10": rep["ens_HR@10"]}
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(rep, indent=2))
    np.savez_compressed(OUT_NPZ, d_jac=d_jac, d_qdisp=d_qdisp,
                        ens_rank=ens_rank, single_rank=single_rank)
    print(json.dumps(rep, indent=2), flush=True)
    print(f"wrote {OUT_JSON} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
