"""J5a/J5b entry — frozen-item query tower (I16/I17), CPU-trainable.

Item side is a FROZEN external embedding matrix (--item-emb; openemb-large stand-in now,
commercial text-embedding-3-large later — same slot, zero code change). Only the query
tower trains (full-softmax CE over 12k items + forbidden-column masking + optional logQ).
--n-query K>1 is the J5b multi-query arm (max-IP loss and serving; I10 semantics).

Selection policy (P19 lesson, codex REV-3): VAL-ONLY. Test is scored ONLY when
J5_TEST_LOOK=1 (or SMOKE=1). Arm sweeps run without it; exactly one val-selected config
takes the look. Per-user hit/dcg vectors are saved for significance on every scored split.

Bars: P21 single-pass deployable frontier (val HR@10 0.0882 / NDCG@5 0.0272;
test 0.0508/0.0821/0.0267/0.0368). TIGER HR@5 0.0454.

Usage:
  python scripts/j5/train_j5a.py --steps 6000 --out outputs/artifacts/j5a
  python scripts/j5/train_j5a.py --n-query 4 --out outputs/artifacts/j5b_k4   # J5b arm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semplan.dense.chn_de import ChnDeDataset, train_popularity  # noqa: E402
from semplan.dense.frozen_query import (  # noqa: E402
    FrozenQueryTower, encode_split_queries, train_frozen_query,
)
from semplan.eval import protocol  # noqa: E402

SEQ = ROOT / "data/amazon_beauty_backbone/Beauty.txt"
DEFAULT_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openemb-large_embeddings.pt"
N_ITEMS = 12101
KS = (5, 10)
BARS = {"P21_single_pass_val": {"HR@10": 0.0882, "NDCG@5": 0.0272},
        "P21_single_pass_test": {"HR@5": 0.0508, "HR@10": 0.0821,
                                 "NDCG@5": 0.0267, "NDCG@10": 0.0368},
        "TIGER_HR@5": 0.0454}


@torch.no_grad()
def per_user_metrics(model: FrozenQueryTower, split, which: str, device: str = "cpu",
                     chunk: int = 512) -> tuple[dict, dict]:
    """Full-corpus max-IP rank of each target -> per-user hit/dcg vectors + summary."""
    q = encode_split_queries(model, split, which=which, device=device)   # [n, K, D]
    data = getattr(split, which)
    tgts = torch.tensor([t for _h, t in data], dtype=torch.long)
    n = len(data)
    pu = {f"{which}_{m}{k}": np.zeros(n, dtype=np.float32)
          for k in KS for m in ("hit", "dcg")}
    item = model.item.cpu()
    for s in range(0, n, chunk):
        sc = torch.einsum("bkd,nd->bkn", q[s: s + chunk], item).max(dim=1).values
        tgt_sc = sc.gather(1, tgts[s: s + chunk].unsqueeze(1))
        rank = (sc > tgt_sc).sum(dim=1).numpy()
        for k in KS:
            hit = rank < k
            pu[f"{which}_hit{k}"][s: s + chunk] = hit
            pu[f"{which}_dcg{k}"][s: s + chunk] = np.where(
                hit, 1.0 / np.log2(rank + 2), 0.0)
    rep = {}
    for k in KS:
        rep[f"HR@{k}"] = round(float(pu[f"{which}_hit{k}"].mean()), 4)
        rep[f"NDCG@{k}"] = round(float(pu[f"{which}_dcg{k}"].mean()), 4)
    cold = np.zeros(n, dtype=bool)
    cold[list(getattr(split, f"cold_{which}_idx", set()) or set())] = True
    if cold.any():
        rep["cold"] = {f"HR@{k}": round(float(pu[f"{which}_hit{k}"][cold].mean()), 4)
                       for k in KS}
    return rep, pu


def main() -> None:
    ap = argparse.ArgumentParser(description="J5a/J5b frozen-item query tower")
    ap.add_argument("--item-emb", default=str(DEFAULT_EMB),
                    help=".pt [n_items, D] frozen item matrix (row = item_id - 1)")
    ap.add_argument("--seq", default=str(SEQ),
                    help="LIGER-style sequence file (P26 inner-validation passes a "
                         "truncated corpus here; canonical Beauty.txt otherwise)")
    ap.add_argument("--input-emb", default=None,
                    help="I24: optional SEPARATE frozen .pt matrix for query-tower "
                         "input tokens (e.g. [own emb ; pred3 centroid]); scoring "
                         "item side stays --item-emb")
    ap.add_argument("--item-name", default="open-embedder-large")
    ap.add_argument("--n-items", type=int, default=N_ITEMS,
                    help="I30: corpus size (12101 Beauty, 11924 Toys)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--n-query", type=int, default=1, help="K query vectors (J5b: 2/4/8)")
    ap.add_argument("--boost-weights", default=None,
                    help="I27: .npy [len(dataset)] per-pair sampling weights (built by "
                         "scripts/j5/build_boost_weights.py against a frozen reference "
                         "ensemble); batches are drawn ∝ weight, loss unchanged")
    ap.add_argument("--band-negs", default=None,
                    help="I31: .npy [len(dataset), 50] reference top-50 wrong items per "
                         "pair (scripts/j5/build_band_negs.py); enables the fixed "
                         "rank-band margin curriculum (40%% CE / 30%% band 20-50 / "
                         "30%% band 5-20, re-mined from the current model at 70%%)")
    ap.add_argument("--distill", default=None,
                    help="I32: .npz {cand [len(ds),C], scores [len(ds),C]} teacher-trio "
                         "candidates + mean-z scores (build_distill_targets.py); trains "
                         "with (1-w)*CE + w*KL at --distill-T")
    ap.add_argument("--distill-T", type=float, default=0.07)
    ap.add_argument("--distill-weight", type=float, default=0.5)
    ap.add_argument("--logq", action="store_true",
                    help="subtract log train-frequency from training logits "
                         "(learned popularity absorption; serving stays raw max-IP)")
    ap.add_argument("--include-val", action="store_true",
                    help="J7/I21: add the (s[:-2] -> s[-2]) val transition to training. "
                         "VAL METRICS BECOME CONTAMINATED — select via the inner world.")
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "outputs/artifacts/j5a"))
    args = ap.parse_args()

    smoke = os.environ.get("SMOKE") == "1"
    test_look = smoke or os.environ.get("J5_TEST_LOOK") == "1"
    if smoke:
        args.steps = min(args.steps, 30)
        args.batch = min(args.batch, 32)
        print("[j5] SMOKE=1 — tiny run", flush=True)

    t0 = time.time()
    item = torch.load(args.item_emb, map_location="cpu", weights_only=True
                      ).float().numpy()
    assert item.shape[0] == args.n_items, item.shape
    split = protocol.build(args.seq, n_items=args.n_items, max_len=args.max_len)
    ds = ChnDeDataset(split, max_len=args.max_len, include_val=args.include_val)
    if args.include_val:
        print("[j5] --include-val: val transitions IN TRAINING -> val metrics "
              "CONTAMINATED; select via inner world only", flush=True)
    if smoke:
        ds.examples = ds.examples[:2000]
        ds.forbidden = ds.forbidden[:2000]
        split.val = split.val[:500]
        split.test = split.test[:500]
        split.cold_val_idx = {i for i in getattr(split, "cold_val_idx", set()) if i < 500}
        split.cold_test_idx = {i for i in split.cold_test_idx if i < 500}
    print(f"[j5] item side {args.item_name} D={item.shape[1]}  train_pairs={len(ds)}  "
          f"K={args.n_query}  logq={args.logq}", flush=True)

    logq = None
    if args.logq:
        freq = train_popularity(split).astype(np.float64)
        logq = np.log((freq + 1.0) / (freq.sum() + args.n_items)).astype(np.float32)

    input_emb = None
    if args.input_emb:
        input_emb = torch.load(args.input_emb, map_location="cpu", weights_only=True
                               ).float().numpy()
        assert input_emb.shape[0] == args.n_items, input_emb.shape
        print(f"[j5] I24 input side: {args.input_emb} D_in={input_emb.shape[1]}",
              flush=True)
    model = FrozenQueryTower(item, d=args.d, n_layers=args.layers, n_heads=args.heads,
                             d_ff=args.d_ff, max_len=args.max_len, tau=args.tau,
                             seed=args.seed, n_query_vectors=args.n_query,
                             input_emb=input_emb)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_p = None
    if args.boost_weights:
        sample_p = np.load(args.boost_weights)
        if smoke:
            sample_p = sample_p[:len(ds)]
        assert len(sample_p) == len(ds), (len(sample_p), len(ds))
        share = float((sample_p > sample_p.min()).mean())
        print(f"[j5] I27 boost weights: {args.boost_weights} "
              f"(upweighted share {share:.3f})", flush=True)
    band_negs = None
    if args.band_negs:
        band_negs = np.load(args.band_negs)
        if smoke:
            band_negs = band_negs[:len(ds)]
        assert len(band_negs) == len(ds), (len(band_negs), len(ds))
        print(f"[j5] I31 band negs: {args.band_negs} shape {band_negs.shape}",
              flush=True)
    distill = None
    if args.distill:
        dz = np.load(args.distill)
        dc, dsc = dz["cand"], dz["scores"]
        if smoke:
            dc, dsc = dc[:len(ds)], dsc[:len(ds)]
        assert len(dc) == len(ds), (len(dc), len(ds))
        distill = (dc, dsc)
        print(f"[j5] I32 distill: {args.distill} cand{dc.shape} "
              f"T={args.distill_T} w={args.distill_weight}", flush=True)
    losses = train_frozen_query(model, ds, steps=args.steps, batch_size=args.batch,
                                lr=args.lr, logq=logq, device=args.device,
                                seed=args.seed, ckpt_path=str(out_dir / "j5.pt"),
                                sample_p=sample_p, band_negs=band_negs,
                                distill=distill, distill_T=args.distill_T,
                                distill_weight=args.distill_weight)

    model.cpu().eval()
    metrics, pu_all = {}, {}
    for which in ("val",) + (("test",) if test_look else ()):
        rep, pu = per_user_metrics(model, split, which)
        metrics[which] = rep
        pu_all |= pu
        print(f"[j5] {which}: {json.dumps(rep)}", flush=True)
    if not test_look:
        print("[j5] test SKIPPED (val-only selection; set J5_TEST_LOOK=1 for the one "
              "selected config)", flush=True)

    result = {"job": "J5a" if args.n_query == 1 else f"J5b_K{args.n_query}",
              "item_side": args.item_name, "item_emb": args.item_emb, "seq": args.seq,
              "steps": args.steps, "batch": args.batch, "lr": args.lr, "d": args.d,
              "layers": args.layers, "heads": args.heads, "d_ff": args.d_ff,
              "tau": args.tau, "n_query": args.n_query, "logq": args.logq,
              "include_val": args.include_val, "input_emb": args.input_emb,
              "boost_weights": args.boost_weights, "band_negs": args.band_negs,
              "distill": args.distill, "distill_T": args.distill_T,
              "distill_weight": args.distill_weight,
              "seed": args.seed, "smoke": smoke, "test_look": test_look,
              "final_loss": losses[-1] if losses else None, "bars": BARS,
              "metrics": metrics, "wall_min": round((time.time() - t0) / 60, 1)}
    (out_dir / "j5_metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out_dir / "j5_per_user.npz", **pu_all)
    print(f"[j5] done in {result['wall_min']} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
