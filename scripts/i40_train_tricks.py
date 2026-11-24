"""I40 — query-tower training-trick arms (BEAT-DENSE plan, briefs 1/3/4 + synthesis).

Four independently-flaggable, composable upgrades — ALL OFF BY DEFAULT, none
changes the serving form (single query vector per head, dot product vs the frozen
catalog):
  T1 --nce_lambda / --nce_tau / --nce_mode {full,scl}   DuoRec contrastive reg
  T2 --ted_beta / --ted_tau                             TED soft-label smoothing
  T3 --longq_L (0=off, recommend 3)                     MQSA long-query branch
  T4 --freq_alpha / --freq_c (alpha=0 off)              BSARec FreqLayer blend
I42 additions (same all-off contract):
  T-GATE  --head_gate {none,softmax} / --head_gate_tau   gated-SUM head aggregation
          (served query = ONE vector, 1 ANN probe; none = max-over-K exactly)
  T-MULTI --multi_target_k (0=off) / --multi_weight      Hungarian next-k per-head
          supervision added to the pooled CE (serving unchanged; composes w/ gate)
plus --input_dropout (SOTA recipes use 0.5; default 0.0 = current behaviour) and
--cumulative_ce (per-position causal supervision; AUDIT: the baseline supervises
one pooled target per encoded sequence — see semplan.dense.tricks_query docstring).

Architecture otherwise IDENTICAL to the winning J5a/I24 recipe (d=256, 2L, 4H,
d_ff=512, tau=0.05, K=4, include-val, [own emb ; pred3 centroid] inputs,
full-softmax CE + forbidden masking). All flags off => the baseline tower bitwise.

Usage (box, one job per arm — co-locate on a busy GPU, ~3 GB VRAM):
  python scripts/i40_train_tricks.py --dataset toys --seed 42 \
      --nce_lambda 0.2 --out-dir outputs/artifacts/i40_toys_nce_s42
CPU smoke (synthetic fixture, ~40 steps, any flag combo):
  python scripts/i40_train_tricks.py --smoke --seed 42 --longq_L 3
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

from semplan.dense.chn_de import ChnDeDataset  # noqa: E402
from semplan.dense.frozen_query import save_checkpoint  # noqa: E402
from semplan.dense.tricks_query import (  # noqa: E402
    TricksQueryTower, build_future_targets, build_ted_teachers,
    load_tricks_checkpoint, mine_forward_adjacency, train_tricks,
)
from semplan.eval import protocol  # noqa: E402

from i36_train_joint import DATASETS, l2, make_smoke_fixture, split_metrics  # noqa: E402

# I24/J5a recipe constants (unchanged)
D, LAYERS, HEADS, D_FF, TAU, N_QUERY, MAX_LEN = 256, 2, 4, 512, 0.05, 4, 20
SMOKE_ROOT = ROOT / "outputs/tmp/i40_smoke"


def main() -> None:
    ap = argparse.ArgumentParser(description="I40 query-tower training tricks")
    ap.add_argument("--dataset", default="toys", choices=sorted(DATASETS),
                    help="BEAT-DENSE primary target is toys")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", default=None,
                    help="default outputs/artifacts/i40_<dataset>_s<seed>")
    # T1 DuoRec-style contrastive regularizer
    ap.add_argument("--nce_lambda", type=float, default=0.0,
                    help="T1 weight; 0=off (plan recommends 0.2, sweep 0.1-0.5)")
    ap.add_argument("--nce_tau", type=float, default=1.0,
                    help="T1 InfoNCE temperature (separate from CE tau)")
    ap.add_argument("--nce_mode", default="full", choices=("full", "scl"),
                    help="full: dropout view vs same-target peer; scl: peer only")
    # T2 TED soft-label smoothing
    ap.add_argument("--ted_beta", type=float, default=0.0,
                    help="T2 teacher mixing weight; 0=off (recommended 0.1)")
    ap.add_argument("--ted_tau", type=float, default=0.1,
                    help="T2 teacher softmax temperature (MQSA-TED best 0.05-0.1)")
    # T3 MQSA long-query branch
    ap.add_argument("--longq_L", type=int, default=0,
                    help="T3 running-mean window; 0=off (recommended 3)")
    # T4 BSARec FreqLayer
    ap.add_argument("--freq_alpha", type=float, default=0.0,
                    help="T4 freq/MHA blend; 0=off (BSARec best 0.3-0.7)")
    ap.add_argument("--freq_c", type=int, default=5,
                    help="T4 low-pass cutoff c (BSARec best 3-5)")
    # I42 T-GATE gated head aggregation
    ap.add_argument("--head_gate", default="none", choices=("none", "softmax"),
                    help="I42 T-GATE: softmax-gated SUM of the K head vectors "
                         "(single served vector, 1 ANN probe); none=max-over-K")
    ap.add_argument("--head_gate_tau", type=float, default=1.0,
                    help="T-GATE gate softmax temperature (plan sweeps ~0.1-1.0)")
    # I42 T-MULTI Hungarian multi-target head supervision
    ap.add_argument("--multi_target_k", type=int, default=0,
                    help="I42 T-MULTI: match the K heads to the next-k future "
                         "items (Hungarian) and add the matched per-head CE; 0=off")
    ap.add_argument("--multi_weight", type=float, default=1.0,
                    help="weight of the T-MULTI matched CE term")
    # audit levers
    ap.add_argument("--input_dropout", type=float, default=0.0,
                    help="extra dropout on projected input tokens (SOTA: 0.5); "
                         "default 0.0 = current behaviour")
    ap.add_argument("--cumulative_ce", action="store_true",
                    help="per-position causal CE (each position predicts its next "
                         "item) added to the pooled CE")
    ap.add_argument("--cum_weight", type=float, default=1.0,
                    help="weight of the cumulative-CE term when --cumulative_ce")
    # capacity levers (defaults == I24 recipe constants; only vary WITH a
    # winning trick — I18 showed raw size alone does not gate through)
    ap.add_argument("--d", type=int, default=D, help="model width (I24: 256)")
    ap.add_argument("--layers", type=int, default=LAYERS,
                    help="transformer layers (I24: 2)")
    ap.add_argument("--heads", type=int, default=HEADS,
                    help="attention heads (I24: 4)")
    ap.add_argument("--d_ff", type=int, default=D_FF, help="FFN width (I24: 512)")
    ap.add_argument("--max_len", type=int, default=MAX_LEN,
                    help="history length (I24: 20; MQSA-TED/BSARec train at 50)")
    # explicit world overrides (real-world runs; default = inner world from
    # DATASETS[--dataset]) — all four must be given together
    ap.add_argument("--seq", default=None,
                    help="override seq txt (e.g. the REAL Toys_and_Games.txt)")
    ap.add_argument("--item-emb", default=None, help="override item-side matrix")
    ap.add_argument("--input-emb", default=None, help="override input-side matrix")
    ap.add_argument("--n-items", type=int, default=None, help="override n_items")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU plumbing test on a synthetic fixture (under "
                         f"{SMOKE_ROOT}); ~40 steps, asserts loss decreases and "
                         "the checkpoint round-trips")
    args = ap.parse_args()
    t0 = time.time()

    if args.smoke:
        cfg = make_smoke_fixture(SMOKE_ROOT)["fake0"]
        args.steps, args.batch, args.device = min(args.steps, 40), 16, "cpu"
        out_dir = Path(args.out_dir) if args.out_dir else SMOKE_ROOT / f"s{args.seed}"
        name = "fake0"
        print(f"[i40] SMOKE — fixture at {SMOKE_ROOT / 'fixture'}", flush=True)
    else:
        name, cfg = args.dataset, DATASETS[args.dataset]
        overrides = (args.seq, args.item_emb, args.input_emb, args.n_items)
        if any(v is not None for v in overrides):
            assert all(v is not None for v in overrides), \
                "--seq/--item-emb/--input-emb/--n-items must be given together"
            cfg = {"seq": args.seq, "item_emb": args.item_emb,
                   "input_emb": args.input_emb, "n_items": args.n_items}
            name = f"{args.dataset}(world-override)"
        out_dir = Path(args.out_dir) if args.out_dir else \
            ROOT / f"outputs/artifacts/i40_{args.dataset}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    item_np = l2(torch.load(cfg["item_emb"], map_location="cpu",
                            weights_only=True).float().numpy())
    input_np = torch.load(cfg["input_emb"], map_location="cpu",
                          weights_only=True).float().numpy()
    assert item_np.shape[0] == cfg["n_items"], item_np.shape
    split = protocol.build(str(cfg["seq"]), n_items=cfg["n_items"],
                           max_len=args.max_len)
    ds = ChnDeDataset(split, max_len=args.max_len, include_val=True)  # I24 recipe
    print(f"[i40] {name}: n_items={cfg['n_items']} train_pairs={len(ds)} "
          f"D={item_np.shape[1]} D_in={input_np.shape[1]}", flush=True)

    ted_teachers = None
    if args.ted_beta > 0:                                          # T2 (sparse)
        adj = mine_forward_adjacency(str(cfg["seq"]), cfg["n_items"])
        ted_teachers = build_ted_teachers(adj, cfg["n_items"], ted_tau=args.ted_tau)
        n_rows = sum(1 for c, _p in ted_teachers if len(c))
        print(f"[i40] TED adjacency: {n_rows}/{cfg['n_items']} items with support",
              flush=True)

    multi_targets = None
    if args.multi_target_k > 0:                                    # I42 T-MULTI
        multi_targets = build_future_targets(ds, args.multi_target_k)
        cov = float((multi_targets >= 0).sum(1).mean())
        print(f"[i40] T-MULTI future windows: k={args.multi_target_k} "
              f"mean-valid-targets {cov:.2f}", flush=True)

    model = TricksQueryTower(
        item_np, d=args.d, n_layers=args.layers, n_heads=args.heads,
        d_ff=args.d_ff, max_len=args.max_len,
        tau=TAU, seed=args.seed, n_query_vectors=N_QUERY, input_emb=input_np,
        longq_L=args.longq_L, freq_alpha=args.freq_alpha, freq_c=args.freq_c,
        input_dropout=args.input_dropout, cumulative_ce=args.cumulative_ce,
        head_gate=args.head_gate, head_gate_tau=args.head_gate_tau,
        multi_target_k=args.multi_target_k)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tricks = {"nce_lambda": args.nce_lambda, "nce_tau": args.nce_tau,
              "nce_mode": args.nce_mode, "ted_beta": args.ted_beta,
              "ted_tau": args.ted_tau, "longq_L": args.longq_L,
              "freq_alpha": args.freq_alpha, "freq_c": args.freq_c,
              "input_dropout": args.input_dropout,
              "cumulative_ce": args.cumulative_ce, "cum_weight": args.cum_weight,
              "head_gate": args.head_gate, "head_gate_tau": args.head_gate_tau,
              "multi_target_k": args.multi_target_k,
              "multi_weight": args.multi_weight}
    print(f"[i40] tricks: {tricks} params {n_params:,}", flush=True)

    ckpt_path = out_dir / "tower.pt"
    hist = train_tricks(model, ds, steps=args.steps, batch_size=args.batch,
                        lr=args.lr, device=args.device, seed=args.seed,
                        ckpt_path=str(ckpt_path), ckpt_every=1000,
                        log_every=50 if args.smoke else 150,
                        nce_lambda=args.nce_lambda, nce_tau=args.nce_tau,
                        nce_mode=args.nce_mode, ted_teachers=ted_teachers,
                        ted_beta=args.ted_beta, cum_weight=args.cum_weight,
                        multi_targets=multi_targets,
                        multi_weight=args.multi_weight)
    save_checkpoint(model, str(ckpt_path), step=args.steps)

    # round-trip check: reloaded tower must reproduce the trained queries
    m2 = load_tricks_checkpoint(str(ckpt_path), item_np, input_emb=input_np)
    model.cpu().eval()
    m2.eval()
    with torch.no_grad():
        from semplan.dense.frozen_query import collate_with_forbidden
        b = collate_with_forbidden(ds, np.arange(min(8, len(ds))), args.max_len)
        q1 = model.encode_queries(b["hist_rows"], b["hist_mask"])
        q2 = m2.encode_queries(b["hist_rows"], b["hist_mask"])
    assert torch.allclose(q1, q2, atol=1e-5), "checkpoint round-trip mismatch"
    print("[i40] checkpoint round-trip OK", flush=True)

    # inner-val metrics (include-val => CONTAMINATED; selection uses pseudo-test gate)
    val_m = split_metrics(m2, split, "val", device="cpu")
    print(f"[i40] inner-val (contaminated): {val_m}", flush=True)

    first, last = float(np.mean(hist["loss"][:5])), float(np.mean(hist["loss"][-5:]))
    result = {"job": "I40_tricks", "dataset": name, "seed": args.seed,
              "steps": args.steps, "batch": args.batch, "lr": args.lr,
              "smoke": args.smoke, "tricks": tricks,
              "arch": {"d": args.d, "layers": args.layers, "heads": args.heads,
                       "d_ff": args.d_ff, "tau": TAU, "n_query": N_QUERY,
                       "max_len": args.max_len, "include_val": True},
              "params_trainable": n_params,
              "loss_first5": round(first, 4), "loss_last5": round(last, 4),
              "uniformity": hist["uniformity"],
              "val_CONTAMINATED": val_m, "ckpt": str(ckpt_path),
              "wall_min": round((time.time() - t0) / 60, 1)}
    (out_dir / "i40_metrics.json").write_text(json.dumps(result, indent=2))
    (out_dir / "i40_config.json").write_text(json.dumps(
        {"args": vars(args) | {"out_dir": str(out_dir)}, "tricks": tricks,
         "model_config": model.config}, indent=2, default=str))
    print(f"[i40] done in {result['wall_min']} min -> {out_dir}", flush=True)
    if args.smoke:
        assert last < first, f"SMOKE FAIL: loss did not decrease ({first} -> {last})"
        print(f"[i40] SMOKE PASS: loss {first:.4f} -> {last:.4f}; "
              "checkpoint round-trips", flush=True)


if __name__ == "__main__":
    main()
