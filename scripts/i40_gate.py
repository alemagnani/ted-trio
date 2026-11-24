"""I40 gate — CLEAN inner-world evaluator for the trick-arm checkpoints (CPU).

The i40_train_tricks.py runner only reports an include-val CONTAMINATED inner-val
number. This gate produces the proper inner-world protocol numbers, with exactly
the i24/i41 conventions:
  * split = protocol.build over the *_inner.txt world (targets are already one
    step inside the real world), inner-val = split.val (real-world -3 targets),
    pseudo-test = split.test (real-world -2 targets);
  * FULL-corpus max-IP ranking per checkpoint (K query heads, max-merge);
  * metrics HR@5/10 + NDCG@5/10 flow through semplan.eval.protocol.evaluate
    (via i41_common.eval_ranked_lists — nothing here computes HR/NDCG on its own);
  * with >1 checkpoint, the i23/i41 mean-of-per-query-z ensemble is also scored.

Checkpoint handling: TricksQueryTower checkpoints (i40 arms — tower.pt) load via
load_tricks_checkpoint, which rebuilds the tower from its STORED config, so any
longq/freq modules and the causal-mask/cumulative-ce eval behaviour match training
exactly. Plain FrozenQueryTower checkpoints (i24/i30/i35 towers — j5.pt) fall back
to load_checkpoint, so the same script scores the baselines side-by-side.

Usage (from the repo root), Toys inner, one arm:
  python scripts/i40_gate.py --dataset toys \
    --seq data/amazon_toys_backbone/Toys_inner.txt --n-items 11924 \
    --item-emb data/amazon_toys_backbone/Toys_openai-3-large_i22inner_embeddings.pt \
    --input-emb data/amazon_toys_backbone/Toys_openai-3-large_i24inner_input.pt \
    --ckpt outputs/artifacts/i40_toys_nce_s42/tower.pt \
    --out results/i40_gate_toys_nce.json
CPU smoke: python scripts/i40_gate.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from i41_common import (ROOT, eval_ranked_lists, l2, load_matrix,
                        make_smoke_fixture, topk_ids, zscore)
sys.path.insert(0, str(ROOT / "src"))
from semplan.dense.frozen_query import (  # noqa: E402
    encode_split_queries, load_checkpoint, save_checkpoint,
)
from semplan.dense.tricks_query import (  # noqa: E402
    TricksQueryTower, load_tricks_checkpoint,
)
from semplan.eval import protocol  # noqa: E402

SMOKE_ROOT = ROOT / "outputs/tmp/i40_gate_smoke"
TOPK = 10
# any of these in the stored config marks a TricksQueryTower checkpoint
TRICK_KEYS = ("longq_L", "freq_alpha", "freq_c", "input_dropout", "cumulative_ce")


def load_gate_checkpoint(path: str, item_l2, input_emb=None):
    """(model, kind): tricks checkpoints via load_tricks_checkpoint (stored config
    rebuilds longq/freq modules AND the causal-mask/cumulative-ce eval behaviour);
    plain FrozenQueryTower checkpoints fall back to load_checkpoint."""
    cfg = torch.load(str(path), map_location="cpu", weights_only=True)["config"]
    if any(k in cfg for k in TRICK_KEYS):
        m = load_tricks_checkpoint(str(path), item_l2, input_emb=input_emb)
        kind = "tricks"
    else:
        m = load_checkpoint(str(path), item_l2, input_emb=input_emb)
        kind = "frozen"
    m.eval()
    return m, kind


def ckpt_names(paths: list[str]) -> list[str]:
    """Short unique display names: <parent-dir>/<file>, deduped by index."""
    names, seen = [], {}
    for p in paths:
        nm = f"{Path(p).parent.name}/{Path(p).name}"
        if nm in seen:
            seen[nm] += 1
            nm = f"{nm}#{seen[nm]}"
        else:
            seen[nm] = 0
        names.append(nm)
    return names


def gate_split(split, which: str, models: dict, item_t: torch.Tensor, t0, tag,
               device: str = "cpu"):
    """Per-checkpoint top-10 full-corpus lists + (if >1 ckpt) the mean-z ensemble
    list, all scored through protocol.evaluate. -> (per_ckpt reports, ens report)."""
    qs = {nm: encode_split_queries(m, split, which=which, device=device).to(device)
          for nm, m in models.items()}
    n = len(getattr(split, which))
    lists = {nm: [] for nm in qs}
    ens_lists = [] if len(qs) > 1 else None
    for qi in range(n):
        zsum = None
        for nm in qs:
            sc = torch.einsum("kd,nd->kn", qs[nm][qi], item_t
                              ).max(dim=0).values.cpu().numpy()
            lists[nm].append(topk_ids(sc, TOPK))
            if ens_lists is not None:
                z = zscore(sc)
                zsum = z if zsum is None else zsum + z
        if ens_lists is not None:
            ens_lists.append(topk_ids(zsum / len(qs), TOPK))
        if (qi + 1) % 5000 == 0:
            print(f"  [i40_gate] {tag}/{which}: {qi+1}/{n} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    reports = {nm: eval_ranked_lists(split, which, lst) for nm, lst in lists.items()}
    ens = eval_ranked_lists(split, which, ens_lists) if ens_lists is not None else None
    return reports, ens


def make_smoke_tricks_ckpt(fix: dict) -> Path:
    """Random-init TricksQueryTower (longq+freq+causal cum-CE ON) over the i41
    fixture matrices — exercises the tricks load path end-to-end on CPU."""
    p = SMOKE_ROOT / "tricks_s42" / "tower.pt"
    if not p.exists():
        item = l2(load_matrix(fix["item_emb"]))
        tower = TricksQueryTower(item, d=16, n_layers=1, n_heads=2, d_ff=32,
                                 max_len=20, seed=42, n_query_vectors=2,
                                 input_emb=load_matrix(fix["input_emb"]),
                                 longq_L=3, freq_alpha=0.5, freq_c=3,
                                 input_dropout=0.5, cumulative_ce=True)
        save_checkpoint(tower, str(p))
    return p


def main():
    ap = argparse.ArgumentParser(description="I40 clean inner-world gate (CPU)")
    ap.add_argument("--dataset", default="toys", choices=("beauty", "toys", "sports"))
    ap.add_argument("--seq", help="inner-world LIGER-style seq txt (*_inner.txt)")
    ap.add_argument("--n-items", type=int)
    ap.add_argument("--item-emb", help="enriched openai catalog matrix "
                                       "(.pt/.npy/.npz), the i22inner scoring side")
    ap.add_argument("--input-emb", default=None,
                    help="i24inner [own;pred3] input matrix (required by i24-recipe "
                         "checkpoints; omit for item-input towers)")
    ap.add_argument("--ckpt", nargs="+",
                    help="one or more checkpoints (tricks tower.pt or plain j5.pt); "
                         ">1 also scores the i23/i41 mean-z ensemble")
    ap.add_argument("--out", default=None,
                    help="default results/i40_gate_<dataset>.json")
    ap.add_argument("--device", default="cpu",
                    help="cpu (default) or cuda — encoding, item matrix and the "
                         "max-IP scoring loop all run on this device; ranked "
                         "lists/metrics are device-independent")
    ap.add_argument("--smoke", action="store_true",
                    help="synthetic fixture under outputs/tmp/i40_gate_smoke "
                         "(i41 fixture + one tricks tower), no real data")
    args = ap.parse_args()
    t0 = time.time()

    dataset = args.dataset
    if args.smoke:
        fix = make_smoke_fixture(SMOKE_ROOT)
        args.seq, args.n_items = str(fix["seq"]), fix["n_items"]
        args.item_emb, args.input_emb = str(fix["item_emb"]), str(fix["input_emb"])
        args.ckpt = [str(make_smoke_tricks_ckpt(fix))] + \
            [str(p) for p in fix["ckpts"][:2]]
        dataset = "smoke"
        out = Path(args.out) if args.out else SMOKE_ROOT / "i40_gate.json"
    else:
        for req in ("seq", "n_items", "item_emb", "ckpt"):
            assert getattr(args, req) is not None, f"--{req.replace('_', '-')} required"
        out = Path(args.out) if args.out else \
            ROOT / f"results/i40_gate_{dataset}.json"

    split = protocol.build(args.seq, n_items=args.n_items, max_len=20)
    emb = l2(load_matrix(args.item_emb))
    item_t = torch.from_numpy(emb).to(args.device)
    inp = None if args.input_emb is None else load_matrix(args.input_emb)

    names = ckpt_names(args.ckpt)
    models, kinds = {}, {}
    for nm, p in zip(names, args.ckpt):
        models[nm], kinds[nm] = load_gate_checkpoint(p, emb, input_emb=inp)
    print(f"[i40_gate] {dataset}: {len(split.val)} val / {len(split.test)} "
          f"pseudo-test events, {len(models)} ckpt(s) "
          f"({', '.join(f'{n}[{kinds[n]}]' for n in names)}) "
          f"[{time.time()-t0:.0f}s]", flush=True)

    val_rep, val_ens = gate_split(split, "val", models, item_t, t0, dataset,
                                  device=args.device)
    test_rep, test_ens = gate_split(split, "test", models, item_t, t0, dataset,
                                    device=args.device)
    for nm in names:
        print(f"[i40_gate] {nm}: inner-val {val_rep[nm]['overall']}  "
              f"pseudo-test {test_rep[nm]['overall']}", flush=True)
    ensemble = None
    if val_ens is not None:
        ensemble = {"n_ckpts": len(names), "method": "mean_per_query_zscore",
                    "inner_val": val_ens, "pseudo_test": test_ens}
        print(f"[i40_gate] z-ensemble({len(names)}): inner-val "
              f"{val_ens['overall']}  pseudo-test {test_ens['overall']}", flush=True)

    json.dump(
        {"experiment": "I40_tricks_gate", "dataset": dataset, "smoke": args.smoke,
         "no_real_test": True, "protocol": "inner-world val(-3)/pseudo-test(-2), "
                                           "full-corpus max-IP, protocol.evaluate",
         "inputs": {"seq": args.seq, "n_items": args.n_items,
                    "item_emb": args.item_emb, "input_emb": args.input_emb,
                    "ckpts": list(args.ckpt)},
         "per_ckpt": {nm: {"path": p, "kind": kinds[nm],
                           "config": models[nm].config,
                           "inner_val": val_rep[nm], "pseudo_test": test_rep[nm]}
                      for nm, p in zip(names, args.ckpt)},
         "ensemble": ensemble,
         "runtime_sec": round(time.time() - t0, 1)},
        open(out, "w"), indent=2)
    print(f"[i40_gate] wrote {out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
