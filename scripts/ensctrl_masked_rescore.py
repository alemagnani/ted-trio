"""E-ENSCTRL masked descriptive rescore — deterministic, NO selection, NO new look.

Prereg basis: PREREG-E-ENSCTRL ("All eval via protocol.py, unmasked primary + masked
descriptive"). The booked one-look fusion runs (ensctrl_sasrec[_ce]_{ds}.json) reported
the unmasked primary only; this script adds the preregistered masked descriptive
companion for the SAME frozen configuration:

  1. Load the 3 per-seed checkpoints saved by the booked run
     (outputs/artifacts/ensctrl_sasrec[_ce]_{ds}_s{seed}/sasrec_best.pt).
  2. Recompute fused per-user z scores with the FROZEN selected_w from the booked JSON
     (no grid, no selection anywhere in this script).
  3. HARD GUARD: recomputed unmasked test metrics must equal the booked test metrics
     exactly (all 4 metrics, tol 5e-5) BEFORE any masked number is read; abort otherwise.
  4. Masked descriptive: identical fused pool, user's history (i43/i41 convention:
     protocol.evaluate exclude_history=True, MQSA-TED/BSARec test-phase semantics)
     excluded from the ranked corpus, never the test target.

Yelp is intentionally OUT of scope: its masked eval was removed from the paper
(6.7% raw repeat-target share makes masked numbers misleading there).

Usage (box, repo root):
  python scripts/ensctrl_masked_rescore.py --dataset beauty --loss ce --device cuda:0
Output: results/ensctrl_masked_{ce|bce}_<dataset>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.eval import protocol  # noqa: E402
import close2_sasrec as c2  # noqa: E402

SEEDS = (42, 43, 44)
MAX_LEN = 20
POOL_K = max(c2.KS) + MAX_LEN  # history is capped at max_len rows => pool stays deep enough
GUARD_TOL = 5e-5


def eval_ranked_lists(split, which, lists, ks=c2.KS, exclude_history=False):
    """Verbatim copy of i41_common.eval_ranked_lists (avoids its scipy dependency):
    protocol.evaluate over precomputed best-first ranked lists (positional)."""
    it = iter(lists)
    kmax = max(ks)

    def rank_fn(_hist, excl):
        r = next(it)
        return [i for i in r if i not in excl][:kmax] if excl else list(r)[:kmax]

    return protocol.evaluate(rank_fn, split, ks=ks, which=which,
                             exclude_history=exclude_history)


def load_seed_model(ckpt_path: Path, n_items: int, device) -> c2.SASRec:
    blob = torch.load(ckpt_path, map_location=device)
    cfg = blob["config"]
    model = c2.SASRec(n_items, cfg["maxlen"], d=cfg["d"], n_blocks=cfg["blocks"],
                      n_heads=cfg["heads"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def fused_pool(models, events, maxlen, device, weights, topk, batch=256):
    """Same fusion as ensctrl_sasrec_fusion.fused_lists, deeper top-k pool."""
    seqs = torch.from_numpy(c2.pad_histories(events, maxlen))
    out = []
    for i in range(0, len(seqs), batch):
        s = seqs[i:i + batch].to(device)
        fused = None
        for w, m in zip(weights, models):
            sc = m.score_full(s)
            z = (sc - sc.mean(dim=1, keepdim=True)) / sc.std(dim=1, keepdim=True).clamp(min=1e-8)
            fused = w * z if fused is None else fused + w * z
        _, idx = torch.topk(fused, k=topk, dim=1)
        out.extend(idx.cpu().numpy().tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("beauty", "toys", "sports"))
    ap.add_argument("--loss", required=True, choices=("bce", "ce"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    t0 = time.time()

    prefix = "ensctrl_sasrec_ce" if args.loss == "ce" else "ensctrl_sasrec"
    booked_path = ROOT / "results" / f"{prefix}_{args.dataset}.json"
    booked = json.load(open(booked_path))
    weights = booked["fusion"]["selected_w"]
    ds = c2.DATASETS[args.dataset]
    device = torch.device(args.device)

    split = protocol.build(str(ds["seq"]), n_items=ds["n_items"], max_len=MAX_LEN)
    models = [load_seed_model(
        ROOT / "outputs/artifacts" / f"{prefix}_{args.dataset}_s{seed}" / "sasrec_best.pt",
        ds["n_items"], device) for seed in SEEDS]
    print(f"[rescore] {args.dataset}/{args.loss}: {len(split.test)} test events, "
          f"frozen w={weights} [{time.time()-t0:.0f}s]", flush=True)

    pool = fused_pool(models, split.test, MAX_LEN, device, weights, topk=POOL_K)

    unmasked = eval_ranked_lists(split, "test", pool, exclude_history=False)
    guard = {"pass": True, "checks": {}}
    for k in ("HR@5", "NDCG@5", "HR@10", "NDCG@10"):
        got, want = unmasked["overall"][k], booked["test"]["overall"][k]
        guard["checks"][k] = {"recomputed": got, "booked": want}
        if abs(got - want) >= GUARD_TOL:
            guard["pass"] = False
    print(f"[rescore] unmasked guard: {guard}", flush=True)
    assert guard["pass"], (
        f"UNMASKED REPRO GUARD FAILED for {args.dataset}/{args.loss}: {guard['checks']} "
        f"— wrong checkpoints or drifted pipeline; refusing to report masked numbers")

    masked = eval_ranked_lists(split, "test", pool, exclude_history=True)
    repeat_share = round(sum(1 for h, t in split.test if t in set(h))
                         / max(len(split.test), 1), 4)
    delta = {m: round(masked["overall"][m] - unmasked["overall"][m], 4)
             for m in unmasked["overall"]}
    print(f"[rescore] masked  : {masked['overall']}", flush=True)
    print(f"[rescore] delta   : {delta} (repeat_target_share={repeat_share})", flush=True)

    out = ROOT / "results" / f"ensctrl_masked_{args.loss}_{args.dataset}.json"
    json.dump({
        "experiment": "E-ENSCTRL_masked_descriptive_rescore",
        "prereg": "PREREG-E-ENSCTRL (unmasked primary + masked descriptive)",
        "dataset": args.dataset, "loss": args.loss,
        "no_selection": True, "configs_scored": 1,
        "booked_source": str(booked_path.relative_to(ROOT)),
        "frozen_w": weights, "seeds": list(SEEDS),
        "masking": "i41/i43 convention: protocol.evaluate exclude_history=True "
                   "(train+val items within the max_len=20 evaluator window)",
        "unmasked_reproduction_guard": guard,
        "repeat_target_share": repeat_share,
        "unmasked": unmasked, "masked": masked,
        "delta_masked_minus_unmasked": delta,
        "runtime_sec": round(time.time() - t0, 1),
    }, open(out, "w"), indent=2)
    print(f"[rescore] wrote {out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
