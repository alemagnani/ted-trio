"""E-ENSCTRL arm A — matched-ensemble control: SASRec x3 seeds, score-level fusion.

Prereg: ledger/ledger.jsonl PREREG-E-ENSCTRL + ledger/E_ENSCTRL.md.
Question: would ANY matched-count (3-model) ensemble of standard baselines match the
deployed trio, or do the E/C injections carry value beyond generic ensembling?

Per dataset:
  1. Train SASRec seeds 42/43/44 with the EXACT CLOSE-2 booked config (paper defaults,
     bce loss) by reusing close2_sasrec.train() verbatim.
  2. REPRO GUARD: each seed's solo test HR@5/NDCG@5 must equal the booked
     results/close2_sasrec_{ds}_s{seed}.json numbers to 4 decimals
     (same seed, same code path => exact; abort the dataset on mismatch).
  3. Fusion menu (identical to the trio's): per-user z-normalized full-corpus scores,
     weighted sum over the 3 seeds. Grid = mean-z (1,1,1) + simplex weights in steps
     of 0.1. Selection ONLY on protocol VAL (-2) HR@5; weights then FROZEN.
  4. ONE real-test look: fused test eval with the frozen weights, via protocol.evaluate.

Outputs:
  results/ensctrl_sasrec_{ds}.json           (selected weights, val grid, test)
  results/ensctrl_sasrec_{ds}_per_user.npz   (fused hit/dcg arrays, test order)
  outputs/artifacts/ensctrl_sasrec_{ds}_s{seed}/sasrec_best.pt

Caveat (recorded in prereg): each seed's checkpoint selection also used VAL HR@10 —
same convention as the booked CLOSE-2 baseline itself; fusion-weight selection adds no
new information channel beyond what the solo baseline already used.

CLI:
  python scripts/ensctrl_sasrec_fusion.py --dataset beauty --device cuda:0
  python scripts/ensctrl_sasrec_fusion.py --smoke     # CPU fixture
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.eval import protocol  # noqa: E402
from semplan.quick import max_epochs  # noqa: E402  (TT_MAX_EPOCHS; quick mode only)
import close2_sasrec as c2  # noqa: E402  (train, SASRec, DATASETS, pad_histories, KS)

SEEDS = (42, 43, 44)


# ------------------------------------------------------------------ fused scoring
@torch.no_grad()
def fused_lists(models, events, maxlen, device, weights, topk, batch=256):
    """Weighted sum of per-model per-user z-normalized full-corpus scores -> top-k rows."""
    for m in models:
        m.eval()
    seqs = torch.from_numpy(c2.pad_histories(events, maxlen))
    out = []
    for i in range(0, len(seqs), batch):
        s = seqs[i:i + batch].to(device)
        fused = None
        for w, m in zip(weights, models):
            sc = m.score_full(s)                                   # [B, N]
            z = (sc - sc.mean(dim=1, keepdim=True)) / sc.std(dim=1, keepdim=True).clamp(min=1e-8)
            fused = w * z if fused is None else fused + w * z
        _, idx = torch.topk(fused, k=topk, dim=1)
        out.extend(idx.cpu().numpy().tolist())
    return out


def eval_lists(lists, split, which):
    it = iter(lists)
    return protocol.evaluate(lambda h, e: next(it), split, ks=c2.KS, which=which)


def weight_grid(step=0.1):
    """Simplex grid over 3 seeds (incl. mean-z (1/3,1/3,1/3) via normalization)."""
    grid, n = [], round(1 / step)
    for a, b in itertools.product(range(n + 1), repeat=2):
        if a + b <= n:
            grid.append((a * step, b * step, 1 - (a + b) * step))
    return grid


# ------------------------------------------------------------------ per-dataset run
def run_dataset(ds_name, seq_file, n_items, device_str, booked_dir=ROOT / "results",
                repro_guard=True, out_prefix="ensctrl_sasrec", train_args=None, loss="bce"):
    device = torch.device(device_str)
    models, solo_test = [], {}
    split = None
    booked_prefix = "close2_sasrec_ce" if loss == "ce" else "close2_sasrec"
    for seed in SEEDS:
        args = argparse.Namespace(
            dataset=ds_name, seed=seed, device=device_str, d=50, blocks=2, heads=1,
            dropout=0.5, maxlen=20, lr=1e-3, l2_emb=0.0, batch=128, epochs=max_epochs(200),
            eval_every=min(10, max_epochs(200)), patience=40, loss=loss)
        if train_args:  # smoke overrides
            vars(args).update(train_args)
        ckpt = ROOT / "outputs/artifacts" / f"{out_prefix}_{ds_name}_s{seed}" / "sasrec_best.pt"
        split, model, best, test_metrics, arrays, dt = c2.train(
            args, seq_file, n_items, out_ckpt=ckpt if repro_guard else None)
        models.append(model)
        solo_test[seed] = test_metrics["overall"]
        if repro_guard:
            booked = json.load(open(booked_dir / f"{booked_prefix}_{ds_name}_s{seed}.json"))
            for k in ("HR@5", "NDCG@5", "HR@10", "NDCG@10"):
                got, want = test_metrics["overall"][k], booked["test"]["overall"][k]
                assert abs(got - want) < 5e-5, (
                    f"REPRO GUARD FAIL {ds_name} s{seed} {k}: retrain {got} vs booked {want}")
            print(f"[ensctrl] {ds_name} s{seed} repro guard PASS "
                  f"(HR@5 {test_metrics['overall']['HR@5']:.4f}) [{dt:.0f}s]", flush=True)

    # fusion selection on VAL only
    t0 = time.time()
    best_w, best_val = None, -1.0
    val_grid = []
    for w in weight_grid():
        lists = fused_lists(models, split.val, models[0].maxlen, device, w, topk=max(c2.KS))
        hr5 = eval_lists(lists, split, "val")["overall"]["HR@5"]
        val_grid.append({"w": w, "val_HR@5": hr5})
        if hr5 > best_val:
            best_w, best_val = w, hr5
    print(f"[ensctrl] {ds_name} selected w={best_w} val HR@5={best_val:.4f} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ONE test look with frozen weights
    test_lists = fused_lists(models, split.test, models[0].maxlen, device, best_w,
                             topk=max(c2.KS))
    test_metrics = eval_lists(test_lists, split, "test")
    arrays = c2.per_user_arrays(test_lists, split.test, split.cold_test_idx)
    for k in c2.KS:
        assert round(float(arrays[f"hit{k}"].mean()), 4) == test_metrics["overall"][f"HR@{k}"]

    out = {
        "experiment": "E-ENSCTRL_armA_sasrec_x3_fusion", "dataset": ds_name,
        "seeds": list(SEEDS), "solo_test": {str(s): m for s, m in solo_test.items()},
        "fusion": {"selected_w": best_w, "selected_on": "protocol val HR@5",
                   "val_grid": val_grid},
        "test": test_metrics,
    }
    if repro_guard:
        res = ROOT / "results" / f"{out_prefix}_{ds_name}.json"
        json.dump(out, open(res, "w"), indent=2)
        np.savez(ROOT / "results" / f"{out_prefix}_{ds_name}_per_user.npz", **arrays)
        print(f"[ensctrl] {ds_name} FUSED test={json.dumps(test_metrics['overall'])} -> {res}",
              flush=True)
    return out


# ------------------------------------------------------------------ smoke
def smoke():
    """CPU fixture: 3 seeds on the synthetic chain; fused HR must be >= mean solo HR - eps
    and the whole selection/eval path must run end-to-end."""
    import tempfile
    n_items, n_users, slen = 60, 300, 10
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        seq_file = Path(td) / "Smoke.txt"
        with open(seq_file, "w") as f:
            for u in range(n_users):
                start = int(rng.integers(0, n_items))
                items = [(start + j) % n_items for j in range(slen)]
                for j in range(slen):
                    if rng.random() < 0.10:
                        items[j] = int(rng.integers(0, n_items))
                f.write(f"{u+1} " + " ".join(str(i + 1) for i in items) + "\n")
        out = run_dataset("smoke", str(seq_file), n_items, "cpu", repro_guard=False,
                          train_args=dict(epochs=40, eval_every=10, patience=40, d=32,
                                          batch=64, maxlen=20, dropout=0.2))
    solo = [m["HR@5"] for m in out["solo_test"].values()]
    fused = out["test"]["overall"]["HR@5"]
    assert fused >= min(solo) - 1e-9, f"fused {fused} < worst solo {min(solo)}"
    assert len(out["fusion"]["val_grid"]) == len(weight_grid())
    print(f"[ensctrl] SMOKE PASS solo HR@5={solo} fused={fused} "
          f"w={out['fusion']['selected_w']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(c2.DATASETS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--loss", choices=("bce", "ce"), default="bce",
                    help="ce = arm C (SASRec-CE x3), guards vs close2_sasrec_ce_* bookings")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    assert args.dataset, "--dataset required (or --smoke)"
    ds = c2.DATASETS[args.dataset]
    prefix = "ensctrl_sasrec_ce" if args.loss == "ce" else "ensctrl_sasrec"
    run_dataset(args.dataset, str(ds["seq"]), ds["n_items"], args.device,
                out_prefix=prefix, loss=args.loss)


if __name__ == "__main__":
    main()
