"""Deterministic regeneration of results/ensctrl_sasrec_instruments.json.

The E-ENSCTRL arm-A Instruments run completed on a cloud GPU box (2026-08-15, log:
outputs/box4_logs/ensctrl_instruments.log) but its summary JSON was not pulled
before the box was destroyed (disclosed in ledger entry E-ENSCTRL-A-INSTRUMENTS).
This script regenerates the JSON from the archived seed checkpoints. It is NOT a
new look: every number is hard-guarded against the run log and the pulled
per-user npz before anything is written.

Guards (all must hold, else abort with nothing written):
  1. per-seed solo test == booked close2_sasrec_instruments_s{42,43,44}.json
  2. val selection reproduces w=(0.5,0.2,0.3) with val HR@5 0.0854 (from log)
  3. fused test == log/npz exactly: HR@5 0.0717, HR@10 0.0961, NDCG@5 0.0389,
     NDCG@10 0.0467; per-user hit/dcg arrays == pulled npz bitwise-close.

Usage (repo root): python scripts/ensctrl_regen_instruments.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.eval import protocol  # noqa: E402
import close2_sasrec as c2  # noqa: E402

SEEDS = (42, 43, 44)
DS = "instruments"
LOG_W = (0.5, 0.2, 0.29999999999999993)     # from run log (verbatim)
LOG_VAL_HR5 = 0.0854                         # from run log
LOG_TEST = {"HR@5": 0.0717, "HR@10": 0.0961, "NDCG@5": 0.0389, "NDCG@10": 0.0467}
TOL = 5e-5


def load_seed_model(ckpt_path: Path, n_items: int) -> c2.SASRec:
    blob = torch.load(ckpt_path, map_location="cpu")
    cfg = blob["config"]
    model = c2.SASRec(n_items, cfg["maxlen"], d=cfg["d"], n_blocks=cfg["blocks"],
                      n_heads=cfg["heads"], dropout=cfg["dropout"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def z_scores(model, events, maxlen, batch=256):
    """Per-user z-normalized full score matrix (float32, [n_users, n_items])."""
    seqs = torch.from_numpy(c2.pad_histories(events, maxlen))
    out = []
    for i in range(0, len(seqs), batch):
        sc = model.score_full(seqs[i:i + batch])
        z = (sc - sc.mean(dim=1, keepdim=True)) / sc.std(dim=1, keepdim=True).clamp(min=1e-8)
        out.append(z)
    return torch.cat(out)


def eval_lists(lists, split, which):
    it = iter(lists)
    return protocol.evaluate(lambda h, e: next(it), split, ks=c2.KS, which=which)


def topk_lists(z_list, w, k):
    fused = sum(wi * zi for wi, zi in zip(w, z_list))
    return torch.topk(fused, k=k, dim=1).indices.numpy().tolist()


def weight_grid(step=0.1):
    grid, n = [], round(1 / step)
    for a, b in itertools.product(range(n + 1), repeat=2):
        if a + b <= n:
            grid.append((a * step, b * step, 1 - (a + b) * step))
    return grid


def main():
    ds = c2.DATASETS[DS]
    split = protocol.build(str(ds["seq"]), n_items=ds["n_items"], max_len=20)
    models, solo_test = [], {}
    kmax = max(c2.KS)

    for seed in SEEDS:
        m = load_seed_model(
            ROOT / "outputs/artifacts" / f"ensctrl_sasrec_{DS}_s{seed}" / "sasrec_best.pt",
            ds["n_items"])
        models.append(m)
        # guard 1: solo test vs booked close2 artifact
        z_t = z_scores(m, split.test, m.maxlen)
        lists = torch.topk(z_t, k=kmax, dim=1).indices.numpy().tolist()
        tm = eval_lists(lists, split, "test")
        booked = json.load(open(ROOT / "results" / f"close2_sasrec_{DS}_s{seed}.json"))
        for k in ("HR@5", "NDCG@5", "HR@10", "NDCG@10"):
            got, want = tm["overall"][k], booked["test"]["overall"][k]
            assert abs(got - want) < TOL, f"GUARD1 FAIL s{seed} {k}: {got} vs {want}"
        solo_test[seed] = tm["overall"]
        print(f"[regen] s{seed} solo guard PASS (HR@5 {tm['overall']['HR@5']:.4f})", flush=True)

    # guard 2: reproduce val selection
    z_val = [z_scores(m, split.val, m.maxlen) for m in models]
    best_w, best_val, val_grid = None, -1.0, []
    for w in weight_grid():
        hr5 = eval_lists(topk_lists(z_val, w, kmax), split, "val")["overall"]["HR@5"]
        val_grid.append({"w": w, "val_HR@5": hr5})
        if hr5 > best_val:
            best_w, best_val = w, hr5
    del z_val
    print(f"[regen] selected w={best_w} val HR@5={best_val:.4f}", flush=True)
    assert all(abs(a - b) < 1e-9 for a, b in zip(best_w, LOG_W)), f"GUARD2 FAIL w {best_w}"
    assert abs(best_val - LOG_VAL_HR5) < TOL, f"GUARD2 FAIL val {best_val}"

    # guard 3: single frozen-w test fusion vs log + pulled npz
    z_test = [z_scores(m, split.test, m.maxlen) for m in models]
    test_lists = topk_lists(z_test, best_w, kmax)
    del z_test
    test_metrics = eval_lists(test_lists, split, "test")
    for k, want in LOG_TEST.items():
        got = test_metrics["overall"][k]
        assert abs(got - want) < TOL, f"GUARD3 FAIL {k}: {got} vs log {want}"
    arrays = c2.per_user_arrays(test_lists, split.test, split.cold_test_idx)
    npz = np.load(ROOT / "results" / f"ensctrl_sasrec_{DS}_per_user.npz")
    for k in ("hit5", "hit10", "dcg5", "dcg10"):
        assert np.allclose(arrays[k], npz[k], atol=1e-6), f"GUARD3 FAIL npz array {k}"
    assert np.array_equal(arrays["cold_mask"], npz["cold_mask"]), "GUARD3 FAIL cold_mask"
    print(f"[regen] fused test={json.dumps(test_metrics['overall'])} — all guards PASS", flush=True)

    out = {
        "experiment": "E-ENSCTRL_armA_sasrec_x3_fusion", "dataset": DS,
        "seeds": list(SEEDS), "solo_test": {str(s): m for s, m in solo_test.items()},
        "fusion": {"selected_w": list(best_w), "selected_on": "protocol val HR@5",
                   "val_grid": val_grid},
        "test": test_metrics,
        "regenerated": ("Deterministically regenerated 2026-08-15 from archived seed "
                        "checkpoints after the original box-4 JSON was lost in teardown; "
                        "guarded against run log (outputs/box4_logs/ensctrl_instruments.log) "
                        "and pulled per-user npz. See ledger E-ENSCTRL-A-INSTRUMENTS."),
    }
    res = ROOT / "results" / f"ensctrl_sasrec_{DS}.json"
    json.dump(out, open(res, "w"), indent=2)
    print(f"[regen] wrote {res}", flush=True)


if __name__ == "__main__":
    main()
