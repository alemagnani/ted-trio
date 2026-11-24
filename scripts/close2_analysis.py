"""CLOSE-2 per-cell paired analysis (prereg CLOSE2-PREREG-FREEZE 2026-08-07T01:33:18Z).

Each (dataset, baseline) cell is analyzed EXACTLY ONCE, as its baseline per-user npz
lands. Trio arms are the frozen SPENT per-user vectors (w1.0_* keys); baseline npz uses
bare keys (hit5/dcg5/hit10/dcg10 + cold_mask). Primary: HR@5 sign-flip two-sided p<0.05
with positive delta (trio minus baseline). Secondary: HR@10/NDCG@5/NDCG@10 bootstrap CIs.

Usage: python close2_analysis.py --dataset beauty --baseline tiger \
           --baseline-npz results/close2_tiger_beauty_s42_per_user.npz
Writes results/close2_cell_{dataset}_{baseline}.json; refuses to overwrite
(one execution per cell).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.eval import significance  # noqa: E402
from semplan.quick import guard  # noqa: E402

TRIO_NPZ = {
    "beauty": ROOT / "results/i24_test_per_user.npz",
    "toys": ROOT / "results/i30_real_per_user.npz",
    "sports": ROOT / "results/i35_real_per_user.npz",
}
METRICS = ("HR@5", "HR@10", "NDCG@5", "NDCG@10")
KEY = {"HR@5": "hit5", "HR@10": "hit10", "NDCG@5": "dcg5", "NDCG@10": "dcg10"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(TRIO_NPZ))
    ap.add_argument("--baseline", required=True, choices=["tiger", "sasrec"])
    ap.add_argument("--baseline-npz", required=True)
    args = ap.parse_args()

    out_path = ROOT / f"results/close2_cell_{args.dataset}_{args.baseline}.json"
    if out_path.exists():
        sys.exit(f"REFUSING: cell already analyzed ({out_path}) — one execution per cell")

    trio = np.load(TRIO_NPZ[args.dataset])
    base = np.load(args.baseline_npz)
    # paired over identical test users: cold_mask equality is the order check
    assert trio["cold_mask"].shape == base["cold_mask"].shape, "n_users mismatch"
    assert bool((trio["cold_mask"] == base["cold_mask"]).all()), "cold_mask mismatch: user order differs"

    arm = "w1.0"  # the pre-registered trio arm (pure trio, w = 1.0)
    if f"{arm}_hit5" not in trio.files:  # only possible on a quick-mode user subset
        sel = sorted({k.split("_")[0] for k in trio.files if k.startswith("w") and k[:4] != "w0.0"})
        guard(False, f"trio npz has no w1.0 arm (look selected {sel}); using {sel[0]}")
        arm = sel[0]
    tv = {m: trio[f"{arm}_{KEY[m]}"] for m in METRICS}
    bv = {m: base[KEY[m]] for m in METRICS}
    cmp_ = significance.compare(tv, bv, metrics=METRICS, cold_mask=trio["cold_mask"])

    primary = cmp_["HR@5"]
    primary_pass = bool(primary["delta"] > 0
                        and primary["randomization"]["p_two_sided"] < 0.05)
    out = {
        "cell": f"{args.dataset}/{args.baseline}",
        "prereg": "CLOSE2-PREREG-FREEZE 2026-08-07T01:33:18Z",
        "baseline_npz": str(args.baseline_npz),
        "n_users": int(trio["cold_mask"].shape[0]),
        "n_cold": int(trio["cold_mask"].sum()),
        "primary_HR@5": {"trio": primary["mean_a"], "baseline": primary["mean_b"],
                         "delta": primary["delta"],
                         "p_sign_flip": primary["randomization"]["p_two_sided"],
                         "pass": primary_pass},
        "compare": cmp_,
    }
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"[close2] {out['cell']}: trio HR@5 {primary['mean_a']:.4f} vs "
          f"{args.baseline} {primary['mean_b']:.4f} (delta {primary['delta']:+.4f}, "
          f"p {primary['randomization']['p_two_sided']:.2e}) -> "
          f"{'PASS' if primary_pass else 'FAIL'}; wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
