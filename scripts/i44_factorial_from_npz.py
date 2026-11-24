"""Re-derive the §4.1 Sports factorial inference from the SHIPPED per-user arrays.

scripts/i44_factorial_inference.py re-scores the four Sports 2x2 cells from their tower
checkpoints (needs embeddings + 12 checkpoints, ~CPU-hours) and then runs the paired
statistics. This helper skips the re-scoring: it loads results/i44_factorial_per_user.npz
(the per-user HR@5 hit vectors that script wrote) and calls the script's own
`effects_from_cells` (sign-flip permutation p, 2x10^4 permutations, seed 0; percentile
bootstrap CI, 10^4 resamples, seed 0), then compares every estimate / CI / p-value with
results/i44_factorial_inference.json. Release addition; numerical code is the original's.

Usage: python scripts/i44_factorial_from_npz.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import i44_factorial_inference as F  # noqa: E402


def main() -> None:
    z = np.load(ROOT / "results/i44_factorial_per_user.npz")
    booked = json.load(open(ROOT / "results/i44_factorial_inference.json"))["datasets"]
    all_same = True
    for ds in sorted({k.rsplit("_", 1)[0] for k in z.files}):   # paper: sports only
        cells = {c: z[f"{ds}_{c}"] for c in ("pp", "Ep", "pC", "EC")}
        print(f"{ds}: n_users={len(cells['EC'])}", flush=True)
        eff = F.effects_from_cells(cells)
        for name, r in booked[ds]["effects"].items():
            a = eff[name]
            same = all(a[f] == r[f] for f in ("estimate", "ci95", "p_signflip"))
            all_same &= same
            print(f"{name:14s} estimate {a['estimate']:+.4f}  CI95 [{a['ci95'][0]:+.4f}, "
                  f"{a['ci95'][1]:+.4f}]  p {a['p_signflip']:.2g}  "
                  f"{'== booked' if same else '!= booked ' + json.dumps(r)}")
    print("ALL IDENTICAL to results/i44_factorial_inference.json" if all_same
          else "MISMATCH vs booked JSON")
    sys.exit(0 if all_same else 1)


if __name__ == "__main__":
    main()
