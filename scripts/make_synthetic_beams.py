"""Write a synthetic TIGER BeamRecord JSONL for a dataset's test split (release addition).

Used only by the quick / smoke pipelines to exercise scripts/close2_tiger_score.py (and the
paired tests downstream of it) without training TIGER. Rows follow LIGER's eval order
(seen test users first, then cold users), each with `target_item` set so the scorer's
alignment guard is exercised; 20 beams per row, the target placed at a random rank for
~30% of the users, plus an occasional invalid beam.

Usage: python scripts/make_synthetic_beams.py --dataset beauty --out outputs/beams.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.eval import protocol  # noqa: E402

from close2_tiger_score import DATASETS, liger_row_to_user  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beams", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds = DATASETS[args.dataset]
    split = protocol.build(str(ds["seq"]), n_items=ds["n_items"], max_len=20)
    rng = np.random.default_rng(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row, u in enumerate(liger_row_to_user(split)):
            tgt = split.test[u][1]
            items = rng.choice(ds["n_items"], size=args.beams, replace=False).tolist()
            if rng.random() < 0.3:
                items[int(rng.integers(0, args.beams))] = tgt
            for r, it in enumerate(items):
                valid = not (r == args.beams - 1 and rng.random() < 0.05)
                f.write(json.dumps({"query_id": f"test:{row}", "beam_rank": r,
                                    "valid": valid,
                                    "mapped_item_id": int(it) + 1 if valid else None,
                                    "target_item": int(tgt) + 1}) + "\n")
    print(f"wrote {args.out} ({len(split.test)} rows x {args.beams} beams)")


if __name__ == "__main__":
    main()
