"""Diff a reproduction's result JSONs against the shipped originals (release addition).

For every *.json present in both directories, flattens all numeric leaves and reports those
that differ by more than --tol (default 0: the metrics are stored rounded to 4 decimals, so
any difference is a real one). Paths, timings, dates and host/environment blocks are
ignored. Also reports files present on only one side.

Usage: python scripts/compare_results.py outputs/full/results results [--tol 1e-4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

IGNORE = ("runtime", "seconds", "_sec", "time", "date", "utc", "path", "ckpt", "checkpoint",
          "env", "host", "platform", "load", "_ms", "ms_", "wall", "threads", "beams",
          "baseline_npz", "item_emb", "input_emb", "seq")


def flatten(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{prefix}{i}.")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix[:-1], float(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mine")
    ap.add_argument("reference")
    ap.add_argument("--tol", type=float, default=0.0)
    args = ap.parse_args()
    mine, ref = Path(args.mine), Path(args.reference)
    a = {p.name for p in mine.glob("*.json")}
    b = {p.name for p in ref.glob("*.json")}
    n_same = n_diff = 0
    for name in sorted(a & b):
        x = dict(flatten(json.load(open(mine / name))))
        y = dict(flatten(json.load(open(ref / name))))
        diffs = [(k, x[k], y[k]) for k in sorted(x.keys() & y.keys())
                 if not any(t in k.lower() for t in IGNORE) and abs(x[k] - y[k]) > args.tol]
        if diffs:
            n_diff += 1
            print(f"DIFF {name}: {len(diffs)} numeric fields differ")
            for k, u, v in diffs[:12]:
                print(f"    {k}: yours {u:g} vs paper {v:g}")
        else:
            n_same += 1
            print(f"SAME {name}")
    for name in sorted(a - b):
        print(f"ONLY-YOURS {name}")
    print(f"\n{n_same} files identical, {n_diff} differ, {len(b - a)} reference files not reproduced")


if __name__ == "__main__":
    main()
