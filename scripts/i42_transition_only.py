"""I42: transition-only heuristic row for the paper (reviewer-requested control).

Ranks items purely by mined directed transition counts from the user's history —
no learned tower, no embeddings. Isolates how much of our method's win is
attributable to raw transition statistics vs. the frozen-space injection.

Leak-safe mining identical to the paper's P3 convention: each user's held-out
final event of the given world is dropped before counting, so no (pseudo-)test
transition contributes. The single free knob (history decay gamma) is selected
in the INNER world (pseudo-test); the real test set is scored ONCE with the
selected gamma, under both the unmasked (primary) and masked conventions.

Usage (from the repo root):
  python scripts/i42_transition_only.py \
      --dataset beauty --n-items 12101 \
      --seq-inner data/amazon_beauty_backbone/Beauty_inner.txt \
      --seq data/amazon_beauty_backbone/Beauty.txt \
      --out results/i42_transition_only_beauty.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from semplan.eval import protocol  # noqa: E402

GAMMAS = [0.0, 0.5, 0.8, 1.0]  # 0.0 = last-item only
KS = (5, 10)


def mine(split) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Directed adjacency counts over consecutive pairs of s[:-1] (final event dropped) + popularity."""
    rows, cols, pop = [], [], np.zeros(split.n_items)
    for hist, _tgt in split.test:  # hist = s[:-1] — exactly the mined world
        for a, b in zip(hist, hist[1:]):
            rows.append(a); cols.append(b)
        for it in hist:
            pop[it] += 1.0
    C = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(split.n_items, split.n_items))
    return C, 1e-6 * pop / max(pop.max(), 1.0)  # pop only pads/breaks ties


def make_rank_fn(C: sparse.csr_matrix, pop: np.ndarray, gamma: float):
    def rank_fn(hist, excl):
        if gamma == 0.0:
            sc = np.asarray(C[hist[-1]].todense()).ravel() + pop
        else:
            w = gamma ** np.arange(len(hist) - 1, -1, -1)
            sc = np.asarray((sparse.csr_matrix(w) @ C[hist]).todense()).ravel() + pop
        if excl:
            sc[list(excl)] = -np.inf
        top = np.argpartition(-sc, max(KS))[: max(KS)]
        return top[np.argsort(-sc[top])].tolist()
    return rank_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-items", type=int, required=True)
    ap.add_argument("--seq-inner", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    # --- inner-world selection (pseudo-test = inner split's test) ---
    inner = protocol.build(args.seq_inner, args.n_items)
    Ci, popi = mine(inner)
    inner_rep = {}
    for g in GAMMAS:
        r = protocol.evaluate(make_rank_fn(Ci, popi, g), inner, ks=KS, which="test")
        inner_rep[f"gamma{g}"] = r["overall"]
        print(f"[inner] gamma={g}: HR@5={r['overall']['HR@5']:.4f}", flush=True)
    best_g = max(GAMMAS, key=lambda g: inner_rep[f"gamma{g}"]["HR@5"])

    # --- ONE real-test look with the selected gamma, both conventions ---
    real = protocol.build(args.seq, args.n_items)
    Cr, popr = mine(real)
    fn = make_rank_fn(Cr, popr, best_g)
    unmasked = protocol.evaluate(fn, real, ks=KS, which="test", exclude_history=False)
    masked = protocol.evaluate(fn, real, ks=KS, which="test", exclude_history=True)

    out = {
        "experiment": "I42_transition_only",
        "dataset": args.dataset,
        "gammas": GAMMAS,
        "inner_selection": inner_rep,
        "selected_gamma": best_g,
        "real_test_unmasked": unmasked,
        "real_test_masked": masked,
        "runtime_sec": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] {args.dataset}: gamma={best_g} "
          f"unmasked HR@5={unmasked['overall']['HR@5']:.4f} "
          f"masked HR@5={masked['overall']['HR@5']:.4f}", flush=True)


if __name__ == "__main__":
    main()
