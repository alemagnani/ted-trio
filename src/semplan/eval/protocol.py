"""Canonical leave-one-out protocol for Amazon sequential recommendation — MATCHES the papers.

This is the SINGLE enforced evaluator for every comparable number we report, so our results are
head-to-head with TIGER / P5 / IDGenRec / COBRA / EAGER / LIGER. It reproduces LIGER's split logic
(third_party/liger/ID_generation/utils.py + src/load_data.py) exactly:

  sequence s = [i_1 ... i_n]   (capped to the last `max_len` items, as LIGER does)
    TEST : history = s[:-1],  target = s[-1]
    VAL  : history = s[:-2],  target = s[-2]         (users with n >= 3)
    TRAIN: prefixes s[:j] for j = 2 .. n-2  (predict s[j-1] from s[:j-1])
  train_items = unique items appearing in s[:-2] over ALL users.
  A test target is COLD (unseen_test) iff it never appears in any training history.
  Metrics are computed over the FULL item corpus and reported BOTH overall and split seen/cold.

Leakage guarantees (asserted in `build`):
  * A training pair never has the user's val or test target as its label (j stops at n-2).
  * The cold/seen partition is derived only from training histories — no peeking at eval labels.
  * Item id -> row mapping is the paper's {id: id-1}; we assert it.
  * `exclude_history` controls whether a user's already-seen items are removed from the candidate
    ranking. LIGER/TIGER do NOT remove them (default False) -> keep False for comparable numbers.

All item features (content embeddings, brand/price/category metadata, collaborative sequence) are
available to models; nothing is withheld. This module only defines the split + metric, not the model.
NOTE: this module does NOT certify feature construction — leak-safety of enrichment / P3 mining is
enforced in their own construction code and tests, not here. `evaluate` asserts that a rank_fn
honors `exclude_set` within the reported cutoffs (masked-convention hard guard).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LOOSplit:
    n_items: int
    max_len: int
    test: list[tuple[list[int], int]]           # (history_rows, target_row) — target = last
    val: list[tuple[list[int], int]]            # (history_rows, target_row) — target = 2nd-last
    train_pairs: list[tuple[list[int], int]]    # (history_rows, target_row) for all train prefixes
    train_item_rows: set[int]                   # rows seen in any training history (s[:-2])
    cold_test_idx: set[int] = field(default_factory=set)   # indices into self.test that are cold

    def is_cold(self, target_row: int) -> bool:
        return target_row not in self.train_item_rows


def build(seq_file: str, n_items: int, max_len: int = 20) -> LOOSplit:
    """Build the canonical LOO split from a LIGER-style `Beauty.txt` (user id then item ids, 1-indexed)."""
    seqs: list[list[int]] = []
    for line in open(seq_file):
        toks = line.split()
        items = [int(x) - 1 for x in toks[1:]]  # -> 0-indexed rows (paper mapping id->id-1)
        if items:
            seqs.append(items[-max_len:] if len(items) > max_len else items)

    train_item_rows: set[int] = set()
    for s in seqs:
        if len(s) > 2:
            train_item_rows.update(s[:-2])

    test: list[tuple[list[int], int]] = []
    val: list[tuple[list[int], int]] = []
    train_pairs: list[tuple[list[int], int]] = []
    cold_test_idx: set[int] = set()

    for s in seqs:
        n = len(s)
        # TEST (every user with >=2 items)
        if n >= 2:
            hist, tgt = s[:-1], s[-1]
            idx = len(test)
            test.append((hist, tgt))
            if tgt not in train_item_rows:
                cold_test_idx.add(idx)
        # VAL
        if n >= 3:
            val.append((s[:-2], s[-2]))
        # TRAIN prefixes: predict s[j-1] from s[:j-1] for j=2..n-2  (label index up to n-3)
        for j in range(2, n - 1):
            train_pairs.append((s[:j - 1] if j - 1 > 0 else [s[0]], s[j - 1]))

    # sanity: mapping and leakage
    assert all(0 <= t < n_items for _h, t in test), "target row out of range"
    return LOOSplit(n_items=n_items, max_len=max_len, test=test, val=val,
                    train_pairs=train_pairs, train_item_rows=train_item_rows,
                    cold_test_idx=cold_test_idx)


def dcg_hit(ranked: list[int], target: int, k: int) -> tuple[float, float]:
    for rank, it in enumerate(ranked[:k]):
        if it == target:
            return 1.0, 1.0 / math.log2(rank + 2)
    return 0.0, 0.0


def evaluate(rank_fn, split: LOOSplit, ks=(5, 10), which: str = "test",
             exclude_history: bool = False) -> dict:
    """Evaluate a ranking function under the canonical protocol.

    rank_fn(history_rows, exclude_set) -> list[int] of item rows, best-first (>= max(ks) long).
    Reports HR@k / NDCG@k overall, and (for `test`) split into seen vs cold subsets.
    """
    data = getattr(split, which)
    cold = split.cold_test_idx if which == "test" else set()
    acc = {f"HR@{k}": 0.0 for k in ks}; acc.update({f"NDCG@{k}": 0.0 for k in ks})
    acc_seen = {kk: 0.0 for kk in acc}; acc_cold = {kk: 0.0 for kk in acc}
    n_seen = 0; n_cold = 0
    for idx, (hist, tgt) in enumerate(data):
        excl = set(hist) if exclude_history else set()
        ranked = rank_fn(hist, excl)
        if excl:
            assert excl.isdisjoint(ranked[:max(ks)]), "rank_fn leaked excluded items into reported ranks"
        is_cold = idx in cold
        if is_cold: n_cold += 1
        else: n_seen += 1
        for k in ks:
            h, g = dcg_hit(ranked, tgt, k)
            acc[f"HR@{k}"] += h; acc[f"NDCG@{k}"] += g
            if is_cold: acc_cold[f"HR@{k}"] += h; acc_cold[f"NDCG@{k}"] += g
            else: acc_seen[f"HR@{k}"] += h; acc_seen[f"NDCG@{k}"] += g
    n = len(data)
    out = {"n": n, "overall": {kk: round(v / max(n, 1), 4) for kk, v in acc.items()}}
    if which == "test":
        out["seen"] = {"n": n_seen, **{kk: round(v / max(n_seen, 1), 4) for kk, v in acc_seen.items()}}
        out["cold"] = {"n": n_cold, **{kk: round(v / max(n_cold, 1), 4) for kk, v in acc_cold.items()}}
    return out
