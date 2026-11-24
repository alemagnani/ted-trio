"""Paired significance utilities for the canonical evaluator (PIPELINES C8).

Every comparable claim ships with paired uncertainty. Two tests, both PAIRED over users:

  * `paired_bootstrap`      — resample users with replacement; mean delta, percentile 95% CI,
                              and p(delta > 0). The workhorse for tables.
  * `paired_randomization`  — exact-style sign-flip test; two-sided p-value. Preferred when the
                              gap is tiny (|delta| < ~0.002), where the bootstrap CI is coarse.

`per_user_metrics` produces the per-user hit / DCG vectors under the SAME split + metric as
`protocol.evaluate` (it reuses `protocol.dcg_hit` and the split's cold partition; averaging the
vectors reproduces `evaluate`'s aggregates exactly). Vectors should be dumped to .npz alongside
every run so later comparisons never need a rerun. `compare` wraps both tests over a family of
metrics into a tidy JSON-ready dict, with seen/cold stratified deltas when a cold mask is given.

All routines are vectorized numpy; 10k resamples over 22k users run in seconds.
"""
from __future__ import annotations

import numpy as np

from . import protocol

_CHUNK = 500  # bootstrap/randomization chunk size (bounds memory at ~chunk*n_users indices)


def per_user_metrics(rank_fn, split: protocol.LOOSplit, ks=(5, 10), which: str = "test",
                     exclude_history: bool = False) -> dict:
    """Per-user binary hit and DCG vectors under the canonical protocol.

    Returns {"HR@k": float64[n_users], "NDCG@k": float64[n_users], ...} plus, for `test`,
    "cold_mask": bool[n_users]. Means of these vectors equal `protocol.evaluate`'s (unrounded)
    aggregates; the vectors are what `paired_bootstrap` / `paired_randomization` consume.
    (With a single relevant item per user, IDCG=1, so per-user DCG == NDCG.)
    """
    data = getattr(split, which)
    n = len(data)
    out = {f"HR@{k}": np.zeros(n) for k in ks}
    out.update({f"NDCG@{k}": np.zeros(n) for k in ks})
    for idx, (hist, tgt) in enumerate(data):
        excl = set(hist) if exclude_history else set()
        ranked = rank_fn(hist, excl)
        for k in ks:
            h, g = protocol.dcg_hit(ranked, tgt, k)
            out[f"HR@{k}"][idx] = h
            out[f"NDCG@{k}"][idx] = g
    if which == "test":
        out["cold_mask"] = np.array([i in split.cold_test_idx for i in range(n)], dtype=bool)
    return out


def paired_bootstrap(a, b, n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired bootstrap over users for mean(a) - mean(b).

    a, b: per-user metric vectors for the SAME users in the SAME order.
    Returns {"mean_delta", "ci95": [lo, hi], "p_gt_zero", "n_users", "n_boot"}.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1, "paired vectors must be same-length 1-D"
    d = a - b
    n = d.size
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for lo in range(0, n_boot, _CHUNK):
        hi = min(lo + _CHUNK, n_boot)
        idx = rng.integers(0, n, size=(hi - lo, n))
        means[lo:hi] = d[idx].mean(axis=1)
    lo_ci, hi_ci = np.percentile(means, [2.5, 97.5])
    return {
        "mean_delta": float(d.mean()),
        "ci95": [float(lo_ci), float(hi_ci)],
        "p_gt_zero": float((means > 0).mean()),
        "n_users": int(n),
        "n_boot": int(n_boot),
    }


def paired_randomization(a, b, n_perm: int = 10000, seed: int = 0) -> dict:
    """Paired sign-flip randomization test; two-sided p-value for mean(a) - mean(b) != 0.

    Use for close results (|delta| < ~0.002) where the bootstrap CI is too coarse to call.
    p uses the add-one estimator (never exactly 0).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1, "paired vectors must be same-length 1-D"
    d = a - b
    n = d.size
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    n_ge = 0
    for lo in range(0, n_perm, _CHUNK):
        hi = min(lo + _CHUNK, n_perm)
        signs = rng.integers(0, 2, size=(hi - lo, n)) * 2 - 1
        n_ge += int((np.abs((signs * d).mean(axis=1)) >= obs - 1e-15).sum())
    return {
        "mean_delta": float(d.mean()),
        "p_two_sided": float((n_ge + 1) / (n_perm + 1)),
        "n_users": int(n),
        "n_perm": int(n_perm),
    }


def compare(system_a_vectors: dict, system_b_vectors: dict, metrics=("HR@5", "HR@10", "NDCG@5", "NDCG@10"),
            cold_mask=None, n_boot: int = 10000, n_perm: int = 10000, seed: int = 0) -> dict:
    """Tidy JSON-ready comparison of two systems' per-user vectors (A minus B).

    For each metric: means, delta, paired bootstrap (CI + p(delta>0)), sign-flip randomization
    p, and a verdict ("A" / "B" / "tie") from whether the 95% CI excludes 0. If `cold_mask`
    (bool per-user, True = cold target) is provided, seen/cold stratified deltas + CIs are added.
    """
    out = {}
    for m in metrics:
        a = np.asarray(system_a_vectors[m], dtype=np.float64)
        b = np.asarray(system_b_vectors[m], dtype=np.float64)
        boot = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)
        rand = paired_randomization(a, b, n_perm=n_perm, seed=seed)
        lo, hi = boot["ci95"]
        entry = {
            "mean_a": float(a.mean()),
            "mean_b": float(b.mean()),
            "delta": boot["mean_delta"],
            "bootstrap": boot,
            "randomization": rand,
            "significant_95": bool(lo > 0 or hi < 0),
            "verdict": "A" if lo > 0 else ("B" if hi < 0 else "tie"),
        }
        if cold_mask is not None:
            mask = np.asarray(cold_mask, dtype=bool)
            entry["stratified"] = {}
            for name, sel in (("seen", ~mask), ("cold", mask)):
                if sel.sum() == 0:
                    continue
                entry["stratified"][name] = paired_bootstrap(a[sel], b[sel],
                                                             n_boot=n_boot, seed=seed)
        out[m] = entry
    return out
