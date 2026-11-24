"""I41 shared plumbing — STAR-static arm, transition diagnostics, masked eval (CPU).

Building blocks shared by i41_star_arms.py / i41_diag_buckets.py / i41_masked_eval.py:
  * sparse binary item-user collab cosine (STAR recipe, Brief 5/5):
      cos_C(a, b) = |U_a n U_b| / sqrt(|U_a| |U_b|), TRAIN interactions only,
      computed via scipy CSR row-normalisation (no dense item x item matrix).
  * STAR-static exponential-decay scoring over the last l history items:
      score(x) = sum_{j=1..l} lambda^j * [a*cosE(x, h_j) + (1-a)*cosC(x, h_j)].
  * z-score fusion, train transition mining + MQSA-TED Fig.5 buckets.
  * trio tower scoring (same checkpoints/machinery as the i23/i24/i30/i35 gates).
  * a deterministic tiny smoke fixture (synthetic seqs + random-init towers) so every
    i41 script runs end-to-end on CPU without touching real artifacts.
All metrics flow through semplan.eval.protocol (evaluate / dcg_hit); nothing here
computes an HR/NDCG on its own.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.dense.frozen_query import (  # noqa: E402
    FrozenQueryTower, encode_split_queries, load_checkpoint, save_checkpoint,
)
from semplan.eval import protocol  # noqa: E402

SEEDS = (42, 43, 44)
KS = (5, 10)
BUCKETS = ("0", "1", "2-4", "5+")


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / max(float(s.std()), 1e-9)


def load_matrix(path) -> np.ndarray:
    """Load an item-embedding matrix from .pt / .npy / .npz (first array in the npz)."""
    p = str(path)
    if p.endswith(".pt"):
        return torch.load(p, map_location="cpu", weights_only=True).float().numpy()
    if p.endswith(".npz"):
        z = np.load(p)
        return np.asarray(z[list(z.files)[0]], np.float32)
    return np.asarray(np.load(p), np.float32)


def train_seqs_of(split) -> list[list[int]]:
    """TRAIN interactions only = the inner val histories (s[:-2] per user), the same
    convention the gates use for the collab space — never touches val/test targets."""
    return [list(h) for h, _t in split.val]


def build_collab_rows(train_seqs: list[list[int]], n_items: int) -> sparse.csr_matrix:
    """Binary item x user CSR, rows scaled by 1/sqrt(|U_i|), from TRAIN only.
    Then cos_C(a, b) = (Mn @ Mn.T)[a, b] exactly (STAR's normalized co-occurrence)."""
    rows, cols = [], []
    for u, seq in enumerate(train_seqs):
        for i in set(seq):
            rows.append(i)
            cols.append(u)
    m = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(n_items, max(len(train_seqs), 1)))
    deg = np.asarray(m.sum(axis=1)).ravel()
    inv = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    return (sparse.diags(inv) @ m).tocsr()


def collab_cos(mn: sparse.csr_matrix, items: list[int]) -> np.ndarray:
    """[n_items, len(items)] collab cosines of every item vs each of ``items``."""
    return np.asarray((mn @ mn[items].T).toarray(), np.float32)


def star_scores(cos_e: np.ndarray, cos_c: np.ndarray, lam: float, l: int,
                a: float) -> np.ndarray:
    """STAR-static: columns of cos_e/cos_c are MOST-RECENT-FIRST history cosines."""
    m = min(l, cos_e.shape[1])
    w = (lam ** np.arange(1, m + 1)).astype(np.float32)
    return (a * cos_e[:, :m] + (1.0 - a) * cos_c[:, :m]) @ w


def zfuse(base_z: np.ndarray, arm: np.ndarray, beta: float) -> np.ndarray:
    """Convex z-fusion (gate convention): (1-beta)*base_z + beta*z(arm)."""
    return (1.0 - beta) * base_z + beta * zscore(arm)


def mine_transitions(train_seqs: list[list[int]]) -> Counter:
    """Directed consecutive transition counts (p -> i) inside TRAIN sequences only."""
    c: Counter = Counter()
    for s in train_seqs:
        for p, i in zip(s, s[1:]):
            c[(p, i)] += 1
    return c


def transition_csr(trans: Counter, n_items: int) -> sparse.csr_matrix:
    """[n_items, n_items] CSR with A[p, i] = count(p -> i)."""
    if not trans:
        return sparse.csr_matrix((n_items, n_items), dtype=np.float32)
    rows, cols, vals = zip(*[(p, i, c) for (p, i), c in trans.items()])
    return sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                             shape=(n_items, n_items))


def transition_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    return "5+"


def topk_ids(sc: np.ndarray, k: int = 10) -> list[int]:
    """Deterministic best-first top-k item rows of a full score vector."""
    k = min(k, len(sc))
    part = np.argpartition(-sc, k - 1)[:k]
    return part[np.argsort(-sc[part], kind="stable")].tolist()


def load_trio(ckpts, item_l2: np.ndarray, input_emb: np.ndarray | None = None) -> dict:
    return {str(s): load_checkpoint(str(p), item_l2, input_emb=input_emb)
            for s, p in zip(SEEDS, ckpts)}


def encode_trio(models: dict, split, which: str) -> dict:
    return {nm: encode_split_queries(m, split, which=which, device="cpu")
            for nm, m in models.items()}


def trio_z(qs: dict, qi: int, item_t: torch.Tensor) -> np.ndarray:
    """Mean of per-query z-scored K4 max-IP tower scores (gate ensemble, w=1.0 arm)."""
    return np.mean([zscore(torch.einsum("kd,nd->kn", qs[nm][qi], item_t)
                           .max(dim=0).values.numpy()) for nm in qs], axis=0)


def eval_ranked_lists(split, which: str, lists: list[list[int]], ks=KS,
                      exclude_history: bool = False) -> dict:
    """protocol.evaluate over precomputed best-first ranked lists (positional).
    With exclude_history the user's seen items are filtered out of the list, so the
    lists must be >= max(ks) long AFTER filtering (pass a deep-enough pool)."""
    it = iter(lists)
    kmax = max(ks)

    def rank_fn(_hist, excl):
        r = next(it)
        return [i for i in r if i not in excl][:kmax] if excl else list(r)[:kmax]

    return protocol.evaluate(rank_fn, split, ks=ks, which=which,
                             exclude_history=exclude_history)


def make_smoke_fixture(root: Path, n_items: int = 60, n_users: int = 48,
                       d: int = 16, seed: int = 0) -> dict:
    """Deterministic tiny fixture mirroring the real layout: LIGER-style seq txt,
    item/input embedding .npy, and 3 random-init (untrained) tower checkpoints —
    enough to exercise every i41 code path on CPU. Rows 6->7 (ids 7->8) form a
    frequent train transition so the Fig.5 buckets are non-degenerate."""
    root = Path(root)
    fix = root / "fixture"
    fix.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    seq_p = fix / "smoke_inner.txt"
    item_p, input_p = fix / "item_emb.npy", fix / "input_emb.npy"
    if not seq_p.exists():
        lines = []
        pool = np.setdiff1d(np.arange(n_items), [6, 7])
        for u in range(n_users):
            length = int(rng.integers(5, 13))
            its = (rng.choice(pool, size=length, replace=False) + 1).tolist()
            if u % 3 == 0:
                its = [7, 8] + its          # frequent (7->8) inside train part
            elif u % 4 == 0:
                its = its + [7, 8]          # ... and as the final (hist->target) pair
            # (elif keeps items unique per user: a repeated target would be masked
            #  away by seen-item masking and rightly break the smoke delta check)
            lines.append(" ".join([str(u + 1)] + [str(i) for i in its]))
        seq_p.write_text("\n".join(lines) + "\n")
        item = rng.standard_normal((n_items, d)).astype(np.float32)
        np.save(item_p, item)
        np.save(input_p, np.concatenate([item, item[::-1]], axis=1))
    ckpts = []
    for s in SEEDS:
        p = fix / f"s{s}" / "j5.pt"
        if not p.exists():
            item = np.load(item_p)
            tower = FrozenQueryTower(l2(item), d=16, n_layers=1, n_heads=2, d_ff=32,
                                     max_len=20, seed=s, n_query_vectors=2,
                                     input_emb=np.load(input_p))
            save_checkpoint(tower, str(p))
        ckpts.append(p)
    return {"seq": seq_p, "item_emb": item_p, "input_emb": input_p,
            "n_items": n_items, "ckpts": ckpts}
