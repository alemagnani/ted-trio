"""A4 — CHN-DE: Counterfactual Hard-Negative Dual Encoder (GPU job J4a).

The deployable TWO-TOWER converter of the P3/P3b finding (DIR-2 ranking-architecture
constraint: NO cross-encoders — query vector and item vector are computed INDEPENDENTLY and
crossed only by cosine, ANN-servable). P3/P3b showed a diagnostic feature re-ranker reaches
test HR@10 0.0822 / NDCG@5 0.0338 out of popularity + same-brand + collab-z signal; CHN-DE
must absorb that signal into learned vectors:

  * ITEM TOWER — a learned per-item embedding table (d=128) initialized from a FROZEN linear
    projection of the sentence-t5-xl content embeddings. Per P13/P13b, callers pass the
    ZCA-WHITENED (shrinkage 0.1, item-side stats, ``semplan.spaces.zca_whiten``) L2-normalised
    embedding as ``content_emb`` — the frozen cold path then rides content_zca's geometry
    (cold HR@10 0.0926 vs 0.0617 raw). Items with zero train-interaction
    count keep the frozen projected-content vector forever (the P1/P3 cold-collapse fix:
    every trained/popularity-driven arm so far buries the 162 cold items; here cold items
    ride the content geometry untouched while warm items learn popularity/brand priors).
  * QUERY TOWER — a lightweight transformer encoder (2 layers, 2 heads, d=128) over the
    last-``max_len`` history ITEM VECTORS (shared item table, so query and item live in one
    space) plus a learned recency embedding; attention-pooled to a single query vector.
    v1 is single-vector; ``n_query_vectors`` is the hook for the URM-style K-vector
    multi-query upgrade (max/LSE pooling) and is asserted ==1 for now — DO NOT widen it
    without the sanctioned multi-vector design.
  * LOSS — InfoNCE at temperature 0.07 over cosine scores: in-batch negatives (duplicate
    positives masked) + per-query HARD negatives sampled from (a) the target's taxonomy
    region excluding the target, (b) precomputed dense near-misses (top blend-space
    candidates that are not the target — exactly the confounders P1 showed occupying the
    top ranks), (c) a base distribution (uniform / frequency / log-uniform over popularity
    rank) with an optional logQ sampled-softmax correction for the in-batch popularity bias.

Protocol safety mirrors ``region_planner.RegionPlanDataset``: training pairs are derived
per-user from the train-visible sequence ``T = s[:-2]`` and re-asserted to be exactly
``LOOSplit.train_pairs`` members; each pair carries a FORBIDDEN set = {its target, the
user's val target, the user's test target} that the hard-negative sampler must never emit.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from semplan.common.io import read_runs, write_jsonl
from semplan.common.run_schema import RunRecord
from semplan.eval.protocol import LOOSplit
from semplan.spaces import RECENCY, user_vec, zscore


# --------------------------------------------------------------------------- towers
class ItemTower(nn.Module):
    """Learned item table warm-started from (and cold-backed by) a frozen content projection.

    ``item_vectors()`` returns the L2-normalised [n_items, d] matrix: the LEARNED table row
    for items with train-interaction-count > 0, the FROZEN projected-content vector for cold
    items. Gradients only ever reach warm rows (``torch.where`` routes them), and the frozen
    projection is stored as a buffer so checkpoints reproduce the cold path exactly.
    """

    def __init__(self, content_emb: np.ndarray, warm_rows: set[int] | np.ndarray,
                 d: int = 128, seed: int = 42):
        super().__init__()
        n_items, content_dim = content_emb.shape
        self.d = d
        gen = torch.Generator().manual_seed(seed)
        # frozen content projection (the cold path); plain gaussian, deterministic
        self.content_proj = nn.Linear(content_dim, d, bias=False)
        with torch.no_grad():
            self.content_proj.weight.copy_(
                torch.randn(d, content_dim, generator=gen) / math.sqrt(content_dim))
        self.content_proj.weight.requires_grad_(False)
        with torch.no_grad():
            init = self.content_proj(torch.from_numpy(np.asarray(content_emb, np.float32)))
        self.table = nn.Embedding(n_items, d)
        with torch.no_grad():
            self.table.weight.copy_(init)
        warm = np.zeros(n_items, dtype=bool)
        warm[np.fromiter(warm_rows, dtype=np.int64)] = True
        self.register_buffer("warm", torch.from_numpy(warm))
        self.register_buffer("cold_vec", init.clone())      # frozen projected content

    def item_vectors(self, normalize: bool = True) -> torch.Tensor:
        v = torch.where(self.warm.unsqueeze(1), self.table.weight, self.cold_vec)
        return F.normalize(v, dim=-1) if normalize else v


class QueryTower(nn.Module):
    """Transformer encoder over history item vectors + learned recency embedding.

    Input positions are indexed by RECENCY (0 = most recent item, counting back), so the
    positional table is shared across history lengths. Output = attention-pooled single
    query vector, L2-normalised. ``n_query_vectors`` is the multi-query hook (URM-style
    K-vector max/LSE pooling) — v1 asserts it is 1; only the pooling head would widen.
    """

    def __init__(self, d: int = 128, n_layers: int = 2, n_heads: int = 2, d_ff: int = 256,
                 max_len: int = 20, dropout: float = 0.1, n_query_vectors: int = 1):
        super().__init__()
        assert n_query_vectors == 1, "multi-vector queries are a later, gated upgrade (v1)"
        self.n_query_vectors = n_query_vectors
        self.max_len = max_len
        self.rec_emb = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = nn.Linear(d, n_query_vectors)           # attention-pool logits per token

    def forward(self, item_vecs: torch.Tensor, hist_rows: torch.Tensor,
                hist_mask: torch.Tensor) -> torch.Tensor:
        """item_vecs [n_items, d] (shared table), hist_rows [B, L] (0-padded),
        hist_mask [B, L] bool -> [B, d] L2-normalised query vectors."""
        h = item_vecs[hist_rows]                                        # [B, L, d]
        lengths = hist_mask.sum(1, keepdim=True)                        # [B, 1]
        pos = torch.arange(hist_rows.shape[1], device=hist_rows.device).unsqueeze(0)
        recency = (lengths - 1 - pos).clamp(min=0, max=self.max_len - 1)
        h = h + self.rec_emb(recency)
        h = self.encoder(h, src_key_padding_mask=~hist_mask)
        att = self.pool(h).squeeze(-1).masked_fill(~hist_mask, float("-inf"))
        w = torch.softmax(att, dim=1).unsqueeze(-1)                     # [B, L, 1]
        q = (w * h).sum(1)                                              # [B, d]  (K=1)
        return F.normalize(q, dim=-1)


class CHNDE(nn.Module):
    """The full dual encoder: shared item table + query tower + InfoNCE loss."""

    def __init__(self, content_emb: np.ndarray, warm_rows, d: int = 128, n_layers: int = 2,
                 n_heads: int = 2, d_ff: int = 256, max_len: int = 20, dropout: float = 0.1,
                 tau: float = 0.07, seed: int = 42, n_query_vectors: int = 1):
        super().__init__()
        self.tau = tau
        self.config = {"d": d, "n_layers": n_layers, "n_heads": n_heads, "d_ff": d_ff,
                       "max_len": max_len, "dropout": dropout, "tau": tau, "seed": seed,
                       "n_query_vectors": n_query_vectors}
        self.item_tower = ItemTower(content_emb, warm_rows, d=d, seed=seed)
        self.query_tower = QueryTower(d=d, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
                                      max_len=max_len, dropout=dropout,
                                      n_query_vectors=n_query_vectors)

    def encode_queries(self, hist_rows: torch.Tensor, hist_mask: torch.Tensor,
                       item_vecs: torch.Tensor | None = None) -> torch.Tensor:
        if item_vecs is None:
            item_vecs = self.item_tower.item_vectors()
        return self.query_tower(item_vecs, hist_rows, hist_mask)

    def forward(self, hist_rows: torch.Tensor, hist_mask: torch.Tensor,
                tgt_rows: torch.Tensor, neg_rows: torch.Tensor,
                logq_in: torch.Tensor | None = None,
                logq_neg: torch.Tensor | None = None) -> torch.Tensor:
        """InfoNCE loss: positives on the diagonal of the in-batch block, negatives =
        in-batch (duplicate positives masked) + per-query hard negatives [B, n_neg].
        ``logq_*`` (optional) applies the sampled-softmax logQ correction: logits -= logQ.
        """
        V = self.item_tower.item_vectors()
        q = self.encode_queries(hist_rows, hist_mask, item_vecs=V)      # [B, d]
        pos = V[tgt_rows]                                               # [B, d]
        logits_in = q @ pos.T / self.tau                                # [B, B]
        if logq_in is not None:
            logits_in = logits_in - logq_in.unsqueeze(0)
        # mask accidental duplicates of the positive off the diagonal
        dup = (tgt_rows.unsqueeze(0) == tgt_rows.unsqueeze(1)) \
            & ~torch.eye(len(tgt_rows), dtype=torch.bool, device=tgt_rows.device)
        logits_in = logits_in.masked_fill(dup, float("-inf"))
        logits_neg = torch.einsum("bd,bnd->bn", q, V[neg_rows]) / self.tau
        if logq_neg is not None:
            logits_neg = logits_neg - logq_neg
        logits = torch.cat([logits_in, logits_neg], dim=1)
        labels = torch.arange(len(tgt_rows), device=tgt_rows.device)
        return F.cross_entropy(logits, labels)


# --------------------------------------------------------------------------- dataset
class ChnDeDataset(Dataset):
    """Protocol-safe (history -> target) pairs with per-pair forbidden-negative sets.

    Derived per-user from the train-visible sequence ``T = s[:-2]`` (exactly each user's
    VAL history under ``protocol.build``'s capping): pairs ``(T[:t], T[t])`` for
    t = 1..len(T)-1 — this reproduces ``split.train_pairs`` membership exactly (asserted,
    like ``region_planner.RegionPlanDataset``). ``forbidden[i]`` = {target, the user's val
    target, the user's test target(s)}: the hard-negative sampler must never emit these for
    pair i (a val/test target used as a *negative* would leak eval information).
    """

    def __init__(self, split: LOOSplit, max_len: int = 20, include_val: bool = False):
        self.max_len = max_len
        self.include_val = include_val
        test_by_hist: dict[tuple, list[int]] = {}
        for h, t in split.test:
            test_by_hist.setdefault(tuple(h), []).append(t)
        train_set = {(tuple(h), t) for h, t in split.train_pairs}
        self.examples: list[tuple[list[int], int]] = []
        self.forbidden: list[frozenset[int]] = []
        for T, vt in split.val:                              # T = s[:-2], vt = val target
            tts = test_by_hist.get(tuple(list(T) + [vt]), [])
            for t in range(1, len(T)):
                hist, tgt = list(T[:t]), T[t]
                assert (tuple(hist), tgt) in train_set, \
                    "derived pair not in protocol train_pairs"
                self.examples.append((hist, tgt))
                self.forbidden.append(frozenset([tgt, vt, *tts]))
            if include_val:
                # J7/I21 train-through-val: add the freshest transition (s[:-2] -> s[-2]).
                # Legal for the FINAL model (LOO retrain-on-train+val); val metrics of such
                # a model are CONTAMINATED -> selection must use the inner world instead.
                self.examples.append((list(T), vt))
                self.forbidden.append(frozenset([vt, *tts]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        hist, tgt = self.examples[i]
        return hist[-self.max_len:], tgt, i

    def collate(self, batch) -> dict:
        hist_rows, hist_mask = pad_histories([h for h, _t, _i in batch], self.max_len)
        return {"hist_rows": hist_rows, "hist_mask": hist_mask,
                "tgt_rows": torch.tensor([t for _h, t, _i in batch], dtype=torch.long),
                "idx": torch.tensor([i for _h, _t, i in batch], dtype=torch.long)}


def pad_histories(hists: list[list[int]], max_len: int = 20
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad row histories (capped to the last ``max_len``) -> (rows [B,L], mask [B,L])."""
    hists = [h[-max_len:] for h in hists]
    L = max(len(h) for h in hists)
    rows = torch.zeros((len(hists), L), dtype=torch.long)
    mask = torch.zeros((len(hists), L), dtype=torch.bool)
    for i, h in enumerate(hists):
        rows[i, : len(h)] = torch.tensor(h, dtype=torch.long)
        mask[i, : len(h)] = True
    return rows, mask


def train_popularity(split: LOOSplit) -> np.ndarray:
    """Item frequencies over the train-visible sequences s[:-2] (== val histories)."""
    counts = np.zeros(split.n_items, dtype=np.int64)
    for T, _vt in split.val:
        for r in T:
            counts[r] += 1
    return counts


# --------------------------------------------------------------------------- near-misses
def precompute_near_misses(dataset: ChnDeDataset, spaces: list[np.ndarray],
                           top_m: int = 50, recency: float | None = RECENCY,
                           chunk: int = 2048, device: str = "cpu") -> np.ndarray:
    """Per-pair dense near-misses: the top-``top_m`` blend-space candidates (E3c z-score-sum
    of cosines across ``spaces``, recency-weighted user vector) EXCLUDING the pair's
    forbidden set (target + the user's val/test targets). Returns int64 [n_pairs, top_m].
    """
    n = len(dataset)
    torch_spaces = [torch.from_numpy(np.asarray(sp, np.float32)).to(device) for sp in spaces]
    out = np.empty((n, top_m), dtype=np.int64)
    for s in range(0, n, chunk):
        idxs = range(s, min(s + chunk, n))
        total = None
        for sp, tsp in zip(spaces, torch_spaces):
            U = torch.from_numpy(np.stack(
                [user_vec(sp, dataset.examples[i][0], recency) for i in idxs])).to(device)
            sc = U @ tsp.T                                              # [c, n_items]
            sc = (sc - sc.mean(1, keepdim=True)) / sc.std(1, keepdim=True).clamp_min(1e-9)
            total = sc if total is None else total + sc
        # over-fetch so the (typically 3-element) forbidden filter still leaves top_m
        k = min(top_m + 16, total.shape[1])
        top = torch.topk(total, k, dim=1).indices.cpu().numpy()
        for j, i in enumerate(idxs):
            forb = dataset.forbidden[i]
            keep = [r for r in top[j].tolist() if r not in forb]
            assert len(keep) >= top_m, "forbidden set unexpectedly large; raise over-fetch"
            out[i] = keep[:top_m]
    return out


# --------------------------------------------------------------------------- sampler
class HardNegativeSampler:
    """Per-query hard negatives: taxonomy-region + dense near-miss + base distribution.

    ``mix = (p_tax, p_near, p_base)`` fractions of ``n_neg`` (shortfalls in the hard
    categories fall through to base). Base distribution flag: ``uniform`` | ``freq``
    (p ∝ train count + 1) | ``log-uniform`` (Zipf over popularity rank — the classic
    sampled-softmax approximation). Never emits a pair's forbidden rows (target, the user's
    val/test targets) — audited in tests per the P7 gate.
    """

    def __init__(self, dataset: ChnDeDataset, row_region: np.ndarray,
                 region_members: dict[int, np.ndarray], near_miss: np.ndarray,
                 pop_counts: np.ndarray,
                 mix: tuple[float, float, float] = (0.3, 0.4, 0.3),
                 base_dist: str = "log-uniform", seed: int = 42):
        assert base_dist in ("uniform", "freq", "log-uniform")
        assert abs(sum(mix) - 1.0) < 1e-6 and all(m >= 0 for m in mix)
        self.ds = dataset
        self.row_region = row_region
        self.members = region_members
        self.near_miss = near_miss
        self.mix = mix
        self.base_dist = base_dist
        self.rng = np.random.default_rng(seed)
        n_items = len(pop_counts)
        freq_p = (pop_counts + 1.0) / (pop_counts + 1.0).sum()
        if base_dist == "uniform":
            self.base_p = np.full(n_items, 1.0 / n_items)
        elif base_dist == "freq":
            self.base_p = freq_p
        else:                                   # log-uniform / Zipf over popularity rank
            order = np.argsort(-pop_counts, kind="stable")
            rank = np.empty(n_items, dtype=np.int64)
            rank[order] = np.arange(n_items)
            p = np.log((rank + 2.0) / (rank + 1.0))
            self.base_p = p / p.sum()
        # empirical target frequency — the logQ correction for the in-batch negatives
        self.freq_logq = np.log(freq_p).astype(np.float32)
        self.base_logq = np.log(self.base_p).astype(np.float32)

    def _base(self, n: int, forbidden: frozenset[int]) -> list[int]:
        out: list[int] = []
        while len(out) < n:
            cand = self.rng.choice(len(self.base_p), size=2 * (n - len(out)) + 4,
                                   p=self.base_p)
            out.extend(int(c) for c in cand if c not in forbidden)
        return out[:n]

    def sample(self, i: int, n_neg: int) -> np.ndarray:
        """n_neg hard negatives for pair i; never returns a forbidden row."""
        _hist, tgt = self.ds.examples[i]
        forbidden = self.ds.forbidden[i]
        n_tax = int(round(self.mix[0] * n_neg))
        n_near = int(round(self.mix[1] * n_neg))
        out: list[int] = []
        # (a) target's taxonomy region, excluding the target (and all forbidden rows)
        reg = int(self.row_region[tgt])
        if reg >= 0 and n_tax > 0:
            pool = self.members[reg]
            pool = pool[~np.isin(pool, list(forbidden))]
            if len(pool) > 0:
                take = self.rng.choice(pool, size=min(n_tax, len(pool)),
                                       replace=len(pool) < n_tax)
                out.extend(int(x) for x in take)
        # (b) dense near-misses (precomputed forbidden-free)
        if n_near > 0:
            row = self.near_miss[i]
            take = self.rng.choice(row, size=min(n_near, len(row)), replace=False)
            out.extend(int(x) for x in take)
        # (c) base distribution fills the remainder (incl. hard-category shortfalls)
        out.extend(self._base(n_neg - len(out), forbidden))
        return np.asarray(out[:n_neg], dtype=np.int64)

    def sample_batch(self, idxs: np.ndarray, n_neg: int) -> np.ndarray:
        return np.stack([self.sample(int(i), n_neg) for i in idxs])


# --------------------------------------------------------------------------- training
def train_chn_de(model: CHNDE, dataset: ChnDeDataset, sampler: HardNegativeSampler,
                 steps: int = 20000, batch_size: int = 256, n_neg: int = 16,
                 lr: float = 1e-3, weight_decay: float = 0.01, warmup_frac: float = 0.05,
                 logq: bool = False, device: str = "cpu", seed: int = 42,
                 ckpt_path: str | None = None, ckpt_every: int = 2000,
                 log_every: int = 50) -> list[float]:
    """AdamW + linear-warmup cosine schedule (mirrors ``train_planner``); returns per-step
    loss history. Batches are sampled with replacement from the pair set (index-level, so
    each pair keeps its own forbidden set / near-miss row)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    warmup = max(int(steps * warmup_frac), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    losses: list[float] = []
    for step in range(steps):
        idxs = rng.choice(len(dataset), size=min(batch_size, len(dataset)), replace=False)
        batch = dataset.collate([dataset[int(i)] for i in idxs])
        neg = sampler.sample_batch(idxs, n_neg)
        tgt = batch["tgt_rows"]
        kw = {}
        if logq:
            kw["logq_in"] = torch.from_numpy(sampler.freq_logq[tgt.numpy()]).to(device)
            kw["logq_neg"] = torch.from_numpy(sampler.base_logq[neg]).to(device)
        loss = model(batch["hist_rows"].to(device), batch["hist_mask"].to(device),
                     tgt.to(device), torch.from_numpy(neg).to(device), **kw)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if log_every and (step + 1) % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"[train_chn_de] step {step + 1}/{steps} loss {avg:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if ckpt_path and ckpt_every and (step + 1) % ckpt_every == 0:
            save_checkpoint(model, ckpt_path, step=step + 1)
    if ckpt_path:
        save_checkpoint(model, ckpt_path, step=steps)
    return losses


def save_checkpoint(model: CHNDE, path: str, step: int = 0) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config, "step": step}, path)


def load_checkpoint(path: str, content_emb: np.ndarray) -> CHNDE:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = CHNDE(content_emb, warm_rows=np.zeros(0, dtype=np.int64), **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])       # warm buffer restored from state_dict
    return model


# --------------------------------------------------------------------------- inference/dumps
@torch.no_grad()
def encode_split_queries(model: CHNDE, split: LOOSplit, which: str = "test",
                         batch_size: int = 512, device: str = "cpu") -> torch.Tensor:
    """Query vectors [n_queries, d] for every (history, target) of ``split.<which>``."""
    model.to(device).eval()
    V = model.item_tower.item_vectors()
    data = getattr(split, which)
    out = []
    for s in range(0, len(data), batch_size):
        hists = [h for h, _t in data[s: s + batch_size]]
        rows, mask = pad_histories(hists, model.config["max_len"])
        out.append(model.encode_queries(rows.to(device), mask.to(device),
                                        item_vecs=V).cpu())
    return torch.cat(out)


def topk_from_vectors(query_vecs: torch.Tensor, item_vecs: torch.Tensor, k: int = 100,
                      chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Exact full-corpus cosine top-k (both inputs L2-normalised): rows + scores [n_q, k]."""
    idxs, scores = [], []
    for s in range(0, len(query_vecs), chunk):
        sc = query_vecs[s: s + chunk] @ item_vecs.T
        top = torch.topk(sc, min(k, sc.shape[1]), dim=1)
        idxs.append(top.indices.numpy())
        scores.append(top.values.numpy())
    return np.concatenate(idxs), np.concatenate(scores)


def dump_run_records(query_vecs: torch.Tensor, item_vecs: torch.Tensor, split: LOOSplit,
                     which: str, path: str, system_name: str = "chn_de",
                     checkpoint: str = "", seed: int = 42, top_k: int = 100) -> int:
    """Dense top-``top_k`` per query -> RunRecord JSONL (item ids are the paper's 1-indexed
    ids, ``str(row + 1)``; query_id = ``<which>:<idx>``). Returns records written."""
    data = getattr(split, which)
    idx, sc = topk_from_vectors(query_vecs, item_vecs, k=top_k)
    recs = (RunRecord(
        query_id=f"{which}:{i}", query_text="",
        relevant_item_ids=[str(tgt + 1)],
        retrieved_item_ids=[str(r + 1) for r in idx[i].tolist()],
        retrieved_scores=[float(x) for x in sc[i].tolist()],
        system_name=system_name, dataset="amazon_beauty", checkpoint=checkpoint, seed=seed)
        for i, (_h, tgt) in enumerate(data))
    return write_jsonl(recs, path)


def rank_fn_from_vectors(query_vecs: torch.Tensor, item_vecs: torch.Tensor,
                         split: LOOSplit, which: str, top_k: int = 100):
    """protocol.evaluate rank_fn scoring the SAME top-k lists the dump writes (query
    identity resolved positionally — protocol.evaluate iterates ``split.<which>`` in order)."""
    idx, _sc = topk_from_vectors(query_vecs, item_vecs, k=top_k)
    it = iter(range(len(idx)))

    def rank_fn(_hist, _excl):
        return idx[next(it)].tolist()
    return rank_fn


def rank_fn_from_run_records(path: str, which: str):
    """protocol.evaluate rank_fn replaying a RunRecord JSONL dump (positional query ids)."""
    ranked = {r.query_id: [int(x) - 1 for x in r.retrieved_item_ids]
              for r in read_runs(path)}
    it = iter(range(len(ranked)))

    def rank_fn(_hist, _excl):
        return ranked[f"{which}:{next(it)}"]
    return rank_fn
