"""J5 (I16/I17) — query-only tower over a FROZEN item embedding matrix.

The item side is a fixed, externally-produced embedding table (openemb-large stand-in now;
the commercial matrix drops into the same slot later) — NEVER trained. Only the query
tower learns: history item vectors -> transformer -> K pooled query vectors projected
back into the ITEM space, L2-normalised. Serving score is ``max_k  q_k . e_i`` (K=1 is
plain cosine), so the system stays two-tower/ANN-servable per DIR-2 (K independent ANN
probes, max-merge).

Training loss is FULL-softmax cross-entropy over all n_items (12k items make this
CPU-feasible; no negative sampling, no sampling-bias correction needed). Two knobs:
  forbidden masking   each train pair's {val target, test target(s)} columns are masked
                      to -inf (a val/test target acting as a softmax NEGATIVE would leak
                      eval information — same rule as chn_de's forbidden sets).
  logQ                optional popularity absorption: logits -= log(train freq) during
                      TRAINING only; serving uses raw max-IP. This is the learned
                      counterpart of the (rejected, P19) static pop prior.

``n_query_vectors`` K>1 is the I17/J5b multi-query arm (I10 max-IP semantics): the loss
itself takes the max over K before the softmax, so each vector specialises on the
targets it already claims (winner-take-all routing).
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .chn_de import ChnDeDataset, pad_histories  # noqa: F401  (re-exported for callers)


class FrozenQueryTower(nn.Module):
    """Trained query tower over a frozen item matrix; scores are max-IP over K vectors."""

    def __init__(self, item_emb: np.ndarray, d: int = 256, n_layers: int = 2,
                 n_heads: int = 4, d_ff: int = 512, max_len: int = 20,
                 dropout: float = 0.1, tau: float = 0.05, seed: int = 42,
                 n_query_vectors: int = 1, input_emb: np.ndarray | None = None):
        super().__init__()
        torch.manual_seed(seed)
        item = torch.from_numpy(np.asarray(item_emb, np.float32))
        item = F.normalize(item, dim=-1)
        self.register_buffer("item", item)                  # [n_items, D] FROZEN
        n_items, D = item.shape
        # I24: optional SEPARATE frozen matrix for the query-tower INPUT tokens (e.g.
        # [own emb ; top-3-predecessor centroid]); scoring stays on ``item``. Both are
        # frozen + external, so ANN-servability is untouched.
        if input_emb is None:
            inp = item
        else:
            inp = torch.from_numpy(np.asarray(input_emb, np.float32))
        self.register_buffer("input_item", inp)             # [n_items, D_in] FROZEN
        D_in = inp.shape[1]
        self.tau = tau
        self.k = n_query_vectors
        self.max_len = max_len
        self.config = {"d": d, "n_layers": n_layers, "n_heads": n_heads, "d_ff": d_ff,
                       "max_len": max_len, "dropout": dropout, "tau": tau, "seed": seed,
                       "n_query_vectors": n_query_vectors, "item_dim": D,
                       "n_items": n_items, "input_dim": D_in}
        self.in_proj = nn.Linear(D_in, d)
        self.rec_emb = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = nn.Linear(d, n_query_vectors)           # K attention-pool heads
        self.out_proj = nn.Linear(d, D)

    def encode_queries(self, hist_rows: torch.Tensor, hist_mask: torch.Tensor
                       ) -> torch.Tensor:
        """[B, L] histories -> [B, K, D] L2-normalised query vectors in the ITEM space."""
        h = self.in_proj(self.input_item[hist_rows])                    # [B, L, d]
        lengths = hist_mask.sum(1, keepdim=True)
        pos = torch.arange(hist_rows.shape[1], device=hist_rows.device).unsqueeze(0)
        recency = (lengths - 1 - pos).clamp(min=0, max=self.max_len - 1)
        h = h + self.rec_emb(recency)
        h = self.encoder(h, src_key_padding_mask=~hist_mask)
        att = self.pool(h).masked_fill(~hist_mask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(att, dim=1)                                   # [B, L, K]
        q = torch.einsum("blk,bld->bkd", w, h)                          # [B, K, d]
        return F.normalize(self.out_proj(q), dim=-1)                    # [B, K, D]

    def full_logits(self, q: torch.Tensor) -> torch.Tensor:
        """[B, K, D] queries -> [B, n_items] max-IP logits (pre-temperature)."""
        s = torch.einsum("bkd,nd->bkn", q, self.item)                   # [B, K, n]
        return s.max(dim=1).values                                      # [B, n]

    def forward(self, hist_rows: torch.Tensor, hist_mask: torch.Tensor,
                tgt_rows: torch.Tensor, forbidden_cols: torch.Tensor | None = None,
                logq: torch.Tensor | None = None,
                band_cols: torch.Tensor | None = None,
                band_margin: float = 0.05, band_weight: float = 0.2,
                distill_cols: torch.Tensor | None = None,
                distill_scores: torch.Tensor | None = None,
                distill_T: float = 0.07, distill_weight: float = 0.5) -> torch.Tensor:
        """Full-softmax CE. ``forbidden_cols`` [B, F] (-1 padded) columns are masked to
        -inf (never a forbidden col == the target; the dataset guarantees that).
        ``logq`` [n_items] is subtracted from ALL logits (popularity absorption).
        ``band_cols`` [B, W] (-1 padded, I31): per-example REFERENCE rank-band wrong
        items; adds a hinge ``band_weight * relu(band_margin - s(pos) + max s(neg))``
        on RAW cosine scores (pre-temperature) against the current-highest-scoring
        band member. Query sampling and the CE objective are untouched.
        ``distill_cols``/``distill_scores`` [B, C] (I32): per-example fixed teacher
        candidate ids + teacher scores (trio mean-z); replaces the loss with
        ``(1-w)*CE + w*KL(student cand logits/T || teacher scores/T)``."""
        q = self.encode_queries(hist_rows, hist_mask)
        raw = self.full_logits(q)                                       # [B, n] cosines
        logits = raw / self.tau
        if logq is not None:
            logits = logits - logq.unsqueeze(0)
        if forbidden_cols is not None:
            rows = torch.arange(len(logits), device=logits.device
                                ).unsqueeze(1).expand_as(forbidden_cols)
            valid = forbidden_cols >= 0
            logits[rows[valid], forbidden_cols[valid]] = float("-inf")
        loss = F.cross_entropy(logits, tgt_rows)
        if band_cols is not None:
            neg = raw.gather(1, band_cols.clamp(min=0))                 # [B, W]
            neg = neg.masked_fill(band_cols < 0, float("-inf")).max(dim=1).values
            pos = raw.gather(1, tgt_rows.unsqueeze(1)).squeeze(1)
            hinge = F.relu(band_margin - pos + neg)
            hinge = hinge[torch.isfinite(neg)]                          # all-pad rows out
            if hinge.numel():
                loss = loss + band_weight * hinge.mean()
        if distill_cols is not None:
            s_log = F.log_softmax(raw.gather(1, distill_cols) / distill_T, dim=1)
            t_log = F.log_softmax(distill_scores / distill_T, dim=1)
            kl = F.kl_div(s_log, t_log, log_target=True, reduction="batchmean")
            loss = (1.0 - distill_weight) * loss + distill_weight * kl
        return loss


def collate_with_forbidden(dataset: ChnDeDataset, idxs: np.ndarray, max_len: int
                           ) -> dict:
    """chn_de-style batch + the forbidden softmax columns [B, F] (-1 padded, target
    itself removed — it must stay a live positive class)."""
    batch = [dataset[int(i)] for i in idxs]
    hist_rows, hist_mask = pad_histories([h for h, _t, _i in batch], max_len)
    tgt = torch.tensor([t for _h, t, _i in batch], dtype=torch.long)
    forb = [sorted(dataset.forbidden[int(i)] - {t})
            for (_h, t, _i), i in zip(batch, idxs)]
    width = max((len(f) for f in forb), default=0)
    cols = torch.full((len(batch), max(width, 1)), -1, dtype=torch.long)
    for r, f in enumerate(forb):
        if f:
            cols[r, : len(f)] = torch.tensor(f, dtype=torch.long)
    return {"hist_rows": hist_rows, "hist_mask": hist_mask, "tgt_rows": tgt,
            "forbidden_cols": cols}


@torch.no_grad()
def mine_band_negs(model: FrozenQueryTower, dataset: ChnDeDataset, top_k: int = 50,
                   device: str = "cpu", chunk: int = 512) -> np.ndarray:
    """I31: per training pair, the reference model's top-``top_k`` WRONG items (target +
    forbidden cols masked out), rank-sorted. Slicing [4:20] / [19:50] at train time
    yields the curriculum bands 5-20 / 20-50."""
    was_training = model.training
    model.to(device).eval()
    out = np.zeros((len(dataset), top_k), dtype=np.int64)
    for s in range(0, len(dataset), chunk):
        idxs = np.arange(s, min(s + chunk, len(dataset)))
        b = collate_with_forbidden(dataset, idxs, model.max_len)
        q = model.encode_queries(b["hist_rows"].to(device), b["hist_mask"].to(device))
        sc = model.full_logits(q)                                       # [B, n]
        rows = torch.arange(len(sc), device=sc.device
                            ).unsqueeze(1).expand_as(b["forbidden_cols"])
        valid = b["forbidden_cols"] >= 0
        sc[rows[valid].to(sc.device),
           b["forbidden_cols"][valid].to(sc.device)] = float("-inf")
        sc.scatter_(1, b["tgt_rows"].unsqueeze(1).to(sc.device), float("-inf"))
        out[s: s + chunk] = sc.topk(top_k, dim=1).indices.cpu().numpy()
    if was_training:
        model.train()
    return out


@torch.no_grad()
def mine_distill_targets(models: list[FrozenQueryTower], dataset: ChnDeDataset,
                         n_cand: int = 256, device: str = "cpu", chunk: int = 512
                         ) -> tuple[np.ndarray, np.ndarray]:
    """I32: per training pair, the TEACHER-TRIO mean-z scores over a fixed top-``n_cand``
    candidate set. Per teacher: raw max-IP cosines z-scored per row; mean over teachers;
    forbidden cols -inf; top-``n_cand`` by mean-z with the TARGET forced into slot 0.
    Returns (cand_ids [N, C] int64, teacher_scores [N, C] float32 mean-z at the cands)."""
    states = [m.training for m in models]
    for m in models:
        m.to(device).eval()
    n = len(dataset)
    cand = np.zeros((n, n_cand), dtype=np.int64)
    tsc = np.zeros((n, n_cand), dtype=np.float32)
    for s in range(0, n, chunk):
        idxs = np.arange(s, min(s + chunk, n))
        b = collate_with_forbidden(dataset, idxs, models[0].max_len)
        zsum = None
        for m in models:
            q = m.encode_queries(b["hist_rows"].to(device), b["hist_mask"].to(device))
            sc = m.full_logits(q)                                       # [B, n_items]
            z = (sc - sc.mean(dim=1, keepdim=True)) / sc.std(dim=1, keepdim=True
                                                             ).clamp(min=1e-9)
            zsum = z if zsum is None else zsum + z
        mz = zsum / len(models)
        rows = torch.arange(len(mz), device=mz.device
                            ).unsqueeze(1).expand_as(b["forbidden_cols"])
        valid = b["forbidden_cols"] >= 0
        mz[rows[valid].to(mz.device),
           b["forbidden_cols"][valid].to(mz.device)] = float("-inf")
        tgt = b["tgt_rows"].to(mz.device)
        tgt_sc = mz.gather(1, tgt.unsqueeze(1))                         # keep target score
        mz.scatter_(1, tgt.unsqueeze(1), float("-inf"))                 # dedup vs top-k
        top = mz.topk(n_cand - 1, dim=1)
        cand[s: s + chunk, 0] = tgt.cpu().numpy()
        cand[s: s + chunk, 1:] = top.indices.cpu().numpy()
        tsc[s: s + chunk, 0] = tgt_sc.squeeze(1).cpu().numpy()
        tsc[s: s + chunk, 1:] = top.values.cpu().numpy()
    for m, was in zip(models, states):
        if was:
            m.train()
    return cand, tsc


def train_frozen_query(model: FrozenQueryTower, dataset: ChnDeDataset,
                       steps: int = 4000, batch_size: int = 256, lr: float = 1e-3,
                       weight_decay: float = 0.01, warmup_frac: float = 0.05,
                       logq: np.ndarray | None = None, device: str = "cpu",
                       seed: int = 42, ckpt_path: str | None = None,
                       ckpt_every: int = 1000, log_every: int = 50,
                       sample_p: np.ndarray | None = None,
                       band_negs: np.ndarray | None = None,
                       band_margin: float = 0.05, band_weight: float = 0.2,
                       band_refresh: bool = True,
                       distill: tuple[np.ndarray, np.ndarray] | None = None,
                       distill_T: float = 0.07,
                       distill_weight: float = 0.5) -> list[float]:
    """AdamW + linear-warmup cosine (mirrors ``train_chn_de``); returns loss history.

    ``sample_p`` [len(dataset)] (I27 boosting): batch examples are drawn with these
    probabilities instead of uniformly — upweights pairs a frozen reference ensemble
    fails on. The loss itself is unchanged.

    ``band_negs`` [len(dataset), 50] (I31 rank-band margin curriculum): reference
    top-50 wrong items per pair (``mine_band_negs``). Fixed schedule, NO sweep:
    steps [0, 40%) CE only; [40%, 70%) hinge vs band ranks 20-50 (cols 19:50);
    [70%, 100%] hinge vs band ranks 5-20 (cols 4:20). At the 70% boundary the bands
    are re-mined from the CURRENT model when ``band_refresh``. Sampling stays
    uniform — this is the anti-I27 design.

    ``distill`` (cand_ids [len(dataset), C], teacher_scores [len(dataset), C])
    (I32 trio→single distillation, ``mine_distill_targets``): every step the loss
    becomes ``(1-distill_weight)*CE + distill_weight*KL`` over the fixed teacher
    candidate set at temperature ``distill_T``. Teachers are FROZEN (pre-mined)."""
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
    logq_t = None if logq is None else torch.from_numpy(
        np.asarray(logq, np.float32)).to(device)
    if sample_p is not None:
        sample_p = np.asarray(sample_p, np.float64)
        assert len(sample_p) == len(dataset) and (sample_p >= 0).all()
        sample_p = sample_p / sample_p.sum()
    if band_negs is not None:
        band_negs = np.asarray(band_negs, np.int64)
        assert band_negs.shape == (len(dataset), 50), band_negs.shape
    d_cand = d_sc = None
    if distill is not None:
        d_cand = np.asarray(distill[0], np.int64)
        d_sc = np.asarray(distill[1], np.float32)
        assert d_cand.shape == d_sc.shape and len(d_cand) == len(dataset), \
            (d_cand.shape, d_sc.shape, len(dataset))
    b1, b2 = int(steps * 0.4), int(steps * 0.7)                 # I31 phase boundaries
    losses: list[float] = []
    for step in range(steps):
        if band_negs is not None and band_refresh and step == b2:
            print(f"[train_frozen_query] I31 refresh: re-mining bands at step {step}",
                  flush=True)
            band_negs = mine_band_negs(model, dataset, device=device)
            model.train()
        idxs = rng.choice(len(dataset), size=min(batch_size, len(dataset)),
                          replace=False, p=sample_p)
        b = collate_with_forbidden(dataset, idxs, model.max_len)
        band_cols = None
        if band_negs is not None and step >= b1:
            lo, hi = (19, 50) if step < b2 else (4, 20)
            band_cols = torch.from_numpy(band_negs[idxs, lo:hi]).to(device)
        dc = ds_ = None
        if d_cand is not None:
            dc = torch.from_numpy(d_cand[idxs]).to(device)
            ds_ = torch.from_numpy(d_sc[idxs]).to(device)
        loss = model(b["hist_rows"].to(device), b["hist_mask"].to(device),
                     b["tgt_rows"].to(device), b["forbidden_cols"].to(device),
                     logq=logq_t, band_cols=band_cols,
                     band_margin=band_margin, band_weight=band_weight,
                     distill_cols=dc, distill_scores=ds_,
                     distill_T=distill_T, distill_weight=distill_weight)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if log_every and (step + 1) % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"[train_frozen_query] step {step + 1}/{steps} loss {avg:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if ckpt_path and ckpt_every and (step + 1) % ckpt_every == 0:
            save_checkpoint(model, ckpt_path, step=step + 1)
    if ckpt_path:
        save_checkpoint(model, ckpt_path, step=steps)
    return losses


def save_checkpoint(model: FrozenQueryTower, path: str, step: int = 0) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sd = {k: v for k, v in model.state_dict().items()
          if k not in ("item", "input_item")}  # frozen, external
    torch.save({"state_dict": sd, "config": model.config, "step": step}, path)


def load_checkpoint(path: str, item_emb: np.ndarray,
                    input_emb: np.ndarray | None = None) -> FrozenQueryTower:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = {k: v for k, v in ckpt["config"].items()
           if k not in ("item_dim", "n_items", "input_dim")}
    model = FrozenQueryTower(item_emb, input_emb=input_emb, **cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model


@torch.no_grad()
def encode_split_queries(model: FrozenQueryTower, split, which: str = "test",
                         batch_size: int = 512, device: str = "cpu") -> torch.Tensor:
    """[n_queries, K, D] query vectors for every (history, target) of ``split.<which>``."""
    model.to(device).eval()
    data = getattr(split, which)
    out = []
    for s in range(0, len(data), batch_size):
        hists = [h for h, _t in data[s: s + batch_size]]
        rows, mask = pad_histories(hists, model.max_len)
        out.append(model.encode_queries(rows.to(device), mask.to(device)).cpu())
    return torch.cat(out)


@torch.no_grad()
def topk_maxip(query_vecs: torch.Tensor, item_vecs: torch.Tensor, k: int = 100,
               chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Exact full-corpus max-IP top-k for [n_q, K, D] queries: rows + scores [n_q, k]."""
    idxs, scores = [], []
    for s in range(0, len(query_vecs), chunk):
        sc = torch.einsum("bkd,nd->bkn", query_vecs[s: s + chunk], item_vecs
                          ).max(dim=1).values
        top = torch.topk(sc, min(k, sc.shape[1]), dim=1)
        idxs.append(top.indices.numpy())
        scores.append(top.values.numpy())
    return np.concatenate(idxs), np.concatenate(scores)


def rank_fn_from_queries(query_vecs: torch.Tensor, item_vecs: torch.Tensor,
                         top_k: int = 100):
    """protocol.evaluate rank_fn over precomputed max-IP top-k (positional queries)."""
    idx, _sc = topk_maxip(query_vecs, item_vecs, k=top_k)
    it = iter(range(len(idx)))

    def rank_fn(_hist, _excl):
        return idx[next(it)].tolist()
    return rank_fn
