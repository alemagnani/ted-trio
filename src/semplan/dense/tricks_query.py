"""I40 — four independently-flaggable training upgrades to the frozen-item query tower.

Each trick is OFF by default; all are composable; NONE changes the serving form:
the tower still emits query vectors scored by dot product against the FROZEN catalog
(two-tower / ANN-servable per DIR-2).

  T1 DuoRec-style contrastive regularizer (loss-level; lives in ``train_tricks``):
     ``nce_mode='full'``: two extra forwards per batch — a fresh-dropout view of the
     same sequences and a same-TARGET peer sequence per example (precomputed inverted
     index; no-peer examples fall back to their own dropout view) — symmetric InfoNCE
     between the two views with in-batch negatives, EXCLUDING any in-batch example
     that shares the anchor's target from the negative set. ``nce_mode='scl'``: the
     main forward's query vs the peer view only (single extra forward).
     Separate temperature ``nce_tau`` (default 1.0) from the CE tau.
     Loss = CE + nce_lambda * NCE. Wang-Isola uniformity of the (L2-normalised)
     query outputs is logged periodically as the collapse diagnostic.

  T2 TED soft-label smoothing (loss-level): forward directed adjacency counts
     A[i,j] = #(i followed by j at gap 1) mined from the TRAIN sequences with the
     i24 leak rule (user id dropped, final held-out event dropped). Teacher for an
     example whose LAST history item is h: softmax( A[h,:]/max(A[h,:]) / ted_tau )
     restricted to the NONZERO support (sparse — the dense |I|x|I| matrix is never
     materialised). Training target = (1-ted_beta)*onehot(next) + ted_beta*teacher.
     Recommended ted_beta 0.1, ted_tau 0.1 (MQSA-TED's best). Examples whose last
     item has an empty adjacency row degrade to plain CE.

  T3 MQSA long-query branch (architecture): a parallel attention stream whose
     per-position attention QUERY is LN(causal running mean of the last ``longq_L``
     input tokens, padding-mask-count normalised) through its OWN W_Q; K,V come
     from the raw (short) stream (SASRec convention: only Q is pre-LN'd).
     Readout u = 0.5*u_short + 0.5*u_long — the blend happens BEFORE the single
     out_proj/L2-norm, so serving is still one query vector. Recommended L=3.

  T4 BSARec FreqLayer (architecture): per block,
     mixed = freq_alpha*FreqBranch(x) + (1-freq_alpha)*MHA_sublayer(x); code-exact
     FreqBranch: xf = rfft(x, dim=seq, norm='ortho'); xf[:, c//2+1:, :] = 0;
     low = irfft(xf, n=seqlen); high = x - low;
     out = LayerNorm(Dropout(low + sqrt_beta**2 * high) + x), sqrt_beta a learned
     (1,1,d) parameter init randn*0.02. freq_alpha=0 bypasses the branch exactly.

  --input_dropout: extra dropout on the projected input tokens (SOTA recipes use
     0.5); default 0.0 == current behaviour.

  I42 T-GATE ``head_gate='softmax'`` (architecture, FAME/SINE gated aggregation):
     a tiny per-head scoring MLP (D -> 32 -> 1, shared across heads) produces K
     logits; gate weights = softmax(logits / head_gate_tau); the SERVED query
     becomes the weighted SUM of the K head vectors — ONE vector, ONE ANN probe
     (cheaper than max-over-K's K probes; still MIPS). Training CE uses the gated
     single vector's logits (== the same convex combination of per-head scores).
     ``head_gate='none'`` (default) keeps max-over-K exactly.

  I42 T-MULTI ``multi_target_k`` (loss-level, AMER optimal matching): each of the
     K heads gets its OWN supervision — the next-k future items of the prefix
     (``build_future_targets``; column 0 is always the example's own target) are
     assigned to heads by Hungarian matching on the cost matrix
     C[h, j] = -log p_h(target_j) under each head's full softmax; the mean matched
     CE is ADDED to the pooled CE with weight ``multi_weight`` (the pooled term
     stays because the pooled readout is the serving path — same convention as
     cumulative_ce). Serving unchanged (max-over-K, or the gated sum if T-GATE is
     also on — the two compose). Forbidden columns are masked per head with the
     candidate-target columns restored (the cumulative_ce keep-positives rule;
     future targets are legal train transitions, incl. the include_val one).

  --cumulative_ce (AUDIT RESULT): the baseline trainer supervises ONE target per
     encoded sequence (``loss = F.cross_entropy(logits, tgt_rows)`` on the pooled
     query) — but ChnDeDataset already enumerates EVERY prefix as its own example,
     so per-prefix supervision exists at the DATA level, each prefix re-encoded
     from scratch. ``cumulative_ce`` adds true SASRec-style per-position supervision
     in a single causal pass: the encoder gets a causal attention mask, each
     position's hidden state is projected/normalised to a query that predicts the
     NEXT input token (the last valid position predicts the example target), and
     loss = pooled_CE + cum_weight * mean-per-position CE. The pooled term stays
     because the K-head pooled readout is the serving path. NOTE: the causal mask
     applies whenever the flag is on (train AND eval) — it is part of the model
     config so serving-time encoding matches training.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from .chn_de import ChnDeDataset
from .frozen_query import FrozenQueryTower, collate_with_forbidden

NEG_INF = float("-inf")


# --------------------------------------------------------------------------- T4
class FreqLayer(nn.Module):
    """BSARec frequency branch (code-exact; Dropout+residual+LN inside)."""

    def __init__(self, d: int, c: int, dropout: float = 0.1):
        super().__init__()
        self.c = c
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seqlen = x.shape[1]
        xf = torch.fft.rfft(x, dim=1, norm="ortho")
        xf = torch.cat([xf[:, : self.c // 2 + 1, :],
                        torch.zeros_like(xf[:, self.c // 2 + 1:, :])], dim=1)
        low = torch.fft.irfft(xf, n=seqlen, dim=1, norm="ortho")
        high = x - low
        return self.ln(self.drop(low + self.sqrt_beta ** 2 * high) + x)


# --------------------------------------------------------------------------- T3
def causal_running_mean(x: torch.Tensor, mask: torch.Tensor, L: int) -> torch.Tensor:
    """[B, T, d] causal mean of the last ``L`` VALID tokens up to each position t
    (window [t-L+1, t] intersected with the padding mask; count-normalised).
    Positions with zero valid tokens in-window return the zero vector."""
    xm = x * mask.unsqueeze(-1).to(x.dtype)
    cs = torch.cumsum(xm, dim=1)                                 # [B, T, d]
    cnt = torch.cumsum(mask.to(x.dtype), dim=1)                  # [B, T]
    if L < x.shape[1]:
        win = cs - F.pad(cs, (0, 0, L, 0))[:, : x.shape[1]]
        wcnt = cnt - F.pad(cnt, (L, 0))[:, : x.shape[1]]
    else:
        win, wcnt = cs, cnt
    return win / wcnt.clamp(min=1.0).unsqueeze(-1)


# --------------------------------------------------------------------------- T1
def duorec_nce(z1: torch.Tensor, z2: torch.Tensor, tgt_rows: torch.Tensor,
               tau: float = 1.0) -> torch.Tensor:
    """Symmetric InfoNCE between two [B, D] L2-normalised views; positive = the
    diagonal; any in-batch pair sharing the SAME target is EXCLUDED from the
    negative set (its similarity masked to -inf on both axes)."""
    sim = z1 @ z2.t() / tau                                      # [B, B]
    same = tgt_rows.unsqueeze(0) == tgt_rows.unsqueeze(1)
    off_diag_same = same & ~torch.eye(len(tgt_rows), dtype=torch.bool,
                                      device=sim.device)
    sim = sim.masked_fill(off_diag_same, NEG_INF)
    labels = torch.arange(len(tgt_rows), device=sim.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


@torch.no_grad()
def wang_isola_uniformity(z: torch.Tensor) -> float:
    """log mean_{i<j} exp(-2*||z_i - z_j||^2) over in-batch pairs (z L2-normalised).
    More negative = more uniform; collapse drives it toward 0."""
    if len(z) < 2:
        return float("nan")
    d2 = torch.pdist(z.float()) ** 2
    return float(torch.logsumexp(-2.0 * d2, dim=0) - np.log(d2.numel()))


def build_peer_index(dataset: ChnDeDataset) -> dict[int, np.ndarray]:
    """target item row -> np.array of example indices with that target."""
    inv: dict[int, list[int]] = defaultdict(list)
    for i, (_h, t) in enumerate(dataset.examples):
        inv[t].append(i)
    return {t: np.asarray(v, np.int64) for t, v in inv.items()}


def sample_peers(peer_index: dict[int, np.ndarray], idxs: np.ndarray,
                 tgts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per example, a DIFFERENT example index with the same target; fallback = own
    index (its fresh-dropout forward then serves as the positive view)."""
    out = np.empty(len(idxs), np.int64)
    for k, (i, t) in enumerate(zip(idxs, tgts)):
        cand = peer_index.get(int(t))
        if cand is None or len(cand) < 2:
            out[k] = i
        else:
            j = rng.choice(cand)
            while j == i:
                j = rng.choice(cand)
            out[k] = j
    return out


# --------------------------------------------------------------------------- T2
def mine_forward_adjacency(seq_file: str, n_items: int) -> dict[int, Counter]:
    """Directed gap-1 counts A[i][j] = #(i followed by j) over TRAIN sequences,
    mirroring i24_build_input_emb leak rule: user id dropped (toks[1:]), final
    held-out event dropped (ids[:-1]). Keys/values are 0-indexed item rows."""
    adj: dict[int, Counter] = defaultdict(Counter)
    for line in open(seq_file):
        ids = [int(x) - 1 for x in line.split()[1:]]
        ids = ids[:-1]
        for a, b in zip(ids, ids[1:]):
            assert 0 <= a < n_items and 0 <= b < n_items
            adj[a][b] += 1
    return adj


def build_ted_teachers(adj: dict[int, Counter], n_items: int, ted_tau: float = 0.1
                       ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per item h: (support cols int64 [S], teacher probs float32 [S]) with
    probs = softmax( counts/max(counts) / ted_tau ) over the NONZERO support only
    (zero elsewhere — never materialises the dense row). Empty rows -> ([], [])."""
    empty = (np.zeros(0, np.int64), np.zeros(0, np.float32))
    out: list[tuple[np.ndarray, np.ndarray]] = [empty] * n_items
    for h, ctr in adj.items():
        cols = np.fromiter(ctr.keys(), np.int64, len(ctr))
        cnts = np.fromiter(ctr.values(), np.float64, len(ctr))
        z = (cnts / cnts.max()) / ted_tau
        z -= z.max()
        p = np.exp(z)
        out[h] = (cols, (p / p.sum()).astype(np.float32))
    return out


def collate_ted(teachers: list[tuple[np.ndarray, np.ndarray]],
                hist_rows: torch.Tensor, hist_mask: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch teacher rows for each example's LAST valid history item.
    Returns (ted_cols [B, S] -1-padded int64, ted_probs [B, S] float32)."""
    lengths = hist_mask.sum(1)
    last = hist_rows.gather(1, (lengths - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
    rows = [teachers[int(h)] for h in last]
    width = max((len(c) for c, _p in rows), default=0)
    cols = torch.full((len(rows), max(width, 1)), -1, dtype=torch.long)
    probs = torch.zeros((len(rows), max(width, 1)), dtype=torch.float32)
    for r, (c, p) in enumerate(rows):
        if len(c):
            cols[r, : len(c)] = torch.from_numpy(c)
            probs[r, : len(c)] = torch.from_numpy(p)
    return cols, probs


# ----------------------------------------------------------------- I42 T-MULTI
def build_future_targets(dataset: ChnDeDataset, k: int) -> np.ndarray:
    """[N, k] int64 (-1 padded): fut[i, j] = the (j+1)-th FUTURE item after
    example i's history (column 0 is always example i's own target). Relies on
    ChnDeDataset's construction order: consecutive examples of one user chain as
    hist[i+1] == hist[i] + [tgt[i]] (prefix enumeration; the include_val example
    (T, vt) extends the chain by one, so vt enters late-prefix windows — the same
    legality class as include_val itself). Test targets NEVER appear: the chain
    ends at vt."""
    assert k >= 1, k
    n = len(dataset.examples)
    chain = np.zeros(max(n - 1, 0), dtype=bool)      # example i+1 continues i
    for i in range(n - 1):
        h, t = dataset.examples[i]
        h2 = dataset.examples[i + 1][0]
        chain[i] = len(h2) == len(h) + 1 and h2[-1] == t and h2[:-1] == h
    fut = np.full((n, k), -1, np.int64)
    for i in range(n):
        fut[i, 0] = dataset.examples[i][1]
        for j in range(1, k):
            if i + j >= n or not chain[i + j - 1]:
                break
            fut[i, j] = dataset.examples[i + j][1]
    return fut


def hungarian_multi_ce(cost: torch.Tensor, multi_cols: torch.Tensor) -> torch.Tensor:
    """cost [B, K, M] = per-head negative log-prob of each candidate target;
    multi_cols [B, M] (-1 padded — pads are excluded). Per example: optimal
    (Hungarian) head<->target assignment, min(K, M_b) matched pairs; returns the
    mean matched cost over ALL matched pairs in the batch (differentiable —
    the assignment itself is computed on detached costs)."""
    terms = []
    cnp = cost.detach().cpu().numpy()
    for b in range(cost.shape[0]):
        valid = (multi_cols[b] >= 0).cpu().numpy().nonzero()[0]
        rows, cols = linear_sum_assignment(cnp[b][:, valid])
        for r, c in zip(rows, cols):
            terms.append(cost[b, r, valid[c]])
    return torch.stack(terms).mean()


# --------------------------------------------------------------------------- tower
class TricksQueryTower(FrozenQueryTower):
    """FrozenQueryTower + T3/T4 architecture flags + T2/cumulative-CE loss hooks.

    With longq_L=0 and freq_alpha=0 and cumulative_ce=False and input_dropout=0
    this IS the baseline tower (same init RNG order: extra modules are constructed
    AFTER the base ones, so base parameters are bitwise identical per seed)."""

    def __init__(self, item_emb: np.ndarray, d: int = 256, n_layers: int = 2,
                 n_heads: int = 4, d_ff: int = 512, max_len: int = 20,
                 dropout: float = 0.1, tau: float = 0.05, seed: int = 42,
                 n_query_vectors: int = 1, input_emb: np.ndarray | None = None,
                 longq_L: int = 0, freq_alpha: float = 0.0, freq_c: int = 5,
                 input_dropout: float = 0.0, cumulative_ce: bool = False,
                 head_gate: str = "none", head_gate_tau: float = 1.0,
                 multi_target_k: int = 0):
        assert head_gate in ("none", "softmax"), head_gate
        super().__init__(item_emb, d=d, n_layers=n_layers, n_heads=n_heads,
                         d_ff=d_ff, max_len=max_len, dropout=dropout, tau=tau,
                         seed=seed, n_query_vectors=n_query_vectors,
                         input_emb=input_emb)
        self.longq_L = int(longq_L)
        self.freq_alpha = float(freq_alpha)
        self.cumulative_ce = bool(cumulative_ce)
        self.head_gate = head_gate
        self.head_gate_tau = float(head_gate_tau)
        self.multi_target_k = int(multi_target_k)
        self.input_drop = nn.Dropout(input_dropout)
        self.config.update({"longq_L": self.longq_L, "freq_alpha": self.freq_alpha,
                            "freq_c": int(freq_c), "input_dropout": float(input_dropout),
                            "cumulative_ce": self.cumulative_ce,
                            "head_gate": self.head_gate,
                            "head_gate_tau": self.head_gate_tau,
                            "multi_target_k": self.multi_target_k})
        # extra modules built strictly AFTER the base tower (init-order invariance)
        self.freq_layers = None
        if self.freq_alpha > 0:
            self.freq_layers = nn.ModuleList(
                [FreqLayer(d, freq_c, dropout) for _ in range(n_layers)])
        self.long_layers = self.long_qln = None
        if self.longq_L > 0:
            self.long_layers = nn.ModuleList(
                [nn.TransformerEncoderLayer(d_model=d, nhead=n_heads,
                                            dim_feedforward=d_ff, dropout=dropout,
                                            batch_first=True)
                 for _ in range(n_layers)])
            self.long_qln = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.gate = None                                    # I42 T-GATE
        if head_gate == "softmax":
            D = self.item.shape[1]
            self.gate = nn.Sequential(nn.Linear(D, 32), nn.Tanh(), nn.Linear(32, 1))

    # ----------------------------------------------------------------- encoding
    def _encode_hidden(self, hist_rows: torch.Tensor, hist_mask: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """-> (short-stream hidden [B,T,d], long-stream hidden [B,T,d] or None)."""
        h = self.in_proj(self.input_item[hist_rows])
        lengths = hist_mask.sum(1, keepdim=True)
        pos = torch.arange(hist_rows.shape[1], device=hist_rows.device).unsqueeze(0)
        recency = (lengths - 1 - pos).clamp(min=0, max=self.max_len - 1)
        h = self.input_drop(h + self.rec_emb(recency))
        kpm = ~hist_mask
        T = h.shape[1]
        attn_mask = None
        if self.cumulative_ce:
            attn_mask = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                              device=h.device), diagonal=1)
        if self.freq_layers is None and self.long_layers is None:
            if attn_mask is None:
                return self.encoder(h, src_key_padding_mask=kpm), None
            return self.encoder(h, mask=attn_mask, src_key_padding_mask=kpm), None
        # custom per-layer path over the SAME parameter containers
        hs, hl = h, h
        for i, layer in enumerate(self.encoder.layers):
            attn_out = layer.norm1(hs + layer._sa_block(hs, attn_mask, kpm))
            if self.freq_layers is not None and self.freq_alpha > 0:
                mixed = self.freq_alpha * self.freq_layers[i](hs) \
                    + (1.0 - self.freq_alpha) * attn_out
            else:
                mixed = attn_out
            new_hs = layer.norm2(mixed + layer._ff_block(mixed))
            if self.long_layers is not None:
                ll = self.long_layers[i]
                qm = self.long_qln[i](
                    causal_running_mean(hl, hist_mask, self.longq_L))
                att, _ = ll.self_attn(qm, hs, hs, key_padding_mask=kpm,
                                      attn_mask=attn_mask, need_weights=False)
                a_out = ll.norm1(hl + ll.dropout1(att))
                hl = ll.norm2(a_out + ll._ff_block(a_out))
            hs = new_hs
        return hs, (hl if self.long_layers is not None else None)

    def _pool_hidden(self, h: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        att = self.pool(h).masked_fill(~hist_mask.unsqueeze(-1), NEG_INF)
        w = torch.softmax(att, dim=1)                                   # [B, L, K]
        return torch.einsum("blk,bld->bkd", w, h)                       # [B, K, d]

    def gate_weights(self, q: torch.Tensor) -> torch.Tensor:
        """[B, K, D] head vectors -> [B, K] softmax gate weights (I42 T-GATE)."""
        return torch.softmax(self.gate(q).squeeze(-1) / self.head_gate_tau, dim=1)

    def encode_queries(self, hist_rows: torch.Tensor, hist_mask: torch.Tensor
                       ) -> torch.Tensor:
        hs, hl = self._encode_hidden(hist_rows, hist_mask)
        q = self._pool_hidden(hs, hist_mask)
        if hl is not None:                                              # T3 blend
            q = 0.5 * q + 0.5 * self._pool_hidden(hl, hist_mask)
        q = F.normalize(self.out_proj(q), dim=-1)                       # [B, K, D]
        if self.gate is not None:              # T-GATE serving: ONE vector, 1 probe
            w = self.gate_weights(q)                                    # [B, K]
            q = torch.einsum("bk,bkd->bd", w, q).unsqueeze(1)           # [B, 1, D]
        return q

    # ----------------------------------------------------------------- loss
    def loss_batch(self, hist_rows: torch.Tensor, hist_mask: torch.Tensor,
                   tgt_rows: torch.Tensor, forbidden_cols: torch.Tensor | None = None,
                   ted_cols: torch.Tensor | None = None,
                   ted_probs: torch.Tensor | None = None, ted_beta: float = 0.0,
                   cum_weight: float = 1.0,
                   multi_cols: torch.Tensor | None = None,
                   multi_weight: float = 1.0) -> dict:
        """-> {'loss','ce','cum','multi','q'}; 'q' [B,K,D] (pre-gate heads) for
        the T1 regularizer anchor. ``multi_cols`` [B, M] (-1 padded, I42 T-MULTI):
        per-example next-k future targets (``build_future_targets`` rows)."""
        hs, hl = self._encode_hidden(hist_rows, hist_mask)
        pooled = self._pool_hidden(hs, hist_mask)
        if hl is not None:
            pooled = 0.5 * pooled + 0.5 * self._pool_hidden(hl, hist_mask)
        q = F.normalize(self.out_proj(pooled), dim=-1)
        s = torch.einsum("bkd,nd->bkn", q, self.item)                    # [B,K,n]
        if self.gate is not None:                                        # T-GATE
            logits = torch.einsum("bk,bkn->bn", self.gate_weights(q), s) / self.tau
        else:
            logits = s.max(dim=1).values / self.tau
        if forbidden_cols is not None:
            rows = torch.arange(len(logits), device=logits.device
                                ).unsqueeze(1).expand_as(forbidden_cols)
            valid = forbidden_cols >= 0
            logits[rows[valid], forbidden_cols[valid]] = NEG_INF
        logp = F.log_softmax(logits, dim=-1)
        lp_tgt = logp.gather(1, tgt_rows.unsqueeze(1)).squeeze(1)
        if ted_cols is not None and ted_beta > 0:                        # T2
            lp_c = logp.gather(1, ted_cols.clamp(min=0))
            fin = (ted_cols >= 0) & torch.isfinite(lp_c)
            p = ted_probs * fin
            psum = p.sum(1)
            soft = (p * lp_c.masked_fill(~fin, 0.0)).sum(1) / psum.clamp(min=1e-12)
            beta = ted_beta * (psum > 0)                                 # renormalised
            ce = (-((1.0 - beta) * lp_tgt + beta * soft)).mean()
        else:
            ce = (-lp_tgt).mean()
        loss, cum, multi = ce, None, None
        if multi_cols is not None and self.multi_target_k > 0:           # T-MULTI
            ph = s / self.tau                                            # [B,K,n]
            # pad cols point at column 0 == the example's own target (never
            # forbidden), so the keep-positives restore below stays a no-op there
            mc = torch.where(multi_cols >= 0, multi_cols,
                             multi_cols[:, :1].expand_as(multi_cols))
            idx = mc.unsqueeze(1).expand(-1, ph.shape[1], -1)            # [B,K,M]
            if forbidden_cols is not None:
                bi = torch.arange(len(hist_rows), device=ph.device)
                forb = torch.zeros(len(hist_rows), self.item.shape[0],
                                   dtype=torch.bool, device=ph.device)
                fvalid = forbidden_cols >= 0
                forb[bi.unsqueeze(1).expand_as(forbidden_cols)[fvalid],
                     forbidden_cols[fvalid]] = True
                saved = ph.gather(2, idx)                    # keep target columns
                ph = ph.masked_fill(forb.unsqueeze(1), NEG_INF)
                ph = ph.scatter(2, idx, saved)
            lph = F.log_softmax(ph, dim=-1)
            multi = hungarian_multi_ce(-lph.gather(2, idx), multi_cols)
            loss = loss + multi_weight * multi
        if self.cumulative_ce:
            hb = hs if hl is None else 0.5 * (hs + hl)
            qpos = F.normalize(self.out_proj(hb), dim=-1)                # [B,T,D]
            pl = torch.einsum("btd,nd->btn", qpos, self.item) / self.tau
            T = hist_rows.shape[1]
            tgt_pos = torch.cat([hist_rows[:, 1:],
                                 torch.zeros_like(hist_rows[:, :1])], dim=1)
            lengths = hist_mask.sum(1)
            b_idx = torch.arange(len(hist_rows), device=hist_rows.device)
            tgt_pos[b_idx, (lengths - 1).clamp(min=0)] = tgt_rows
            if forbidden_cols is not None:
                forb = torch.zeros(len(hist_rows), self.item.shape[0],
                                   dtype=torch.bool, device=pl.device)
                valid = forbidden_cols >= 0
                forb[b_idx.unsqueeze(1).expand_as(forbidden_cols)[valid],
                     forbidden_cols[valid]] = True
                saved = pl.gather(2, tgt_pos.unsqueeze(-1))              # keep positives
                pl = pl.masked_fill(forb.unsqueeze(1), NEG_INF)
                pl = pl.scatter(2, tgt_pos.unsqueeze(-1), saved)         # restore
            lp_pos = F.log_softmax(pl, dim=-1)
            nll = -lp_pos.gather(2, tgt_pos.unsqueeze(-1)).squeeze(-1)   # [B,T]
            cum = (nll * hist_mask).sum() / hist_mask.sum().clamp(min=1)
            loss = loss + cum_weight * cum
        return {"loss": loss, "ce": ce, "cum": cum, "multi": multi, "q": q}

    def nce_embed(self, hist_rows: torch.Tensor, hist_mask: torch.Tensor
                  ) -> torch.Tensor:
        """[B, D] single L2-normalised view for T1 (mean over the K query heads)."""
        return F.normalize(self.encode_queries(hist_rows, hist_mask).mean(1), dim=-1)


def load_tricks_checkpoint(path: str, item_emb: np.ndarray,
                           input_emb: np.ndarray | None = None) -> TricksQueryTower:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = {k: v for k, v in ckpt["config"].items()
           if k not in ("item_dim", "n_items", "input_dim")}
    model = TricksQueryTower(item_emb, input_emb=input_emb, **cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model


# --------------------------------------------------------------------------- trainer
def train_tricks(model: TricksQueryTower, dataset: ChnDeDataset,
                 steps: int = 4000, batch_size: int = 256, lr: float = 1e-3,
                 weight_decay: float = 0.01, warmup_frac: float = 0.05,
                 device: str = "cpu", seed: int = 42,
                 ckpt_path: str | None = None, ckpt_every: int = 1000,
                 log_every: int = 50,
                 nce_lambda: float = 0.0, nce_tau: float = 1.0,
                 nce_mode: str = "full",
                 ted_teachers: list | None = None, ted_beta: float = 0.0,
                 cum_weight: float = 1.0,
                 multi_targets: np.ndarray | None = None,
                 multi_weight: float = 1.0) -> dict:
    """train_frozen_query twin with the T1/T2 loss hooks (T3/T4/input_dropout/
    cumulative_ce/head_gate live in the model construction). ``multi_targets``
    [len(dataset), k] (-1 padded, ``build_future_targets``) enables the I42
    T-MULTI matched per-head CE when the model has ``multi_target_k > 0``.
    Returns {'loss': [...], 'ce': [...], 'nce': [...], 'uniformity': [...]}."""
    from .frozen_query import save_checkpoint
    import math as _math
    assert nce_mode in ("full", "scl"), nce_mode
    if multi_targets is not None:
        multi_targets = np.asarray(multi_targets, np.int64)
        assert len(multi_targets) == len(dataset), multi_targets.shape
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
        return 0.5 * (1.0 + _math.cos(_math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    peer_index = build_peer_index(dataset) if nce_lambda > 0 else None
    hist = {"loss": [], "ce": [], "nce": [], "uniformity": []}
    for step in range(steps):
        idxs = rng.choice(len(dataset), size=min(batch_size, len(dataset)),
                          replace=False)
        b = collate_with_forbidden(dataset, idxs, model.max_len)
        hr = b["hist_rows"].to(device)
        hm = b["hist_mask"].to(device)
        tg = b["tgt_rows"].to(device)
        tc = tp = None
        if ted_teachers is not None and ted_beta > 0:                    # T2
            tc, tp = collate_ted(ted_teachers, b["hist_rows"], b["hist_mask"])
            tc, tp = tc.to(device), tp.to(device)
        mc = None
        if multi_targets is not None and model.multi_target_k > 0:       # T-MULTI
            mc = torch.from_numpy(multi_targets[idxs]).to(device)
        out = model.loss_batch(hr, hm, tg, b["forbidden_cols"].to(device),
                               ted_cols=tc, ted_probs=tp, ted_beta=ted_beta,
                               cum_weight=cum_weight, multi_cols=mc,
                               multi_weight=multi_weight)
        loss, nce_val = out["loss"], float("nan")
        if nce_lambda > 0:                                               # T1
            tgts = b["tgt_rows"].numpy()
            peers = sample_peers(peer_index, idxs, tgts, rng)
            pb = collate_with_forbidden(dataset, peers, model.max_len)
            z_peer = model.nce_embed(pb["hist_rows"].to(device),
                                     pb["hist_mask"].to(device))
            if nce_mode == "full":
                z_anchor = model.nce_embed(hr, hm)   # 2nd forward, fresh dropout
            else:                                    # scl: reuse the main forward
                z_anchor = F.normalize(out["q"].mean(1), dim=-1)
            nce = duorec_nce(z_anchor, z_peer, tg, tau=nce_tau)
            loss = loss + nce_lambda * nce
            nce_val = float(nce.item())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        hist["loss"].append(float(loss.item()))
        hist["ce"].append(float(out["ce"].item()))
        hist["nce"].append(nce_val)
        if log_every and (step + 1) % log_every == 0:
            with torch.no_grad():
                was = model.training
                model.eval()
                u = wang_isola_uniformity(model.nce_embed(hr, hm))
                model.train(was)
            hist["uniformity"].append({"step": step + 1, "uniformity": round(u, 4)})
            avg = sum(hist["loss"][-log_every:]) / log_every
            print(f"[train_tricks] step {step + 1}/{steps} loss {avg:.4f} "
                  f"nce {nce_val:.4f} unif {u:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if ckpt_path and ckpt_every and (step + 1) % ckpt_every == 0:
            save_checkpoint(model, ckpt_path, step=step + 1)
    if ckpt_path:
        save_checkpoint(model, ckpt_path, step=steps)
    return hist
