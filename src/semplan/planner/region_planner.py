"""J2 — interpretable region-planner MVP (ideas I1/I3, roadmap E4).

A small T5-style encoder-decoder whose INPUT is the user's history encoded as the sequence of
interpretable REGION tokens (taxonomy+price regions from ``attribute_semantic_ids``, or any
row->region map, e.g. collaborative k-means regions), optionally enriched by the items' content
embeddings projected through a small MLP and ADDED to the region token embeddings (LIGER's
``flag_add_input_embedding`` trick — item-level tokens are too sparse on Beauty, region tokens
carry the coarse signal, content embeddings restore item identity). The OUTPUT is a true AUTOREGRESSIVE region plan
(upgraded 2026-08-02 per the accepted codex review, REV-1): for a train pair built from the
prefix ``s[:j]`` (target ``s[j]``) of a user sequence ``s`` of length n (protocol-capped to the
last ``max_len`` items), the label is the ORDERED UNIQUE region tokens of the training-visible
FUTURE WINDOW ``[s[j], s[j+1], ..., s[n-3]]`` — everything from the pair's target up to but
EXCLUDING the val target ``s[n-2]`` and test target ``s[n-1]`` — in first-occurrence order,
capped at ``max_plan_len`` (default 5, per E4c hit@5=0.245 -> K in 1..5), followed by EOS.
The FIRST label token is always ``region(s[j])`` (the immediate next item's region), so step-1
supervision stays comparable to the old length-1 objective; later steps train true step-2+
conditioning (I4) and EOS position trains a learned stop (used by J3's gate).

Vocab layout (custom, tiny): PAD=0, EOS=1, BOS=2 (decoder start), region r -> 3+r.

Everything here is protocol-safe: training examples are derived per-user from the train-visible
sequence ``s[:-2]`` (== each user's VAL history), so no label token can ever derive from the
val/test targets; the derived (history, target) pairs are re-asserted to be exactly
``LOOSplit.train_pairs`` members for defense in depth.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from semplan.eval.protocol import LOOSplit
from semplan.ids.attribute_semantic_ids import build_attribute_sids

PAD_ID, EOS_ID, BOS_ID, N_SPECIAL = 0, 1, 2, 3


# --------------------------------------------------------------------------- region maps
@dataclass
class RegionMap:
    """Row-level view of a region system: item row -> region id, region -> member rows."""
    row_region: np.ndarray                  # [n_items] int64, -1 = unmapped
    n_regions: int
    members: dict[int, np.ndarray]          # region id -> np.array of item rows
    region_parent: np.ndarray | None = None  # [n_regions] int64 parent-group id (siblings share
    #                                          one) — REV-2 sibling label smoothing; None = n/a

    def region_token(self, row: int) -> int:
        return int(self.row_region[row]) + N_SPECIAL

    @property
    def vocab_size(self) -> int:
        return self.n_regions + N_SPECIAL

    def sibling_groups_vocab(self) -> torch.Tensor:
        """[vocab_size] int64: parent-group id per VOCAB token (-1 for specials). Requires
        ``region_parent`` (built by the taxonomy/collab constructors)."""
        assert self.region_parent is not None, "region map carries no parent groups"
        g = torch.full((self.vocab_size,), -1, dtype=torch.long)
        g[N_SPECIAL:] = torch.from_numpy(self.region_parent.astype(np.int64))
        return g


def build_taxonomy_region_map(id2meta_path: str, n_items: int, depth: int = 3) -> RegionMap:
    """The 820 taxonomy+price regions on Beauty (E1's exact derivation: region key = path[:4])."""
    id2meta = json.load(open(id2meta_path))
    sid = build_attribute_sids(id2meta, depth=depth, use_price=True)
    row_region = np.full(n_items, -1, dtype=np.int64)
    region_ids: dict[tuple, int] = {}
    parent_of: dict[int, tuple] = {}
    for k, path in sid.ids.items():
        row = int(k) - 1                    # paper mapping item id -> row (id-1)
        if 0 <= row < n_items:
            rid = region_ids.setdefault(path[: depth + 1], len(region_ids))
            row_region[row] = rid
            parent_of[rid] = path[:depth]   # siblings share the taxonomy path above the leaf
    return _finish_region_map(row_region, len(region_ids),
                              region_parent=_parent_ids(parent_of, len(region_ids)))


def build_collab_region_map(embeddings: np.ndarray, branching: int = 32, depth: int = 2,
                            seed: int = 0) -> RegionMap:
    """Collaborative/geometric regions: hierarchical k-means leaves over a trained item space."""
    from semplan.ids.hierarchical_kmeans import build_hierarchical_kmeans

    n_items = embeddings.shape[0]
    hk = build_hierarchical_kmeans([str(i) for i in range(n_items)], embeddings,
                                   branching=branching, depth=depth, seed=seed)
    row_region = np.full(n_items, -1, dtype=np.int64)
    region_ids: dict[tuple, int] = {}
    parent_of: dict[int, tuple] = {}
    for iid, path in hk.ids.items():
        rid = region_ids.setdefault(tuple(path), len(region_ids))
        row_region[int(iid)] = rid
        parent_of[rid] = tuple(path)[:-1]   # siblings share the k-means parent node
    return _finish_region_map(row_region, len(region_ids),
                              region_parent=_parent_ids(parent_of, len(region_ids)))


def _parent_ids(parent_of: dict[int, tuple], n_regions: int) -> np.ndarray:
    """Densify {region id -> parent key} into a [n_regions] int64 parent-GROUP id array."""
    out = np.full(n_regions, -1, dtype=np.int64)
    gid: dict[tuple, int] = {}
    for rid, key in parent_of.items():
        out[rid] = gid.setdefault(key, len(gid))
    return out


def _finish_region_map(row_region: np.ndarray, n_regions: int,
                       region_parent: np.ndarray | None = None) -> RegionMap:
    members: dict[int, list[int]] = {}
    for row, r in enumerate(row_region.tolist()):
        if r >= 0:
            members.setdefault(r, []).append(row)
    return RegionMap(row_region=row_region, n_regions=n_regions,
                     members={r: np.array(v, dtype=np.int64) for r, v in members.items()},
                     region_parent=region_parent)


def build_item_feats(regions: RegionMap, split: LOOSplit, n_buckets: int = 10) -> np.ndarray:
    """[n_items, 2] int64 inference-safe per-ITEM feature buckets (REV-2 J2 arm):
    col 0 = train-history popularity decile (0..9, from ``split.train_pairs`` histories only),
    col 1 = region-size decile of the item's region (0..9; unmapped items get 0).
    The third feature (seen-earlier-in-history repeat flag) is per-POSITION and is computed by
    the dataset/encoder, not here."""
    n_items = len(regions.row_region)
    pop = np.zeros(n_items, dtype=np.int64)
    for hist, _tgt in split.train_pairs:
        for row in hist:
            pop[row] += 1
    rsize = np.zeros(n_items, dtype=np.int64)
    for r, mem in regions.members.items():
        rsize[mem] = len(mem)

    def _decile(x: np.ndarray) -> np.ndarray:
        qs = np.quantile(x, np.linspace(0, 1, n_buckets + 1)[1:-1])
        return np.searchsorted(qs, x, side="right").astype(np.int64)

    return np.stack([_decile(pop), _decile(rsize)], axis=1)


def region_popularity(regions: RegionMap, split: LOOSplit) -> np.ndarray:
    """Train-history region frequencies (used to pad beam output to k unique regions)."""
    counts = np.zeros(regions.n_regions, dtype=np.int64)
    for hist, _tgt in split.train_pairs:
        for row in hist:
            r = regions.row_region[row]
            if r >= 0:
                counts[r] += 1
    return np.argsort(-counts)


# --------------------------------------------------------------------------- model
class RegionPlanner(nn.Module):
    """T5ForConditionalGeneration over the region vocab + optional content-embedding input MLP."""

    def __init__(self, n_regions: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
                 d_ff: int = 256, content_dim: int | None = 768, dropout: float = 0.1,
                 use_item_feats: bool = False):
        super().__init__()
        from transformers import T5Config, T5ForConditionalGeneration

        self.n_regions = n_regions
        self.use_item_feats = use_item_feats
        cfg = T5Config(
            vocab_size=n_regions + N_SPECIAL,
            d_model=d_model, d_kv=max(d_model // n_heads, 8), d_ff=d_ff,
            num_layers=n_layers, num_decoder_layers=n_layers, num_heads=n_heads,
            dropout_rate=dropout, feed_forward_proj="relu",
            pad_token_id=PAD_ID, eos_token_id=EOS_ID,
            decoder_start_token_id=BOS_ID, tie_word_embeddings=False,
        )
        self.t5 = T5ForConditionalGeneration(cfg)
        self.use_content = content_dim is not None
        if self.use_content:
            self.content_proj = nn.Sequential(
                nn.Linear(content_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        if use_item_feats:                          # REV-2 arm: pop decile / region-size
            self.feat_embs = nn.ModuleList([        # decile / seen-earlier repeat flag
                nn.Embedding(10, d_model), nn.Embedding(10, d_model), nn.Embedding(2, d_model)])

    def _encoder_inputs(self, input_ids: torch.Tensor, content: torch.Tensor | None,
                        feats: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.t5.shared(input_ids)
        if self.use_content and content is not None:
            emb = emb + self.content_proj(content)
        if self.use_item_feats and feats is not None:
            for f, fe in enumerate(self.feat_embs):
                emb = emb + fe(feats[..., f])
        return emb

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor, content: torch.Tensor | None = None,
                feats: torch.Tensor | None = None):
        emb = self._encoder_inputs(input_ids, content, feats)
        return self.t5(inputs_embeds=emb, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def beam_regions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                     content: torch.Tensor | None, k: int, num_beams: int | None = None,
                     max_plan_len: int = 5, return_scores: bool = False,
                     feats: torch.Tensor | None = None):
        """Beam-search the region vocab; return per-query RANKED unique region ids (>= k long
        unless the vocab itself is smaller), collected in emission order across beams (each beam
        is now a genuine multi-token plan trained on future-window targets).

        ``return_scores=True`` (J3's gate needs this — ``output_scores=True`` path) additionally
        returns a dict with:
          * ``seq_logprobs``   [bsz, num_beams] length-normalised per-sequence logprobs, sorted
                               best-first along the beam dim (transformers' return order);
          * ``step_entropy``   [bsz, num_beams, steps] entropy (nats) of the per-step token
                               distribution (softmax over the processed logits at each decode
                               step), nan-safe for finished beams.
        Returns ``regions`` alone when ``return_scores=False`` (backward compatible), else
        ``(regions, scores_dict)``.
        """
        self.eval()
        num_beams = max(num_beams or 2 * k, k)
        emb = self._encoder_inputs(input_ids, content, feats)
        out = self.t5.generate(
            inputs_embeds=emb, attention_mask=attention_mask,
            num_beams=num_beams, num_return_sequences=num_beams,
            max_new_tokens=max_plan_len + 1, early_stopping=True,
            output_scores=return_scores, return_dict_in_generate=return_scores,
        )
        bsz = input_ids.shape[0]
        raw = out.sequences if return_scores else out       # [batch*num_beams, seq]
        seqs = raw.view(bsz, num_beams, -1)
        results: list[list[int]] = []
        for b in range(bsz):
            ranked: list[int] = []
            for beam in seqs[b].tolist():
                for tok in beam:
                    if tok >= N_SPECIAL and (tok - N_SPECIAL) not in ranked:
                        ranked.append(tok - N_SPECIAL)
                if len(ranked) >= k:
                    break
            results.append(ranked[:k] if len(ranked) >= k else ranked)
        if not return_scores:
            return results
        seq_logprobs = out.sequences_scores.view(bsz, num_beams)
        ents = []
        for step_scores in out.scores:                       # each [bsz*num_beams, vocab]
            logp = torch.log_softmax(step_scores.float(), dim=-1)
            p = logp.exp()
            ent = -(p * torch.where(p > 0, logp, torch.zeros_like(logp))).sum(-1)
            ents.append(torch.nan_to_num(ent, nan=0.0).view(bsz, num_beams))
        step_entropy = (torch.stack(ents, dim=-1) if ents
                        else torch.zeros(bsz, num_beams, 0))  # [bsz, num_beams, steps]
        return results, {"seq_logprobs": seq_logprobs, "step_entropy": step_entropy}


# --------------------------------------------------------------------------- dataset
def future_window_label(future_rows: list[int], regions: RegionMap,
                        max_plan_len: int = 5) -> list[int]:
    """Multi-region autoregressive label (REV-1 rule): ordered UNIQUE region tokens of the
    future window (first-occurrence order), capped at ``max_plan_len``, followed by EOS.
    ``future_rows[0]`` is the pair's immediate target, so the first token is its region.
    Future items with no region are skipped (never happens on Beauty)."""
    toks: list[int] = []
    for row in future_rows:
        if regions.row_region[row] < 0:
            continue
        t = regions.region_token(row)
        if t not in toks:
            toks.append(t)
        if len(toks) >= max_plan_len:
            break
    return toks + [EOS_ID]


class RegionPlanDataset(Dataset):
    """Protocol-safe (history -> future-window region plan) examples.

    Examples are derived per-user from the train-visible sequence ``T = s[:-2]`` (exactly each
    user's VAL history under ``protocol.build``'s max_len capping): for every target index
    ``t`` in 1..len(T)-1, encoder input = region tokens of ``T[:t]`` (last ``max_len``, with
    optional per-position content embedding) and decoder label =
    ``future_window_label(T[t:])`` — the ordered unique region tokens of
    ``[s[t], ..., s[n-3]]`` capped at ``max_plan_len`` + EOS. This reproduces
    ``split.train_pairs`` (history, target) pairs exactly (asserted), and by construction no
    label token can derive from the val target ``s[n-2]`` or test target ``s[n-1]``.

    History items with no region are dropped from the input (never happens on Beauty).
    Examples whose TARGET has no region are skipped (they cannot supervise the planner).
    ``pairs`` (legacy escape hatch) supplies raw (history, target) pairs with no future
    window — labels degrade to the old length-1 ``[region(target), EOS]``.
    """

    def __init__(self, split: LOOSplit, regions: RegionMap,
                 content_emb: np.ndarray | None = None, max_len: int = 20,
                 max_plan_len: int = 5,
                 pairs: list[tuple[list[int], int]] | None = None,
                 item_feats: np.ndarray | None = None):
        self.regions = regions
        self.content = content_emb
        self.max_len = max_len
        self.max_plan_len = max_plan_len
        self.item_feats = item_feats            # [n_items, 2] from build_item_feats, or None
        if pairs is not None:
            src = [(hist, [tgt]) for hist, tgt in pairs]        # no future window known
        else:
            # Train-visible sequence per user == the user's VAL history (s[:-2], capped).
            src = [(list(T[:t]), list(T[t:]))
                   for T, _vt in split.val for t in range(1, len(T))]
            # Defense in depth: the derived (history, target) pairs must be exactly members of
            # protocol.build's train_pairs. Deriving both history and future window from the
            # SAME user's val history T = s[:-2] makes per-user leakage structurally impossible
            # (every label token comes from T; the user's val/test targets are outside T, and a
            # derived history is strictly shorter than the val history). A derived pair may
            # coincidentally equal ANOTHER user's val/test pair — that is not leakage under the
            # per-user positional LOO guarantee (REV-1: cross-split dedup rejected as N/A).
            train_set = {(tuple(h), t) for h, t in split.train_pairs}
            for hist, future in src:
                assert (tuple(hist), future[0]) in train_set, \
                    "derived pair not in protocol train_pairs"
        # examples: (input history rows, label token seq incl. EOS, target row)
        self.examples: list[tuple[list[int], list[int], int]] = []
        for hist, future in src:
            tgt = future[0]
            if regions.row_region[tgt] < 0:
                continue
            h = [r for r in hist[-max_len:] if regions.row_region[r] >= 0]
            if h:
                label = future_window_label(future, regions, max_plan_len=max_plan_len)
                self.examples.append((h, label, tgt))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        hist, label, _tgt = self.examples[i]
        ids = [self.regions.region_token(r) for r in hist]
        cont = self.content[hist] if self.content is not None else None
        return ids, cont, label, hist

    def collate(self, batch):
        max_h = max(len(ids) for ids, _c, _l, _h in batch)
        max_l = max(len(label) for _ids, _c, label, _h in batch)
        bsz = len(batch)
        input_ids = torch.full((bsz, max_h), PAD_ID, dtype=torch.long)
        attn = torch.zeros((bsz, max_h), dtype=torch.long)
        labels = torch.full((bsz, max_l), -100, dtype=torch.long)
        content = None
        if self.content is not None:
            content = torch.zeros((bsz, max_h, self.content.shape[1]), dtype=torch.float32)
        feats = None
        if self.item_feats is not None:
            feats = torch.zeros((bsz, max_h, 3), dtype=torch.long)
        for i, (ids, cont, label, hist) in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, : len(ids)] = 1
            labels[i, : len(label)] = torch.tensor(label, dtype=torch.long)
            if content is not None:
                content[i, : len(ids)] = torch.from_numpy(np.asarray(cont, dtype=np.float32))
            if feats is not None:
                feats[i, : len(ids)] = torch.from_numpy(position_feats(hist, self.item_feats))
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels,
                "content": content, "feats": feats}


def position_feats(hist: list[int], item_feats: np.ndarray) -> np.ndarray:
    """[len(hist), 3] int64 per-POSITION features: item pop decile, region-size decile
    (both from ``build_item_feats``), and the seen-earlier-in-history repeat flag."""
    f = np.zeros((len(hist), 3), dtype=np.int64)
    f[:, :2] = item_feats[hist]
    seen: set[int] = set()
    for i, r in enumerate(hist):
        f[i, 2] = 1 if r in seen else 0
        seen.add(r)
    return f


def encode_histories(histories: list[list[int]], regions: RegionMap,
                     content_emb: np.ndarray | None, max_len: int = 20,
                     device: str = "cpu", item_feats: np.ndarray | None = None) -> dict:
    """Batch-encode raw history rows into planner inputs (inference-side mirror of collate)."""
    rows = [[r for r in h[-max_len:] if regions.row_region[r] >= 0] or [h[-1]] for h in histories]
    max_h = max(len(h) for h in rows)
    bsz = len(rows)
    input_ids = torch.full((bsz, max_h), PAD_ID, dtype=torch.long)
    attn = torch.zeros((bsz, max_h), dtype=torch.long)
    content = None
    if content_emb is not None:
        content = torch.zeros((bsz, max_h, content_emb.shape[1]), dtype=torch.float32)
    feats = None
    if item_feats is not None:
        feats = torch.zeros((bsz, max_h, 3), dtype=torch.long)
    for i, h in enumerate(rows):
        toks = [regions.region_token(r) if regions.row_region[r] >= 0 else PAD_ID for r in h]
        input_ids[i, : len(h)] = torch.tensor(toks, dtype=torch.long)
        attn[i, : len(h)] = 1
        if content is not None:
            content[i, : len(h)] = torch.from_numpy(content_emb[h].astype(np.float32))
        if feats is not None:
            feats[i, : len(h)] = torch.from_numpy(position_feats(h, item_feats))
    out = {"input_ids": input_ids.to(device), "attention_mask": attn.to(device)}
    out["content"] = content.to(device) if content is not None else None
    out["feats"] = feats.to(device) if feats is not None else None
    return out


def predict_regions(model: RegionPlanner, histories: list[list[int]], regions: RegionMap,
                    k: int, content_emb: np.ndarray | None = None, max_len: int = 20,
                    device: str = "cpu", batch_size: int = 64,
                    pad_order: np.ndarray | None = None,
                    item_feats: np.ndarray | None = None) -> list[list[int]]:
    """Beam-decode the top-k regions for each history; pad with globally popular regions so
    every query gets exactly k unique valid region ids (pad_order defaults to region id order;
    pass an EMPTY array for the REV-2 no-padding policy — queries may then return < k)."""
    if pad_order is None:
        pad_order = np.arange(regions.n_regions)
    out: list[list[int]] = []
    for s in range(0, len(histories), batch_size):
        chunk = histories[s: s + batch_size]
        enc = encode_histories(chunk, regions, content_emb, max_len=max_len, device=device,
                               item_feats=item_feats)
        ranked = model.beam_regions(enc["input_ids"], enc["attention_mask"], enc["content"], k=k,
                                    feats=enc["feats"])
        for r in ranked:
            for p in pad_order.tolist():
                if len(r) >= k:
                    break
                if p not in r:
                    r.append(int(p))
            out.append(r[:k])
    return out


# --------------------------------------------------------------------------- training
def planner_loss(logits: torch.Tensor, labels: torch.Tensor,
                 pos_weights: list[float] | None = None,
                 dup_coef: float = 0.0,
                 sibling_eps: float = 0.0,
                 sibling_groups: torch.Tensor | None = None) -> torch.Tensor:
    """REV-2 cheap-arm loss over teacher-forced decoder ``logits`` [B, L, V] and ``labels``
    [B, L] (-100 = pad). With all arms off this equals HF's mean token CE exactly.

    pos_weights    : per-label-POSITION CE weights (e.g. [1.0, .7, .5, .35, .25]); positions past
                     the list reuse the last weight; the loss is a weighted mean over valid
                     tokens (weights sum in the denominator, so all-off == plain CE).
    dup_coef       : coefficient for the duplicate-region unlikelihood
                     ``L_dup = -log(1 - p_j(token in labels[:, :j]))`` (codex REV-2), averaged
                     over positions with a nonempty prefix; specials never counted.
    sibling_eps    : label-smoothing mass spread UNIFORMLY OVER SIBLING regions (same
                     ``region_parent`` group, needs ``sibling_groups`` from
                     ``RegionMap.sibling_groups_vocab``) instead of the whole vocab. EOS/special
                     targets and singleton-sibling targets fall back to hard CE.
    """
    B, L, V = logits.shape
    valid = labels != -100                                      # [B, L]
    safe = labels.clamp_min(0)
    logp = torch.log_softmax(logits.float(), dim=-1)            # [B, L, V]
    nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)      # [B, L]

    if sibling_eps > 0.0:
        assert sibling_groups is not None, "sibling smoothing needs sibling_groups"
        g = sibling_groups.to(labels.device)                    # [V], -1 for specials
        tgt_group = g[safe]                                     # [B, L]
        sib_mask = (g.view(1, 1, V) == tgt_group.unsqueeze(-1)) & (tgt_group.unsqueeze(-1) >= 0)
        n_sib = sib_mask.sum(-1).clamp_min(1)                   # incl. the target itself
        # smoothed NLL: (1-eps)*NLL(target) + eps * mean NLL over the sibling set; targets with
        # no siblings beyond themselves (n_sib==1) reduce to plain NLL exactly
        sib_nll = -(logp * sib_mask).sum(-1) / n_sib
        smooth = torch.where(n_sib > 1,
                             (1.0 - sibling_eps) * nll + sibling_eps * sib_nll, nll)
        nll = smooth

    if pos_weights:
        w = torch.tensor(pos_weights, dtype=nll.dtype, device=nll.device)
        idx = torch.arange(L, device=nll.device).clamp_max(len(pos_weights) - 1)
        pw = w[idx].unsqueeze(0).expand_as(nll)
    else:
        pw = torch.ones_like(nll)
    loss = (nll * pw * valid).sum() / (pw * valid).sum().clamp_min(1e-9)

    if dup_coef > 0.0:
        p = logp.exp()
        prev = torch.zeros(B, V, dtype=torch.bool, device=logits.device)
        dup_terms, dup_norm = [], 0
        for j in range(L):
            has_prev = prev.any(-1) & valid[:, j]               # [B]
            if has_prev.any():
                mass = (p[:, j] * prev).sum(-1).clamp(max=1 - 1e-6)
                dup_terms.append((-torch.log1p(-mass) * has_prev).sum())
                dup_norm += int(has_prev.sum())
            tok = safe[:, j]
            is_region = valid[:, j] & (tok >= N_SPECIAL)
            prev = prev.clone()
            prev[torch.arange(B, device=logits.device)[is_region], tok[is_region]] = True
        if dup_norm:
            loss = loss + dup_coef * torch.stack(dup_terms).sum() / dup_norm
    return loss


def train_planner(model: RegionPlanner, dataset: RegionPlanDataset, steps: int = 2000,
                  batch_size: int = 128, lr: float = 1e-3, weight_decay: float = 0.01,
                  warmup_frac: float = 0.05, device: str = "cpu", seed: int = 42,
                  ckpt_path: str | None = None, ckpt_every: int = 1000,
                  log_every: int = 50,
                  pos_weights: list[float] | None = None,
                  dup_coef: float = 0.0,
                  sibling_eps: float = 0.0,
                  sibling_groups: torch.Tensor | None = None) -> list[float]:
    """AdamW + linear-warmup cosine schedule, device-agnostic, step-based. Returns loss history
    (one entry per step). SMOKE-sized runs just pass tiny ``steps``. The REV-2 cheap-arm loss
    knobs (``pos_weights``/``dup_coef``/``sibling_eps``) default OFF == HF mean-token CE."""
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup = max(int(steps * warmup_frac), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=dataset.collate, drop_last=len(dataset) > batch_size)
    losses: list[float] = []
    it = iter(loader)
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        content = batch["content"].to(device) if batch["content"] is not None else None
        feats = batch.get("feats")
        feats = feats.to(device) if feats is not None else None
        labels = batch["labels"].to(device)
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    labels, content=content, feats=feats)
        if pos_weights or dup_coef > 0.0 or sibling_eps > 0.0:
            out.loss = planner_loss(out.logits, labels, pos_weights=pos_weights,
                                    dup_coef=dup_coef, sibling_eps=sibling_eps,
                                    sibling_groups=sibling_groups)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(out.loss.item()))
        if log_every and (step + 1) % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"[train_planner] step {step + 1}/{steps} loss {avg:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if ckpt_path and ckpt_every and (step + 1) % ckpt_every == 0:
            save_checkpoint(model, ckpt_path, step=step + 1)
    if ckpt_path:
        save_checkpoint(model, ckpt_path, step=steps)
    return losses


def save_checkpoint(model: RegionPlanner, path: str, step: int = 0) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_regions": model.n_regions,
                "step": step}, path)


def load_checkpoint(path: str, **model_kwargs) -> RegionPlanner:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = RegionPlanner(n_regions=ckpt["n_regions"], **model_kwargs)
    model.load_state_dict(ckpt["state_dict"])
    return model
