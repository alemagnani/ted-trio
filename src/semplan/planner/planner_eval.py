"""J2 glue: beam regions -> candidate union -> cosine rank -> ``protocol.evaluate`` rank_fn.

Inference chain per query (idea I1, DENSE-RESIDUAL default per the accepted codex review REV-1 +
E3c evidence: k=2 pure-region 0.045 loses to plain dense 0.0751): the DENSE ARM — plain cosine
top-N in the provided rank space — is ALWAYS on; the trained planner beam-decodes its top-K
interpretable regions, whose member items contribute ADDITIONAL candidates; the two ranked lists
are merged with RRF (``fusion/rrf.py``). Pure-region mode (planner arm only, dense used just as
tail padding so lists cover the metric cutoffs) is an explicit opt-out (``pure_region=True``) kept
as an ablation arm.

``rank_space`` is pluggable: a single [n_items, d] matrix (content embeddings today, LIGER trained
embeddings once J1 syncs) scored with a mean user vector, OR a list of [n_items, d_i] spaces —
the E3c ``blend_recency`` convenience: per space, cosine to a recency-weighted user vector
(RECENCY=0.8, ``semplan.spaces.user_vec``), z-scored, then summed (content + collab lifted dense
HR@10 0.0472 -> 0.0751 in the diagnostic).
"""

from __future__ import annotations

import numpy as np

from semplan import spaces as spaces_mod
from semplan.eval import protocol
from semplan.fusion.rrf import reciprocal_rank_fusion
from semplan.planner.region_planner import (
    RegionMap, RegionPlanner, encode_histories, region_popularity,
)


def _user_vecs(histories: list[list[int]], space: np.ndarray) -> np.ndarray:
    uv = np.zeros((len(histories), space.shape[1]), dtype=np.float32)
    for i, h in enumerate(histories):
        v = space[h].mean(0)
        n = np.linalg.norm(v)
        uv[i] = v / n if n > 0 else v
    return uv


def normalize_space(space: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot product = cosine. Works with any [n_items, d] space."""
    space = np.asarray(space, dtype=np.float32)
    return space / np.maximum(np.linalg.norm(space, axis=1, keepdims=True), 1e-9)


def make_planner_rank_fn(
    model: RegionPlanner,
    regions: RegionMap,
    rank_space,
    k_regions: int = 3,
    content_emb: np.ndarray | None = None,
    max_len: int = 20,
    device: str = "cpu",
    pure_region: bool = False,
    dense_topn: int = 100,
    rrf_k: int = 60,
    depth: int = 100,
    pad_order: np.ndarray | None = None,
    recency: float | None = spaces_mod.RECENCY,
    item_feats: np.ndarray | None = None,
    whiten: bool = False,
    covered: np.ndarray | None = None,
    deeper_dense_control: bool = False,
):
    """Build a ``protocol.evaluate``-compatible ``rank_fn(history_rows, exclude_set)``.

    rank_space  : [n_items, d] array (mean-user-vector cosine) or LIST of arrays (E3c
                  blend_recency: recency-weighted user vec per space, z-score-sum of cosines).
    pure_region : True disables the dense arm's RRF contribution (ablation); the ranked list is
                  then the planner arm alone, padded with the dense ranking only so it covers
                  the metric cutoffs. Default False = dense-residual (dense arm always on).
    dense_topn  : dense-arm list depth fed into the RRF merge (dense-residual mode).
    depth       : per-arm list depth / returned ranking length target.
    recency     : decay for the blended-space user vector (only used when rank_space is a list).
    item_feats  : [n_items, 2] ``build_item_feats`` buckets (REV-2 arm; model must be built with
                  ``use_item_feats=True``).
    whiten      : score the blend with ``blended_scores_whitened`` (P13b winner: whitened
                  spaces + cold content-fallback). rank_space must then be the 2-list
                  [content_w, collab_w] from ``spaces.whiten_blend_spaces`` and ``covered``
                  the collab coverage mask.
    deeper_dense_control : P4's mandatory control (first-class per codex REV-3 / ledger P4):
                  the planner still runs, but ONLY to measure each query's fused candidate
                  budget B = |planner-union ∪ dense_topn|; the returned ranking is plain
                  dense top-B. Any planner claim must beat THIS arm, not just dense_topn —
                  P4 showed training-free region probes lose to deeper dense at matched
                  budget.
    """
    is_blend = isinstance(rank_space, (list, tuple))
    sp_list = [normalize_space(s) for s in (rank_space if is_blend else [rank_space])]
    n_items = sp_list[0].shape[0]
    assert all(s.shape[0] == n_items for s in sp_list), "spaces disagree on n_items"
    if whiten:
        assert is_blend and len(sp_list) == 2 and covered is not None, \
            "whiten=True needs rank_space=[content_w, collab_w] + covered mask"
    if pad_order is None:
        pad_order = np.arange(regions.n_regions)

    def _scores(hist: list[int]) -> np.ndarray:
        if whiten:
            return spaces_mod.blended_scores_whitened(sp_list[0], sp_list[1], covered, hist,
                                                      recency=recency)
        if is_blend:
            return spaces_mod.blended_scores(sp_list, hist, recency=recency)
        return sp_list[0] @ _user_vecs([hist], sp_list[0])[0]

    def rank_fn(hist: list[int], excl: set[int]) -> list[int]:
        enc = encode_histories([hist], regions, content_emb, max_len=max_len, device=device,
                               item_feats=item_feats)
        planned = model.beam_regions(enc["input_ids"], enc["attention_mask"], enc["content"],
                                     k=k_regions, feats=enc["feats"])[0]
        for p in pad_order.tolist():
            if len(planned) >= k_regions:
                break
            if p not in planned:
                planned.append(int(p))

        scores = _scores(hist)                                  # [n_items]
        if excl:
            scores[list(excl)] = -1e9

        cand = [regions.members[r] for r in planned if r in regions.members]
        cand = np.unique(np.concatenate(cand)) if cand else np.array([], dtype=np.int64)
        if excl and cand.size:
            cand = cand[~np.isin(cand, list(excl))]
        planner_arm = cand[np.argsort(-scores[cand])[:depth]].tolist() if cand.size else []

        # dense arm: ALWAYS computed (default inference route; also the tail padding source)
        top = np.argpartition(-scores, min(depth, n_items - 1))[:depth]
        dense_arm = top[np.argsort(-scores[top])].tolist()

        if deeper_dense_control:
            budget = len(set(planner_arm) | set(dense_arm[:dense_topn]))
            ranked = dense_arm[:budget]
        elif pure_region or not planner_arm:
            ranked = planner_arm
        else:
            fused = reciprocal_rank_fusion([planner_arm, dense_arm[:dense_topn]], k=rrf_k)
            ranked = [i for i, _s in fused]
        # pad with dense so the list always covers max(ks)
        seen = set(ranked)
        ranked.extend(i for i in dense_arm if i not in seen)
        return ranked

    return rank_fn


def evaluate_planner(
    model: RegionPlanner,
    regions: RegionMap,
    split: protocol.LOOSplit,
    rank_space,
    which: str = "test",
    ks=(5, 10),
    **rank_kwargs,
) -> dict:
    """End-to-end: planner rank_fn -> full ``protocol.evaluate`` metrics dict."""
    if "pad_order" not in rank_kwargs:
        rank_kwargs["pad_order"] = region_popularity(regions, split)
    rank_fn = make_planner_rank_fn(model, regions, rank_space, **rank_kwargs)
    return protocol.evaluate(rank_fn, split, ks=ks, which=which)
