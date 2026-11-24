"""Predecessor mining, verbalization (E) and predecessor-centroid query inputs (C).

Single implementation of the paper's two injection sites, shared by the prep scripts
(i22_enrich_embed_catalog, i24_build_input_emb, i30_prep_toys, i35_prep_sports,
i50_prep_expand, i44_ablation_prep, i25_dump_enriched_texts). It is a verbatim
factoring-out of the code those scripts carried inline; tests/test_real_data.py checks
that it reproduces the paper's enriched texts and input matrices exactly.

Conventions (paper §2.1 / §2.2):
  * sequence files are LIGER-style lines "user_id item item ...", ids 1-indexed;
  * mining drops the user id AND the world's held-out final event (ids[1:][:-1]) before
    counting consecutive pairs p -> i, so no (pseudo-)test transition is ever counted;
  * verbalization appends " Often bought after: 'T1'; 'T2'; 'T3'." (Yelp: "Often visited
    after") using the top-3 predecessors that have a title; items with none keep plain text;
  * query-input token = [L2(e_i) ; L2(mean(L2(e_p) for the top-3 predecessors p))], with
    the item's own L2 embedding as the centroid fallback when it has no predecessor.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

TOPK = 3
AFTER_BOUGHT = "Often bought after"
AFTER_VISITED = "Often visited after"
_TITLE = re.compile(r"called '(.*?)'\.")


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def read_lines(seq: str | Path | Iterable[str]) -> Iterable[str]:
    if isinstance(seq, (str, Path)):
        with open(seq) as f:
            yield from f
    else:
        yield from seq


def pred_counts(seq) -> dict[int, Counter]:
    """pred[i][p] = number of p -> i consecutive transitions (1-indexed item ids), counted on
    each user's sequence with the user id and the held-out FINAL event removed."""
    pred: dict[int, Counter] = defaultdict(Counter)
    for line in read_lines(seq):
        ids = [int(x) for x in line.split()]
        ids = ids[1:] if len(ids) > 1 else ids   # drop user id
        ids = ids[:-1]                            # drop held-out final event
        for p, i in zip(ids, ids[1:]):
            pred[i][p] += 1
    return pred


def base_texts_from_meta(meta: dict[str, str], n_items: int) -> list[str]:
    """id2meta json (1-indexed string keys) -> row-ordered list (row = id - 1)."""
    texts = [""] * n_items
    for k, t in meta.items():
        texts[int(k) - 1] = t
    return texts


def titles_from_texts(base_texts: list[str]) -> list[str]:
    out = []
    for t in base_texts:
        m = _TITLE.search(t)
        out.append(m.group(1) if m else "")
    return out


def enriched_texts(base_texts: list[str], titles: list[str], pred: dict[int, Counter],
                   after_phrase: str = AFTER_BOUGHT, topk: int = TOPK) -> tuple[list[str], int]:
    """E site: append the titles of the top-k predecessors. Returns (texts, n_enriched)."""
    texts, enriched_n = [], 0
    for i in range(len(base_texts)):
        t = base_texts[i] if base_texts[i] else " "
        tops = [p for p, _ in pred.get(i + 1, Counter()).most_common(topk) if titles[p - 1]]
        if tops:
            t = t + f" {after_phrase}: " + "; ".join(f"'{titles[p - 1]}'" for p in tops) + "."
            enriched_n += 1
        texts.append(t)
    return texts, enriched_n


def centroid_inputs(emb: np.ndarray, pred: dict[int, Counter], k: int = TOPK
                    ) -> tuple[np.ndarray, int]:
    """C site: [N, 2d] matrix [L2(e) ; L2(mean of the top-k predecessors' L2(e))].
    Returns (matrix float32, number of items with >=1 predecessor)."""
    emb = l2(np.asarray(emb, dtype=np.float32))
    aug = emb.copy()  # fallback: own embedding
    n_aug = 0
    for i in range(emb.shape[0]):
        ps = [p - 1 for p, _ in pred.get(i + 1, Counter()).most_common(k)]
        if ps:
            aug[i] = emb[ps].mean(axis=0)
            n_aug += 1
    return np.concatenate([emb, l2(aug)], axis=1).astype(np.float32), n_aug


def drop_last_event(lines: Iterable[str]) -> list[str]:
    """Inner world: each user's LAST event removed (all users kept)."""
    out = []
    for line in lines:
        toks = line.split()
        if toks:
            out.append(" ".join(toks[:-1]))
    return out
