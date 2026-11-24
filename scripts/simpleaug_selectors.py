"""Shared augmentation selectors (task #87) — used by both the training-free proxy
(simpleaug_explore.py) and the real trained-tower prep (simpleaug_prep.py) so selection
is byte-identical across paths.

Each selector maps item_id (1-indexed) -> up to 3 neighbour ids (1-indexed), excluding self,
built from INNER-world sequences only (leak-safe for the inner pseudo-test: we use ids[:-1],
dropping the held-out −2 target the same way i33_prep does).

Selectors:
  pred3       top-3 predecessors by inner transition count            [OURS]
  contentnn3  top-3 cosine neighbours in the frozen base OpenAI space  (semantic, no graph)
  bm25_3      top-3 BM25 lexical neighbours of item text               (lexical, no graph)
  random3     3 seeded-random items                                    (null)
  popular3    global top-3 by inner train count                        (popularity prior)
  upred3      3 uniform-random from the item's positive predecessors    (graph membership, not rank)
  translift3  top-3 incoming edges by count(p->i)/outdeg(p)  ∝ P(i|p)/P(i)  (directional specificity)
  ppmi3       top-3 symmetric windowed-co-occurrence PPMI neighbours    (undirected assoc.)
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np


def load_titles(base_texts):
    titles = [""] * len(base_texts)
    for i, t in enumerate(base_texts):
        m = re.search(r"called '(.*?)'\.", t)
        titles[i] = m.group(1) if m else ""
    return titles


def pred_counts(seqs):
    pred = defaultdict(Counter)
    for ids in seqs:
        s = ids[:-1]
        for p, i in zip(s, s[1:]):
            pred[i][p] += 1
    return pred


def train_counts(seqs):
    c = Counter()
    for ids in seqs:
        for i in ids[:-1]:
            c[i] += 1
    return c


def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def build_all(seqs, base_emb, titles, N, seed=42):
    """Return {name: selector_fn}. base_emb: np [N, D] frozen base (for contentnn3)."""
    pred = pred_counts(seqs)
    cnt = train_counts(seqs)
    rng = np.random.default_rng(seed)
    pop3_all = [p for p, _ in cnt.most_common(4)]

    def has(p):
        return 1 <= p <= N and titles[p - 1]

    def pred3(i):
        return [p for p, _ in pred.get(i, Counter()).most_common(3) if has(p)]

    def pred1(i):
        return [p for p, _ in pred.get(i, Counter()).most_common(1) if has(p)]

    def random3(i):
        out = []
        while len(out) < 3:
            c = int(rng.integers(1, N + 1))
            if c != i and has(c) and c not in out:
                out.append(c)
        return out

    def popular3(i):
        return [p for p in pop3_all if p != i and has(p)][:3]

    def upred3(i):
        ps = [p for p in pred.get(i, Counter()) if has(p)]
        if not ps:
            return []
        k = min(3, len(ps))
        return [int(x) for x in rng.choice(ps, size=k, replace=False)]

    def translift3(i):
        ps = pred.get(i, Counter())
        scored = []
        for p, c in ps.items():
            if not has(p):
                continue
            scored.append((c / max(cnt.get(p, 1), 1), p))
        scored.sort(reverse=True)
        return [p for _s, p in scored[:3]]

    # content-NN over frozen base embeddings (chunked cosine)
    be = _l2(base_emb.astype(np.float32))
    cnn = {}
    for s in range(0, N, 512):
        sims = be[s:s + 512] @ be.T
        for r in range(sims.shape[0]):
            i0 = s + r
            sims[r, i0] = -2.0
            top = np.argpartition(-sims[r], 5)[:5]
            top = top[np.argsort(-sims[r][top])]
            cnn[i0 + 1] = [int(t) + 1 for t in top if has(int(t) + 1)][:3]

    def contentnn3(i):
        return cnn.get(i, [])

    # PPMI over symmetric windowed co-occurrence
    co = defaultdict(Counter)
    tot = Counter()
    W = 5
    for ids in seqs:
        s = ids[:-1]
        for a in range(len(s)):
            for b in range(a + 1, min(a + 1 + W, len(s))):
                x, y = s[a], s[b]
                if x != y:
                    co[x][y] += 1
                    co[y][x] += 1
        for i in s:
            tot[i] += 1
    total = sum(tot.values()) or 1
    ppmi = {}
    for i, neigh in co.items():
        scored = []
        for j, c in neigh.items():
            if not has(j):
                continue
            v = np.log(max(c * total / (max(tot[i], 1) * max(tot[j], 1)), 1e-9))
            if v > 0:
                scored.append((v, j))
        scored.sort(reverse=True)
        ppmi[i] = [j for _s, j in scored[:3]]

    def ppmi3(i):
        return ppmi.get(i, [])

    # reverse-graph personalized PageRank: seed at i on the reversed transition graph
    # (edges i->p, i.e. "who preceded i"), 3 power iterations, restart 0.15, top-3.
    radj = defaultdict(dict)  # radj[i][p] = weight of i->p (reversed)
    for i, pc in pred.items():
        tot_i = sum(pc.values()) or 1
        for p, c in pc.items():
            radj[i][p] = c / tot_i
    ppr_cache = {}

    def ppr3(i):
        if i in ppr_cache:
            return ppr_cache[i]
        alpha, scores = 0.15, {i: 1.0}
        for _ in range(3):
            nxt = defaultdict(float)
            for node, s in scores.items():
                nxt[i] += alpha * s
                nbrs = radj.get(node, {})
                if nbrs:
                    for p, w in nbrs.items():
                        nxt[p] += (1 - alpha) * s * w
                else:
                    nxt[i] += (1 - alpha) * s
            scores = dict(nxt)
        ranked = sorted((p for p in scores if p != i and has(p)),
                        key=lambda p: -scores[p])[:3]
        ppr_cache[i] = ranked
        return ranked

    return {"pred3": pred3, "pred1": pred1, "random3": random3, "popular3": popular3,
            "upred3": upred3, "translift3": translift3, "contentnn3": contentnn3,
            "ppmi3": ppmi3, "ppr3": ppr3}


def build_catbrand(attrs_path, titles, N, seed=42):
    """Category-stratified random-3: 3 uniform-random items sharing a category attribute."""
    import json
    attrs = json.load(open(attrs_path))
    attr2items = {}
    item_attrs = [[] for _ in range(N + 1)]
    for k, alist in attrs.items():
        i = int(k)
        item_attrs[i] = alist
        for a in alist:
            attr2items.setdefault(a, []).append(i)
    rng = np.random.default_rng(seed)

    def sel(i):
        pool = set()
        for a in item_attrs[i] if i <= N else []:
            pool.update(attr2items.get(a, []))
        pool.discard(i)
        pool = [p for p in pool if 1 <= p <= N and titles[p - 1]]
        if not pool:
            return []
        k = min(3, len(pool))
        return [int(x) for x in rng.choice(pool, size=k, replace=False)]

    return sel


def build_bm25(base_texts, N):
    from scipy.sparse import csr_matrix
    tok_re = re.compile(r"[a-z0-9]+")
    STOP = set("the a an of and or to for with is it this item called priced at manufactured "
               "by in on set pack".split())
    docs = [[w for w in tok_re.findall(t.lower()) if w not in STOP and len(w) > 1]
            for t in base_texts]
    vocab = {}
    for d in docs:
        for w in d:
            vocab.setdefault(w, len(vocab))
    V = len(vocab)
    rows, cols, vals, dl = [], [], [], np.zeros(N)
    for i, d in enumerate(docs):
        c = Counter(vocab[w] for w in d)
        dl[i] = sum(c.values())
        for t, f in c.items():
            rows.append(i); cols.append(t); vals.append(f)
    C = csr_matrix((vals, (rows, cols)), shape=(N, V), dtype=np.float32)
    df = np.asarray((C > 0).sum(0)).ravel()
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    avgdl = dl.mean() or 1.0
    k1, b = 1.5, 0.75
    cx = C.tocoo()
    wdata = idf[cx.col] * cx.data * (k1 + 1) / (cx.data + k1 * (1 - b + b * dl[cx.row] / avgdl))
    Wt = csr_matrix((wdata, (cx.row, cx.col)), shape=(N, V), dtype=np.float32).T.tocsr()
    Q = csr_matrix((np.ones_like(cx.data), (cx.row, cx.col)), shape=(N, V), dtype=np.float32)
    titles_ok = [bool(re.search(r"called '(.*?)'\.", t)) for t in base_texts]
    nn = {}
    for s in range(0, N, 256):
        sc = (Q[s:s + 256] @ Wt).toarray()
        for r in range(sc.shape[0]):
            i0 = s + r
            sc[r, i0] = -1.0
            top = np.argpartition(-sc[r], 5)[:5]
            top = top[np.argsort(-sc[r][top])]
            nn[i0 + 1] = [int(t) + 1 for t in top if titles_ok[int(t)]][:3]
    return lambda i: nn.get(i, [])
