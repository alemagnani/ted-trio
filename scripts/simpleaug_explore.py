"""SIMPLE-AUG explore (task #87) — INNER-WORLD ONLY, zero real-test looks.

Matched selector sweep: hold the frozen OpenAI encoder + K=3 budget + E-text template +
training-free content-cosine scorer fixed; vary ONLY which 3 items augment each item's text.
Answers "does the directed engagement-graph selection beat naive / looser selectors" in the
development world. Training-free content-cosine is a fast PROXY for the deployed tower — used
here only for RELATIVE selector ordering. Promotion to trained tower + one real-test look is a
separate, later step (only if the ordering is favorable).

World: Beauty_inner.txt (real test event already dropped). protocol.build(...).test = inner
pseudo-test (the −2 event). All selectors built from inner sequences (leak-safe for pseudo-test).

Selectors (K=3, neighbor ids 1-indexed):
  none        : no enrichment (base embeddings)
  pred3       : top-3 predecessors by inner transition count            [OURS]
  contentnn3  : top-3 cosine neighbours in the frozen base OpenAI space  (semantic, no graph)
  bm25_3      : top-3 BM25 lexical neighbours of item text               (lexical, no graph)
  random3     : 3 seeded-random items                                    (null)
  popular3    : global top-3 by inner train count                        (popularity prior)
  upred3      : 3 uniform-random from the item's positive predecessors    (graph membership, not rank)
  translift3  : top-3 incoming edges by P(i|p)/P(i)                       (directional specificity)
  ppmi3       : top-3 symmetric co-purchase PPMI neighbours               (undirected assoc.)

Reuse: none -> Beauty_openai-3-large_embeddings.pt; pred3 -> ..._i22inner_embeddings.pt
(both already on disk with the exact template). New selectors are embedded here and cached.

Embedder: --embedder openai (default; $OPENAI_API_KEY) | fake | st:<model>. Output: results/simpleaug_explore_beauty.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan import spaces as sp  # noqa: E402
from semplan.eval import protocol  # noqa: E402
from semplan.embedders import embed_texts  # noqa: E402

DATA = ROOT / "data/amazon_beauty_backbone"
SEQ = DATA / "Beauty_inner.txt"
META = ROOT / "data/amazon_meta/beauty_id2meta.json"
BASE_EMB = DATA / "Beauty_openai-3-large_embeddings.pt"
PRED3_EMB = DATA / "Beauty_openai-3-large_i22inner_embeddings.pt"
CACHE_DIR = DATA / "simpleaug_cache"
OUT = ROOT / "results/simpleaug_explore_beauty.json"
N = 12101
EMB_MODEL = "text-embedding-3-large"
BATCH = 256
KS = (5, 10)

# ---- catalog text + titles ---------------------------------------------------
meta = json.load(open(META))
base_texts = [""] * N
for k, t in meta.items():
    base_texts[int(k) - 1] = t
titles = [""] * N
for i, t in enumerate(base_texts):
    m = re.search(r"called '(.*?)'\.", t)
    titles[i] = m.group(1) if m else ""


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


# ---- inner-world graph statistics (leak-safe: ids[:-1] drops the pseudo-test target) ----
def load_seqs():
    seqs = []
    for line in open(SEQ):
        ids = [int(x) for x in line.split()][1:]
        seqs.append(ids)
    return seqs


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


def cooc_counts(seqs, window=5):
    """symmetric windowed co-occurrence over inner train events (for PPMI)."""
    co = defaultdict(Counter)
    tot = Counter()
    for ids in seqs:
        s = ids[:-1]
        for a in range(len(s)):
            for b in range(a + 1, min(a + 1 + window, len(s))):
                x, y = s[a], s[b]
                if x != y:
                    co[x][y] += 1
                    co[y][x] += 1
        for i in s:
            tot[i] += 1
    return co, tot


# ---- selectors: item_id(1-idx) -> up to 3 neighbour ids (1-idx), excluding self ----
def build_selectors(seqs, base_emb):
    pred = pred_counts(seqs)
    cnt = train_counts(seqs)
    rng = np.random.default_rng(42)
    pop3 = [p for p, _ in cnt.most_common(3)]

    def sel_pred3(i):
        return [p for p, _ in pred.get(i, Counter()).most_common(3) if titles[p - 1]]

    def sel_random3(i):
        return [int(x) for x in rng.integers(1, N + 1, size=3)]

    def sel_popular3(i):
        return [p for p in pop3 if p != i][:3]

    def sel_upred3(i):
        ps = [p for p in pred.get(i, Counter()) if titles[p - 1]]
        if not ps:
            return []
        k = min(3, len(ps))
        return [int(x) for x in rng.choice(ps, size=k, replace=False)]

    def sel_translift3(i):
        ps = pred.get(i, Counter())
        if not ps:
            return []
        # lift = P(i|p)/P(i) ∝ count(p->i)/count(p_out); rank incoming by count/source-outdeg
        scored = []
        for p, c in ps.items():
            if not titles[p - 1]:
                continue
            outdeg = max(cnt.get(p, 1), 1)
            scored.append((c / outdeg, p))
        scored.sort(reverse=True)
        return [p for _s, p in scored[:3]]

    # content-NN via chunked cosine over frozen base embeddings
    be = l2(base_emb.astype(np.float32))
    cnn = {}

    def build_cnn():
        for s in range(0, N, 512):
            q = be[s:s + 512]
            sims = q @ be.T           # (chunk, N)
            for r in range(q.shape[0]):
                i0 = s + r
                sims[r, i0] = -2.0
                top = np.argpartition(-sims[r], 4)[:4]
                top = top[np.argsort(-sims[r][top])]
                cnn[i0 + 1] = [int(t) + 1 for t in top if int(t) != i0][:3]

    def sel_contentnn3(i):
        return cnn.get(i, [])

    return {
        "pred3": sel_pred3, "random3": sel_random3, "popular3": sel_popular3,
        "upred3": sel_upred3, "translift3": sel_translift3, "contentnn3": sel_contentnn3,
    }, build_cnn


def build_bm25(seqs):
    """BM25 top-3 lexical neighbours of each item's own text (query = item terms)."""
    from scipy.sparse import csr_matrix
    tok_re = re.compile(r"[a-z0-9]+")
    STOP = set("the a an of and or to for with is it this item called priced at manufactured by "
               "in on set of pack".split())
    docs = [[w for w in tok_re.findall(t.lower()) if w not in STOP and len(w) > 1]
            for t in base_texts]
    vocab = {}
    for d in docs:
        for w in d:
            if w not in vocab:
                vocab[w] = len(vocab)
    V = len(vocab)
    rows, cols, vals = [], [], []
    dl = np.zeros(N)
    for i, d in enumerate(docs):
        c = Counter(vocab[w] for w in d)
        dl[i] = sum(c.values())
        for t, f in c.items():
            rows.append(i); cols.append(t); vals.append(f)
    C = csr_matrix((vals, (rows, cols)), shape=(N, V), dtype=np.float32)
    df = np.asarray((C > 0).sum(0)).ravel()
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    avgdl = dl.mean() if dl.mean() > 0 else 1.0
    k1, b = 1.5, 0.75
    Ccoo = C.tocoo()
    denom = Ccoo.data + k1 * (1 - b + b * dl[Ccoo.row] / avgdl)
    wdata = idf[Ccoo.col] * Ccoo.data * (k1 + 1) / denom
    W = csr_matrix((wdata, (Ccoo.row, Ccoo.col)), shape=(N, V), dtype=np.float32)
    Q = csr_matrix((np.ones_like(Ccoo.data), (Ccoo.row, Ccoo.col)), shape=(N, V), dtype=np.float32)
    Wt = W.T.tocsr()
    nn = {}
    for s in range(0, N, 256):
        sc = (Q[s:s + 256] @ Wt).toarray()   # (chunk, N)
        for r in range(sc.shape[0]):
            i0 = s + r
            sc[r, i0] = -1.0
            top = np.argpartition(-sc[r], 4)[:4]
            top = top[np.argsort(-sc[r][top])]
            nn[i0 + 1] = [int(t) + 1 for t in top if int(t) != i0][:3]
    return lambda i: nn.get(i, [])


def build_ppmi(seqs):
    co, tot = cooc_counts(seqs)
    total = sum(tot.values()) or 1
    nn = {}
    for i, neigh in co.items():
        scored = []
        for j, c in neigh.items():
            if not titles[j - 1]:
                continue
            pmi = np.log(max(c * total / (max(tot[i], 1) * max(tot[j], 1)), 1e-9))
            if pmi > 0:
                scored.append((pmi, j))
        scored.sort(reverse=True)
        nn[i] = [j for _s, j in scored[:3]]
    return lambda i: nn.get(i, [])


def enrich_texts(selector):
    out = []
    for i in range(1, N + 1):
        t = base_texts[i - 1] if base_texts[i - 1] else " "
        tops = [p for p in selector(i) if 1 <= p <= N and titles[p - 1]][:3]
        if tops:
            t = t + " Often bought after: " + "; ".join(f"'{titles[p - 1]}'" for p in tops) + "."
        out.append(t)
    return out


def embed_openai(texts, cache_path, embedder="openai"):
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True).float().numpy()
    arr = embed_texts(texts, embedder).numpy().astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(arr), cache_path)
    return arr


def _pop_embedder(argv):
    """`--embedder X` (openai | fake | st:<model>) may appear anywhere in argv."""
    argv = list(argv)
    if "--embedder" in argv:
        i = argv.index("--embedder")
        emb = argv[i + 1]
        del argv[i:i + 2]
        return emb, argv
    return "openai", argv


def eval_trainfree(emb, split, which="test", chunk=1000):
    """training-free recency-weighted content-cosine; HR@5/NDCG@5 unmasked + masked.

    Vectorised: precompute user vectors, then batched BLAS matmul against the item matrix.
    """
    e = l2(emb.astype(np.float32))
    data = getattr(split, which)
    hists, tgts = [], []
    for hist, tgt in data:
        if len(hist) >= 1:
            hists.append(list(hist)); tgts.append(tgt)
    m = len(hists)
    uvs = np.stack([sp.user_vec(e, h) for h in hists]).astype(np.float32) if m else \
        np.zeros((0, e.shape[1]), np.float32)
    tgts = np.asarray(tgts, np.int64)
    accU = {f"HR@{k}": 0.0 for k in KS} | {f"NDCG@{k}": 0.0 for k in KS}
    accM = {kk: 0.0 for kk in accU}
    for s in range(0, m, chunk):
        S = uvs[s:s + chunk] @ e.T                      # (ch, N)
        tg = tgts[s:s + chunk]
        tsc = S[np.arange(S.shape[0]), tg].copy()
        ranks_u = (S > tsc[:, None]).sum(1)             # unmasked ranks (before masking)
        for r, h in enumerate(hists[s:s + chunk]):
            S[r, h] = -1e30                             # tgt is held out, never in hist
        ranks_m = (S > tsc[:, None]).sum(1)
        for ranks, acc in ((ranks_u, accU), (ranks_m, accM)):
            for k in KS:
                hit = ranks < k
                acc[f"HR@{k}"] += float(hit.sum())
                acc[f"NDCG@{k}"] += float((hit / np.log2(ranks + 2)).sum())
    return {"unmasked": {kk: round(v / max(m, 1), 4) for kk, v in accU.items()} | {"n": m},
            "masked": {kk: round(v / max(m, 1), 4) for kk, v in accM.items()} | {"n": m}}


def main():
    embedder, argv = _pop_embedder(sys.argv[1:])
    selectors = argv or ["none", "pred3", "contentnn3", "bm25_3", "random3", "popular3"]
    t0 = time.time()
    split = protocol.build(str(SEQ), n_items=N, max_len=20)
    print(f"inner split: {len(split.test)} pseudo-test users [{time.time()-t0:.0f}s]", flush=True)
    seqs = load_seqs()
    base_emb = torch.load(BASE_EMB, map_location="cpu", weights_only=True).float().numpy()
    sel_map, build_cnn = build_selectors(seqs, base_emb)

    results = json.load(open(OUT)) if OUT.exists() else {"experiment": "simpleaug_explore",
        "world": "inner pseudo-test (Beauty_inner.txt), training-free content-cosine PROXY",
        "note": "relative selector ordering only; zero real-test looks", "selectors": {}}

    for name in selectors:
        if name in results["selectors"]:
            print(f"[{name}] cached in results, skip", flush=True)
            continue
        print(f"[{name}] building...", flush=True)
        if name == "none":
            emb = base_emb
        elif name == "pred3":
            emb = torch.load(PRED3_EMB, map_location="cpu", weights_only=True).float().numpy()
        else:
            if name == "contentnn3":
                build_cnn(); sel = sel_map["contentnn3"]
            elif name == "bm25_3":
                sel = build_bm25(seqs)
            elif name == "ppmi3":
                sel = build_ppmi(seqs)
            elif name in sel_map:
                sel = sel_map[name]
            else:
                print(f"  unknown selector {name}, skip"); continue
            texts = enrich_texts(sel)
            cov = sum(1 for i in range(1, N + 1) if "Often bought after" in texts[i - 1])
            print(f"  {name}: {cov}/{N} items augmented; embedding...", flush=True)
            emb = embed_openai(texts, CACHE_DIR / f"beauty_{name}_inner.pt", embedder)
        res = eval_trainfree(emb, split, "test")
        results["selectors"][name] = res
        print(f"  [{name}] pseudo-test HR@5 unmasked {res['unmasked']['HR@5']} / "
              f"masked {res['masked']['HR@5']}  [{time.time()-t0:.0f}s]", flush=True)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(OUT, "w"), indent=2)

    print("\n=== inner pseudo-test summary (HR@5 unmasked / masked) ===", flush=True)
    for name, r in results["selectors"].items():
        print(f"  {name:12s} {r['unmasked']['HR@5']:.4f} / {r['masked']['HR@5']:.4f}", flush=True)
    print(f"wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
