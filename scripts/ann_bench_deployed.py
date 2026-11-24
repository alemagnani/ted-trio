"""ANN-BENCH v2 — million-vector servability benchmark of the DEPLOYED trio path.

Replaces the flawed ann_bench_scale.py (never booked; three defects confirmed
2026-08-15 and the run discarded):
  1. WRONG CONFIG: benched a single i22 s42 tower (4 probes). Deployed serving is
     the i24 trio: 3 seeds x K=4 heads = 12 probes, union rescore
     (close3_serving_bench.py convention).
  2. UNCALIBRATED DISTRACTORS: sigma=0.05 iid noise has norm 0.05*sqrt(3072)=2.77
     vs mixture norm ~0.86 -> post-normalization the distractors were ~random
     directions (cos to parent 0.25 vs 0.45 real-pair), trivially prunable, so
     ANN recall was inflated. Here noise is scaled RELATIVE to the mixture norm
     and sigma_rel is CALIBRATED so distractor<->random-real cosine matches the
     real<->real cosine distribution; calibration stats are reported in the JSON.
  3. UNFAIR BASELINE: exact scan was timed with all cores vs single-thread ANN.
     Here the headline comparison is single-thread vs single-thread batch-1
     (exact scan timed on a query subsample); the multi-thread exact number is
     also reported, labeled as such.

Scoring semantics (mirrors close3_serving_bench 2c + offline eval):
  Truth   = full-corpus exact: per-tower max over its K=4 head IPs -> per-tower
            z-score over the FULL corpus -> mean over 3 towers -> top-10.
  Serving = per tower: ANN search each of 12 probes top-CAND; union all ids;
            exact rescore of the union (per-tower max-over-K -> per-tower z over
            the UNION -> mean) -> top-10. Recall@10 vs truth.
            (The union-z vs full-z approximation is thus measured end-to-end.)

This is a systems benchmark, not a recommendation-quality test look: no test
target is scored; one-look discipline untouched.

CLI:
  python scripts/ann_bench_deployed.py --smoke      # 50k, quick check
  python scripts/ann_bench_deployed.py              # full 1M
Output: results/ann_bench_deployed.json (written incrementally).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.dense.frozen_query import load_checkpoint  # noqa: E402
from semplan.eval import protocol  # noqa: E402

SEQ = ROOT / "data/amazon_beauty_backbone/Beauty.txt"
ITEM_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i22real_embeddings.pt"
INPUT_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i24real_input.pt"
SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/i24_real_openai_k4_s{s}_iv/j5.pt" for s in SEEDS}
OUT = ROOT / "results/ann_bench_deployed.json"
N_ITEMS, K_EVAL, CAND = 12101, 10, 500          # CAND=500/probe = deployed TOPK
MIX_LO, MIX_HI = 0.3, 0.7
N_EXACT_ST = 20                                  # queries timed for 1-thread exact scan
N_THREADS = faiss.omp_get_max_threads()


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


# ------------------------------------------------------------------ distractors
def calibrate_sigma_rel(real: np.ndarray, rng, n_probe=2000):
    """Pick relative noise scale so distractor<->random-real cosine matches the
    real<->real pairwise cosine (mean). Returns (sigma_rel, stats dict)."""
    i, j = rng.integers(0, len(real), n_probe), rng.integers(0, len(real), n_probe)
    real_cos = float(np.mean(np.sum(real[i] * real[j], axis=1)))
    best = None
    for sig in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5):
        d = make_distractors(real, n_probe, rng, sig)
        k = rng.integers(0, len(real), n_probe)
        cos = float(np.mean(np.sum(d * real[k], axis=1)))
        if best is None or abs(cos - real_cos) < abs(best[1] - real_cos):
            best = (sig, cos)
    sig = best[0]
    d = make_distractors(real, n_probe, rng, sig)
    stats = {
        "real_pair_cos_mean": round(real_cos, 4),
        "sigma_rel": sig,
        "distractor_random_real_cos_mean": round(
            float(np.mean(np.sum(d * real[rng.integers(0, len(real), n_probe)], axis=1))), 4),
    }
    return sig, stats


def make_distractors(real, m, rng, sigma_rel):
    i, j = rng.integers(0, len(real), m), rng.integers(0, len(real), m)
    a = rng.uniform(MIX_LO, MIX_HI, (m, 1)).astype(np.float32)
    v = a * real[i] + (1 - a) * real[j]
    if sigma_rel > 0:
        eps = rng.standard_normal(v.shape, dtype=np.float32)
        eps /= np.maximum(np.linalg.norm(eps, axis=1, keepdims=True), 1e-9)
        v += sigma_rel * np.linalg.norm(v, axis=1, keepdims=True) * eps
    return l2(v)


def build_corpus(real, n_total, rng, sigma_rel):
    n_real, d = real.shape
    out = np.empty((n_total, d), dtype=np.float32)
    out[:n_real] = real
    t0, chunk = time.time(), 50_000
    for s in range(n_real, n_total, chunk):
        m = min(chunk, n_total - s)
        out[s:s + m] = make_distractors(real, m, rng, sigma_rel)
        if (s - n_real) % 500_000 < chunk:
            print(f"  [corpus] {s + m}/{n_total} [{time.time()-t0:.0f}s]", flush=True)
    # post-hoc verification of calibration on the actual corpus
    k = rng.integers(0, n_real, 2000)
    j = rng.integers(n_real, n_total, 2000) if n_total > n_real else k
    return out, round(float(np.mean(np.sum(out[j] * out[k], axis=1))), 4)


# ------------------------------------------------------------------ exact truth
def fused_z_scores(corpus_chunkT, q_flat, nq):
    """[nq*12, m] IPs -> per-tower max over K=4 -> [3, nq, m]."""
    sc = q_flat @ corpus_chunkT                              # [nq*12, m]
    return sc.reshape(nq, 3, 4, -1).max(axis=2)              # [nq, 3, m]


def exact_truth(corpus, q, k, chunk=100_000):
    """Full-corpus per-tower max-over-K scores (streamed), then full z + mean.
    Memory: keeps [nq, 3, N] fp16 per-tower maxima (nq=2000, N=1e6 -> 12 GB fp16
    would be too much; instead stream twice: pass 1 mean/std, pass 2 top-k)."""
    nq = len(q)
    q_flat = q.reshape(nq * 12, -1)
    # pass 1: per-tower mean/std over full corpus (Welford by chunks)
    n_seen = 0
    s1 = np.zeros((nq, 3), dtype=np.float64)
    s2 = np.zeros((nq, 3), dtype=np.float64)
    t0 = time.time()
    for s in range(0, len(corpus), chunk):
        pt = fused_z_scores(corpus[s:s + chunk].T, q_flat, nq)   # [nq, 3, m]
        s1 += pt.sum(axis=2)
        s2 += (pt.astype(np.float64) ** 2).sum(axis=2)
        n_seen += pt.shape[2]
    mu = (s1 / n_seen)[:, :, None]
    sd = np.sqrt(np.maximum(s2 / n_seen - (s1 / n_seen) ** 2, 1e-18))[:, :, None]
    # pass 2: fused z-mean scores, streamed top-k
    best_s = np.full((nq, k), -np.inf, dtype=np.float32)
    best_i = np.zeros((nq, k), dtype=np.int64)
    for s in range(0, len(corpus), chunk):
        pt = fused_z_scores(corpus[s:s + chunk].T, q_flat, nq)   # [nq, 3, m]
        fused = ((pt - mu) / sd).mean(axis=1).astype(np.float32)  # [nq, m]
        m = fused.shape[1]
        take = min(k, m)
        part = np.argpartition(-fused, take - 1, axis=1)[:, :take]
        cand_s = np.take_along_axis(fused, part, axis=1)
        cat_s = np.concatenate([best_s, cand_s], axis=1)
        cat_i = np.concatenate([best_i, part + s], axis=1)
        sel = np.argpartition(-cat_s, k - 1, axis=1)[:, :k]
        best_s = np.take_along_axis(cat_s, sel, axis=1)
        best_i = np.take_along_axis(cat_i, sel, axis=1)
    order = np.argsort(-best_s, axis=1)
    dt = time.time() - t0
    return np.take_along_axis(best_i, order, axis=1), dt / nq * 1000.0 / 2.0  # per pass


def exact_singlethread_ms(corpus, q, n_time):
    """Batch-1 single-thread exact fused scan (torch, 1 thread), ms/query."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    ct = torch.from_numpy(corpus)
    ts = []
    for u in range(min(n_time, len(q))):
        qs = torch.from_numpy(q[u].reshape(12, -1))
        t0 = time.perf_counter()
        sc = (qs @ ct.T).reshape(3, 4, -1).max(dim=1).values      # [3, N]
        z = (sc - sc.mean(1, keepdim=True)) / sc.std(1, keepdim=True).clamp(min=1e-9)
        z.mean(0).topk(K_EVAL)
        ts.append((time.perf_counter() - t0) * 1000.0)
    torch.set_num_threads(old)
    return {"n_timed": len(ts), "ms_mean": round(float(np.mean(ts)), 1),
            "ms_p50": round(float(np.median(ts)), 1)}


# ------------------------------------------------------------------ serving path
def union_rescore(q_u, corpus, ids):
    """Deployed merge: exact rescore of union ids -> per-tower max/z -> mean -> top10."""
    uni = np.unique(ids[ids >= 0])
    sc = (q_u @ corpus[uni].T).reshape(3, 4, -1).max(axis=1)       # [3, U]
    z = (sc - sc.mean(1, keepdims=True)) / np.maximum(sc.std(1, keepdims=True), 1e-9)
    return uni[np.argsort(-z.mean(0))[:K_EVAL]], len(uni)


def ann_eval(index, corpus, q, gt, label, sweep_key, sweep_vals, set_param):
    rows = []
    faiss.omp_set_num_threads(1)                     # single-thread serving latency
    for val in sweep_vals:
        set_param(val)
        lat, rec, usz = [], [], []
        for u in range(len(q)):
            t0 = time.perf_counter()
            _, ids = index.search(q[u], CAND)        # 12 probes in one batched call
            top, u_sz = union_rescore(q[u], corpus, ids)
            lat.append((time.perf_counter() - t0) * 1000.0)
            rec.append(len(np.intersect1d(top, gt[u])) / K_EVAL)
            usz.append(u_sz)
        lat = np.array(lat)
        row = {"index": label, sweep_key: val,
               "recall@10_vs_exact": round(float(np.mean(rec)), 4),
               "ms_mean": round(float(lat.mean()), 2),
               "ms_p50": round(float(np.median(lat)), 2),
               "ms_p95": round(float(np.percentile(lat, 95)), 2),
               "union_size_mean": int(np.mean(usz))}
        rows.append(row)
        print(f"  [{label} {sweep_key}={val}] recall@10={row['recall@10_vs_exact']} "
              f"ms mean/p50/p95={row['ms_mean']}/{row['ms_p50']}/{row['ms_p95']} "
              f"union~{row['union_size_mean']}", flush=True)
    faiss.omp_set_num_threads(N_THREADS)
    return rows


def index_mb(index):
    return len(faiss.serialize_index(index)) / 1e6


def dump(results):
    dump.path.write_text(json.dumps(results, indent=2))


dump.path = OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=int, default=1_000_000)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-hnsw", action="store_true")
    ap.add_argument("--skip-ivfflat", action="store_true",
                    help="IVF-Flat stores a second full vector copy; skip when RAM-bound")
    ap.add_argument("--out", type=str, default=None,
                    help="override output JSON path (default results/ann_bench_deployed.json)")
    args = ap.parse_args()
    if args.smoke:
        args.corpus, args.queries = 50_000, 200
    if args.out:
        dump.path = Path(args.out)
    t0 = time.time()
    rng = np.random.default_rng(0)

    # --- queries: deployed i24 trio (12 probes/user) over sampled test histories ---
    real = l2(torch.load(ITEM_EMB, map_location="cpu", weights_only=True).float().numpy())
    inp = torch.load(INPUT_EMB, map_location="cpu", weights_only=True).float().numpy()
    d = real.shape[1]
    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)
    models = [load_checkpoint(str(CKPTS[s]), real, input_emb=inp).eval() for s in SEEDS]
    from semplan.dense.frozen_query import pad_histories
    sel = rng.choice(len(split.test), size=min(args.queries, len(split.test)),
                     replace=False)
    sub = [split.test[i] for i in sel]     # sample FIRST, encode only the sample

    @torch.no_grad()
    def enc(m):
        outs = []
        for s in range(0, len(sub), 512):
            hists = [h for h, _t in sub[s:s + 512]]
            rows, mask = pad_histories(hists, m.max_len)
            outs.append(m.encode_queries(rows, mask).cpu())
        return torch.cat(outs).numpy()                            # [nq, 4, d]

    q_seed = [enc(m) for m in models]
    q = np.ascontiguousarray(
        np.concatenate(q_seed, axis=1), dtype=np.float32)         # [nq, 12, d]
    del models, q_seed
    gc.collect()
    print(f"[bench] queries {q.shape} (trio 12-probe) corpus target {args.corpus} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # --- calibrated distractors ---
    sigma_rel, cal = calibrate_sigma_rel(real, rng)
    print(f"[bench] distractor calibration: {cal}", flush=True)
    corpus, corpus_cos = build_corpus(real, args.corpus, rng, sigma_rel)
    cal["corpus_distractor_random_real_cos_mean"] = corpus_cos
    del real
    gc.collect()

    # --- exact truth + baselines ---
    gt, exact_mt_ms = exact_truth(corpus, q, K_EVAL)
    print(f"[bench] exact GT done; multi-thread exact ~{exact_mt_ms:.0f} ms/query/pass "
          f"[{time.time()-t0:.0f}s]", flush=True)
    exact_st = exact_singlethread_ms(corpus, q, N_EXACT_ST)
    print(f"[bench] single-thread batch-1 exact scan: {exact_st}", flush=True)

    results = {
        "config": "deployed i24 trio (3 seeds x K=4 = 12 probes), union rescore "
                  "(per-tower max-over-K -> per-tower z over union -> mean -> top-10)",
        "truth": "full-corpus exact with per-tower z over FULL corpus (offline eval "
                 "semantics); recall thus includes the union-z approximation error",
        "n_corpus": int(args.corpus), "n_real_items": N_ITEMS, "d": int(d),
        "n_queries": int(len(q)), "cand_per_probe": CAND,
        "distractor_calibration": cal,
        "exact_singlethread_batch1": exact_st,
        "exact_multithread_ms_per_query_per_pass": round(exact_mt_ms, 1),
        "exact_flat_mem_MB": round(corpus.nbytes / 1e6, 1),
        "threads_note": f"ANN + single-thread exact use 1 thread; multithread exact "
                        f"uses {N_THREADS}", "rows": []}
    dump(results)

    nlist = 4096 if args.corpus >= 500_000 else 256
    faiss.omp_set_num_threads(N_THREADS)
    # --- IVF Flat (second full vector copy; skippable when RAM-bound) ---
    if not args.skip_ivfflat:
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(corpus[rng.choice(len(corpus), min(200_000, len(corpus)), replace=False)])
        ivf.add(corpus)
        print(f"[bench] IVF{nlist},Flat built ({index_mb(ivf):.0f} MB) [{time.time()-t0:.0f}s]",
              flush=True)
        results["rows"] += ann_eval(ivf, corpus, q, gt, f"IVF{nlist},Flat", "nprobe",
                                    [8, 16, 32, 64, 128], lambda v: setattr(ivf, "nprobe", v))
        results["ivf_flat_mem_MB"] = round(index_mb(ivf), 1)
        del ivf
        gc.collect()
        dump(results)

    # --- IVF PQ ---
    ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, 64, 8,
                             faiss.METRIC_INNER_PRODUCT)
    ivfpq.train(corpus[rng.choice(len(corpus), min(200_000, len(corpus)), replace=False)])
    ivfpq.add(corpus)
    print(f"[bench] IVF{nlist},PQ64 built ({index_mb(ivfpq):.0f} MB) [{time.time()-t0:.0f}s]",
          flush=True)
    results["rows"] += ann_eval(ivfpq, corpus, q, gt, f"IVF{nlist},PQ64", "nprobe",
                                [8, 16, 32, 64, 128], lambda v: setattr(ivfpq, "nprobe", v))
    results["ivf_pq_mem_MB"] = round(index_mb(ivfpq), 1)
    del ivfpq
    gc.collect()
    dump(results)

    # --- HNSW (memory-heavy: second copy of the vectors) ---
    if not args.skip_hnsw:
        hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 80
        hnsw.add(corpus)
        print(f"[bench] HNSW32 built ({index_mb(hnsw):.0f} MB) [{time.time()-t0:.0f}s]",
              flush=True)
        results["rows"] += ann_eval(hnsw, corpus, q, gt, "HNSW32,Flat", "efSearch",
                                    [16, 32, 64, 128],
                                    lambda v: setattr(hnsw.hnsw, "efSearch", v))
        results["hnsw_mem_MB"] = round(index_mb(hnsw), 1)
        del hnsw
        gc.collect()

    results["runtime_sec"] = round(time.time() - t0, 1)
    dump(results)
    print(f"[bench] wrote {dump.path} [{time.time()-t0:.0f}s]", flush=True)
    if args.smoke:
        best = max(r["recall@10_vs_exact"] for r in results["rows"])
        assert best >= 0.85, f"SMOKE: best recall {best} < 0.85 — pipeline suspect"
        print(f"[bench] SMOKE PASS best recall@10={best}")


if __name__ == "__main__":
    main()
