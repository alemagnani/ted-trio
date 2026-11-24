"""CLOSE-3-EXPAND — serving bench for the paper-2 datasets (Instruments, Yelp).

Same measurement structure as close3_serving_bench.py (CPU-only, no training,
no test looks: VAL histories used for lengths/ids only). Per dataset:
exact tower param counts (i50/i51 real trios), query-encode latency,
IndexFlatIP probe cost at the dataset's catalog size, union-rescore merge,
memory footprint, batch-1 end-to-end sum.

Usage: python scripts/close3_serving_bench_expand.py
Output: results/serving_bench_expand.json
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semplan.dense.frozen_query import load_checkpoint  # noqa: E402
from semplan.dense.chn_de import pad_histories  # noqa: E402
from semplan.eval import protocol  # noqa: E402

from close3_serving_bench import l2, timeit, count_params  # noqa: E402

SEEDS = (42, 43, 44)
D_ITEM, D_IN, TOPK = 3072, 6144, 500
N_REQ, BATCH = 200, 64

DATASETS = {
    "instruments": dict(
        seq=ROOT / "data/amazon_instruments_backbone/Musical_Instruments.txt",
        item_emb=ROOT / "data/amazon_instruments_backbone/"
                        "Instruments_openai-3-large_i22real_embeddings.pt",
        input_emb=ROOT / "data/amazon_instruments_backbone/"
                         "Instruments_openai-3-large_i24real_input.pt",
        ckpt="i50_instruments_real_openai_k4_s{s}_iv",
        n_items=9922),
    "yelp": dict(
        seq=ROOT / "data/yelp_backbone/Yelp.txt",
        item_emb=ROOT / "data/yelp_backbone/Yelp_openai-3-large_i22real_embeddings.pt",
        input_emb=ROOT / "data/yelp_backbone/Yelp_openai-3-large_i24real_input.pt",
        ckpt="i51_yelp_real_openai_k4_s{s}_iv",
        n_items=20033),
}


def bench_dataset(name, cfg, t0):
    n_items = cfg["n_items"]
    split = protocol.build(str(cfg["seq"]), n_items=n_items, max_len=20)
    hists = [list(h) for h, _t in split.val]  # lengths/ids only, no metrics
    rng = np.random.default_rng(0)
    req_idx = rng.choice(len(hists), size=min(len(hists) // BATCH * BATCH, max(N_REQ, BATCH * 8)),  # min(): subsets only
                         replace=False)

    item = l2(torch.load(cfg["item_emb"], map_location="cpu",
                         weights_only=True).float().numpy())
    inp = torch.load(cfg["input_emb"], map_location="cpu",
                     weights_only=True).float().numpy()
    ckpts = {s: ROOT / "outputs/artifacts" / cfg["ckpt"].format(s=s) / "j5.pt"
             for s in SEEDS}
    models = {s: load_checkpoint(str(ckpts[s]), item, input_emb=inp).eval()
              for s in SEEDS}
    m0 = models[42]
    K = m0.k

    params = {str(s): count_params(ckpts[s]) for s in SEEDS}
    per_tower = params["42"]["total_trainable"]
    assert all(v["total_trainable"] == per_tower for v in params.values())

    def encode_batch(model, idxs):
        rows, mask = pad_histories([hists[i] for i in idxs], model.max_len)
        return model.encode_queries(rows, mask)

    enc = {}
    it_b1 = iter(np.tile(req_idx, 10))
    enc["single_batch1"] = timeit(lambda: encode_batch(m0, [next(it_b1)]), N_REQ)
    it_trio = iter(np.tile(req_idx, 10))

    def trio_one():
        i = next(it_trio)
        for s in SEEDS:
            encode_batch(models[s], [i])
    enc["trio_batch1"] = timeit(trio_one, N_REQ)

    index = faiss.IndexFlatIP(D_ITEM)
    index.add(item.astype(np.float32))
    probes_trio = []
    for i in req_idx[:N_REQ]:
        qs = [encode_batch(models[s], [i])[0].numpy() for s in SEEDS]
        probes_trio.append(np.concatenate(qs, axis=0).astype(np.float32))  # [12, D]
    it_p12 = iter(probes_trio * 5)
    ann = {"trio_12probe_top500_batch1": timeit(
        lambda: index.search(next(it_p12), TOPK), N_REQ)}
    big12 = np.concatenate(probes_trio, axis=0)
    r = timeit(lambda: index.search(big12, TOPK), 10, warmup=2)
    ann["trio_batched_throughput"] = {
        "requests_per_call": N_REQ, "call_ms": r["median_ms"],
        "per_request_ms": r["median_ms"] / N_REQ,
        "qps": 1000.0 * N_REQ / r["median_ms"]}
    ann["index"] = {"type": "IndexFlatIP (exact IP)", "n_items": n_items,
                    "dim": D_ITEM, "memory_mb": n_items * D_ITEM * 4 / 2**20}

    def merge_one(probe, ids):
        union = np.unique(ids.ravel())
        sub = item[union]
        sc = probe @ sub.T
        per_tower = sc.reshape(3, K, -1).max(axis=1)
        z = (per_tower - per_tower.mean(1, keepdims=True)) / \
            np.maximum(per_tower.std(1, keepdims=True), 1e-9)
        return union[np.argsort(-z.mean(0))[:10]]
    pre = []
    for p in probes_trio[:50]:
        _, ids = index.search(p, TOPK)
        pre.append((p, ids))
    it_m = iter(pre * 10)
    merge = {"union_rescore_z_mean_top10": timeit(
        lambda: merge_one(*next(it_m)), 100)}
    merge["union_size_typical"] = int(np.mean(
        [len(np.unique(ids.ravel())) for _p, ids in pre]))

    mem = {
        "faiss_index_mb": n_items * D_ITEM * 4 / 2**20,
        "query_input_table_mb": n_items * D_IN * 4 / 2**20,
        "trio_params_mb_fp32": 3 * per_tower * 4 / 2**20,
        "total_serving_mb_trio": (n_items * D_ITEM * 4 + n_items * D_IN * 4 +
                                  3 * per_tower * 4) / 2**20,
    }
    e2e = {
        "trio_batch1_ms": enc["trio_batch1"]["median_ms"] +
        ann["trio_12probe_top500_batch1"]["median_ms"] +
        merge["union_rescore_z_mean_top10"]["median_ms"],
        "components": "encode + ANN probes + merge (batch-1 medians summed)",
    }
    print(f"[close3x] {name}: trio e2e {e2e['trio_batch1_ms']:.1f} ms/query, "
          f"serving {mem['total_serving_mb_trio']:.0f} MB, "
          f"per-tower {per_tower/1e6:.2f}M params [{time.time()-t0:.0f}s]",
          flush=True)
    return {"params_per_tower": per_tower, "params_trio": 3 * per_tower,
            "query_encoding_ms": enc, "ann": ann, "merge": merge,
            "memory_mb": mem, "end_to_end_batch1_ms": e2e}


def main():
    t0 = time.time()
    torch.set_grad_enabled(False)
    nt = int(os.environ.get("BENCH_THREADS", "0"))
    if nt > 0:
        torch.set_num_threads(nt)
        faiss.omp_set_num_threads(nt)
    env = {
        "machine": "local workstation (disclosed)",
        "cpu": "13th Gen Intel Core i5-1340P, 16 logical cores",
        "torch": torch.__version__, "faiss": faiss.__version__,
        "torch_threads": torch.get_num_threads(),
        "faiss_threads": faiss.omp_get_max_threads(),
        "python": platform.python_version(),
        "loadavg_at_start": list(os.getloadavg()),
        "note": "CPU-only; batch-1 numbers are single-request wall clock",
    }
    out = {"experiment": "CLOSE3_serving_bench_EXPAND",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "env": env,
           "datasets": {nm: bench_dataset(nm, cfg, t0)
                        for nm, cfg in DATASETS.items()},
           "runtime_sec": round(time.time() - t0, 1)}
    out_path = ROOT / "results/serving_bench_expand.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[close3x] wrote {out_path} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
