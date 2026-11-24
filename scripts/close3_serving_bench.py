"""CLOSE-3 (part 1) — make the efficiency claims measurable.

CPU-only, no training, no test looks (only VAL histories are used, and only their
LENGTHS/ids matter — no metric is computed). Produces:

  results/serving_bench.json

Contents:
  1. EXACT trainable parameter counts for the I24 frontier towers (J5a arch:
     d=256/2L/4H/dff=512/K=4 over 6144-d inputs), per tower and x3 trio; the I32
     distilled single; and the training-free free arm (0 params; whitening
     matrices disclosed as fixed serving-side linear maps, not learned params).
  2. Serving microbenchmark on THIS machine (disclosed below): query-encoding
     latency (batch 1 / batch 64, single tower and trio), faiss-cpu IndexFlatIP
     probe cost over the 12,101-item Beauty catalog (3072-d) at top-500 for
     12 probes (trio, 3 seeds x K=4) and 4 probes (single tower), merge overhead
     (union -> per-tower max-over-K -> per-tower z-score -> mean -> top-10),
     exact full-catalog scoring as the brute-force alternative, memory
     footprints, and an end-to-end per-request estimate.
  3. TIGER-style generative decode: NO measured wall-clock decode time exists in
     RESULTS.md/ledger (J1 beams were dumped on a GPU box without per-request
     timing), so we emit a clearly-labelled ANALYTIC estimate parameterised by
     the exact J1 checkpoint (params counted from outputs/artifacts/j1/
     tiger_ckpt_best.pt).
  4. Per-seed HR@5 spread tables copied from existing gate/look artifacts
     (no new eval).

Usage:  python scripts/close3_serving_bench.py
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semplan.dense.frozen_query import FrozenQueryTower, load_checkpoint  # noqa: E402
from semplan.dense.chn_de import pad_histories  # noqa: E402
from semplan.eval import protocol  # noqa: E402

SEQ = ROOT / "data/amazon_beauty_backbone/Beauty.txt"
ITEM_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i22real_embeddings.pt"
INPUT_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i24real_input.pt"
SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/i24_real_openai_k4_s{s}_iv/j5.pt" for s in SEEDS}
I32C_CKPT = ROOT / "outputs/artifacts/i32_real_openai_k4_s43_distill/j5.pt"  # selected seed 43
TIGER_CKPT = ROOT / "outputs/artifacts/j1/tiger_ckpt_best.pt"
OUT = ROOT / "results/serving_bench.json"

N_ITEMS, D_ITEM, D_IN, TOPK = 12101, 3072, 6144, 500
N_REQ = 200          # per-request latency sample size
BATCH = 64


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def pct(xs, p):
    return float(np.percentile(np.asarray(xs), p))


def timeit(fn, n, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    return {"mean_ms": float(np.mean(ts)), "median_ms": float(np.median(ts)),
            "p95_ms": pct(ts, 95), "n": n}


def count_params(ckpt_path: Path) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ck["state_dict"]
    per = {k: int(v.numel()) for k, v in sd.items()}
    total = sum(per.values())
    groups = {}
    for k, v in per.items():
        g = k.split(".")[0]
        groups[g] = groups.get(g, 0) + v
    return {"total_trainable": total, "by_module": groups, "config": ck["config"]}


def tiger_params() -> dict:
    ck = torch.load(TIGER_CKPT, map_location="cpu", weights_only=False)
    seen, tot = set(), 0
    enc = dec = 0
    for k, v in ck.items():
        if not hasattr(v, "data_ptr"):
            continue
        if v.data_ptr() in seen:
            continue  # T5 weight tying (shared/embed_tokens/lm_head)
        seen.add(v.data_ptr())
        tot += v.numel()
        if k.startswith("encoder."):
            enc += v.numel()
        elif k.startswith("decoder."):
            dec += v.numel()
    return {"total_unique": int(tot), "encoder": int(enc), "decoder": int(dec),
            "d_model": 128, "d_ff": 1024, "enc_layers": 6, "dec_layers": 6,
            "vocab": int(ck["shared.weight"].shape[0]),
            "source": str(TIGER_CKPT.relative_to(ROOT))}


def main():
    t0 = time.time()
    torch.set_grad_enabled(False)
    nt = int(os.environ.get("BENCH_THREADS", "0"))
    if nt > 0:
        torch.set_num_threads(nt)
        faiss.omp_set_num_threads(nt)
    n_threads = torch.get_num_threads()

    env = {
        "machine": "local workstation (disclosed)",
        "cpu": "13th Gen Intel Core i5-1340P, 16 logical cores",
        "ram_gb": 30,
        "torch": torch.__version__, "faiss": faiss.__version__,
        "torch_threads": n_threads, "faiss_threads": faiss.omp_get_max_threads(),
        "python": platform.python_version(),
        "loadavg_at_start": list(os.getloadavg()),
        "co_tenant_disclosure": "benchmark ran co-located with an active research "
                                "job (i38_soft_router, ~10 of 16 cores; lane "
                                "discipline forbids stopping it); latencies are "
                                "therefore CONSERVATIVE upper bounds under load, "
                                "not idle-machine numbers",
        "note": "CPU-only; batch-1 numbers are single-request wall clock",
    }

    # ---------------- 1. exact parameter counts ----------------
    params = {"i24_towers": {}, }
    for s in SEEDS:
        params["i24_towers"][str(s)] = count_params(CKPTS[s])
    per_tower = params["i24_towers"]["42"]["total_trainable"]
    assert all(v["total_trainable"] == per_tower for v in params["i24_towers"].values())
    params["per_tower_trainable"] = per_tower
    params["trio_trainable"] = 3 * per_tower
    if I32C_CKPT.exists():
        params["i32c_distilled_single"] = count_params(I32C_CKPT)
    params["free_arm"] = {
        "trainable": 0,
        "note": "training-free whitened content+collab blend; the ZCA whitening maps "
                "are fixed linear transforms computed in closed form from item-side "
                "statistics (no gradient training)",
        "whitening_matrices": {
            "content_zca": {"shape": [768, 768], "plus_mean": 768,
                            "floats": 768 * 768 + 768,
                            "source": "sentence-t5-xl 768-d content space"},
            "collab_zca": {"shape": [128, 128], "plus_mean": 128,
                           "floats": 128 * 128 + 128,
                           "source": "PPMI+SVD 128-d collab space"},
        },
    }
    frozen = {
        "note": "frozen EXTERNAL tables excluded from trainable counts (never trained; "
                "the scoring table doubles as the ANN index)",
        "item_scoring_table": {"shape": [N_ITEMS, D_ITEM],
                               "fp32_mb": N_ITEMS * D_ITEM * 4 / 2**20},
        "query_input_table": {"shape": [N_ITEMS, D_IN],
                              "fp32_mb": N_ITEMS * D_IN * 4 / 2**20},
    }

    # ---------------- data + models ----------------
    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)
    hists = [list(h) for h, _t in split.val]  # VAL histories only; lengths/ids only
    rng = np.random.default_rng(0)
    req_idx = rng.choice(len(hists), size=min(len(hists) // BATCH * BATCH, max(N_REQ, BATCH * 8)),  # min(): subsets only
                         replace=False)

    item = l2(torch.load(ITEM_EMB, map_location="cpu", weights_only=True)
              .float().numpy())
    inp = torch.load(INPUT_EMB, map_location="cpu", weights_only=True).float().numpy()
    models = {s: load_checkpoint(str(CKPTS[s]), item, input_emb=inp).eval()
              for s in SEEDS}
    m0 = models[42]
    K = m0.k

    def encode_batch(model, idxs):
        rows, mask = pad_histories([hists[i] for i in idxs], model.max_len)
        return model.encode_queries(rows, mask)

    # ---------------- 2a. query encoding latency ----------------
    it_b1 = iter(np.tile(req_idx, 10))
    enc = {}
    enc["single_batch1"] = timeit(lambda: encode_batch(m0, [next(it_b1)]), N_REQ)
    it_trio = iter(np.tile(req_idx, 10))

    def trio_one():
        i = next(it_trio)
        for s in SEEDS:
            encode_batch(models[s], [i])
    enc["trio_batch1"] = timeit(trio_one, N_REQ)
    it_b64 = iter(np.tile(req_idx, 20).reshape(-1, BATCH))
    enc["single_batch64"] = timeit(lambda: encode_batch(m0, next(it_b64)), 50)
    enc["single_batch64"]["per_user_ms"] = enc["single_batch64"]["median_ms"] / BATCH
    it_b64t = iter(np.tile(req_idx, 30).reshape(-1, BATCH))

    def trio_b64():
        idxs = next(it_b64t)
        for s in SEEDS:
            encode_batch(models[s], idxs)
    enc["trio_batch64"] = timeit(trio_b64, 30)
    enc["trio_batch64"]["per_user_ms"] = enc["trio_batch64"]["median_ms"] / BATCH

    # ---------------- 2b. faiss ANN probe cost ----------------
    index = faiss.IndexFlatIP(D_ITEM)
    index.add(item.astype(np.float32))
    # pre-encode probe sets for N_REQ requests
    probes_trio, probes_single = [], []
    for i in req_idx[:N_REQ]:
        qs = [encode_batch(models[s], [i])[0].numpy() for s in SEEDS]  # 3 x [K, D]
        probes_trio.append(np.concatenate(qs, axis=0).astype(np.float32))   # [12, D]
        probes_single.append(qs[0].astype(np.float32))                       # [4, D]
    it_p12 = iter(probes_trio * 5)
    it_p4 = iter(probes_single * 5)
    ann = {}
    ann["trio_12probe_top500_batch1"] = timeit(
        lambda: index.search(next(it_p12), TOPK), N_REQ)
    ann["single_4probe_top500_batch1"] = timeit(
        lambda: index.search(next(it_p4), TOPK), N_REQ)
    # throughput: all requests' probes in one call
    big12 = np.concatenate(probes_trio, axis=0)   # [N_REQ*12, D]
    big4 = np.concatenate(probes_single, axis=0)  # [N_REQ*4, D]
    r = timeit(lambda: index.search(big12, TOPK), 10, warmup=2)
    ann["trio_batched_throughput"] = {
        "requests_per_call": N_REQ, "call_ms": r["median_ms"],
        "per_request_ms": r["median_ms"] / N_REQ,
        "qps": 1000.0 * N_REQ / r["median_ms"]}
    r = timeit(lambda: index.search(big4, TOPK), 10, warmup=2)
    ann["single_batched_throughput"] = {
        "requests_per_call": N_REQ, "call_ms": r["median_ms"],
        "per_request_ms": r["median_ms"] / N_REQ,
        "qps": 1000.0 * N_REQ / r["median_ms"]}
    ann["index"] = {"type": "IndexFlatIP (exact IP; 12k catalog needs no quantiser)",
                    "n_items": N_ITEMS, "dim": D_ITEM,
                    "memory_mb": N_ITEMS * D_ITEM * 4 / 2**20}

    # ---------------- 2c. merge overhead ----------------
    # production merge: union of 12x500 candidate ids -> exact rescoring of union
    # against each tower's K probes (max over K) -> per-tower z over union -> mean
    # -> top-10.  (Offline eval z-scored over the full catalog; over the retrieved
    # union is the deployable equivalent — flagged in the appendix.)
    _, I12 = index.search(probes_trio[0], TOPK)
    item_t = torch.from_numpy(item)

    def merge_one(probe, ids):
        union = np.unique(ids.ravel())                       # <= 6000
        sub = item[union]                                    # gather [U, D]
        sc = probe @ sub.T                                   # [12, U]
        per_tower = sc.reshape(3, K, -1).max(axis=1)         # [3, U] max over K
        z = (per_tower - per_tower.mean(1, keepdims=True)) / \
            np.maximum(per_tower.std(1, keepdims=True), 1e-9)
        final = z.mean(0)
        top = union[np.argsort(-final)[:10]]
        return top
    pre = []
    for p in probes_trio[:50]:
        _, ids = index.search(p, TOPK)
        pre.append((p, ids))
    it_m = iter(pre * 10)
    merge = {"union_rescore_z_mean_top10": timeit(
        lambda: merge_one(*next(it_m)), 100)}
    merge["union_size_typical"] = int(np.mean(
        [len(np.unique(ids.ravel())) for _p, ids in pre]))

    # exact full-catalog alternative (what the offline eval does; viable at 12k items)
    it_f = iter(probes_trio * 5)

    def full_exact():
        p = next(it_f)
        sc = torch.einsum("kd,nd->kn", torch.from_numpy(p), item_t)  # [12, n]
        per_tower = sc.reshape(3, K, -1).max(dim=1).values
        z = (per_tower - per_tower.mean(1, keepdim=True)) / \
            per_tower.std(1, keepdim=True).clamp(min=1e-9)
        return z.mean(0).topk(10)
    merge["exact_full_catalog_trio_top10"] = timeit(full_exact, N_REQ)

    # ---------------- 2d. memory ----------------
    mem = {
        "faiss_index_mb": N_ITEMS * D_ITEM * 4 / 2**20,
        "query_input_table_mb": N_ITEMS * D_IN * 4 / 2**20,
        "tower_params_mb_fp32": per_tower * 4 / 2**20,
        "trio_params_mb_fp32": 3 * per_tower * 4 / 2**20,
        "total_serving_mb_trio": (N_ITEMS * D_ITEM * 4 + N_ITEMS * D_IN * 4 +
                                  3 * per_tower * 4) / 2**20,
        "note": "input table is shared by all towers; index is shared; fp32 throughout",
    }

    # ---------------- end-to-end ----------------
    e2e = {
        "trio_batch1_ms": enc["trio_batch1"]["median_ms"] +
        ann["trio_12probe_top500_batch1"]["median_ms"] +
        merge["union_rescore_z_mean_top10"]["median_ms"],
        "single_batch1_ms": enc["single_batch1"]["median_ms"] +
        ann["single_4probe_top500_batch1"]["median_ms"] +
        merge["union_rescore_z_mean_top10"]["median_ms"] / 3,
        "components": "encode + ANN probes + merge (batch-1 medians summed; "
                      "single-tower merge scaled 1/3 as a bound)",
    }

    # ---------------- 3. TIGER analytic decode estimate ----------------
    if not TIGER_CKPT.exists():  # release: optional (quick mode / no TIGER run yet)
        tiger = {"skipped": f"TIGER checkpoint not found at {TIGER_CKPT.relative_to(ROOT)}"}
    else:
        tp = tiger_params()
        d, dff = 128, 1024
        hist_tokens = 20 * 4          # 20-item history x 4 SID tokens
        beam, sid_len = 20, 4
        enc_flops = 2 * (tp["encoder"] - 0) * hist_tokens
        # decode: sid_len sequential steps, beam width live after step 1; with KV cache
        # each step is ~2*dec_params per token per beam (+cross-attn over 80 enc states)
        dec_tokens = sid_len * beam
        dec_flops = 2 * tp["decoder"] * dec_tokens
        tower_flops_per_user = (
            2 * (D_IN * 256) * 20 +                       # in_proj over 20 tokens
            2 * 2 * (4 * 256 * 256 + 2 * 256 * 512) * 20 +  # 2 transformer layers
            2 * (256 * D_ITEM) * K)                       # out_proj for K queries
        tiger = {
            "measured_decode_time": None,
            "why": "no per-request decode wall-clock recorded in RESULTS.md or "
                   "ledger/ledger.jsonl for J1 (beams dumped offline on a GPU box)",
            "analytic_estimate_LABELLED": {
                "model": tp,
                "assumptions": {"history_tokens": hist_tokens, "beam": beam,
                                "sid_len": sid_len, "kv_cache": True},
                "encoder_flops": float(enc_flops),
                "decoder_flops": float(dec_flops),
                "total_flops": float(enc_flops + dec_flops),
                "tower_encode_flops_per_user": float(tower_flops_per_user),
                "trio_encode_flops_per_user": float(3 * tower_flops_per_user),
                "flops_ratio_tiger_over_trio_encode":
                    float((enc_flops + dec_flops) / (3 * tower_flops_per_user)),
                "structural_note": "beyond FLOPs, TIGER decode is 4 SEQUENTIAL "
                                   "beam-search steps (latency floor of 4 dependent "
                                   "forward passes) and cannot emit unseen semantic "
                                   "IDs (J1: cold HR 0.0000); the two-tower path is "
                                   "one forward pass + one parallel ANN lookup",
            },
        }

    # ---------------- 4. per-seed HR@5 spread (existing artifacts only) --------
    def spread(vals):
        v = list(vals.values())
        return {"per_seed": vals, "mean": round(float(np.mean(v)), 4),
                "min": min(v), "max": max(v), "range": round(max(v) - min(v), 4),
                "sd": round(float(np.std(v, ddof=1)), 5)}
    res = json.load(open(ROOT / "results/i24_gate.json"))
    b_seeds = {s: res["pseudo_test"][s]["1.0"]["HR@5"] for s in ("42", "43", "44")}
    b_trio = res["pseudo_test"]["42+43+44"]["1.0"]["HR@5"]
    r30 = json.load(open(ROOT / "results/i30_gate.json"))
    t_seeds = {s: r30["pseudo_test"][s]["1.0"]["HR@5"] for s in ("42", "43", "44")}
    t_trio = r30["pseudo_test"]["toys_trio"]["1.0"]["HR@5"]
    r35 = json.load(open(ROOT / "results/i35_gate.json"))
    s_seeds = {s: r35["pseudo_test"][s]["1.0"]["HR@5"] for s in ("42", "43", "44")}
    s_trio = r35["pseudo_test"]["sports_trio"]["1.0"]["HR@5"]
    look35 = json.load(open(ROOT / "results/i35_real_look.json"))
    s_real = {s: look35["arms"][s]["HR@5"] for s in ("42", "43", "44")}
    seeds_tab = {
        "protocol_note": "pseudo-test = inner-world held-out (selection world); real = "
                         "the single preregistered real-test look already spent — no "
                         "new test looks taken here, numbers copied from artifacts",
        "beauty": {"pseudo_test_singles": spread(b_seeds), "pseudo_test_trio": b_trio,
                   "real_test_trio": 0.0710,
                   "real_singles": "not recorded (look budget spent on trio only)"},
        "toys": {"pseudo_test_singles": spread(t_seeds), "pseudo_test_trio": t_trio,
                 "real_test_trio": 0.0804,
                 "real_singles": "not recorded (look budget spent on trio only)"},
        "sports": {"pseudo_test_singles": spread(s_seeds), "pseudo_test_trio": s_trio,
                   "real_test_singles": spread(s_real),
                   "real_test_trio": look35["arms"]["trio"]["HR@5"]},
    }

    out = {"experiment": "CLOSE3_serving_bench",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "env": env, "params": params, "frozen_tables": frozen,
           "query_encoding_ms": enc, "ann": ann, "merge": merge, "memory_mb": mem,
           "end_to_end_batch1_ms": e2e, "tiger_generative_comparison": tiger,
           "seed_spread_HR@5": seeds_tab,
           "runtime_sec": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[close3] wrote {OUT} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
