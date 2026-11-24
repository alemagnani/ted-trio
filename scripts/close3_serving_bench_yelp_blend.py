"""CLOSE-3-YELP-BLEND — serving bench for the FULL deployed Yelp configuration
(trio + training-free arm, w=0.9 blend), closing the Table-5 gap where only the
trio arm was measured.

Same measurement discipline as close3_serving_bench_expand.py: CPU-only, no
training, ZERO test looks (VAL histories used for lengths/ids only). Deployed
semantics per the booked i51 real look: per-tower full-corpus max-over-K IP ->
per-tower z -> mean -> z_trio; free arm = whitened content+collab blend at two
recencies (0.9*b03 + 0.1*b01) -> z_free; final = 0.9*z_trio + 0.1*z_free ->
top-10. At 20,033 items full-corpus scoring IS the serving path (exact FlatIP
regime, cf. Table 5 note), so components are timed as dense matvecs.

Usage: BENCH_THREADS=1 python scripts/close3_serving_bench_yelp_blend.py
Output: results/serving_bench_yelp_blend.json
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

from semplan import spaces as sp  # noqa: E402
from semplan.dense.frozen_query import load_checkpoint  # noqa: E402
from semplan.dense.chn_de import pad_histories  # noqa: E402
from semplan.eval import protocol  # noqa: E402

from close3_serving_bench import timeit, count_params  # noqa: E402

SEEDS = (42, 43, 44)
N_ITEMS, W = 20033, 0.9
N_REQ = 200
DATA = ROOT / "data/yelp_backbone"
SEQ = DATA / "Yelp.txt"
ITEM_EMB = DATA / "Yelp_openai-3-large_i22real_embeddings.pt"
INPUT_EMB = DATA / "Yelp_openai-3-large_i24real_input.pt"
EMB_T5 = DATA / "Yelp_sentence-t5-xl_embeddings.pt"
COLLAB_CACHE = ROOT / "outputs/artifacts/collab_space_yelp_real.npz"
CKPTS = {s: ROOT / f"outputs/artifacts/i51_yelp_real_openai_k4_s{s}_iv/j5.pt"
         for s in SEEDS}
OUT = ROOT / "results/serving_bench_yelp_blend.json"


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s):
    st = float(s.std())
    return (s - s.mean()) / st if st > 1e-9 else s - s.mean()


def main():
    t0 = time.time()
    torch.set_grad_enabled(False)
    nt = int(os.environ.get("BENCH_THREADS", "0"))
    if nt > 0:
        torch.set_num_threads(nt)
        faiss.omp_set_num_threads(nt)

    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)
    hists = [list(h) for h, _t in split.val]  # lengths/ids only, no metrics
    rng = np.random.default_rng(0)
    req_idx = rng.choice(len(hists), size=min(len(hists), N_REQ), replace=False)  # min(): subsets only

    item = l2(torch.load(ITEM_EMB, map_location="cpu",
                         weights_only=True).float().numpy())
    item_t = torch.from_numpy(item)
    inp = torch.load(INPUT_EMB, map_location="cpu",
                     weights_only=True).float().numpy()
    models = {s: load_checkpoint(str(CKPTS[s]), item, input_emb=inp).eval()
              for s in SEEDS}
    per_tower = count_params(CKPTS[42])["total_trainable"]

    # free-arm spaces, exactly as the booked real look built them (train
    # histories = val split up to -2; cache is the same artifact)
    content = l2(torch.load(EMB_T5, map_location="cpu").float().numpy()
                 .astype(np.float32))
    collab, covered = sp.build_collab_embeddings(hists, N_ITEMS,
                                                 cache_path=str(COLLAB_CACHE))
    content_w, collab_w = sp.whiten_blend_spaces(content, collab, covered)
    print(f"[yblend] spaces ready: content_w {content_w.shape}, collab_w "
          f"{collab_w.shape} [{time.time()-t0:.0f}s]", flush=True)

    def encode_one(model, i):
        rows, mask = pad_histories([hists[i]], model.max_len)
        return model.encode_queries(rows, mask)[0]

    def trio_scores(i):
        zs = []
        for s in SEEDS:
            q = encode_one(models[s], i)                    # [K, D]
            ip = torch.einsum("kd,nd->kn", q, item_t).numpy()
            zs.append(zscore(ip.max(axis=0)))
        return np.mean(zs, axis=0)

    def free_scores(i):
        h = hists[i]
        b03 = sp.blended_scores_whitened(content_w, collab_w, covered, h,
                                         recency=sp.WHITEN_RECENCY)
        b01 = sp.blended_scores_whitened(content_w, collab_w, covered, h,
                                         recency=0.1)
        return zscore(0.9 * b03 + 0.1 * b01)

    def blend_one(i):
        s = W * trio_scores(i) + (1.0 - W) * free_scores(i)
        return np.argpartition(-s, 10)[:10]

    it1 = iter(np.tile(req_idx, 10))
    comp_trio = timeit(lambda: trio_scores(next(it1)), N_REQ)
    it2 = iter(np.tile(req_idx, 10))
    comp_free = timeit(lambda: free_scores(next(it2)), N_REQ)
    it3 = iter(np.tile(req_idx, 10))
    e2e = timeit(lambda: blend_one(next(it3)), N_REQ)

    mem = {
        "item_matrix_mb": item.nbytes / 2**20,
        "query_input_table_mb": inp.nbytes / 2**20,
        "trio_params_mb_fp32": 3 * per_tower * 4 / 2**20,
        "free_arm_content_w_mb": content_w.nbytes / 2**20,
        "free_arm_collab_w_mb": collab_w.nbytes / 2**20,
        "free_arm_covered_mb": np.asarray(covered).nbytes / 2**20,
    }
    mem["total_serving_mb_trio_only"] = (mem["item_matrix_mb"] +
                                         mem["query_input_table_mb"] +
                                         mem["trio_params_mb_fp32"])
    mem["total_serving_mb_full_blend"] = sum(
        v for k, v in mem.items() if not k.startswith("total"))

    env = {
        "machine": "local workstation (disclosed)",
        "cpu": "13th Gen Intel Core i5-1340P, 16 logical cores",
        "torch": torch.__version__, "faiss": faiss.__version__,
        "torch_threads": torch.get_num_threads(),
        "faiss_threads": faiss.omp_get_max_threads(),
        "python": platform.python_version(),
        "loadavg_at_start": list(os.getloadavg()),
        "note": "CPU-only; batch-1 single-request wall clock; VAL histories, "
                "lengths/ids only, zero test looks",
    }
    out = {
        "experiment": "CLOSE3_serving_bench_YELP_BLEND",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": env,
        "config": "deployed Yelp blend: 0.9 * z(trio full-corpus max-over-K "
                  "mean-z) + 0.1 * z(free arm 0.9*b03+0.1*b01), top-10",
        "n_items": N_ITEMS,
        "params_per_tower": per_tower,
        "components_batch1_ms": {
            "trio_fullcorpus_score_z": comp_trio,
            "free_arm_score_z": comp_free,
        },
        "end_to_end_blend_batch1_ms": e2e,
        "memory_mb": mem,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[yblend] blend e2e {e2e['median_ms']:.1f} ms/query "
          f"(trio {comp_trio['median_ms']:.1f} + free {comp_free['median_ms']:.1f}); "
          f"full-blend serving {mem['total_serving_mb_full_blend']:.0f} MB "
          f"(trio-only {mem['total_serving_mb_trio_only']:.0f}); "
          f"wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
