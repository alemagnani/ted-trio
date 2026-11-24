"""I44 factorial inference — paired per-user analysis of the booked 2x2 injection
ablation (analysis addendum, NOT a new test look).

Re-scores each of the four I44 cells (pp, Ep, pC, EC) per dataset EXACTLY as
i44_eval_cell.py did (same i41_common trio machinery, same protocol.py split,
unmasked convention, POOL=200), but keeps the per-user HR@5 hit indicator (0/1)
in test-user order.  HARD REPRO GUARD: the aggregate unmasked HR@5 recomputed
from the per-user array must equal the booked results/i44_{ds}_{cell}.json
unmasked overall HR@5 to 4 decimals — any mismatch aborts that dataset (numbers
are never "fixed").  Cells whose checkpoints are not present locally (the Beauty
pp/pC and Toys pp/pC/Ep trios existed only on since-destroyed boxes) abort the
dataset with an explicit missing-artifact report.

From the paired 0/1 arrays (all four cells share the same test users) it computes
per-user contrasts:
    E main effect      ((Ep + EC) - (pp + pC)) / 2
    C main effect      ((pC + EC) - (pp + Ep)) / 2
    interaction        EC - Ep - pC + pp
    standalone         Ep - pp,  pC - pp,  EC - pp
each with a two-sided paired sign-flip randomization p-value (20000 permutations,
seed 0) and a 95% percentile bootstrap CI over users (10000 resamples, seed 0).
All outcomes are reported regardless of sign/size — no selection happens here.

Outputs:
    results/i44_factorial_inference.json   (committed)
    results/i44_factorial_per_user.npz     (gitignored; per-user arrays)
"""
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import i41_common as C
from semplan.dense.frozen_query import encode_split_queries
from semplan.eval import protocol

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
POOL = 200          # identical to i44_eval_cell.py
SEEDS = (42, 43, 44)
N_PERM = 20000
N_BOOT = 10000
STAT_SEED = 0
CHUNK = 1000        # permutation/bootstrap vectorization chunk

# Cell configs = local mirrors of the exact artifacts recorded in the booked
# results/i44_{ds}_{cell}.json files (which carry box-side <box>
# paths).  item_emb / input_emb basenames match the booked JSONs one-to-one.
DATASETS = {
    "beauty": {
        "seq": "data/amazon_beauty_backbone/Beauty.txt",
        "n_items": 12101,
        "plain_item": "data/amazon_beauty_backbone/Beauty_openai-3-large_embeddings.pt",
        "enr_item": "data/amazon_beauty_backbone/Beauty_openai-3-large_i22real_embeddings.pt",
        "plain_input": "data/amazon_beauty_backbone/Beauty_openai-3-large_i44plain_input.pt",
        "enr_input": "data/amazon_beauty_backbone/Beauty_openai-3-large_i24real_input.pt",
        "cells": {
            "pp": [f"outputs/artifacts/i44_beauty_pp_s{s}_iv/j5.pt" for s in SEEDS],
            "Ep": [f"outputs/artifacts/i22_real_openai_k4_s{s}_iv/j5.pt" for s in SEEDS],
            "pC": [f"outputs/artifacts/i44_beauty_pc_s{s}_iv/j5.pt" for s in SEEDS],
            "EC": [f"outputs/artifacts/i24_real_openai_k4_s{s}_iv/j5.pt" for s in SEEDS],
        },
    },
    "toys": {
        "seq": "data/amazon_toys_backbone/Toys_and_Games.txt",
        "n_items": 11924,
        "plain_item": "data/amazon_toys_backbone/Toys_openai-3-large_embeddings.pt",
        "enr_item": "data/amazon_toys_backbone/Toys_openai-3-large_i22real_embeddings.pt",
        "plain_input": "data/amazon_toys_backbone/Toys_openai-3-large_i44plain_input.pt",
        "enr_input": "data/amazon_toys_backbone/Toys_openai-3-large_i24real_input.pt",
        "cells": {
            "pp": [f"outputs/artifacts/i44_toys_pp_s{s}_iv/j5.pt" for s in SEEDS],
            "Ep": [f"outputs/artifacts/i44_toys_ep_s{s}_iv/j5.pt" for s in SEEDS],
            "pC": [f"outputs/artifacts/i44_toys_pc_s{s}_iv/j5.pt" for s in SEEDS],
            "EC": [f"outputs/artifacts/i30_toys_real_openai_k4_s{s}_iv/j5.pt" for s in SEEDS],
        },
    },
    "sports": {
        "seq": "data/amazon_sports_backbone/Sports_and_Outdoors.txt",
        "n_items": 18357,
        "plain_item": "data/amazon_sports_backbone/Sports_openai-3-large_embeddings.pt",
        "enr_item": "data/amazon_sports_backbone/Sports_openai-3-large_i22real_embeddings.pt",
        "plain_input": "data/amazon_sports_backbone/Sports_openai-3-large_i44plain_input.pt",
        "enr_input": "data/amazon_sports_backbone/Sports_openai-3-large_i24real_input.pt",
        "cells": {
            "pp": [f"outputs/artifacts/i44_sports_pp_s{s}_iv/j5.pt" for s in SEEDS],
            "Ep": [f"outputs/artifacts/i44_sports_ep_s{s}_iv/j5.pt" for s in SEEDS],
            "pC": [f"outputs/artifacts/i44_sports_pc_s{s}_iv/j5.pt" for s in SEEDS],
            "EC": [f"outputs/artifacts/i35_sports_real_openai_k4_s{s}_iv/j5.pt" for s in SEEDS],
        },
    },
}

# item-side / input-side matrix per cell (the matrix the cell's towers trained
# against; identical to the booked i44_eval_cell invocations).
CELL_MATS = {
    "pp": ("plain_item", None),
    "Ep": ("enr_item", None),
    "pC": ("plain_item", "plain_input"),
    "EC": ("enr_item", "enr_input"),
}


def score_cell(split, cfg: dict, cell: str, ckpts: list[str]) -> np.ndarray:
    """Per-user unmasked HR@5 hit indicators, machinery verbatim from i44_eval_cell.py."""
    t0 = time.time()
    item_key, input_key = CELL_MATS[cell]
    item = C.l2(C.load_matrix(ROOT / cfg[item_key]))
    # input matrix loaded RAW (concat of per-half l2 blocks) — matches training
    # and the i43/i44 convention exactly; do NOT l2 the concatenated vector.
    inp = C.load_matrix(ROOT / cfg[input_key]) if input_key else None
    item_t = torch.from_numpy(item)
    models = C.load_trio([str(ROOT / p) for p in ckpts], item, input_emb=inp)
    del inp
    gc.collect()

    # Same computation as C.encode_trio + the i44_eval_cell per-user loop, but the
    # [n_users, K, D] query cache (K=4 vectors in the 3072-dim item space, ~1.75GB
    # PER TOWER on Sports) is built in USER BATCHES instead of all at once — this
    # host has far less RAM than the eval boxes.  encode_split_queries and
    # C.trio_z/C.topk_ids are reused verbatim, so every number is identical.
    n = len(split.test)
    hits = np.zeros(n, dtype=np.uint8)
    B = 2048
    for s0 in range(0, n, B):
        sub = SimpleNamespace(test=split.test[s0:s0 + B])
        qs = {nm: encode_split_queries(m, sub, which="test", device="cpu")
              for nm, m in models.items()}
        for j, (_hist, tgt) in enumerate(sub.test):
            ranked = C.topk_ids(C.trio_z(qs, j, item_t), POOL)
            hits[s0 + j] = 1 if tgt in ranked[:5] else 0
        del qs
        gc.collect()
        print(f"    {cell} {min(s0 + B, n)}/{n} [{time.time()-t0:.0f}s]", flush=True)
    del models
    gc.collect()
    print(f"  [{cell}] per-user HR@5 mean={hits.mean():.6f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return hits


def sign_flip_p(d: np.ndarray, n_perm: int = N_PERM, seed: int = STAT_SEED) -> float:
    """Two-sided paired sign-flip randomization p-value for mean(d) = 0."""
    rng = np.random.default_rng(seed)
    obs = abs(float(d.mean()))
    n = len(d)
    d32 = d.astype(np.float32)
    ge = 0
    for start in range(0, n_perm, CHUNK):
        m = min(CHUNK, n_perm - start)
        signs = rng.integers(0, 2, size=(m, n), dtype=np.int8).astype(np.float32) * 2 - 1
        perm = np.abs(signs @ d32) / n
        ge += int((perm >= obs - 1e-12).sum())
    return (1 + ge) / (1 + n_perm)


def boot_ci(d: np.ndarray, n_boot: int = N_BOOT, seed: int = STAT_SEED) -> tuple[float, float]:
    """95% percentile bootstrap CI over users for mean(d)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    d32 = d.astype(np.float32)
    means = np.empty(n_boot, np.float64)
    for start in range(0, n_boot, CHUNK):
        m = min(CHUNK, n_boot - start)
        idx = rng.integers(0, n, size=(m, n), dtype=np.int32)
        means[start:start + m] = d32[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def effects_from_cells(h: dict[str, np.ndarray]) -> dict:
    pp, ep, pc, ec = (h[c].astype(np.float64) for c in ("pp", "Ep", "pC", "EC"))
    contrasts = {
        "E_main": ((ep + ec) - (pp + pc)) / 2.0,
        "C_main": ((pc + ec) - (pp + ep)) / 2.0,
        "interaction": ec - ep - pc + pp,
        "Ep_minus_pp": ep - pp,
        "pC_minus_pp": pc - pp,
        "EC_minus_pp": ec - pp,
    }
    out = {}
    for name, d in contrasts.items():
        lo, hi = boot_ci(d)
        out[name] = {"estimate": float(d.mean()),
                     "ci95": [lo, hi],
                     "p_signflip": sign_flip_p(d),
                     "n_perm": N_PERM, "n_boot": N_BOOT, "stat_seed": STAT_SEED}
        print(f"  {name}: {d.mean():+.5f}  CI95 [{lo:+.5f}, {hi:+.5f}]  "
              f"p={out[name]['p_signflip']:.5f}", flush=True)
    return out


def run_dataset(ds: str, cfg: dict, npz_store: dict) -> dict:
    print(f"=== {ds} ===", flush=True)
    rep = {"status": None, "cells": {}, "effects": None}

    missing = [p for ckpts in cfg["cells"].values() for p in ckpts
               if not (ROOT / p).exists()]
    if missing:
        rep["status"] = "aborted_missing_artifacts"
        rep["missing"] = missing
        print(f"  ABORT {ds}: {len(missing)} checkpoint(s) not present locally "
              f"(existed only on since-destroyed boxes):", flush=True)
        for p in missing:
            print(f"    {p}", flush=True)
        return rep

    split = protocol.build(str(ROOT / cfg["seq"]), cfg["n_items"])
    rep["n"] = len(split.test)
    hits: dict[str, np.ndarray] = {}
    for cell in ("pp", "Ep", "pC", "EC"):
        booked_file = RESULTS / f"i44_{ds}_{cell}.json"
        booked = json.loads(booked_file.read_text())["unmasked"]["overall"]["HR@5"]
        h = score_cell(split, cfg, cell, cfg["cells"][cell])
        recomputed = round(float(h.mean()), 4)
        guard = abs(recomputed - round(booked, 4)) < 5e-9
        rep["cells"][cell] = {"HR@5_per_user_mean": float(h.mean()),
                              "HR@5_recomputed_4dp": recomputed,
                              "HR@5_booked": booked,
                              "booked_file": str(booked_file.relative_to(ROOT)),
                              "repro_guard": "PASS" if guard else "FAIL"}
        print(f"  [{cell}] guard: recomputed {recomputed:.4f} vs booked {booked:.4f} "
              f"-> {'PASS' if guard else 'FAIL'}", flush=True)
        if not guard:
            rep["status"] = "aborted_repro_guard_fail"
            rep["failed_cell"] = cell
            return rep
        hits[cell] = h

    for cell, h in hits.items():
        npz_store[f"{ds}_{cell}"] = h
    rep["effects"] = effects_from_cells(hits)
    rep["status"] = "ok"
    return rep


def main():
    t0 = time.time()
    torch.manual_seed(0)
    report = {"experiment": "I44_factorial_inference",
              "kind": "analysis_addendum (no new test look; all outcomes reported)",
              "pool": POOL, "metric": "unmasked HR@5 per-user hit indicator",
              "inference": {"sign_flip_permutations": N_PERM,
                            "bootstrap_resamples": N_BOOT,
                            "seed": STAT_SEED, "ci": "95% percentile"},
              "datasets": {}}
    npz_store: dict[str, np.ndarray] = {}
    for ds, cfg in DATASETS.items():
        report["datasets"][ds] = run_dataset(ds, cfg, npz_store)

    report["runtime_sec"] = round(time.time() - t0, 1)
    out_json = RESULTS / "i44_factorial_inference.json"
    out_json.write_text(json.dumps(report, indent=2))
    if npz_store:
        np.savez_compressed(RESULTS / "i44_factorial_per_user.npz", **npz_store)
    print(f"[done] {out_json} ({report['runtime_sec']}s); statuses: "
          f"{ {ds: r['status'] for ds, r in report['datasets'].items()} }", flush=True)


if __name__ == "__main__":
    main()
