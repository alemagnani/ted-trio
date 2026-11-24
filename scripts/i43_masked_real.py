"""I43 — protocol-alignment rescore of the frozen REAL-test trio under seen-item masking.

LITERATURE_NOTES.md §7 (block "2026-08-07 masked-eval protocol verification"): the
MQSA-TED/BSARec/TV-Rec lineage masks each user's train+val items out of the candidate
set at test time; our published real-test numbers (I24 Beauty 0.0710 / I30 Toys 0.0804 /
I35 Sports 0.0401, HR@5) rank the FULL corpus unmasked — a strictly harder protocol.

This script does NO selection. It rescores exactly the already-selected, already
real-tested frozen trio config per dataset (I24 recipe: mean per-query z of K4 max-IP
tower scores, seeds 42/43/44, [own;pred3] input side, i22real-enriched scoring side):
  (a) unmasked  — MUST reproduce the published HR@5 within 1e-3 (hard fail otherwise);
  (b) masked    — the user's train+val items excluded from the ranked corpus, never
                  the test target (i41_masked_eval semantics: protocol.evaluate with
                  exclude_history=True; repeat-purchase targets are guaranteed misses
                  and their share is reported as repeat_target_share).

Usage on the box (repo root):
  python scripts/i43_masked_real.py --dataset toys
  python scripts/i43_masked_real.py --dataset beauty
  python scripts/i43_masked_real.py --dataset sports
CPU smoke: python scripts/i43_masked_real.py --smoke
Output: results/i43_masked_real_<dataset>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from i41_common import (KS, ROOT, SEEDS, encode_trio, eval_ranked_lists, l2,
                        load_matrix, load_trio, make_smoke_fixture, topk_ids,
                        trio_z)
sys.path.insert(0, str(ROOT / "src"))
from semplan.quick import guard  # noqa: E402  (TT_QUICK relaxes booked-number guards)
from semplan.dense.frozen_query import load_checkpoint  # noqa: E402
from semplan.dense.tricks_query import load_tricks_checkpoint  # noqa: E402
from semplan.eval import protocol  # noqa: E402


def load_trio_any(ckpts, item_l2, input_emb=None) -> dict:
    """load_trio with the i40_gate dual loader: tricks checkpoints (stored config,
    e.g. the toys_ted i40 towers) first, plain FrozenQueryTower fallback."""
    def _load(p):
        try:
            return load_tricks_checkpoint(str(p), item_l2, input_emb=input_emb)
        except (KeyError, TypeError):
            return load_checkpoint(str(p), item_l2, input_emb=input_emb)
    return {str(s): _load(p) for s, p in zip(SEEDS, ckpts)}

SMOKE_ROOT = ROOT / "outputs/tmp/i43_smoke"
MAX_LEN = 20
POOL_K = max(KS) + MAX_LEN  # filtering <=20 seen items still leaves >= max(KS)
REPRO_TOL = 1e-3

NOTE = ("protocol-alignment rescore of the frozen preregistered config; masking "
        "convention = MQSA-TED/BSARec test-phase (train+val excluded); no selection "
        "performed")

# Frozen trio configs — the exact artifacts behind the published real-test numbers
# (i24_test_look.json / i30_real_look.json / i35_real_look.json, w=1.0 arm).
DATASETS = {
    "beauty": {
        "seq": "data/amazon_beauty_backbone/Beauty.txt",
        "n_items": 12101,
        "item_emb": "data/amazon_beauty_backbone/Beauty_openai-3-large_i22real_embeddings.pt",
        "input_emb": "data/amazon_beauty_backbone/Beauty_openai-3-large_i24real_input.pt",
        "ckpts": [f"outputs/artifacts/i24_real_openai_k4_s{s}_iv/j5.pt"
                  for s in (42, 43, 44)],
        "published": {"HR@5": 0.0710, "HR@10": 0.1064,
                      "NDCG@5": 0.0459, "NDCG@10": 0.0572},
        "published_source": "results/i24_test_look.json (w=1.0)",
        "mqsa_ted_bar_HR@5": 0.0752,
    },
    "toys": {
        "seq": "data/amazon_toys_backbone/Toys_and_Games.txt",
        "n_items": 11924,
        "item_emb": "data/amazon_toys_backbone/Toys_openai-3-large_i22real_embeddings.pt",
        "input_emb": "data/amazon_toys_backbone/Toys_openai-3-large_i24real_input.pt",
        "ckpts": [f"outputs/artifacts/i30_toys_real_openai_k4_s{s}_iv/j5.pt"
                  for s in (42, 43, 44)],
        "published": {"HR@5": 0.0804, "HR@10": 0.1210,
                      "NDCG@5": 0.0510, "NDCG@10": 0.0641},
        "published_source": "results/i30_real_look.json (w=1.0)",
        "mqsa_ted_bar_HR@5": 0.0834,
    },
    # I40-TED-TOYS-REAL-LOOK PASS (2026-08-13): plain-ted trio, new Toys headline
    # candidate. Guard reference = the booked one-look numbers; no selection here.
    "toys_ted": {
        "seq": "data/amazon_toys_backbone/Toys_and_Games.txt",
        "n_items": 11924,
        "item_emb": "data/amazon_toys_backbone/Toys_openai-3-large_i22real_embeddings.pt",
        "input_emb": "data/amazon_toys_backbone/Toys_openai-3-large_i24real_input.pt",
        "ckpts": [f"outputs/artifacts/i40_toys_ted_REAL_s{s}/tower.pt"
                  for s in (42, 43, 44)],
        "published": {"HR@5": 0.0838, "HR@10": 0.1232,
                      "NDCG@5": 0.0557, "NDCG@10": 0.0685},
        "published_source": "results/i40_ted_real_look_toys.json (trio mean-z)",
        "mqsa_ted_bar_HR@5": 0.0834,
    },
    "sports": {
        "seq": "data/amazon_sports_backbone/Sports_and_Outdoors.txt",
        "n_items": 18357,
        "item_emb": "data/amazon_sports_backbone/Sports_openai-3-large_i22real_embeddings.pt",
        "input_emb": "data/amazon_sports_backbone/Sports_openai-3-large_i24real_input.pt",
        "ckpts": [f"outputs/artifacts/i35_sports_real_openai_k4_s{s}_iv/j5.pt"
                  for s in (42, 43, 44)],
        "published": {"HR@5": 0.0401, "HR@10": 0.0622,
                      "NDCG@5": 0.0259, "NDCG@10": 0.0330},
        "published_source": "results/i35_real_look.json (trio arm)",
        "mqsa_ted_bar_HR@5": 0.0455,
    },
    "sports_ted": {
        "seq": "data/amazon_sports_backbone/Sports_and_Outdoors.txt",
        "n_items": 18357,
        "item_emb": "data/amazon_sports_backbone/Sports_openai-3-large_i22real_embeddings.pt",
        "input_emb": "data/amazon_sports_backbone/Sports_openai-3-large_i24real_input.pt",
        "ckpts": [f"outputs/artifacts/i40_sports_ted_REAL_s{s}/tower.pt"
                  for s in (42, 43, 44)],
        "published": {"HR@5": 0.0431, "HR@10": 0.0665,
                      "NDCG@5": 0.0284, "NDCG@10": 0.0359},
        "published_source": "results/i40_ted_real_look_sports.json (trio mean-z)",
        "mqsa_ted_bar_HR@5": 0.0455,
    },
}


def main():
    ap = argparse.ArgumentParser(
        description="I43 masked-protocol rescore of the frozen real-test trio")
    ap.add_argument("--dataset", choices=sorted(DATASETS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--published-hr5", type=float, default=None,
                    help="override the reproduction-guard reference HR@5 "
                         "(tests / smoke only; real datasets default to the "
                         "published number and MUST NOT be overridden)")
    ap.add_argument("--out", default=None,
                    help="default results/i43_masked_real_<dataset>.json")
    args = ap.parse_args()
    t0 = time.time()

    if args.smoke:
        fix = make_smoke_fixture(SMOKE_ROOT)
        cfg = {"seq": str(fix["seq"]), "n_items": fix["n_items"],
               "item_emb": str(fix["item_emb"]), "input_emb": str(fix["input_emb"]),
               "ckpts": [str(p) for p in fix["ckpts"]],
               "published": None, "published_source": "none (smoke fixture)",
               "mqsa_ted_bar_HR@5": 0.05}
        dataset = "smoke"
        out = Path(args.out) if args.out else SMOKE_ROOT / "i43_masked_real.json"
        published_hr5 = args.published_hr5  # guard exercised only when given
    else:
        assert args.dataset, "--dataset required (or --smoke)"
        dataset = args.dataset
        cfg = dict(DATASETS[dataset])
        cfg["seq"] = str(ROOT / cfg["seq"])
        cfg["item_emb"] = str(ROOT / cfg["item_emb"])
        cfg["input_emb"] = str(ROOT / cfg["input_emb"])
        cfg["ckpts"] = [str(ROOT / p) for p in cfg["ckpts"]]
        assert args.published_hr5 is None, \
            "--published-hr5 override is smoke/test-only; real guards are frozen"
        published_hr5 = cfg["published"]["HR@5"]
        out = Path(args.out) if args.out else \
            ROOT / f"results/i43_masked_real_{dataset}.json"

    split = protocol.build(cfg["seq"], n_items=cfg["n_items"], max_len=MAX_LEN)
    emb = l2(load_matrix(cfg["item_emb"]))
    item_t = torch.from_numpy(emb)
    inp = load_matrix(cfg["input_emb"])
    models = load_trio_any(cfg["ckpts"], emb, input_emb=inp)
    q_test = encode_trio(models, split, "test")
    print(f"[i43] {dataset}: {len(split.test)} real-test events "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # single scoring pass over the frozen trio; top-POOL_K per user serves both modes
    pool = []
    for qi, (_hist, _tgt) in enumerate(split.test):
        pool.append(topk_ids(trio_z(q_test, qi, item_t), k=POOL_K))
        if (qi + 1) % 5000 == 0:
            print(f"  [i43] test: {qi+1}/{len(split.test)} [{time.time()-t0:.0f}s]",
                  flush=True)

    repeat_share = round(sum(1 for h, t in split.test if t in set(h))
                         / max(len(split.test), 1), 4)

    unmasked = eval_ranked_lists(split, "test", pool, exclude_history=False)
    masked = eval_ranked_lists(split, "test", pool, exclude_history=True)
    delta = {m: round(masked["overall"][m] - unmasked["overall"][m], 4)
             for m in unmasked["overall"]}
    print(f"[i43] unmasked: {unmasked['overall']}", flush=True)
    print(f"[i43] masked  : {masked['overall']}", flush=True)
    print(f"[i43] delta (masked - unmasked): {delta}", flush=True)

    # --- unmasked reproduction guard (hard fail on mismatch) ---
    repro = None
    if published_hr5 is not None:
        got = unmasked["overall"]["HR@5"]
        diff = abs(got - published_hr5)
        repro = {"published_HR@5": published_hr5, "computed_HR@5": got,
                 "abs_diff": round(diff, 6), "tol": REPRO_TOL,
                 "pass": bool(diff <= REPRO_TOL)}
        print(f"[i43] reproduction guard: {repro}", flush=True)
        guard(diff <= REPRO_TOL, (
            f"REPRODUCTION GUARD FAILED: unmasked HR@5 {got} vs published "
            f"{published_hr5} (|diff| {diff:.4f} > {REPRO_TOL}) — wrong artifacts "
            f"or drifted pipeline; refusing to report a masked number"))

    bar = cfg["mqsa_ted_bar_HR@5"]
    report = {
        "experiment": "I43_masked_real",
        "dataset": dataset,
        "smoke": args.smoke,
        "no_selection": True,
        "configs_scored": 1,
        "note": NOTE,
        "inputs": {"seq": cfg["seq"], "n_items": cfg["n_items"],
                   "item_emb": cfg["item_emb"], "input_emb": cfg["input_emb"],
                   "ckpts": list(cfg["ckpts"])},
        "published": cfg["published"],
        "published_source": cfg["published_source"],
        "unmasked_reproduction": repro,
        "repeat_target_share": repeat_share,
        "unmasked": unmasked,
        "masked": masked,
        "delta_masked_minus_unmasked": delta,
        "mqsa_ted_bar_HR@5": bar,
        "delta_vs_bar_HR@5": {
            "unmasked": round(unmasked["overall"]["HR@5"] - bar, 4),
            "masked": round(masked["overall"]["HR@5"] - bar, 4)},
        "masked_beats_bar": bool(masked["overall"]["HR@5"] > bar),
        "runtime_sec": round(time.time() - t0, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"[i43] masked HR@5 {masked['overall']['HR@5']} vs MQSA-TED bar {bar} "
          f"-> masked_beats_bar={report['masked_beats_bar']}", flush=True)
    print(f"[i43] wrote {out} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
