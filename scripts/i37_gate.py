"""I37 — pretrained-LM query-tower gate (Beauty inner world only; modeled on
i36_gate.py, single dataset).

Towers: the 3-seed LM-trunk towers (scripts/i37_train_lmq.py,
outputs/artifacts/i37_lmq_s{42,43,44}/lmq.pt), loaded with Beauty's frozen sides
via i37_train_lmq.load_lm_checkpoint (state_dict excludes item/input_item; the LM
is rebuilt from the stored config — no hub access needed).

PRIMARY criterion (single, numeric): the LM trio (mean per-query z of K4 max-IP
scores, blend w selected on Beauty inner val over the standard grid) must have
Beauty inner pseudo-test HR@5 STRICTLY GREATER than the per-dataset I24 trio bar
BEAUTY_INNER_TRIO_BAR = 0.0802. Pass => the pretrained trunk beats the from-scratch
d256 trunk at identical serving cost; scale (--lm openemb-base) / real-stage decisions
go back to the ledger. Fail => book "pretrained-LM trunk does not beat J5a on
Beauty inner"; singles/union stay descriptive either way.
Secondary (descriptive, NO veto): per-seed singles, trio union recall@500.

Free arm = whitened t5-xl + collab blend (same as i30/i35/i36 gates).
Output: results/i37_gate.json
CPU smoke (after i37_train_lmq.py --smoke for seeds 42/43/44):
  python scripts/i37_gate.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from semplan import spaces as sp  # noqa: E402
from semplan.dense.frozen_query import encode_split_queries  # noqa: E402
from semplan.eval import protocol  # noqa: E402

from i37_train_lmq import SMOKE_ROOT, load_lm_checkpoint  # noqa: E402

SEEDS = (42, 43, 44)
CKPTS = {s: ROOT / f"outputs/artifacts/i37_lmq_s{s}/lmq.pt" for s in SEEDS}
OUT_JSON = ROOT / "results/i37_gate.json"

KS = (5, 10)
WEIGHTS = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
BEAUTY_INNER_TRIO_BAR = 0.0802          # I24 per-dataset trio pseudo-test HR@5
BARS = {"primary": "beauty LM-trio pseudo-test HR@5 at inner-val-selected w > "
                   f"per-dataset I24 trio bar {BEAUTY_INNER_TRIO_BAR}"}

BEAUTY = {
    "seq": ROOT / "data/amazon_beauty_backbone/Beauty_inner.txt",
    "item_emb": ROOT / "data/amazon_beauty_backbone/"
                       "Beauty_openai-3-large_i22inner_embeddings.pt",
    "input_emb": ROOT / "data/amazon_beauty_backbone/"
                        "Beauty_openai-3-large_i24inner_input.pt",
    "content_emb": ROOT / "data/amazon_beauty_backbone/"
                          "Beauty_sentence-t5-xl_embeddings.pt",
    "collab_cache": ROOT / "outputs/artifacts/collab_space_beauty_inner.npz",
    "n_items": 12101,
}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s):
    return (s - s.mean()) / max(float(s.std()), 1e-9)


def sweep_split(split, which, qs, combos, item_t, content_w, collab_w, covered,
                n_items, t0, tag, probe_rank_out=None):
    events = [(list(h), t) for h, t in getattr(split, which)]
    n = len(events)
    names = list(qs)
    hits = {(c, w, k): np.zeros(n, dtype=np.float32)
            for c in combos for w in WEIGHTS for k in KS}
    dcgs = {(c, w, k): np.zeros(n, dtype=np.float32)
            for c in combos for w in WEIGHTS for k in KS}
    for qi, (hist, tgt) in enumerate(events):
        b03 = sp.blended_scores_whitened(content_w, collab_w, covered, hist,
                                         recency=sp.WHITEN_RECENCY)
        b01 = sp.blended_scores_whitened(content_w, collab_w, covered, hist, recency=0.1)
        z_free = zscore(0.9 * b03 + 0.1 * b01)
        z_tow = {}
        for nm in names:
            ip = torch.einsum("kd,nd->kn", qs[nm][qi], item_t).numpy()
            z_tow[nm] = zscore(ip.max(axis=0))
            if probe_rank_out is not None:
                best = int((ip > ip[:, tgt:tgt + 1]).sum(axis=1).min())
                probe_rank_out.setdefault(nm, np.full(n, n_items, np.int64))
                probe_rank_out[nm][qi] = best
        for c, members in combos.items():
            z_ens = np.mean([z_tow[m] for m in members], axis=0)
            for w in WEIGHTS:
                sc = (1.0 - w) * z_free + w * z_ens
                rank = int((sc > sc[tgt]).sum())
                for k in KS:
                    if rank < k:
                        hits[(c, w, k)][qi] = 1.0
                        dcgs[(c, w, k)][qi] = 1.0 / np.log2(rank + 2)
        if (qi + 1) % 5000 == 0:
            print(f"  [i37] {tag}/{which}: {qi+1}/{n} [{time.time()-t0:.0f}s]",
                  flush=True)
    return {c: {str(w): {f"HR@{k}": round(float(hits[(c, w, k)].mean()), 4) for k in KS} |
                        {f"NDCG@{k}": round(float(dcgs[(c, w, k)].mean()), 4) for k in KS}
                for w in WEIGHTS} for c in combos}


def run_dataset(name: str, cfg: dict, ckpts: dict, collab_d: int, t0: float) -> dict:
    split = protocol.build(str(cfg["seq"]), n_items=cfg["n_items"], max_len=20)
    item = l2(torch.load(cfg["item_emb"], map_location="cpu",
                         weights_only=True).float().numpy())
    item_t = torch.from_numpy(item)
    inp = torch.load(cfg["input_emb"], map_location="cpu",
                     weights_only=True).float().numpy()
    models = {str(s): load_lm_checkpoint(str(ckpts[s]), item, input_emb=inp)
              for s in SEEDS}
    combos = {"trio": [str(s) for s in SEEDS]} | {str(s): [str(s)] for s in SEEDS}

    content = l2(torch.load(cfg["content_emb"], map_location="cpu"
                            ).float().numpy().astype(np.float32))
    train_seqs = [list(h) for h, _t in split.val]
    collab, covered = sp.build_collab_embeddings(train_seqs, cfg["n_items"], d=collab_d,
                                                 cache_path=str(cfg["collab_cache"]))
    content_w, collab_w = sp.whiten_blend_spaces(content, collab, covered)
    print(f"[i37] {name} spaces ready [{time.time()-t0:.0f}s]", flush=True)

    q_val = {nm: encode_split_queries(m, split, which="val", device="cpu")
             for nm, m in models.items()}
    val_sweep = sweep_split(split, "val", q_val, combos, item_t, content_w, collab_w,
                            covered, cfg["n_items"], t0, name)
    sel = {}
    for c, sweep in val_sweep.items():
        best = max(WEIGHTS, key=lambda w: (sweep[str(w)]["HR@10"],
                                           sweep[str(w)]["NDCG@5"]))
        sel[c] = best
        print(f"[i37] {name}/{c} inner-val winner w={best}: {sweep[str(best)]}",
              flush=True)

    probe_rank = {}
    q_test = {nm: encode_split_queries(m, split, which="test", device="cpu")
              for nm, m in models.items()}
    test_sweep = sweep_split(split, "test", q_test, combos, item_t, content_w, collab_w,
                             covered, cfg["n_items"], t0, name,
                             probe_rank_out=probe_rank)
    for c in combos:
        print(f"[i37] {name}/{c} pseudo-test w={sel[c]}: {test_sweep[c][str(sel[c])]}",
              flush=True)
    t_rank = np.minimum.reduce([probe_rank[str(s)] for s in SEEDS])
    union_recall_500 = round(float((t_rank < 500).mean()), 4)
    print(f"[i37] {name} trio union recall@500 = {union_recall_500}", flush=True)
    return {"selected_w": sel, "inner_val": val_sweep, "pseudo_test": test_sweep,
            "trio_pseudo_HR@5": test_sweep["trio"][str(sel["trio"])]["HR@5"],
            "trio_pseudo_w1.0": test_sweep["trio"]["1.0"],
            "union_recall_500_trio": union_recall_500}


def main():
    ap = argparse.ArgumentParser(description="I37 LM-trunk inner gate (Beauty)")
    ap.add_argument("--smoke", action="store_true",
                    help="run on the synthetic fixture + smoke towers produced by "
                         "i37_train_lmq.py --smoke (seeds 42/43/44)")
    ap.add_argument("--ckpt-prefix", default=None,
                    help="I48: tower dir prefix (e.g. outputs/artifacts/i48_lmqted_s"
                         ") — seeds 42/43/44 appended, /lmq.pt inside; default I37")
    ap.add_argument("--out", default=None, help="output json path override")
    args = ap.parse_args()
    t0 = time.time()

    if args.smoke:
        from i36_train_joint import make_smoke_fixture
        cfg = make_smoke_fixture(SMOKE_ROOT, n_datasets=1)["fake0"]
        cfg = cfg | {"content_emb": cfg["seq"].parent / "content_emb.pt",
                     "collab_cache": SMOKE_ROOT / "collab_fake0.npz"}
        ckpts = {s: SMOKE_ROOT / f"s{s}/lmq.pt" for s in SEEDS}
        out_json = SMOKE_ROOT / "i37_gate.json"
        collab_d = 8
        name = "fake0"
        bar = 0.0                                       # smoke: plumbing only
    else:
        cfg, ckpts, out_json = BEAUTY, CKPTS, OUT_JSON
        if args.ckpt_prefix:                                    # I48 override
            ckpts = {s: ROOT / f"{args.ckpt_prefix}{s}/lmq.pt" for s in SEEDS}
        if args.out:
            out_json = ROOT / args.out
        collab_d = 128
        name = "beauty"
        bar = BEAUTY_INNER_TRIO_BAR
    for s in SEEDS:
        assert ckpts[s].exists(), f"missing tower checkpoint {ckpts[s]}"

    report = run_dataset(name, cfg, ckpts, collab_d, t0)

    trio_hr5 = report["trio_pseudo_HR@5"]
    verdict = trio_hr5 > bar
    json.dump({"experiment": "I37_lmq_gate", "smoke": args.smoke,
               "bars": BARS, "primary_dataset": name,
               "primary_bar_HR@5": bar, "primary_pass": bool(verdict),
               "primary_trio_pseudo_HR@5": trio_hr5,
               "weights": WEIGHTS, "seeds": list(SEEDS),
               "ckpts": {s: str(ckpts[s]) for s in SEEDS},
               "datasets": {name: report},
               "runtime_sec": round(time.time() - t0, 1)},
              open(out_json, "w"), indent=2)
    print(f"[i37] PRIMARY {'PASS' if verdict else 'FAIL'} ({name} trio "
          f"{trio_hr5} vs bar {bar}); wrote {out_json} [{time.time()-t0:.0f}s]",
          flush=True)


if __name__ == "__main__":
    main()
