"""I45 — ensemble-confound controls for the WSDM paper (codex review item):
(a) each SINGLE seed tower evaluated alone (no z-avg trio ensemble), reported as
    mean +/- SD across the 3 seeds, and
(b) a uniform WEIGHT SOUP of the 3 seed state_dicts served as ONE tower
    (one encode, one probe budget — no ensemble at inference).

These are descriptive rows quantifying how much of the headline comes from
3-seed z-ensembling vs the method itself. Machinery identical to the i44 cells
(item matrix l2'd, [own;pred3] input matrix RAW); all metrics via
semplan.eval.protocol. Pre-declared cells: {beauty,toys,sports} x {s42,s43,s44,soup},
unmasked + masked in one pass per cell.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import i41_common as C
from semplan.dense.frozen_query import FrozenQueryTower, encode_split_queries
from semplan.eval import protocol

POOL = 200  # deep enough to survive history masking


def load_soup(ckpts, item_l2: np.ndarray, input_emb: np.ndarray | None):
    """Uniform average of the 3 seed state_dicts, loaded into a single tower."""
    raws = [torch.load(str(p), map_location="cpu", weights_only=True) for p in ckpts]
    keys = raws[0]["state_dict"].keys()
    avg = {k: (sum(r["state_dict"][k].float() for r in raws) / len(raws))
           .to(raws[0]["state_dict"][k].dtype) for k in keys}
    cfg = {k: v for k, v in raws[0]["config"].items()
           if k not in ("item_dim", "n_items", "input_dim")}
    model = FrozenQueryTower(item_l2, input_emb=input_emb, **cfg)
    model.load_state_dict(avg, strict=False)
    return model


def eval_queries(qs: torch.Tensor, item_t: torch.Tensor, split) -> dict:
    """Single-tower max-IP full ranking -> POOL lists -> unmasked+masked protocol."""
    lists = []
    for qi in range(len(split.test)):
        sc = torch.einsum("kd,nd->kn", qs[qi], item_t).max(dim=0).values.numpy()
        lists.append(C.topk_ids(sc, POOL))
    return {"unmasked": C.eval_ranked_lists(split, "test", lists, exclude_history=False),
            "masked": C.eval_ranked_lists(split, "test", lists, exclude_history=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--n-items", type=int, required=True)
    ap.add_argument("--item-emb", required=True)
    ap.add_argument("--input-emb", required=True)
    ap.add_argument("--ckpt", nargs=3, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    split = protocol.build(args.seq, args.n_items)
    item = C.l2(C.load_matrix(args.item_emb))
    # input matrix RAW (concat of per-half l2 blocks) — i43/i44 convention;
    # do NOT l2 the concatenated vector.
    inp = C.load_matrix(args.input_emb)
    item_t = torch.from_numpy(item)

    models = C.load_trio(args.ckpt, item, input_emb=inp)
    results = {}
    for nm, m in models.items():
        qs = encode_split_queries(m, split, which="test", device="cpu")
        results[f"seed_{nm}"] = eval_queries(qs, item_t, split)
        print(f"  [seed {nm}] unmasked HR@5="
              f"{results[f'seed_{nm}']['unmasked']['overall']['HR@5']:.4f} "
              f"masked HR@5="
              f"{results[f'seed_{nm}']['masked']['overall']['HR@5']:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    soup = load_soup(args.ckpt, item, inp)
    qs = encode_split_queries(soup, split, which="test", device="cpu")
    results["soup"] = eval_queries(qs, item_t, split)
    print(f"  [soup] unmasked HR@5="
          f"{results['soup']['unmasked']['overall']['HR@5']:.4f} "
          f"masked HR@5={results['soup']['masked']['overall']['HR@5']:.4f}",
          flush=True)

    # mean +/- SD across single seeds (population SD, n=3) for paper reporting
    summary = {}
    for side in ("unmasked", "masked"):
        for met in ("HR@5", "NDCG@5", "HR@10", "NDCG@10"):
            vals = [results[f"seed_{s}"][side]["overall"][met] for s in C.SEEDS]
            summary[f"{side}.{met}"] = {
                "single_mean": float(np.mean(vals)),
                "single_sd": float(np.std(vals)),
                "single_seeds": [float(v) for v in vals],
                "soup": results["soup"][side]["overall"][met]}

    out = {"experiment": "I45_seed_soup", "dataset": args.tag,
           "ckpts": list(args.ckpt), "item_emb": args.item_emb,
           "input_emb": args.input_emb, "pool": POOL,
           "results": results, "summary": summary,
           "runtime_sec": round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] {args.tag} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
