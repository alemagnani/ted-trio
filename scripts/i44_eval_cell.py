"""I44 — evaluate one 2x2-ablation cell: trio z-avg over a K4 tower triple, REAL test,
unmasked + masked in one pass (one test look per cell; cells are pre-declared ablation rows).

Scoring item side = the matrix the cell's towers trained against (plain cells score the
plain catalog). Machinery identical to the i23/i24 gates via i41_common; all metrics flow
through semplan.eval.protocol.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import i41_common as C
from semplan.eval import protocol

POOL = 200  # deep enough to survive history masking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--n-items", type=int, required=True)
    ap.add_argument("--item-emb", required=True)
    ap.add_argument("--input-emb", default=None)
    ap.add_argument("--ckpt", nargs=3, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    split = protocol.build(args.seq, args.n_items)
    item = C.l2(C.load_matrix(args.item_emb))
    # input matrix loaded RAW (concat of per-half l2 blocks) — must match training
    # and the i43 replay convention exactly; do NOT l2 the concatenated vector.
    inp = C.load_matrix(args.input_emb) if args.input_emb else None
    item_t = torch.from_numpy(item)
    models = C.load_trio(args.ckpt, item, input_emb=inp)
    qs = C.encode_trio(models, split, "test")

    lists = []
    for qi in range(len(split.test)):
        lists.append(C.topk_ids(C.trio_z(qs, qi, item_t), POOL))
        if qi % 2000 == 0:
            print(f"  {qi}/{len(split.test)} [{time.time()-t0:.0f}s]", flush=True)

    unmasked = C.eval_ranked_lists(split, "test", lists, exclude_history=False)
    masked = C.eval_ranked_lists(split, "test", lists, exclude_history=True)
    out = {"experiment": "I44_2x2_cell", "cell": args.tag,
           "ckpts": list(args.ckpt), "item_emb": args.item_emb,
           "input_emb": args.input_emb, "pool": POOL,
           "unmasked": unmasked, "masked": masked,
           "runtime_sec": round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] {args.tag}: unmasked HR@5={unmasked['overall']['HR@5']:.4f} "
          f"masked HR@5={masked['overall']['HR@5']:.4f}", flush=True)


if __name__ == "__main__":
    main()
