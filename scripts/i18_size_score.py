"""I18 size-sweep control — score M/L/L5 single towers (inner, diagnostic ONLY).

No gate, no look, no selection: pure-tower (w=1.0) pseudo-val/test HR/NDCG for
i18_size_{m,l,l5}_s42_iv vs the base d256 singles (0.0745-0.0751 pseudo-test
HR@5, all of which selected w=1.0 on inner val) and the trio bar 0.0802.
Question (codex prereq for the I18 scale ask): does parameter scale alone lift a
single K4 tower toward the seed-diversity trio? Output: results/i18_size_score.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.dense.frozen_query import encode_split_queries, load_checkpoint  # noqa: E402
from semplan.eval import protocol  # noqa: E402

SEQ = ROOT / "data/amazon_beauty_backbone/Beauty_inner.txt"
ITEM_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i22inner_embeddings.pt"
INPUT_EMB = ROOT / "data/amazon_beauty_backbone/Beauty_openai-3-large_i24inner_input.pt"
TOWERS = {nm: ROOT / f"outputs/artifacts/i18_size_{nm}_s42_iv/j5.pt"
          for nm in ("m", "l", "l5")}
OUT_JSON = ROOT / "results/i18_size_score.json"
N_ITEMS = 12101
KS = (5, 10)
REFS = {"base_d256_singles_pseudo_HR@5": [0.0745, 0.0751, 0.0749],
        "trio_pseudo_HR@5": 0.0802,
        "distilled_singles_pseudo_HR@5": [0.0786, 0.0787, 0.0782]}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def main():
    t0 = time.time()
    split = protocol.build(str(SEQ), n_items=N_ITEMS, max_len=20)
    item = l2(torch.load(ITEM_EMB, map_location="cpu", weights_only=True).float().numpy())
    item_t = torch.from_numpy(item)
    inp = torch.load(INPUT_EMB, map_location="cpu", weights_only=True).float().numpy()

    out = {"experiment": "I18_size_sweep_score", "refs": REFS, "towers": {}}
    for nm, ckpt in TOWERS.items():
        model = load_checkpoint(str(ckpt), item, input_emb=inp)
        n_par = sum(p.numel() for p in model.parameters())
        rep = {"params": n_par}
        for which in ("val", "test"):
            qs = encode_split_queries(model, split, which=which, device="cpu")
            events = [(list(h), t) for h, t in getattr(split, which)]
            hits = {k: 0.0 for k in KS}
            dcgs = {k: 0.0 for k in KS}
            for qi, (_h, tgt) in enumerate(events):
                sc = torch.einsum("kd,nd->kn", qs[qi], item_t).max(dim=0).values.numpy()
                rank = int((sc > sc[tgt]).sum())
                for k in KS:
                    if rank < k:
                        hits[k] += 1.0
                        dcgs[k] += 1.0 / np.log2(rank + 2)
            n = len(events)
            rep[which] = {f"HR@{k}": round(hits[k] / n, 4) for k in KS} | \
                         {f"NDCG@{k}": round(dcgs[k] / n, 4) for k in KS}
            print(f"[i18size] {nm} ({n_par/1e6:.1f}M) {which}: {json.dumps(rep[which])} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        out["towers"][nm] = rep
    out["runtime_sec"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"[i18size] wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
