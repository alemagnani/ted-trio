"""SIMPLE-AUG box runner (task #87) — train J5b K4 towers (3 seeds) per augmentation
selector on GPU, eval INNER pseudo-test, trio z-score ensemble. Matched to I24.

Per selector arm: item side = E embeddings (--emb), query-tower input = C matrix (--inp).
Inner world = Beauty_inner.txt; ChnDeDataset(include_val=True) as in I24. Trio HR@5 gated
vs pred3 bar 0.0802. NO real-test look (inner split.test = original -2 pseudo-test target).

Run (GPU recommended):  python scripts/simpleaug_box.py <name1> <name2> ...
Files expected under DATA:  Beauty_openai-3-large_aug_{name}_inner_{embeddings,input}.pt
Anchors: pred3 -> i22inner emb + i24inner input; none -> base emb (input = base).
Output: results/simpleaug_box_{name}.json  (+ merged simpleaug_box_all.json)
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
from semplan.dense.chn_de import ChnDeDataset  # noqa: E402
from semplan.dense.frozen_query import (  # noqa: E402
    FrozenQueryTower, encode_split_queries, train_frozen_query,
)
from semplan.eval import protocol  # noqa: E402
from semplan.quick import max_steps  # noqa: E402

DATA = ROOT / "data/amazon_beauty_backbone"
SEQ = DATA / "Beauty_inner.txt"
N = 12101
KS = (5, 10)
SEEDS = (42, 43, 44)
STEPS = max_steps(6000)  # TT_MAX_STEPS caps it in quick mode only
DEV = "cuda" if torch.cuda.is_available() else "cpu"
BAR = 0.0802  # pred3 (i24) trio inner pseudo-test HR@5


def files(name):
    if name == "pred3":
        return (DATA / "Beauty_openai-3-large_i22inner_embeddings.pt",
                DATA / "Beauty_openai-3-large_i24inner_input.pt")
    if name == "none":
        e = DATA / "Beauty_openai-3-large_embeddings.pt"
        return (e, e)
    return (DATA / f"Beauty_openai-3-large_aug_{name}_inner_embeddings.pt",
            DATA / f"Beauty_openai-3-large_aug_{name}_inner_input.pt")


def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-8)


def load(p):
    return torch.load(p, map_location="cpu", weights_only=True).float().numpy()


def run_selector(name, split, ds):
    emb_f, inp_f = files(name)
    if not emb_f.exists() or not inp_f.exists():
        print(f"[{name}] MISSING {emb_f.name}/{inp_f.name} — skip", flush=True)
        return None
    item = load(emb_f)
    inp = load(inp_f)
    tgts = torch.tensor([t for _h, t in split.test], dtype=torch.long)
    item_t = torch.from_numpy(item).to(DEV)
    per_seed_scores, per_seed_hr = {}, {}
    t0 = time.time()
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        model = FrozenQueryTower(item, input_emb=inp, n_query_vectors=4,
                                 d=256, n_layers=2, n_heads=4, d_ff=512, tau=0.05)
        train_frozen_query(model, ds, steps=STEPS, batch_size=256, lr=1e-3,
                           device=DEV, seed=sd, log_every=2000)
        model.eval()
        q = encode_split_queries(model, split, which="test", device=DEV)  # [n,K,D]
        with torch.no_grad():
            sc = torch.einsum("bkd,nd->bkn", q.to(DEV), item_t).max(1).values.cpu()  # [n,N]
        per_seed_scores[sd] = sc
        rank = (sc > sc.gather(1, tgts[:, None])).sum(1).numpy()
        hr5 = float((rank < 5).mean())
        per_seed_hr[sd] = round(hr5, 4)
        print(f"[{name}] seed {sd} HR@5 {hr5:.4f} [{time.time()-t0:.0f}s]", flush=True)
    # trio: mean of per-seed z-scored scores
    ztrio = sum(torch.stack([zscore(per_seed_scores[sd][i]) for sd in SEEDS]).mean(0)
                for i in range(0)) if False else None
    Z = torch.stack([torch.stack([zscore(per_seed_scores[sd][i]) for sd in SEEDS]).mean(0)
                     for i in range(len(tgts))])
    res = {}
    for masked in (False, True):
        S = Z.clone()
        if masked:
            for i, (h, _t) in enumerate(split.test):
                S[i, list(h)] = -1e30
        rank = (S > S.gather(1, tgts[:, None])).sum(1).numpy()
        tag = "masked" if masked else "unmasked"
        res[tag] = {f"HR@{k}": round(float((rank < k).mean()), 4) for k in KS}
        res[tag] |= {f"NDCG@{k}": round(float(np.where(rank < k, 1/np.log2(rank+2), 0).mean()), 4)
                     for k in KS}
    out = {"selector": name, "per_seed_HR5": per_seed_hr, "trio": res,
           "trio_HR5_unmasked": res["unmasked"]["HR@5"], "bar_pred3": BAR,
           "pass": res["unmasked"]["HR@5"] >= BAR, "seconds": round(time.time()-t0, 1)}
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / f"results/simpleaug_box_{name}.json", "w"), indent=2)
    print(f"[{name}] TRIO HR@5 unmasked {res['unmasked']['HR@5']} / masked "
          f"{res['masked']['HR@5']}  (bar {BAR}) [{time.time()-t0:.0f}s]", flush=True)
    return out


def main():
    names = sys.argv[1:] or ["pred3", "contentnn3", "ppmi3", "translift3", "upred3",
                             "bm25_3", "popular3", "random3", "none"]
    split = protocol.build(str(SEQ), n_items=N, max_len=20)
    ds = ChnDeDataset(split, max_len=20, include_val=True)
    print(f"inner split {len(split.test)} pseudo-test; ds {len(ds)} pairs; dev {DEV}", flush=True)
    allres = {}
    for name in names:
        r = run_selector(name, split, ds)
        if r:
            allres[name] = {"trio_HR5_unmasked": r["trio_HR5_unmasked"],
                            "trio_HR5_masked": r["trio"]["masked"]["HR@5"],
                            "pass": r["pass"]}
            json.dump(allres, open(ROOT / "results/simpleaug_box_all.json", "w"), indent=2)
    print("\n=== TRIO inner pseudo-test HR@5 (unmasked / masked) vs pred3 bar 0.0802 ===", flush=True)
    for n, r in sorted(allres.items(), key=lambda x: -x[1]["trio_HR5_unmasked"]):
        print(f"  {n:12s} {r['trio_HR5_unmasked']:.4f} / {r['trio_HR5_masked']:.4f} "
              f"{'PASS' if r['pass'] else 'below'}", flush=True)


if __name__ == "__main__":
    main()
