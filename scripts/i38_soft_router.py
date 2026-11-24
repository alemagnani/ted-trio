"""I38 — learned SOFT routing over arms (inner world ONLY, zero test looks, CPU).

Question: can a SERVABLE learned router (features computable user-side at query time,
NO target knowledge) recover a meaningful fraction of the hard cold-target rule's gain
(RESULTS.md #60), which is NOT servable user-side (it reads the target's train count)?

Setup (Beauty inner world, i24 inner trio + free arm — same arms as i24/i36 gates):
  * Per-user target-free features on inner val + inner pseudo-test:
      history length, trio retrieval-confidence (raw max-IP top-1 / top-10 mean,
      per-seed top-1 std), seed disagreement (mean pairwise top-10 overlap across the
      3 seeds — the I29 d_jac signal, AUC 0.73 for failure), free-arm z top-1 /
      top-10 mean, trio-vs-free z top-1 gap, history popularity stats (train-count
      table IS servable — it's an item-side table independent of the target).
  * Labels on inner val: free per-user HR@5 > trio per-user HR@5.
  * Router: standardized logistic regression (balanced classes).
  * Policies (threshold / soft scale swept on INNER VAL only):
      router-hard : p > thr -> free arm else trio
      router-soft : per-user free-weight wf = nearest grid to min(1, alpha*p)
  * Inner pseudo-test comparison: always-trio, always-free, hard cold-target rule
    (non-servable reference), router-hard, router-soft, oracle per-user arm (bound).

Bar for interesting: router beats always-trio HR@5 on Beauty inner pseudo-test.
Optional --dataset toys/sports once Beauty pipeline is validated.
Output: results/i38_inner.json (+ per-user npz cache for reuse).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan import spaces as sp  # noqa: E402
from semplan.dense.frozen_query import encode_split_queries, load_checkpoint  # noqa: E402
from semplan.eval import protocol  # noqa: E402

SEEDS = (42, 43, 44)
KS = (5, 10)
# wf = weight on the FREE arm: 0.0 == pure trio, 1.0 == pure free.
WF_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
THR_QUANTILES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99)
TOPK_CONF = 10

FEATURE_NAMES = [
    "hist_len", "log1p_hist_len",
    "tow_top1_mean", "tow_top10_mean", "tow_top1_std",
    "seed_overlap10",
    "free_top1_z", "free_top10_mean_z",
    "gap_top1_z",
    "pop_log_mean", "pop_log_min", "pop_log_last", "frac_hist_cold",
]

DATASETS = {
    "beauty": {
        "seq": ROOT / "data/amazon_beauty_backbone/Beauty_inner.txt",
        "item_emb": ROOT / "data/amazon_beauty_backbone/"
                           "Beauty_openai-3-large_i22inner_embeddings.pt",
        "input_emb": ROOT / "data/amazon_beauty_backbone/"
                            "Beauty_openai-3-large_i24inner_input.pt",
        "content_emb": ROOT / "data/amazon_beauty_backbone/"
                              "Beauty_sentence-t5-xl_embeddings.pt",
        "collab_cache": ROOT / "outputs/artifacts/collab_space_beauty_inner.npz",
        "ckpt": "i24_inner_openai_k4_s{s}_iv",
        "n_items": 12101},
    "toys": {
        "seq": ROOT / "data/amazon_toys_backbone/Toys_inner.txt",
        "item_emb": ROOT / "data/amazon_toys_backbone/"
                           "Toys_openai-3-large_i22inner_embeddings.pt",
        "input_emb": ROOT / "data/amazon_toys_backbone/"
                            "Toys_openai-3-large_i24inner_input.pt",
        "content_emb": ROOT / "data/amazon_toys_backbone/"
                              "Toys_sentence-t5-xl_embeddings.pt",
        "collab_cache": ROOT / "outputs/artifacts/collab_space_toys_inner.npz",
        "ckpt": "i30_toys_inner_openai_k4_s{s}_iv",
        "n_items": 11924},
    "sports": {
        "seq": ROOT / "data/amazon_sports_backbone/Sports_inner.txt",
        "item_emb": ROOT / "data/amazon_sports_backbone/"
                           "Sports_openai-3-large_i22inner_embeddings.pt",
        "input_emb": ROOT / "data/amazon_sports_backbone/"
                            "Sports_openai-3-large_i24inner_input.pt",
        "content_emb": ROOT / "data/amazon_sports_backbone/"
                              "Sports_sentence-t5-xl_embeddings.pt",
        "collab_cache": ROOT / "outputs/artifacts/collab_space_sports_inner.npz",
        "ckpt": "i35_sports_inner_openai_k4_s{s}_iv",
        "n_items": 18357},
    # EXPAND (next paper): same inner-world arms as the i50/i51 gates
    "instruments": {
        "seq": ROOT / "data/amazon_instruments_backbone/Instruments_inner.txt",
        "item_emb": ROOT / "data/amazon_instruments_backbone/"
                           "Instruments_openai-3-large_i22inner_embeddings.pt",
        "input_emb": ROOT / "data/amazon_instruments_backbone/"
                            "Instruments_openai-3-large_i24inner_input.pt",
        "content_emb": ROOT / "data/amazon_instruments_backbone/"
                              "Instruments_sentence-t5-xl_embeddings.pt",
        "collab_cache": ROOT / "outputs/artifacts/collab_space_instruments_inner.npz",
        "ckpt": "i50_instruments_inner_openai_k4_s{s}_iv",
        "n_items": 9922},
    "yelp": {
        "seq": ROOT / "data/yelp_backbone/Yelp_inner.txt",
        "item_emb": ROOT / "data/yelp_backbone/"
                           "Yelp_openai-3-large_i22inner_embeddings.pt",
        "input_emb": ROOT / "data/yelp_backbone/"
                            "Yelp_openai-3-large_i24inner_input.pt",
        "content_emb": ROOT / "data/yelp_backbone/"
                              "Yelp_sentence-t5-xl_embeddings.pt",
        "collab_cache": ROOT / "outputs/artifacts/collab_space_yelp_inner.npz",
        "ckpt": "i51_yelp_inner_openai_k4_s{s}_iv",
        "n_items": 20033},
}


def l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def zscore(s):
    return (s - s.mean()) / max(float(s.std()), 1e-9)


def scan_split(split, which, qs, item_t, content_w, collab_w, covered, train_count,
               n_items, t0, tag, limit=0):
    """One pass: per-event ranks for trio/free/every wf blend + target-free features."""
    events = [(list(h), t) for h, t in getattr(split, which)]
    if limit:
        events = events[:limit]
    n = len(events)
    names = [str(s) for s in SEEDS]
    ranks = {("wf", wf): np.zeros(n, dtype=np.int32) for wf in WF_GRID}
    feats = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    cold = np.zeros(n, dtype=bool)
    logc = np.log1p(train_count.astype(np.float64))
    for qi, (hist, tgt) in enumerate(events):
        cold[qi] = train_count[tgt] == 0
        b03 = sp.blended_scores_whitened(content_w, collab_w, covered, hist,
                                         recency=sp.WHITEN_RECENCY)
        b01 = sp.blended_scores_whitened(content_w, collab_w, covered, hist,
                                         recency=0.1)
        z_free = zscore(0.9 * b03 + 0.1 * b01)
        z_tow, top1_raw, top10_mean_raw, tops = {}, [], [], []
        for nm in names:
            ip = torch.einsum("kd,nd->kn", qs[nm][qi], item_t).numpy()
            smax = ip.max(axis=0)
            z_tow[nm] = zscore(smax)
            idx = np.argpartition(smax, -TOPK_CONF)[-TOPK_CONF:]
            tops.append(set(idx.tolist()))
            t10 = smax[idx]
            top1_raw.append(float(t10.max()))
            top10_mean_raw.append(float(t10.mean()))
        z_trio = np.mean([z_tow[nm] for nm in names], axis=0)
        for wf in WF_GRID:
            sc = (1.0 - wf) * z_trio + wf * z_free
            ranks[("wf", wf)][qi] = int((sc > sc[tgt]).sum())
        overlap = np.mean([len(a & b) / TOPK_CONF
                           for a, b in combinations(tops, 2)])
        hl = len(hist)
        hp = logc[hist]
        f_top10 = np.partition(z_free, -TOPK_CONF)[-TOPK_CONF:]
        feats[qi] = [
            hl, np.log1p(hl),
            float(np.mean(top1_raw)), float(np.mean(top10_mean_raw)),
            float(np.std(top1_raw)),
            float(overlap),
            float(z_free.max()), float(f_top10.mean()),
            float(z_trio.max() - z_free.max()),
            float(hp.mean()), float(hp.min()),
            float(hp[-1]), float((train_count[hist] == 0).mean()),
        ]
        if (qi + 1) % 2000 == 0:
            print(f"  [i38] {tag}/{which}: {qi+1}/{n} [{time.time()-t0:.0f}s]",
                  flush=True)
    return {"ranks": {f"wf{wf}": ranks[("wf", wf)] for wf in WF_GRID},
            "feats": feats, "cold": cold, "n": n}


def metrics_from_rank(rank, mask=None):
    m = np.ones(len(rank), dtype=bool) if mask is None else mask
    r = rank[m].astype(np.float64)
    nn = max(int(m.sum()), 1)
    out = {}
    for k in KS:
        hit = r < k
        out[f"HR@{k}"] = round(float(hit.mean()), 4)
        out[f"NDCG@{k}"] = round(float(np.where(hit, 1.0 / np.log2(r + 2), 0.0)
                                       .sum()) / nn, 4)
    return out


def policy_rank(store, wf_per_user):
    """Per-user rank when each user is served the grid weight in wf_per_user."""
    n = store["n"]
    out = np.zeros(n, dtype=np.int32)
    for wf in WF_GRID:
        sel = wf_per_user == wf
        if sel.any():
            out[sel] = store["ranks"][f"wf{wf}"][sel]
    return out


def snap_to_grid(w):
    grid = np.asarray(WF_GRID)
    return grid[np.abs(grid[None, :] - w[:, None]).argmin(axis=1)]


def run_dataset(name, cfg, args, t0):
    cache = ROOT / f"results/i38_{name}_inner_per_user.npz"
    if cache.exists() and not args.limit:
        d = np.load(cache)
        print(f"[i38] {name}: loaded cached per-user arrays from {cache}", flush=True)
        stores = {}
        for which in ("val", "test"):
            stores[which] = {
                "ranks": {f"wf{wf}": d[f"{which}_rank_wf{wf}"] for wf in WF_GRID},
                "feats": d[f"{which}_feats"], "cold": d[f"{which}_cold"],
                "n": len(d[f"{which}_cold"])}
    else:
        split = protocol.build(str(cfg["seq"]), n_items=cfg["n_items"], max_len=20)
        item = l2(torch.load(cfg["item_emb"], map_location="cpu",
                             weights_only=True).float().numpy())
        item_t = torch.from_numpy(item)
        inp = torch.load(cfg["input_emb"], map_location="cpu",
                         weights_only=True).float().numpy()
        models = {str(s): load_checkpoint(
            str(ROOT / "outputs/artifacts" / cfg["ckpt"].format(s=s) / "j5.pt"),
            item, input_emb=inp) for s in SEEDS}
        content = l2(torch.load(cfg["content_emb"], map_location="cpu"
                                ).float().numpy().astype(np.float32))
        train_seqs = [list(h) for h, _t in split.val]
        collab, covered = sp.build_collab_embeddings(
            train_seqs, cfg["n_items"], d=128, cache_path=str(cfg["collab_cache"]))
        content_w, collab_w = sp.whiten_blend_spaces(content, collab, covered)
        train_count = np.zeros(cfg["n_items"], dtype=np.int64)
        for h in train_seqs:
            for it in h:
                train_count[it] += 1
        print(f"[i38] {name} spaces ready; cold items "
              f"{int((train_count == 0).sum())} [{time.time()-t0:.0f}s]", flush=True)
        stores = {}
        for which in ("val", "test"):
            qs = {nm: encode_split_queries(m, split, which=which, device="cpu")
                  for nm, m in models.items()}
            stores[which] = scan_split(split, which, qs, item_t, content_w, collab_w,
                                       covered, train_count, cfg["n_items"], t0,
                                       name, limit=args.limit)
        if not args.limit:
            np.savez_compressed(
                cache,
                **{f"{w}_rank_wf{wf}": stores[w]["ranks"][f"wf{wf}"]
                   for w in ("val", "test") for wf in WF_GRID},
                **{f"{w}_feats": stores[w]["feats"] for w in ("val", "test")},
                **{f"{w}_cold": stores[w]["cold"] for w in ("val", "test")},
                feature_names=np.array(FEATURE_NAMES))
            print(f"[i38] {name}: cached per-user arrays -> {cache}", flush=True)

    # ---- router: train on inner val, select policy on inner val ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    val, test = stores["val"], stores["test"]
    r_trio_v, r_free_v = val["ranks"]["wf0.0"], val["ranks"]["wf1.0"]
    y = ((r_free_v < 5).astype(int) > (r_trio_v < 5).astype(int)).astype(int)
    print(f"[i38] {name}: val n={val['n']} label(free HR@5 beats trio) "
          f"pos={int(y.sum())} ({y.mean():.4f})", flush=True)

    scaler = StandardScaler().fit(val["feats"])
    Xv = scaler.transform(val["feats"])
    Xt = scaler.transform(test["feats"])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(Xv, y)
    p_val = lr.predict_proba(Xv)[:, 1]
    p_test = lr.predict_proba(Xt)[:, 1]
    auc_val = float(roc_auc_score(y, p_val))
    y_test = ((test["ranks"]["wf1.0"] < 5).astype(int) >
              (test["ranks"]["wf0.0"] < 5).astype(int)).astype(int)
    auc_test = float(roc_auc_score(y_test, p_test)) if 0 < y_test.sum() < len(y_test) \
        else float("nan")
    coefs = dict(zip(FEATURE_NAMES, [round(float(c), 4) for c in lr.coef_[0]]))
    print(f"[i38] {name}: router AUC val(in-sample)={auc_val:.4f} "
          f"test={auc_test:.4f}", flush=True)
    print(f"[i38] {name}: coefs {coefs}", flush=True)

    def hard_wf(p, thr):
        return np.where(p > thr, 1.0, 0.0)

    # threshold sweep (inner val, in-sample by design)
    hard_sweep = {}
    best_hard = None
    for q in THR_QUANTILES:
        thr = float(np.quantile(p_val, q))
        m = metrics_from_rank(policy_rank(val, hard_wf(p_val, thr)))
        hard_sweep[f"q{q}"] = {"thr": round(thr, 4), **m,
                               "frac_free": round(float((p_val > thr).mean()), 4)}
        key = (m["HR@5"], m["NDCG@5"])
        if best_hard is None or key > best_hard[0]:
            best_hard = (key, q, thr)
    _, best_q, best_thr = best_hard

    soft_sweep = {}
    best_soft = None
    for a in ALPHAS:
        wfv = snap_to_grid(np.minimum(1.0, a * p_val))
        m = metrics_from_rank(policy_rank(val, wfv))
        soft_sweep[f"alpha{a}"] = {**m,
                                   "mean_wf": round(float(wfv.mean()), 4)}
        key = (m["HR@5"], m["NDCG@5"])
        if best_soft is None or key > best_soft[0]:
            best_soft = (key, a)
    _, best_alpha = best_soft
    print(f"[i38] {name}: selected hard thr q={best_q} ({best_thr:.4f}), "
          f"soft alpha={best_alpha}", flush=True)

    # ---- inner pseudo-test evaluation ----
    r_trio = test["ranks"]["wf0.0"]
    r_free = test["ranks"]["wf1.0"]
    cold_t = test["cold"]
    r_hardrule = np.where(cold_t, r_free, r_trio)
    wf_rh = hard_wf(p_test, best_thr)
    r_router_hard = policy_rank(test, wf_rh)
    wf_rs = snap_to_grid(np.minimum(1.0, best_alpha * p_test))
    r_router_soft = policy_rank(test, wf_rs)
    r_oracle = np.minimum(r_trio, r_free)

    policies = {
        "always_trio": r_trio, "always_free": r_free,
        "hard_cold_rule_nonservable": r_hardrule,
        "router_hard": r_router_hard, "router_soft": r_router_soft,
        "oracle_per_user_arm": r_oracle,
    }
    report = {pn: metrics_from_rank(r) for pn, r in policies.items()}
    cold_report = {pn: metrics_from_rank(r, cold_t)
                   for pn, r in policies.items() if pn != "oracle_per_user_arm"}
    for pn, m in report.items():
        print(f"[i38] {name} pseudo-test {pn}: {m}", flush=True)
    print(f"[i38] {name} cold slice (n={int(cold_t.sum())}): {cold_report}",
          flush=True)

    beats_trio = report["router_hard"]["HR@5"] > report["always_trio"]["HR@5"] or \
        report["router_soft"]["HR@5"] > report["always_trio"]["HR@5"]
    gain_hard_rule = report["hard_cold_rule_nonservable"]["HR@5"] - \
        report["always_trio"]["HR@5"]
    best_router_hr5 = max(report["router_hard"]["HR@5"],
                          report["router_soft"]["HR@5"])
    frac_recovered = ((best_router_hr5 - report["always_trio"]["HR@5"]) /
                      gain_hard_rule) if gain_hard_rule > 0 else None
    return {
        "n_val": val["n"], "n_test": test["n"],
        "n_cold_test": int(cold_t.sum()),
        "label_pos_rate_val": round(float(y.mean()), 4),
        "router_auc_val_insample": round(auc_val, 4),
        "router_auc_test": round(auc_test, 4),
        "router_coefs": coefs,
        "router_intercept": round(float(lr.intercept_[0]), 4),
        "selected": {"hard_thr_quantile": best_q, "hard_thr": round(best_thr, 4),
                     "soft_alpha": best_alpha,
                     "test_frac_routed_free_hard":
                         round(float((wf_rh == 1.0).mean()), 4),
                     "test_mean_wf_soft": round(float(wf_rs.mean()), 4)},
        "val_hard_sweep": hard_sweep, "val_soft_sweep": soft_sweep,
        "pseudo_test": report, "pseudo_test_cold_slice": cold_report,
        "router_beats_always_trio_HR@5": bool(beats_trio),
        "hard_rule_gain_HR@5": round(gain_hard_rule, 4),
        "router_frac_of_hard_rule_gain":
            round(frac_recovered, 3) if frac_recovered is not None else None,
    }


def main():
    ap = argparse.ArgumentParser(description="I38 learned soft router (inner only)")
    ap.add_argument("--dataset", nargs="+", default=["beauty"],
                    choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only first N events per split (no cache/json)")
    args = ap.parse_args()
    t0 = time.time()
    reports = {nm: run_dataset(nm, DATASETS[nm], args, t0) for nm in args.dataset}
    out = {"experiment": "I38_learned_soft_router_inner",
           "note": "inner worlds only, zero test looks; features target-free "
                   "(two-tower/ANN-servable); hard cold-target rule included as "
                   "NON-servable reference",
           "feature_names": FEATURE_NAMES, "wf_grid": list(WF_GRID),
           "seeds": list(SEEDS), "datasets": reports,
           "runtime_sec": round(time.time() - t0, 1)}
    if not args.limit:
        out_json = ROOT / "results/i38_inner.json"
        if out_json.exists():  # merge: keep previously booked datasets
            prev = json.load(open(out_json))
            merged = dict(prev.get("datasets", {}))
            merged.update(out["datasets"])
            out["datasets"] = merged
        json.dump(out, open(out_json, "w"), indent=2)
        print(f"[i38] wrote {out_json} [{time.time()-t0:.0f}s]", flush=True)
    else:
        print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
