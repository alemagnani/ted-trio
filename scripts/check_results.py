"""Sanity check: every headline number printed in the paper == the value stored in results/.

Part 1 compares each printed number (rounded the way the paper prints it) with the field of
the result JSON that produced it: Table 1 (masked + unmasked rows, paired deltas and p),
the §4 text numbers (paired SASRec deltas, matched-ensemble controls, cold slices, single
towers, soup, TED disclosures), Table 2 (2x2 cells) and the §4.1 factorial inference,
Table 3 (selector controls), Table 4/5 (Instruments/Yelp + their ablations), the efficiency
numbers (Table 5 / §6 text) and Table 6 (1M ANN benchmark), plus the supplement's router and
TED/scale numbers.

Part 2 RE-RUNS the statistics from the shipped per-user vectors: the six confirmatory paired
sign-flip tests (trio vs same-protocol TIGER and best-seed SASRec on Beauty/Toys/Sports, same
`significance.compare` call as scripts/close2_analysis.py) and the Sports factorial
inference (`i44_factorial_inference.effects_from_cells`), and checks the recomputed deltas,
CIs and p-values are identical to the booked JSONs and match the paper.

Usage: python scripts/check_results.py [--results DIR]   (exit code 0 iff all checks pass)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semplan.eval import significance  # noqa: E402

RES = ROOT / "results"
N_OK = N_BAD = 0
DS3 = ("beauty", "toys", "sports")


def get(fname: str, path: str):
    x = json.load(open(RES / fname))
    parts = path.split(".")
    i = 0
    while i < len(parts):
        k = parts[i]
        if isinstance(x, list):
            x = x[int(k)]
        else:
            j = i  # keys may themselves contain dots (e.g. blend weight "1.0")
            while k not in x and j + 1 < len(parts):
                j += 1
                k = k + "." + parts[j]
            x = x[k]
            i = j
        i += 1
    return x


def check(desc: str, value, paper, nd: int | None = 4):
    """value (from results/) rounded to nd decimals must equal the printed paper number."""
    global N_OK, N_BAD
    v = round(float(value), nd) if nd is not None else value
    ok = (v == paper) if nd is None else abs(v - paper) < 10 ** (-nd) / 2
    N_OK += ok
    N_BAD += not ok
    print(f"  [{'OK' if ok else 'FAIL'}] {desc}: results={value!r} paper={paper}")


def single(fname: str) -> float:
    """pseudo-test HR@5 of a one-checkpoint i40_gate JSON (no ensemble block)."""
    per = get(fname, "per_ckpt")
    (only,) = per.values()
    return only["pseudo_test"]["overall"]["HR@5"]


def section(t):
    print(f"\n== {t}")


def hr5(f, key="test.overall.HR@5"):
    return get(f, key)


def part1() -> None:
    section("Table 1 - masked (dense convention)")
    for ds, h, t in zip(DS3, (0.0648, 0.0738, 0.0353), (0.0824, 0.0969, 0.0464)):
        check(f"heuristic masked {ds}", hr5(f"i42_transition_only_{ds}.json",
                                            "real_test_masked.overall.HR@5"), h)
        check(f"trio masked {ds}", hr5(f"i43_masked_real_{ds}.json", "masked.overall.HR@5"), t)
        check(f"trio masked {ds}: unmasked-reproduction guard abs diff",
              hr5(f"i43_masked_real_{ds}.json", "unmasked_reproduction.abs_diff"), 0.0)
    for ds, v in (("toys", 0.1009), ("sports", 0.0496)):
        check(f"TED trio masked {ds}", hr5(f"i43_masked_real_{ds}_ted.json",
                                           "masked.overall.HR@5"), v)
    # delta vs best dense row (MQSA-TED 0.0752 / 0.0834 / 0.0455)
    best = {"beauty": hr5("i43_masked_real_beauty.json", "masked.overall.HR@5"),
            "toys": hr5("i43_masked_real_toys_ted.json", "masked.overall.HR@5"),
            "sports": hr5("i43_masked_real_sports_ted.json", "masked.overall.HR@5")}
    for ds, bar, d in zip(DS3, (0.0752, 0.0834, 0.0455), (0.0072, 0.0175, 0.0041)):
        check(f"masked delta vs MQSA-TED {ds}", best[ds] - bar, d)

    section("Table 1 - unmasked (generative convention)")
    for ds, h in zip(DS3, (0.0617, 0.0709, 0.0343)):
        check(f"heuristic unmasked {ds}", hr5(f"i42_transition_only_{ds}.json",
                                              "real_test_unmasked.overall.HR@5"), h)
    for ds, v in zip(DS3, (0.0453, 0.0409, 0.0275)):
        check(f"TIGER (ours) {ds}", hr5(f"close2_tiger_{ds}_s42.json", "overall.HR@5"), v)
    for ds, v in zip(DS3, (0.0265, 0.0347, 0.0149)):
        check(f"SASRec best seed {ds}",
              max(hr5(f"close2_sasrec_{ds}_s{s}.json") for s in (42, 43, 44)), v)
    for ds, v in zip(DS3, (0.0526, 0.0587, 0.0290)):
        check(f"SASRec-CE best seed {ds}",
              max(hr5(f"close2_sasrec_ce_{ds}_s{s}.json") for s in (42, 43, 44)), v)
    for ds, v in zip(DS3, (0.0551, 0.0634, 0.0316)):
        check(f"SASRec-CE x3 matched fusion {ds}", hr5(f"ensctrl_sasrec_ce_{ds}.json"), v)
    check("trio unmasked beauty", hr5("i24_test_look.json", "test.1.0.HR@5"), 0.0710)
    check("trio unmasked toys", hr5("i30_real_look.json", "test.1.0.HR@5"), 0.0804)
    check("trio unmasked sports", hr5("i35_real_look.json", "arms.trio.HR@5"), 0.0401)
    for ds, v in (("toys", 0.0838), ("sports", 0.0431)):
        check(f"TED trio unmasked {ds}", hr5(f"i40_ted_real_look_{ds}.json",
                                             "ensemble.pseudo_test.overall.HR@5"), v)
    for ds, v in zip(DS3, (0.0257, 0.0396, 0.0126)):
        check(f"trio vs TIGER delta {ds}", hr5(f"close2_cell_{ds}_tiger.json",
                                               "primary_HR@5.delta"), v)
        check(f"trio vs TIGER p {ds} (paper 1e-4)", hr5(f"close2_cell_{ds}_tiger.json",
                                                         "primary_HR@5.p_sign_flip"), 1e-4)

    section("§4 text")
    for ds, v in zip(DS3, (0.0445, 0.0457, 0.0253)):
        check(f"trio vs best-seed SASRec delta {ds}", hr5(f"close2_cell_{ds}_sasrec.json",
                                                          "primary_HR@5.delta"), v)
        check(f"trio vs SASRec p {ds}", hr5(f"close2_cell_{ds}_sasrec.json",
                                            "primary_HR@5.p_sign_flip"), 1e-4)
    for ds, v in zip(DS3, (0.0328, 0.0429, 0.0203)):
        check(f"BCE-SASRec fused control {ds}", hr5(f"ensctrl_sasrec_{ds}.json"), v)
    check("BCE fused control instruments", hr5("ensctrl_sasrec_instruments.json"), 0.0717)
    check("BCE fused control yelp", hr5("ensctrl_sasrec_yelp.json"), 0.0359)
    for ds, inc in zip(DS3, (0.0025, 0.0047, 0.0026)):
        best_ce = max(hr5(f"close2_sasrec_ce_{ds}_s{s}.json") for s in (42, 43, 44))
        check(f"control gain over best single seed {ds}",
              hr5(f"ensctrl_sasrec_ce_{ds}.json") - best_ce, inc)
    for ds, v in zip(DS3, (0.0689, 0.0772, 0.0379)):
        check(f"CE control masked {ds}", hr5(f"ensctrl_masked_ce_{ds}.json",
                                             "masked.overall.HR@5"), v)
    for ds, v in zip(DS3, (0.0062, 0.0236, 0.0088)):
        check(f"cold slice unmasked {ds}", hr5(f"i44_{ds}_EC.json", "unmasked.cold.HR@5"), v)
    for ds, v in (("toys", 0.0372), ("sports", 0.0146)):
        check(f"TED cold slice unmasked {ds}", hr5(f"i43_masked_real_{ds}_ted.json",
                                                   "unmasked.cold.HR@5"), v)
    for ds, m in zip(DS3, (0.0114, 0.0165, 0.0063)):
        f = f"i43_masked_real_{ds}.json"
        check(f"masking worth {ds}", hr5(f, "masked.overall.HR@5")
              - hr5(f, "unmasked.overall.HR@5"), m)
        check(f"repeat-target share {ds}", hr5(f, "repeat_target_share"), 0.0)
    for ds, um, usd, mm, msd, soup in zip(
            DS3, (0.0671, 0.0747, 0.0383), (0.0005, 0.0006, 0.0004),
            (0.0788, 0.0894, 0.0439), (0.0005, 0.0004, 0.0008), (0.0339, 0.0478, 0.0188)):
        f = f"i45_{ds}_seed_soup.json"
        check(f"single-tower mean unmasked {ds}", hr5(f, "summary.unmasked.HR@5.single_mean"), um)
        check(f"single-tower SD unmasked {ds}", hr5(f, "summary.unmasked.HR@5.single_sd"), usd)
        check(f"single-tower mean masked {ds}", hr5(f, "summary.masked.HR@5.single_mean"), mm)
        check(f"single-tower SD masked {ds}", hr5(f, "summary.masked.HR@5.single_sd"), msd)
        check(f"uniform weight soup {ds}", hr5(f, "results.soup.unmasked.overall.HR@5"), soup)
    check("Beauty TED inner gate (fail)", hr5("i39_gate_beauty_ted_trio.json",
                                              "ensemble.pseudo_test.overall.HR@5"), 0.0797)
    check("second Toys config (fail)", hr5("i40_lqted_real_look_toys.json",
                                           "ensemble.pseudo_test.overall.HR@5"), 0.0794)

    section("Table 2 - 2x2 injection ablation (unmasked / masked)")
    t2 = {"beauty": ((0.0656, 0.0754), (0.0647, 0.0762), (0.0690, 0.0824), (0.0710, 0.0824)),
          "toys": ((0.0726, 0.0858), (0.0762, 0.0887), (0.0779, 0.0947), (0.0804, 0.0969)),
          "sports": ((0.0344, 0.0381), (0.0368, 0.0404), (0.0399, 0.0446), (0.0401, 0.0464))}
    for ds, rows in t2.items():
        for cell, (u, m) in zip(("pp", "pC", "Ep", "EC"), rows):
            check(f"{ds} {cell} unmasked", hr5(f"i44_{ds}_{cell}.json", "unmasked.overall.HR@5"), u)
            check(f"{ds} {cell} masked", hr5(f"i44_{ds}_{cell}.json", "masked.overall.HR@5"), m)

    section("§4.1 factorial inference (Sports)")
    eff = "datasets.sports.effects"
    for name, est, lo, hi, p, pnd in (("E_main", 0.0044, 0.0031, 0.0057, 5e-5, 5),
                                      ("C_main", 0.0013, 0.0004, 0.0022, 0.003, 3),
                                      ("interaction", -0.0022, -0.0040, -0.0005, 0.011, 3)):
        f = "i44_factorial_inference.json"
        check(f"{name} estimate", get(f, f"{eff}.{name}.estimate"), est)
        check(f"{name} CI low", get(f, f"{eff}.{name}.ci95.0"), lo)
        check(f"{name} CI high", get(f, f"{eff}.{name}.ci95.1"), hi)
        check(f"{name} p", get(f, f"{eff}.{name}.p_signflip"), p, pnd)
    check("factorial n users", get("i44_factorial_inference.json",
                                   "datasets.sports.n"), 35598, None)

    section("Table 3 - selector controls (Beauty inner, masked HR@5)")
    for sel, v in (("ppr3", 0.0814), ("pred3", 0.0803), ("pred1", 0.0760), ("ppmi3", 0.0751),
                   ("contentnn3", 0.0749), ("bm25_3", 0.0744), ("translift3", 0.0741),
                   ("upred3", 0.0740), ("none", 0.0647), ("catbrand3", 0.0597),
                   ("popular3", 0.0591), ("random3", 0.0587)):
        check(f"selector {sel}", hr5(f"simpleaug_box_{sel}.json", "trio.masked.HR@5"), v)

    section("Table 4 - Instruments / Yelp")
    f = "i50_real_look_instruments.json"
    for k, v in (("HR@5", 0.0995), ("HR@10", 0.1254), ("NDCG@5", 0.0799), ("NDCG@10", 0.0882)):
        check(f"Instruments trio {k}", get(f, f"arms.trio.{k}"), v)
    check("Instruments free arm", get(f, "arms.free.HR@5"), 0.0654)
    check("Instruments delta vs free", get(f, "significance_trio_vs_free.HR@5.delta"), 0.0342)
    check("Instruments delta CI low", get(f, "significance_trio_vs_free.HR@5.bootstrap.ci95.0"), 0.0310)
    check("Instruments delta CI high", get(f, "significance_trio_vs_free.HR@5.bootstrap.ci95.1"), 0.0375)
    check("Instruments p", get(f, "significance_trio_vs_free.HR@5.randomization.p_two_sided"), 1e-4)
    singles = [get(f, f"arms.{s}.HR@5") for s in ("42", "43", "44")]
    check("Instruments single seeds min", min(singles), 0.0960)
    check("Instruments single seeds max", max(singles), 0.0974)
    f = "i51_real_look_yelp.json"
    for k, v in (("HR@5", 0.0461), ("HR@10", 0.0648), ("NDCG@5", 0.0340)):
        check(f"Yelp deployed blend {k}", get(f, f"arms.trio.{k}"), v)
    check("Yelp free arm", get(f, "arms.free.HR@5"), 0.0366)
    check("Yelp delta vs free", get(f, "significance_trio_vs_free.HR@5.delta"), 0.0095)
    check("Yelp delta CI low", get(f, "significance_trio_vs_free.HR@5.bootstrap.ci95.0"), 0.0076)
    check("Yelp delta CI high", get(f, "significance_trio_vs_free.HR@5.bootstrap.ci95.1"), 0.0114)
    singles = [get(f, f"arms.{s}.HR@5") for s in ("42", "43", "44")]
    check("Yelp single seeds min", min(singles), 0.0284)
    check("Yelp single seeds max", max(singles), 0.0291)
    check("Yelp cold slice free arm", get(f, "cold_slice.free.HR@5"), 0.0424)
    check("Yelp cold slice trio", get(f, "cold_slice.trio.HR@5"), 0.0254)
    check("Instruments inner pseudo-test trio", get("i50_gate_instruments.json",
                                                    "trio_pseudo_HR@5"), 0.1109)
    check("Yelp inner blend w=0.9", get("i51_gate_yelp.json", "pseudo_test.yelp_trio.0.9.HR@5"), 0.0397)
    check("Yelp inner pure trio", get("i51_gate_yelp.json", "pseudo_test.yelp_trio.1.0.HR@5"), 0.0272)
    check("Yelp inner free arm", get("i51_gate_yelp.json", "free_arm_pseudo_HR@5"), 0.0309)
    for ds, vals, mean in (("instruments", (0.0639, 0.0629, 0.0637), 0.0635),
                           ("yelp", (0.0278, 0.0270, 0.0279), 0.0276)):
        got = [hr5(f"close2_sasrec_{ds}_s{s}.json") for s in (42, 43, 44)]
        for s, g, v in zip((42, 43, 44), got, vals):
            check(f"SASRec {ds} s{s}", g, v)
        check(f"SASRec {ds} 3-seed mean", np.mean(got), mean)
    for ds, vals, mean, sd in (("instruments", (0.0886, 0.0861, 0.0901), 0.0883, 0.0020),
                               ("yelp", (0.0198, 0.0224, 0.0218), 0.0213, 0.0014)):
        got = [hr5(f"close2_tiger_{ds}_s{s}.json", "overall.HR@5") for s in (42, 43, 44)]
        for s, g, v in zip((42, 43, 44), got, vals):
            check(f"TIGER {ds} s{s}", g, v)
        check(f"TIGER {ds} mean", np.mean(got), mean)
        check(f"TIGER {ds} SD", np.std(got, ddof=1), sd)

    section("Table 5 - Instruments / Yelp injection ablations")
    for cell, u, m in zip(("pp", "pC", "Ep", "EC"), (0.0937, 0.0963, 0.0967, 0.0995),
                          (0.0989, 0.1020, 0.1046, 0.1066)):
        check(f"Instruments {cell} unmasked", hr5(f"i44_instruments_{cell}.json",
                                                  "unmasked.overall.HR@5"), u)
        check(f"Instruments {cell} masked", hr5(f"i44_instruments_{cell}.json",
                                                "masked.overall.HR@5"), m)
    for cell, u in zip(("pp", "pC", "Ep"), (0.0173, 0.0200, 0.0266)):
        check(f"Yelp {cell} unmasked", hr5(f"i44_yelp_{cell}.json", "unmasked.overall.HR@5"), u)

    section("Efficiency (Table 5 efficiency rows, §6 text)")
    f = "serving_bench_expand.json"
    check("Instruments trio ms/query", get(f, "datasets.instruments.end_to_end_batch1_ms.trio_batch1_ms"), 22.9, 1)
    check("Yelp trio ms/query", get(f, "datasets.yelp.end_to_end_batch1_ms.trio_batch1_ms"), 35.2, 1)
    check("Instruments serving MB", get(f, "datasets.instruments.memory_mb.total_serving_mb_trio"), 388, 0)
    check("Yelp serving MB", get(f, "datasets.yelp.memory_mb.total_serving_mb_trio"), 743, 0)
    f = "serving_bench.json"
    check("params per tower", get(f, "params.per_tower_trainable"), 3422980, None)
    check("trio params (10.3M)", get(f, "params.trio_trainable") / 1e6, 10.3, 1)
    check("B/T/S footprint MB", get(f, "memory_mb.total_serving_mb_trio"), 465, 0)
    check("one tower ms (1 thread)", get(f, "end_to_end_batch1_ms.single_batch1_ms"), 84, 0)
    check("trio ms (1 thread)", get(f, "end_to_end_batch1_ms.trio_batch1_ms"), 198, 0)
    f = "serving_bench_yelp_blend.json"
    check("Yelp full blend ms", get(f, "end_to_end_blend_batch1_ms.median_ms"), 104.4, 1)
    check("Yelp trio-alone ms", get(f, "components_batch1_ms.trio_fullcorpus_score_z.median_ms"), 84.6, 1)
    # paper: "the free arm adds 69 MB" = 812 - 743 (difference of the rounded totals)
    check("Yelp full-blend MB", get(f, "memory_mb.total_serving_mb_full_blend"), 812, 0)
    check("Yelp trio-only MB", get(f, "memory_mb.total_serving_mb_trio_only"), 743, 0)
    check("TIGER ms/query (co-tenant)", get("close2_tiger_latency.json", "gpu_batch1.mean_ms"), 739, 0)
    check("ANN replay exact == served", get("i23_ann_replay.json", "served.HR@5"),
          get("i23_ann_replay.json", "exact_reference.HR@5"))

    section("Table 6 - 1M-item ANN benchmark")
    f = "ann_bench_deployed_1m.json"
    rows = get(f, "rows")

    def row(index, key, val):
        return next(r for r in rows if r["index"] == index and r.get(key) == val)
    check("exact FlatIP p50 ms", get(f, "exact_singlethread_batch1.ms_p50"), 1291.7, 1)
    check("exact FlatIP MB", get(f, "exact_flat_mem_MB"), 12288, 0)
    for (idx, key, val), (rec, ms) in {
            ("IVF4096,Flat", "nprobe", 16): (0.952, 161.5),
            ("IVF4096,Flat", "nprobe", 128): (0.971, 297.6),
            ("IVF4096,PQ64", "nprobe", 64): (0.959, 145.0),
            ("HNSW32,Flat", "efSearch", 128): (0.955, 132.7)}.items():
        r = row(idx, key, val)
        check(f"{idx} {key}={val} recall@10", r["recall@10_vs_exact"], rec, 3)
        check(f"{idx} {key}={val} p50 ms", r["ms_p50"], ms, 1)
    check("IVF-Flat MB", get(f, "ivf_flat_mem_MB"), 12346, 0)
    check("IVF-PQ MB", get(f, "ivf_pq_mem_MB"), 126, 0)
    check("HNSW MB", get(f, "hnsw_mem_MB"), 12560, 0)
    check("distractor-to-real cosine", get(f, "distractor_calibration.distractor_random_real_cos_mean"), 0.447, 3)
    check("real-pair cosine", get(f, "distractor_calibration.real_pair_cos_mean"), 0.446, 3)

    section("Supplement D/E (inner world)")
    f = "i38_inner.json"
    for ds, key, v in (("instruments", "always_trio", 0.1109), ("instruments", "router_hard", 0.1104),
                       ("instruments", "router_soft", 0.1029), ("instruments", "oracle_per_user_arm", 0.1269),
                       ("yelp", "router_soft", 0.0393), ("yelp", "always_trio", 0.0272),
                       ("yelp", "always_free", 0.0309), ("yelp", "oracle_per_user_arm", 0.0509),
                       ("beauty", "router_hard", 0.0787), ("beauty", "router_soft", 0.0711)):
        check(f"router {ds} {key}", get(f, f"datasets.{ds}.pseudo_test.{key}.HR@5"), v)
    check("seed-disagreement AUC", get("i29_adaptive_k.json", "auc_d_jac"), 0.73, 2)
    bars = {"toys": ("i40_gate_toys_ted_trio.json", 0.0041), "sports": ("i40_gate_sports_ted_trio.json", 0.0041)}
    for ds, (fn, d) in bars.items():
        ted = get(fn, "ensemble.pseudo_test.overall.HR@5")
        gate = {"toys": ("i30_gate.json", "toys_trio"), "sports": ("i35_gate.json", "sports_trio")}[ds]
        base = get(gate[0], f"pseudo_test.{gate[1]}.1.0.HR@5")
        check(f"TED inner margin {ds}", ted - base, d)
    check("TED inner margin beauty", get("i39_gate_beauty_ted_trio.json",
                                         "ensemble.pseudo_test.overall.HR@5") - 0.0802, -0.0005)
    check("TED inner margin instruments",
          get("i46_gate_instruments_ted_trio.json", "ensemble.pseudo_test.overall.HR@5")
          - get("i46_gate_instruments_i50_trio.json", "ensemble.pseudo_test.overall.HR@5"), 0.0005)
    check("TED inner margin yelp",
          get("i47_gate_yelp_ted_trio.json", "ensemble.pseudo_test.overall.HR@5")
          - get("i47_gate_yelp_i51_trio.json", "ensemble.pseudo_test.overall.HR@5"), 0.0048)
    check("Yelp TED-in-blend", get("i47_blend_yelp_ted.json", "trio_pseudo_HR@5"), 0.0398)
    check("M-tower Beauty", get("i18_size_score.json", "towers.m.test.HR@5"), 0.0463)
    check("M-tower Toys (hard)", single("i18hard_gate_toys_m.json"), 0.0638)
    check("M+TED Beauty", single("i18ted_gate_beauty_m.json"), 0.0578)
    check("M+TED Toys", single("i18ted_gate_toys_m.json"), 0.0724)
    check("M+TED Sports", single("i18ted_gate_sports_m.json"), 0.0313)
    check("MiniLM + TED trio", get("i48_gate.json", "primary_trio_pseudo_HR@5"), 0.0778)


def part2() -> None:
    section("Re-run: six confirmatory paired tests from the shipped per-user vectors")
    trio = {"beauty": "i24_test_per_user.npz", "toys": "i30_real_per_user.npz",
            "sports": "i35_real_per_user.npz"}
    metrics = ("HR@5", "HR@10", "NDCG@5", "NDCG@10")
    key = {"HR@5": "hit5", "HR@10": "hit10", "NDCG@5": "dcg5", "NDCG@10": "dcg10"}
    for ds in DS3:
        for base in ("tiger", "sasrec"):
            booked = json.load(open(RES / f"close2_cell_{ds}_{base}.json"))
            bnpz = RES / Path(booked["baseline_npz"]).name
            t, b = np.load(RES / trio[ds]), np.load(bnpz)
            cmp_ = significance.compare({m: t[f"w1.0_{key[m]}"] for m in metrics},
                                        {m: b[key[m]] for m in metrics},
                                        metrics=metrics, cold_mask=t["cold_mask"])
            same = json.loads(json.dumps(cmp_)) == booked["compare"]
            check(f"{ds} vs {base}: recomputed comparison identical to booked JSON "
                  f"(delta {cmp_['HR@5']['delta']:+.4f}, p {cmp_['HR@5']['randomization']['p_two_sided']:.1e})",
                  same, True, None)

    section("Re-run: Sports factorial inference from i44_factorial_per_user.npz")
    sys.path.insert(0, str(ROOT / "scripts"))
    import i44_factorial_inference as F
    z = np.load(RES / "i44_factorial_per_user.npz")
    eff = F.effects_from_cells({c: z[f"sports_{c}"] for c in ("pp", "Ep", "pC", "EC")})
    ref = get("i44_factorial_inference.json", "datasets.sports.effects")
    for name in ref:
        same = all(eff[name][f] == ref[name][f] for f in ("estimate", "ci95", "p_signflip"))
        check(f"factorial {name} recomputed == booked", same, True, None)


def main() -> None:
    global RES
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(RES))
    ap.add_argument("--skip-rerun", action="store_true", help="only part 1 (seconds)")
    args = ap.parse_args()
    RES = Path(args.results)
    part1()
    if not args.skip_rerun:
        part2()
    print(f"\n{N_OK} checks passed, {N_BAD} failed")
    sys.exit(1 if N_BAD else 0)


if __name__ == "__main__":
    main()
