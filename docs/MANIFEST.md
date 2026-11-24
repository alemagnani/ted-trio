# MANIFEST — release/wsdm2027

> **Layout note (public release).** This manifest was written for the review-time artifact
> package. In this repository `code/` is `scripts/` (plus `baselines/tiger/`), and the
> `research/results/` / `research/experiments/` paths that appear inside result JSONs and
> ledger entries are `results/` / `scripts/` here. File-to-ledger mappings are unchanged.

One line per file: what it is and which append-only ledger entry (`ledger/ledger.jsonl`,
field `id`) booked it. "Plan:" = the frozen prereg spec governing it. Files marked
**[no dedicated ledger entry]** are claim-bearing artifacts whose numbers appear in the
paper but for which no individual ledger booking exists (disclosed here rather than
papered over; the JSON-internal guards/prereg fields are noted).

## src/ — the `semplan` source package

- `src/semplan/**` — the complete project source the runners import (35 modules): query
  towers and training (`dense/`), evaluation protocol (`eval/protocol.py`), whitened
  content/collab spaces (`spaces.py`), enrichment, routing, metrics. Anonymized; no key
  material (API embedders read `OPENAI_API_KEY` etc.\ from the environment only).
- `pyproject.toml` — dependency list for `semplan`.

## ledger/

- `ledger.jsonl` — full append-only experiment ledger, 231 entries (anonymized copy; see README "Anonymization").

## SUPPLEMENT.md

- `SUPPLEMENT.md` — submission supplement: the TIGER per-seed reproduction table (moved out of the paper to meet the 9-page limit) + the paper-to-artifact map for §4.1 factorial inference and Table 6 (1M ANN benchmark).
- `close2_prereg_draft.md` — FROZEN confirmatory analysis plan for the six baseline comparisons; booked `CLOSE2-PREREG-FREEZE` (2026-08-07T01:33:18Z).
- `E_ENSCTRL.md` — matched-ensemble-control prereg spec; booked `PREREG-E-ENSCTRL` + `PREREG-E-ENSCTRL-ARMC`.

## results/ — per-user metric vectors

- `*_per_user.npz` (67 files) — per-user hit/metric vectors behind every paired
  significance test: `i44_factorial_per_user.npz` (the four Sports 2×2 cells, inputs to
  the §4.1 permutation/bootstrap inference), `close2_*_per_user.npz` (same-protocol
  SASRec/TIGER re-runs, per seed), `ensctrl_*_per_user.npz` (matched-ensemble control
  arms), and the booked real-look/gate vectors (`i30_real`, `i35_real`, `p*`, `i2*`,
  `j7`, `i38_*_inner`). Numeric arrays only.

## results/ — headline trios (Beauty/Toys/Sports)

- `i24_gate.json` — Beauty inner-world gate for the deployed trio; `I24-GATE-PASS`.
- `i24_test_look.json` — Beauty trio single real look (unmasked HR@5 0.071); `I24-LOOK` (prereg `I24-LOOK-PREREG`).
- `i30_gate.json` — Toys inner gate; `I30-GATE-PASS`.
- `i30_real_look.json` — Toys trio single real look (0.0804); `I30-REAL-LOOK` (prereg `I30-REAL-PREREG`).
- `i35_gate.json` — Sports inner gate; `I35-GATE-PASS`.
- `i35_real_look.json` — Sports trio single real look (0.0401); `I35-REAL-LOOK` (prereg `I35-REAL-LOOK-PREREG`).
- `i43_masked_real_beauty.json` — masked-convention rescore of the frozen Beauty trio (0.0824), guarded by exact unmasked repro (0.071); **[no dedicated ledger entry]** (TED-era companions were booked; the three plain rescores carry the guard inside the JSON).
- `i43_masked_real_toys.json` — masked rescore Toys trio (0.0969), same guard; **[no dedicated ledger entry]**.
- `i43_masked_real_sports.json` — masked rescore Sports trio (0.0464), same guard; **[no dedicated ledger entry]**.

## results/ — TED trios and gates

- `i39_gate_beauty_ted_trio.json` — Beauty TED trio inner gate FAIL (0.0797 < 0.0802); `I39-GATE-FAIL`.
- `i39_gate_beauty_ted_gpu.json` — GPU/CPU cross-check of the i39 gate (matches to 4 decimals); `I39-CPU-CROSSCHECK`.
- `i40_gate_toys_ted_trio.json` — Toys TED trio inner gate; `I40-TED-TRIO-GATE-TOYS`.
- `i40_gate_sports_ted_trio.json` — Sports TED trio inner gate; `I40-TED-TRIO-GATE-SPORTS`.
- `i40_ted_real_look_toys.json` — Toys TED trio single real look (0.0838; `no_real_test` field is a runner artifact, clarified in `CLARIFY-TED-NO-REAL-TEST-FLAG`); `I40-TED-TOYS-REAL-LOOK` (prereg `I40-TED-TOYS-REAL-PREREG`).
- `i40_ted_real_look_sports.json` — Sports TED trio single real look (0.0431); `I40-TED-SPORTS-REAL-LOOK` (prereg `I40-TED-SPORTS-REAL-PREREG`).
- `i43_masked_real_toys_ted.json` — masked rescore Toys TED (0.1009); `I43-MASKED-TED-TOYS`.
- `i43_masked_real_sports_ted.json` — masked rescore Sports TED (0.0496); `I43-MASKED-TED-SPORTS`.
- `i40_gate_toys_lqted_trio.json` — lqted stack inner gates; `I40-LQTED-TRIO-GATES`.
- `i40_lqted_real_look_toys.json` — second Toys real look, FAILED (0.0794 < 0.0804); `I40-LQTED-TOYS-REAL-LOOK` (prereg `I40-LQTED-TOYS-REAL-PREREG`).
- `i46_gate_instruments_ted_trio.json` — Instruments TED inner gate (transfer FAIL); `I46-TED-INSTRUMENTS-GATE`.
- `i46_gate_instruments_i50_trio.json` — same-scorer deployed-trio anchor for that gate; `I46-TED-INSTRUMENTS-GATE`.
- `i47_gate_yelp_ted_trio.json` — Yelp TED inner gate (+0.0048); `I47-TED-YELP-GATE`.
- `i47_gate_yelp_i51_trio.json` — same-scorer anchor; `I47-TED-YELP-GATE`.
- `i47_blend_yelp_ted.json` — Yelp blend rescore with TED trio substituted (0.0398 vs 0.0397 → no real-stage TED run); `I47-BLEND-RESCORE-YELP`.

## results/ — 2×2 injection ablation (i44)

- `i44_beauty_pp.json` / `i44_beauty_pC.json` / `i44_beauty_Ep.json` — Beauty non-EC cells (one look each); **[no dedicated ledger entries found for the Beauty cells]**.
- `i44_beauty_EC.json` — Beauty EC cell = deployed trio re-eval (reproduces 0.071/0.0824 exactly); **[no dedicated ledger entry]**.
- `i44_toys_pp.json` / `i44_toys_pC.json` / `i44_toys_Ep.json` / `i44_toys_EC.json` — Toys cells (EC reproduces 0.0804/0.0969); **[no dedicated ledger entries found for the Toys cells]**.
- `i44_sports_pp.json` — Sports pp cell; `I44-SPORTS-PP`.
- `i44_sports_pC.json` — Sports pC cell; `I44-SPORTS-PC`.
- `i44_sports_Ep.json` — Sports Ep cell; `I44-SPORTS-EP`.
- `i44_sports_EC.json` — Sports EC cell (deployed trio re-eval); `I44-SPORTS-EC`.
- `i44_instruments_pp.json` / `i44_instruments_pC.json` / `i44_instruments_Ep.json` — Instruments cells; `I44-INSTRUMENTS-CELLS`.
- `i44_instruments_EC.json` — Instruments EC (0.0995 reproduced; masked 0.1066 single look); `I44-INSTRUMENTS-CELLS` + `I44-INSTRUMENTS-EC-MASKED` (prereg `PREREG-I44-INSTRUMENTS-EC-MASKED`).
- `i44_yelp_pp.json` / `i44_yelp_pC.json` / `i44_yelp_Ep.json` — Yelp standalone injection controls; `I44-YELP-CELLS`.
- `i44_factorial_inference.json` — paired per-user factorial inference over the booked 2×2 cells (Sports n=35,598: E/C main effects, interaction; sign-flip permutation p, bootstrap CIs; each cell under an exact-reproduction guard vs its booked JSON; Beauty/Toys `aborted_missing_artifacts`, disclosed); `I44-FACTORIAL-INFERENCE`.

## results/ — matched-ensemble controls

- `ensctrl_sasrec_beauty.json` — BCE SASRec ×3 matched fusion, Beauty (0.0328); `E-ENSCTRL-A-BEAUTY` (plan `PREREG-E-ENSCTRL`).
- `ensctrl_sasrec_toys.json` — same, Toys (0.0429); `E-ENSCTRL-A-TOYS`.
- `ensctrl_sasrec_sports.json` — same, Sports (0.0203); `E-ENSCTRL-A-SPORTS`.
- `ensctrl_sasrec_yelp.json` — same, Yelp (0.0359, −22.1% vs the trio's 0.0461); `E-ENSCTRL-A-YELP`.
- `ensctrl_sasrec_instruments.json` — same, Instruments (0.0717, −27.9% vs the trio's 0.0995); `E-ENSCTRL-A-INSTRUMENTS`. Deterministically regenerated from the archived seed checkpoints under three hard guards after the original file was lost with its compute instance (full disclosure in the ledger entry and in the JSON's `regenerated` field).
- `ensctrl_sasrec_ce_beauty.json` — SASRec-CE ×3 matched fusion, Beauty (0.0551); `E-ENSCTRL-C-BEAUTY` (plan `PREREG-E-ENSCTRL-ARMC`).
- `ensctrl_sasrec_ce_toys.json` — same, Toys (0.0634); `E-ENSCTRL-C-TOYS`.
- `ensctrl_sasrec_ce_sports.json` — same, Sports (0.0316); `E-ENSCTRL-C-SPORTS`.
- Note: the `experiment` tag inside all six fusion JSONs reads `E-ENSCTRL_armA_sasrec_x3_fusion` — the arm-A runner was reused for arm C with `--loss ce`; the `_ce_` files are the **arm-C** looks. Correction of record in the `E-ENSCTRL-C-{BEAUTY,TOYS,SPORTS}` ledger entries (artifacts left unmodified, append-only discipline).
- `ensctrl_masked_ce_beauty.json` / `ensctrl_masked_ce_toys.json` / `ensctrl_masked_ce_sports.json` — masked (exclude_history) descriptive rescore of the frozen arm-C CE fusions (0.0689/0.0772/0.0379), each guarded by exact unmasked reproduction of the booked fusion; `E-ENSCTRL-MASKED` (plan `PREREG-E-ENSCTRL`, "unmasked primary + masked descriptive").
- `ensctrl_masked_bce_beauty.json` / `ensctrl_masked_bce_toys.json` / `ensctrl_masked_bce_sports.json` — same for the arm-A BCE fusions (0.0422/0.0541/0.0245); `E-ENSCTRL-MASKED`.

## results/ — same-protocol baselines (CLOSE-2)

- `close2_cell_beauty_tiger.json` — confirmatory cell 1/6, trio vs TIGER Beauty (Δ +0.0257, p 1e-4); `CLOSE2-CELL-BEAUTY-TIGER` (plan `CLOSE2-PREREG-FREEZE`; retrospective, disclosed).
- `close2_cell_beauty_sasrec.json` — cell trio vs SASRec Beauty; plan `CLOSE2-PREREG-FREEZE`, prereg stamp inside JSON; **[no per-cell ledger entry found]**.
- `close2_cell_toys_sasrec.json` / `close2_cell_toys_tiger.json` — Toys cells; same plan; **[no per-cell ledger entries found]**.
- `close2_cell_sports_sasrec.json` / `close2_cell_sports_tiger.json` — Sports cells; same plan; **[no per-cell ledger entries found]**.
- `close2_sasrec_beauty_s42.json` / `_s43` / `_s44` — faithful BCE SASRec seeds, Beauty (best 0.0265); harness `CLOSE2-CODE-READY`, plan `CLOSE2-PREREG-FREEZE`.
- `close2_sasrec_toys_s42.json` / `_s43` / `_s44` — Toys (best 0.0347); same.
- `close2_sasrec_sports_s42.json` / `_s43` / `_s44` — Sports (best 0.0149); same.
- `close2_sasrec_instruments_s42.json` / `_s43` / `_s44` — Instruments (0.0639/0.0629/0.0637); `CLOSE2-EXPAND-SASREC`.
- `close2_sasrec_yelp_s42.json` / `_s43` / `_s44` — Yelp (0.0278/0.027/0.0279); `CLOSE2-EXPAND-SASREC`.
- `close2_sasrec_ce_beauty_s42.json` / `_s43` / `_s44` — SASRec-CE descriptive seeds, Beauty (best 0.0526); harness `CLOSE2-CODE-READY`; repro-guarded in `E-ENSCTRL-C-BEAUTY`.
- `close2_sasrec_ce_toys_s42.json` / `_s43` / `_s44` — Toys (best 0.0587); ditto (`E-ENSCTRL-C-TOYS`).
- `close2_sasrec_ce_sports_s42.json` / `_s43` / `_s44` — Sports (best 0.029); ditto (`E-ENSCTRL-C-SPORTS`).
- `close2_tiger_beauty_s42.json` — same-protocol TIGER Beauty (0.0453; reproduces published 0.0454); training `J1-COMPLETE`, scoring `CLOSE2-CODE-READY`.
- `close2_tiger_toys_s42.json` — TIGER Toys (0.0409); same harness.
- `close2_tiger_sports_s42.json` — TIGER Sports (0.0275); same harness.
- `close2_tiger_instruments_s42.json` / `_s43` / `_s44` — TIGER Instruments 3-seed repro (0.0886/0.0861/0.0901); `CLOSE2-EXPAND-TIGER` (checkpoint provenance `CLOSE2-EXPAND-TIGER-SEEDS`).
- `close2_tiger_yelp_s42.json` / `_s43` / `_s44` — TIGER Yelp 3-seed repro (0.0198/0.0224/0.0218, repro-low caveat); same entries.
- `close2_tiger_latency.json` — TIGER Beauty batch-1 beam-20 latency (RTX 4090 + 1-thread CPU); `TIGER-LAT-VALIDATED`.
- `close2_tiger_latency_reconstructed_cpu.json` / `_gpu.json` — reconstruction companions; `TIGER-LAT-VALIDATED`.

## results/ — transition heuristic

- `i42_transition_only_beauty.json` — zero-parameter transition-count heuristic, Beauty (unmasked 0.0617 / masked 0.0648; γ selected inner-world); **[no dedicated ledger entry found]**.
- `i42_transition_only_toys.json` — Toys (0.0709 / 0.0738); **[no dedicated ledger entry found]**.
- `i42_transition_only_sports.json` — Sports (0.0343 / 0.0353); **[no dedicated ledger entry found]**.

## results/ — Instruments / Yelp transfer

- `i50_gate_instruments.json` — Instruments inner-world gate (pass before any real training); `I50-GATE-PASS` (prereg `I50-GATE-PREREG`).
- `i50_real_look_instruments.json` — Instruments single real look (trio 0.0995); `I50-REAL-LOOK-INSTRUMENTS` (prereg `I50-REAL-LOOK-PREREG`; audit `I50-REAL-LOOK-CODEX-AUDIT`).
- `i51_gate_yelp.json` — Yelp inner gate (w=0.9 selected inner-world); `I51-GATE-YELP` (prereg `I51-GATE-PREREG`).
- `i51_real_look_yelp.json` — Yelp single real look (deployed blend 0.0461); `I51-REAL-LOOK-YELP` (prereg `I51-REAL-LOOK-PREREG`).

## results/ — routing and diagnostics

- `i38_inner.json` — learned servable soft router, inner worlds only (beauty/instruments/yelp); `I38-ROUTER-INNER` + `I38-EXPAND-INNER`.
- `i29_adaptive_k.json` — seed-disagreement failure-prediction diagnostic (AUC 0.731); `I29-ADAPTIVE-K`.
- `i22b_variant_gate.json` — enrichment-verbalization variant ablation (successors/top-5/both, all <0.002); `I22B-KILL`.
- `i23_ann_replay.json` — serving-fidelity ANN replay (top-500/probe; exact HR@5 0.069 reproduced); `I23-ANN-REPLAY`.

## results/ — efficiency

- `serving_bench.json` — B/T/S CPU batch-1 bench, 1 Torch/FAISS thread (84 ms single / 198 ms trio; co-tenancy disclosed in `env`); `CLOSE3-SERVING-BENCH-T1`.
- `serving_bench_multithread.json` — superseded multithread run, kept for transparency; `CLOSE3-SERVING-BENCH`.
- `serving_bench_expand.json` — Instruments/Yelp bench (22.899541072547436 / 35.248879343271255 ms; 388/743 MB); `CLOSE3-EXPAND-SERVING`.
- `serving_bench_yelp_blend.json` — full deployed Yelp blend bench (trio + free arm, w=0.9): 104.4 ms/query batch-1 single-thread, 812 MB full-blend vs 743 MB trio-only; `CLOSE3-YELP-BLEND-SERVING`.
- `ann_bench_deployed_1m.json` — 1M-item recall–latency benchmark of the deployed 12-probe trio path (calibrated synthetic distractors; IVF-Flat/IVF-PQ/HNSW sweeps vs single-thread exact scan; paper Table 6); `ANN-BENCH-1M`.

## results/ — scale study (inner-world)

- `i18_size_score.json` — from-scratch size sweep M/L (13.1M / 59.9M) vs d256 refs; `I18-SIZE-CONTROL`.
- `i18ted_gate_beauty_m.json` — M+TED Beauty cell (decision cell); `I18-TED-BEAUTY-M-PROBE`.
- `i18ted_gate_toys_m.json` — M+TED Toys cell; `I18-TED-TOYS-M-PROBE`.
- `i18ted_gate_sports_m.json` — M+TED Sports cell; `I18-TED-SPORTS-M-PROBE`.
- `i18hard_gate_toys_m.json` — M hard-label attribution control, Toys; `I18-TOYS-M-HARD-CONTROL`.
- `i48_gate.json` — MiniLM pretrained-trunk + TED trio gate (0.0756/0.0778 vs 0.0802); `I48-LMQTED-BEAUTY-GATE`.
- `i45_ted_sweep_beauty_s42.json` — TED β/τ sweep, Beauty inner singles; `I45-TED-SWEEP-BEAUTY`.
- `i45_hetero_v2_beauty_s42.json` — heterogeneous-loss trio v2 gate; `I45-HETERO-V2-BEAUTY`.
- `i45_beauty_seed_soup.json` / `i45_toys_seed_soup.json` / `i45_sports_seed_soup.json` — uniform weight-soup collapse controls (0.0339/0.0478/0.0188); **[no dedicated ledger entries found]** (descriptive controls cited in §6).

## code/ (copied from research/experiments/, plus scripts/j5/train_j5a.py; anonymized, key reads switched to env vars)

- `train_j5a.py` — tower trainer for the headline I24-recipe trios (bit-deterministic per `REPRO-RETRAIN-VARIANCE`).
- `embed_openai_catalog.py` — plain-catalog OpenAI embedding (key from `$OPENAI_API_KEY`).
- `i22_enrich_embed_catalog.py` — collaborative-text enrichment + embedding (item side, E).
- `i22b_variant_embeds.py` — enrichment-variant embeddings (successors/top-5/both).
- `i22_test_look.py` — I22 (enrichment-only) single test look runner.
- `i24_build_input_emb.py` — predecessor-centroid 6144-d input table (query side, C).
- `i25_dump_enriched_texts.py` — dumps the exact enriched texts (released-artifact aid).
- `i24_gate.py` / `i24_test_look.py` — Beauty trio inner gate + single real look.
- `i30_prep_toys.py` / `i30_embed_t5_toys.py` / `i30_gate.py` / `i30_real_look.py` — Toys world prep, free-arm t5-xl embeddings, gate, real look.
- `i35_prep_sports.py` / `i35_embed_t5_sports.py` / `i35_gate.py` / `i35_real_look.py` — Sports equivalents.
- `i40_train_tricks.py` — TED / lqted / hetero trainer (TED = `--ted_beta 0.1 --ted_tau 0.1`).
- `i40_gate.py` — trio gate/scorer for TRICKS towers (also produced the TED real looks).
- `i43_masked_real.py` — masked-convention rescore with exact unmasked-reproduction guard.
- `i42_transition_only.py` — zero-parameter transition-count heuristic.
- `i44_ablation_prep.py` / `i44_eval_cell.py` / `i44_factorial_inference.py` — 2×2 ablation prep, cell evaluation, factorial contrasts.
- `i45_seed_soup.py` — uniform weight-soup control.
- `i18_size_score.py` — size-sweep scorer (M/L towers).
- `i37_train_lmq.py` — MiniLM LM-query-trunk trainer (I48 cells).
- `i38_soft_router.py` — learned servable soft router (inner worlds).
- `i29_adaptive_k.py` — seed-disagreement adaptive-K diagnostic.
- `i23_ann_replay.py` — serving-fidelity ANN replay.
- `close2_sasrec.py` — faithful SASRec / SASRec-CE trainer+scorer (`--loss bce|ce`).
- `close2_tiger_score.py` — TIGER beam scorer through the pinned metric implementation.
- `close2_tiger_runner.md` — box runbook for TIGER training/dumping (vendor pipeline).
- `close2_analysis.py` — paired per-user sign-flip randomization cells (CLOSE-2 plan).
- `close2_sports_replay.py` — deterministic replay recovering Sports per-user vectors; `CLOSE2-SPORTS-REPLAY`.
- `ensctrl_sasrec_fusion.py` — matched-ensemble control fusion (per-user z-scores, simplex weights, repro guards).
- `ensctrl_masked_rescore.py` — deterministic masked descriptive rescore of the booked fusions (frozen weights, exact unmasked-reproduction guard; no selection).
- `close3_serving_bench.py` / `close3_serving_bench_expand.py` — CPU serving benchmarks.
- `close3_serving_bench_yelp_blend.py` — full deployed Yelp blend serving bench (trio + free arm end-to-end).
- `i44_factorial_inference.py` — paired factorial inference runner (per-user re-scoring under exact-repro guards + permutation/bootstrap contrasts).
- `ann_bench_deployed.py` — 1M ANN benchmark runner (calibrated distractor corpus, deployed 12-probe union-rescore semantics, single-thread timing).
- `i50_prep_instruments_data.py` / `i51_prep_yelp_data.py` / `i50_prep_expand.py` / `i50_embed_t5_expand.py` — transfer dataset prep + embeddings.
- `i50_gate_expand.py` / `i50_real_look_expand.py` — transfer inner gates + single real looks (both datasets).

## Excluded from this package — available on request

Binary artifacts are excluded (size; the project `.gitignore` excludes them from the
repo as well). Every excluded per-user vector corresponds 1:1 to an included JSON:

- Per-user metric vectors (`*_per_user.npz`) for: `i24_test`, `i30_real`, `i35_real`, `i23_test`, `i23b_test`, `i22_test`,
  `close2_sasrec_{beauty,toys,sports,instruments,yelp}_s{42,43,44}`,
  `close2_sasrec_ce_{beauty,toys,sports}_s{42,43,44}`,
  `close2_tiger_{beauty,toys,sports}_s42`, `close2_tiger_{instruments,yelp}_s{42,43,44}`,
  `ensctrl_sasrec_{beauty,toys,sports}`, `i38_{beauty,instruments,yelp}_inner`.
- Tower checkpoints (`outputs/artifacts/**/j5.pt`, `tower.pt`), TIGER checkpoints
  (`close2_tiger_seed_{42,43,44}` + per-seed SID pkls), frozen embedding tables
  (`data/*_backbone/*_openai-3-large_*.pt`, t5-xl free-arm tables).
- Raw dataset backbones (public: Amazon-2014 B/T/S, LC-Rec Instruments snapshot,
  LETTER Yelp snapshot) — rebuildable with the prep scripts in `code/`.

## Augmentation-selector controls (§4.2, Table 3)
- `code/simpleaug_{selectors,prep,box,explore}.py` — selector definitions, E/C prep, trio training+eval, proxy.
- `results/simpleaug_box_*.json` (12) + `results/simpleaug_explore_beauty.json` — per-selector inner pseudo-test trios (both masking conventions) + proxy sweep.
