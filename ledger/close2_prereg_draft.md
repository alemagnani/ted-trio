# CLOSE-2 prereg DRAFT — same-protocol generative/attentive baselines (TIGER, SASRec)

Status: **FROZEN** — ledger CLOSE2-PREREG-FREEZE (UTC timestamp in ledger.jsonl), before any
baseline test scoring beyond the already-spent TIGER Beauty look. RecSys 2027 repair
program, hole 2. No edits below this line after the freeze timestamp.

## 1. Primary claim

On each of Beauty, Toys, Sports (Amazon 2014, LOO full-corpus protocol =
`src/semplan/eval/protocol.py`, real test split, no history exclusion, full-corpus ranking,
no sampled negatives):

> **Trio real-test HR@5 > same-protocol TIGER HR@5, and trio real-test HR@5 > same-protocol
> SASRec HR@5, with paired sign-flip (randomization) two-sided p < 0.05 per comparison**
> (`semplan.eval.significance.paired_randomization` on per-user HR@5 vectors, default
> n_perm; paired over the identical test users).

6 comparisons (2 baselines x 3 datasets); the primary claim passes only if ALL 6 deltas are
positive and significant. Holm-corrected p-values reported as robustness, not gating.
Secondary (reported, not gating): HR@10, NDCG@5/10 with paired bootstrap CIs; seen/cold
slices descriptive only (TIGER cold is structurally 0; SASRec cold is structurally ~0).

Trio = frozen mean-of-3-seed z-score ensemble at w=1.0 (i24 Beauty / i30 Toys / i35 Sports).

## 2. What is already SPENT (frozen; zero new looks for our arms)

Our real-test looks for the trio are **SPENT** and their outputs frozen. Baselines are scored
fresh and compared against the SAME saved per-user arrays — our arms are never re-run,
re-selected, or re-inspected:

| dataset | frozen trio HR@5 | per-user array (trio = `w1.0_*` keys) | look marker |
|---------|-----------------|----------------------------------------|-------------|
| Beauty  | 0.0710 | `research/results/i24_test_per_user.npz` (n=22363: `w1.0_hit5/dcg5/hit10/dcg10`, `cold_mask`) | i24_test_look.json (1 look) |
| Toys    | 0.0804 | `research/results/i30_real_per_user.npz` (n=19412, same keys) | i30_real_look.json (1 look) |
| Sports  | 0.0401 | `research/results/i35_real_per_user.npz` (n=35598, same keys; zero-look replay DONE, hard-assert exact — ledger CLOSE2-SPORTS-REPLAY 2026-08-07T01:31:54Z) | i35_real_look.json (1 look, marker `.i35_real_look_taken`) |

## 3. Sports gap: zero-look recompute of trio per-user ranks — **RESOLVED 2026-08-07**

Executed as specified below: `research/experiments/close2_sports_replay.py` ran 2026-08-07,
hard-assert passed EXACTLY (0.0401/0.0622/0.0259/0.0330; routed secondary 0.0405 also exact),
npz written, zero looks. Ledger: CLOSE2-SPORTS-REPLAY 2026-08-07T01:31:54Z. Original spec kept
below for the record.

I35 saved only aggregates (no per-user npz), so Sports paired tests require recomputing the
trio per-user hit/dcg vectors **zero-look from the frozen checkpoints** — a deterministic
replay of the already-taken look, not a new look. Inputs (all frozen on disk):

- checkpoints: `outputs/artifacts/i35_sports_real_openai_k4_s{42,43,44}_iv/j5.pt`
- item emb: `data/amazon_sports_backbone/Sports_openai-3-large_i22real_embeddings.pt`
- input emb: `data/amazon_sports_backbone/Sports_openai-3-large_i24real_input.pt`
- collab cache: `outputs/artifacts/collab_space_sports_real.npz`
- sequences: `data/amazon_sports_backbone/Sports_and_Outdoors.txt` (n_items 18357)

Procedure (replay `research/experiments/i35_real_look.py` scoring path verbatim at w=1.0):
the replay script writes `research/results/i35_real_per_user.npz` and **hard-asserts** its
aggregate HR@5/HR@10/NDCG@5/NDCG@10 equal the frozen i35_real_look.json trio row
(0.0401/0.0622/0.0259/0.0330) before anything else is printed or read. Any mismatch =
abort, no numbers reported. No metric other than the already-published aggregates is
inspected from this replay; only the per-user vectors feed the paired tests.

## 4. Baseline arms (scored fresh, one look each, this prereg is their prereg)

**TIGER** (seed 42, matching J1/published single-seed convention):
- Beauty: already scored — `research/results/close2_tiger_beauty_s42{.json,_per_user.npz}`
  (HR@5 0.0453, from the J1 beam dump; reproduces published 0.0454).
- Toys, Sports: LIGER vendor pipeline per `research/experiments/close2_tiger_runner.md`
  (sentence-t5-xl content model, RQ-VAE 256x3+collision, T5 128d 6+6, 200k steps, 20-beam
  test dump) then `close2_tiger_score.py` -> per-user npz in `split.test` order.
- No tuning on our side; vendor defaults; the test dump is scored ONCE.

**SASRec** (seeds 42, 43, 44; per-seed comparisons vs trio, plus seed-mean reported):
- `research/experiments/close2_sasrec.py` — paper defaults (d=50, 2 blocks, 1 head,
  dropout 0.5, BCE + 1 uniform negative, Adam 1e-3, batch 128, maxlen 20 = protocol cap),
  trained on the protocol train side (asserted == `split.train_pairs`), model selection on
  protocol VAL HR@10 only, TEST scored once through `protocol.evaluate`.
- Primary SASRec comparison vector = per-seed; the claim uses the **best single SASRec seed
  by test HR@5** (conservative in our favor's disfavor: trio must beat SASRec's best seed).

Both baselines get full feature access rules of the protocol (nothing withheld); they simply
don't use content embeddings, as in their papers.

## 5. Test procedure

For each dataset and baseline: load frozen trio per-user `hit5` (and secondary vectors),
load baseline `hit5` from its npz (identical `split.test` user order — asserted via
`cold_mask` equality), run `significance.paired_randomization` (primary, HR@5) and
`significance.paired_bootstrap` (CIs, all metrics). One analysis script
(`research/experiments/close2_analysis.py`); each (dataset, baseline) cell is analyzed
EXACTLY ONCE, as soon as its baseline npz exists (cells are independent paired tests on
frozen vectors — no cross-cell selection is possible since the claim requires ALL 6 to
pass). Results appended to ledger + RESULTS.md per cell.

## 6. Spend / schedule

- GPU: TIGER Toys+Sports ≈ 8-10 h on RTX 4090 (RQ-VAE 2x0.5 h, train 2-3 h + 4-5 h,
  dumps ~0.5 h). SASRec 3 datasets x 3 seeds ≈ 2-3 h total (tiny model; full-corpus evals
  dominate). Sports trio replay: CPU-only, ~1-2 h. Total new GPU ≈ 10-13 h.
- Launch order when the box frees: Sports TIGER (longest pole) -> Toys TIGER -> SASRec x9.
- Zero new looks for trio arms; one look per baseline cell; no RESULTS.md/ledger writes
  until the freeze timestamp exists.

## 7. Falsification / reporting

If any of the 6 comparisons fails (delta <= 0 or p >= 0.05), the primary claim FAILS and is
reported as such per dataset — no post-hoc metric switching, no re-run of baselines with
different hyperparameters, no dropping datasets. Cold-slice and NDCG stories may be
discussed but cannot rescue the claim.
