# E-ENSCTRL — matched-ensemble control baseline (prereg)

**Booked 2026-08-15, BEFORE any training. Licensed by codex WWW-pivot review (W4: "a matched
heterogeneous-ensemble baseline that separates the value of E/C from ordinary fusion") + user GO.**

## Question
Are the deployed trio's gains attributable to the E (content-enrichment) and C (collaborative
input) injections, or would ANY matched-count ensemble of standard baselines under the identical
protocol do the same? A reviewer-facing control for papers 1+2.

## Arms (per dataset)
- **A (primary): SASRec x3 seeds (42/43/44)**, identical config to the booked CLOSE-2 SASRec runs
  for that dataset. Score-level fusion, same menu as our trio: mean-z and inner-val-weighted blend;
  fusion variant + weights selected ONLY on the inner world (train->-3, select on -2), then frozen.
- **B (secondary): heterogeneous SASRec + TIGER fusion** reusing CLOSE-2 checkpoints/score dumps
  where archived (retrain only what is missing, same booked configs). Same inner-world fusion
  selection. DuoRec is NOT included unless a faithful implementation is first validated against its
  paper on Beauty (would be a separate prereg).
- **C (addendum, prereg'd 2026-08-15 BEFORE any arm-C training; codex xhigh-prep review):
  SASRec-CE x3 seeds (42/43/44)** — the STRONGER control. Codex flagged that arm A (BCE) could be
  dismissed as underpowered since our own CLOSE-2 bookings show the CE recipe is ~2x BCE
  (Beauty 0.0511 / Toys 0.0587 / Sports 0.0288 vs BCE 0.0231/0.0334/0.0149, s42). Identical
  config to the booked close2_sasrec_ce_* runs; repro guard against those JSONs (all 3 seeds
  booked for Beauty/Toys/Sports); same fusion menu + inner-val weight selection as arm A;
  ONE prereg'd real-test look per dataset. Datasets: Beauty, Toys, Sports ONLY (CE not booked on
  Instruments/Yelp; extending would need new solo baselines = separate prereg).
  Runner: ensctrl_sasrec_fusion.py --loss ce  (outputs ensctrl_sasrec_ce_{ds}.json).

## Datasets & budget
Primary: Beauty, Toys, Sports (Amazon-2014) — the headline trio datasets.
Secondary (run only if box time allows within ~3 days): Instruments, Yelp.
Box: fresh 4090 instance <id> (label ensctrl, ~$0.30/hr).

## Eval & looks
All numbers via src/semplan/eval/protocol.py, unmasked primary, masked descriptive.
**ONE prereg'd real-test look per dataset per arm** (max 6 looks primary+secondary-A; arm-B looks
only on datasets where arm A completes; arm-C adds max 3 looks, B/T/S only). Inner-world numbers unlimited as usual.

## Primary read (per dataset)
Best control ensemble (arm A) real-test unmasked HR@5 vs deployed trio unmasked HR@5
(0.0710 / 0.0804 / 0.0401 B/T/S; 0.0995 Instruments; 0.0461 Yelp).
- Control < trio: E/C value confirmed beyond generic ensembling (expected).
- Control >= trio on any dataset: booked as-is, honesty item in both papers, framing revised.
No optimization against the test read; all outcomes published.

## Deliverables
Per-dataset JSON + per-user npz dumps pulled local; ledger booking per dataset; paper-2 table row
"Matched ensemble (SASRec x3)" + prose; box destroyed at completion (idle-boxes-down rule).
