# Supplementary material (WSDM 2027 submission)

## A. TIGER reproduction detail (moved from the paper appendix)

TIGER 3-seed reproduction detail on Instruments/Yelp (seeds 42/43/44).
*Notes:* cold slice 0.0 for all seeds (TIGER cannot emit unseen SIDs;
Instruments n=45, Yelp n=118). The s43/s44 result-JSON `seed` field was
corrected post-hoc (the scorer's `--seed` defaulted to 42); checkpoint
provenance verified in dump logs (per-seed checkpoints and per-seed SID
pkls).

| Metric (overall) | Instruments mean±SD | per-seed | Yelp mean±SD | per-seed |
|---|---|---|---|---|
| HR@5    | 0.0883±0.0020 | 0.0886/0.0861/0.0901 | 0.0213±0.0014 | 0.0198/0.0224/0.0218 |
| HR@10   | 0.1118±0.0017 | 0.1121/0.1099/0.1133 | 0.0347±0.0017 | 0.0331/0.0365/0.0346 |
| NDCG@5  | 0.0732±0.0019 | 0.0735/0.0711/0.0749 | 0.0137±0.0007 | 0.0129/0.0143/0.0139 |
| NDCG@10 | 0.0807±0.0018 | 0.0810/0.0788/0.0824 | 0.0180±0.0009 | 0.0171/0.0189/0.0180 |

Per-seed artifacts: `close2_tiger_{instruments,yelp}_s{42,43,44}.json`
(+ `_per_user.npz`).

## B. Paper-to-artifact map for the two newest pillars

- **§4.1 paired factorial inference (Sports)** →
  `results/i44_factorial_inference.json` (effects, permutation p-values,
  bootstrap CIs, exact-reproduction guard values; Beauty/Toys entries record
  `aborted_missing_artifacts` — checkpoints existed only on since-destroyed
  cloud instances and were never archived; disclosed in the paper).
- **§6 Table "Recall–latency at 1M items"** →
  `results/ann_bench_deployed_1m.json` (all 14 sweep rows, distractor
  calibration, memory footprints, timing methodology) +
  `code/ann_bench_deployed.py` (the benchmark runner; deployed 12-probe
  union-rescore semantics, single-thread timing).

## C. Matched augmentation-selector controls (paper §4.2, Table 3)

The augmentation-selector sweep isolates *which* context the E+C injection needs
by holding the frozen encoder, K=3 budget, E/C sites, tower recipe, and
evaluation fixed and varying only the selector P3(i). Development-world (Beauty
inner pseudo-test), three tower seeds, zero outer-test looks.

Reference code (`code/`):
- `simpleaug_selectors.py` — the 12 selectors (graph: pred3/pred1/ppr3/ppmi3/
  translift3/upred3; content: contentnn3/bm25_3; controls: random3/popular3/
  catbrand3; plus `none`).
- `simpleaug_prep.py` — enrich (E) + [own;centroid] input (C) per selector.
- `simpleaug_box.py` — train J5b K4 trio (seeds 42/43/44), inner pseudo-test,
  trio z-score ensemble; writes the per-selector JSONs.
- `simpleaug_explore.py` — training-free content-cosine proxy (documents why the
  proxy cannot discriminate selectors; the signal is tower-borne).

Per-selector results (`results/simpleaug_box_<selector>.json`): each has
`per_seed_HR5`, `trio` (HR@5/NDCG@5 at 5/10, both masking conventions), and the
gate outcome. Table 3 (masked HR@5) is reproduced exactly by these files:
ppr3 0.0814, pred3 0.0803, pred1 0.0760, ppmi3 0.0751, contentnn3 0.0749,
bm25_3 0.0744, translift3 0.0741, upred3 0.0740, none 0.0647, catbrand3 0.0597,
popular3 0.0591, random3 0.0587.


---

# Moved from the main paper (page budget; content preserved here)

The following was in earlier drafts of the main paper and relocated here to meet the
9-page limit. It is supporting analysis, not a claimed contribution.


## D. Training-free arm: learned soft router (was §2.5 / §5.3)

**Learned Soft Router.**
Static blending applies one weight to every user, although which arm is better
varies across users. To replace it with a per-user policy, we train a router
over the two arms in the inner world with zero test looks. All features are
target-free, namely history statistics available at serving time, so the
routed system remains two-tower/ANN-servable. More precisely, the router is a
standardized, class-balanced logistic regression over 13 per-user features,
all computable at serving time from the query-side pass itself: (a) history
length (raw and log1p); (b) trio retrieval confidence, including the raw
max-IP top-1 and top-10-mean scores and the per-seed top-1 std; (c) seed
disagreement, measured by the mean pairwise top-10 overlap across the n_e
trio seeds (the adaptive-K diagnostic signal, AUC 0.73 for trio failure on
Beauty); (d) free-arm z-scores (top-1 and top-10 mean) and the trio-vs-free
top-1 z gap; and (e) history popularity statistics from the training
item-count table, including log-mean/min/last and the fraction of cold history
items. Note that the item-count table is an item-side table independent of the
target, and is hence servable. Each user is labeled on inner validation by
whether the free arm beats the trio, that is, whether free-arm per-user HR@5
> trio per-user HR@5. We sweep two policies on inner validation only,
including router-hard (p > thr → free arm) and router-soft
(per-user free weight w_f = nearest grid point to (1 − p),
again mixing the arms' per-query z-normalized scores). As a result, deployment
costs one logistic evaluation per query on features that the two arms already
produce. Finally, this router replaces an earlier oracle cold-routing
diagnostic that branched on target identity, which we never cite as deployable.


**Routing**

All routing numbers in this subsection are inner-world (pseudo-test) HR@5
with zero test looks.

**Instruments:** always-trio 0.1109 / router_hard 0.1104 /
router_soft 0.1029 / oracle 0.1269 (inner pseudo HR@5). The free arm wins
for only 1.1% of users; routing adds nothing over always-trio, which is the
right deployment. Oracle headroom (+0.016) exists but a router at test AUC
0.65 cannot harvest it. The cold slice is tiny (n=73 inner).

**Yelp:** router_soft 0.0393 vs. pure arms always-trio 0.0272 /
always-free 0.0309, oracle 0.0509, router AUC 0.62. The servable learned
router *recovers* the tuned static blend (inner, w=0.9: 0.0397) without
oracle or target knowledge—a deployable per-user policy matching a weight
that otherwise needs a tuning sweep—but it does *not beat* the blend;
oracle headroom +0.011 is documented, not claimed. On the cold slice,
router_soft matches the free arm (0.0353).

The claim shape is deliberate: per-user learned routing servably
*recovers* the blend, it does not beat it. The oracle-routed numbers
from the real looks (Instruments routed 0.0996, Yelp routed 0.0462) are
diagnostic-only and labeled as such. On Beauty, an earlier learned router
over ensemble arms *failed* its inner-validation gate (hard 0.0787,
soft 0.0711, vs. bar 0.0802) and was dropped—an honest negative.

## E. When does TED transfer — and does scaling help? (was §7)

### When does TED transfer — and does scaling help?

All numbers in this section are INNER-WORLD pseudo-test HR@5 (zero real
test looks); the real-stage TED numbers live in the main-paper BTS table.
The TED upgrade of the main paper is loss-only and free at serving time, so
we probed it on every dataset at trio level in the inner worlds (one
preregistered gate each, zero test looks). It does *not* transfer
uniformly. Across the five cells, the TED trio moved inner pseudo-test HR@5
against the same-scorer deployed-trio bar by: Beauty -0.0005, Toys
+0.0041, Sports +0.0041, Instruments +0.0005, Yelp +0.0048
(same-scorer anchors reproduced the deployed bars exactly before each
comparison). The split tracks headroom, not dataset family: the two failures
are precisely the cells whose trios were most heavily tuned toward their
inner ceiling (Beauty after its long optimization lineage; Instruments, whose
singles already sit within 0.004 of the trio), while the largest gain lands
on Yelp, the weakest trio arm of the five. TED redistributes probability mass
toward observed transitions; where the ensemble has already absorbed that
structure, it has nothing left to add. The same redundancy argument applies
*across arms*: rescoring the full Yelp blend with the TED trio
substituted in, the +0.0048 trio-level gain evaporates at the
inner-val-selected blend weight (0.0398 vs. the deployed blend's 0.0397
at w=0.9)—the training-free arm already carries the transition signal
TED injects, so the deployed Yelp configuration stands and no real-stage TED
run was licensed there.

The same mechanism explains our scale results. From-scratch scale
*collapses* at these data sizes: a 13.1M-parameter (d512/L4) tower under
the identical recipe loses 21% (Toys, 0.0638) to 38–40% (Beauty,
0.0463) of inner pseudo-test hard-label HR@5 versus the d256 recipe
(0.0749–0.0771/0.0808); on Sports only the TED variant was trained at
this size, and even with TED it sits 31–33% below d256 (0.0313 vs.\
0.0452–0.0465). Where both cells exist, TED recovers 37–40%
(Beauty, 0.0578) and 51% (Toys, 0.0724) of the hard-label
degradation—its largest gains appear exactly where the model is most
under-trained for its size—but closes the gap nowhere. A
pretrained trunk does not rescue scale either: swapping the d256 trunk for a
fine-tuned MiniLM-L6 encoder (22.7M encoder parameters, 26.3M trainable with
heads; identical recipe and serving form) reaches 0.0756 plain and 0.0778 with TED on Beauty inner—TED
again helps the under-trained trunk, and again falls short of the
from-scratch d256 trio's 0.0802. At 20–120k training sequences, parameter
count is a liability that transition-aware soft labels mitigate but do not
repay; the efficient frozen-catalog trio is not a compromise forced by our
budget, it is the best-performing point we can find on the scale axis.

## F. Full honesty and limitations — nine disclosures (was §8)

### Honesty and limitations

Nine caveats materially qualify our claims. The core caveats are disclosed in
situ in the main paper where the affected number appears (the main paper's
"Evaluation scope" section summarizes three); the full list with per-number
detail follows here.

**(1) Yelp margin composition.** Single towers on Yelp reach only
0.0284–0.0291, ≈ our same-protocol SASRec (0.0276); the injection
controls of the main paper show the single-seed injected towers already reach
SASRec, so the injections carry standalone value and the matched-ensemble
control rules out generic ensembling as sufficient. The deployed 0.0461
additionally includes the w=0.9 free-arm blend; from a single booked Yelp look
we do **not** claim a clean injection/ensembling/blend decomposition. Both the
free-blend contribution and the development-selection asymmetry remain
disclosed wherever the headline appears, including the abstract and the main
results table.

**(2) Yelp cold slice prefers the free arm.** On the n=118 cold-user
slice, the free arm (0.0424) beats the trio (0.0254). The oracle-routed
0.0462 is a diagnostic only (target-conditioned); the deployable answer is
the learned router, whose cold behavior matches the free arm
(main paper).

**(3) The BSARec Yelp bar is not seq-len-matched.** The published 0.0275
uses max-len 50; the TIGER-line (and our) Yelp numbers use max-len 20. The
independent reproduction band for BSARec on this snapshot is
0.0178–0.0193 [bsarecrepro]. Both facts stay attached to the bar
everywhere it appears.

**(4) Development-selection asymmetry.** The (E,C) cells of the ablation
tables reuse the deployed trios
that were developed and selected in the inner worlds; the other cells were
trained once for the ablation. The Beauty/Toys/Sports headline margins over
published scalars are likewise development-selected and read descriptively
(main paper).

**(5) TIGER Yelp repro-low.** Our TIGER Yelp reproduction
(0.0213±0.0014, 3 seeds) is below the published 0.0253, consistent with
the field-wide Yelp repro-low pattern (BSARec 0.0275 → 0.0182).
Both numbers appear in the main results table; conclusions on Yelp never rest
on the repro number alone. (The Instruments repro, by contrast, lands just
above the published band: 0.0883±0.0020 vs. 0.0863–0.0870, seed range
straddling it.)

**(6) Include-val training and enrichment provenance.** Final real
towers train through the validation transition (one extra labeled transition
vs. LC-Rec's stated convention); the real-world pred3 enrichment is
test-target-safe but built from train+validation interactions—stated as
such, never as ``train-only''.

**(7) Published-bar protocol documentation.** Only LC-Rec fully
documents full-corpus LOO among the Instruments LLM-band papers (OD-LLM
reversed val/test as written; SemConv/ LOHRec/
EAGER-LLM underspecified);
``same protocol'' is qualified wherever published bars are cited. Tie
handling in our scorer is strictly-greater rank (optimistic).

**(8) GRAM's Yelp HR@5 exceeds ours.** GRAM [gram] prints 0.0476 on
our exact Yelp snapshot (statistics match
30,431/20,033/316,354;
full-ranking; 3 seeds; max-len 20), 0.0015 above our deployed configuration's
0.0461—so we do *not* claim ``above every published bar'' on Yelp.
Context: our NDCG@5 0.0340 > their 0.0326; GRAM is
LLM-assisted (LLM-derived multi-granular semantic alignment) vs. our
10.3M-parameter CPU-servable trio; their masked/unmasked convention is
unspecified (their SASRec repro 0.0284 ≈ our same-protocol 0.0276
suggests comparability); their TIGER repro (0.0164) is deflated. Our Yelp
claim is therefore: best efficiency-class number, within 0.0015 of the best
published number overall.

**(9) Cold items and scope.** Items with no interaction history carry no
collaborative signal (Beauty cold HR@5 0.006; the Yelp cold slice is
near zero in every trio cell—pp 0.0169 / pC 0.0085 / Ep 0.0 unmasked,
n=118); composition with distillation [mqsated] is explored only in the
reported TED study; the domains studied are text-rich; results depend on a
single commercial embedder family; shortcut concerns [shortcut] apply to the
benchmark family as a whole. The 1M ANN study (main-paper ANN benchmark) measures
retrieval fidelity against exact model scores on a calibrated
synthetic-distractor corpus, not recommendation quality on a real
million-item catalog.