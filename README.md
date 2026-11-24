# TED-Trio

Code, per-user outputs, and the experiment ledger for

> **TED-Trio: Injecting Directed Transition Signal into Frozen Text-Embedding Retrievers
> for Sequential Recommendation.**
> Alessandro Magnani, Min Xie, Ming-Hen Tsai, Jyun-Yu Jiang.
> *Proceedings of the 20th ACM International Conference on Web Search and Data Mining
> (WSDM '27)*, Hong Kong, February 15-19, 2027.
> **Paper:** [`paper/ted-trio-wsdm2027.pdf`](paper/ted-trio-wsdm2027.pdf) (camera-ready)
> · **DOI:** [10.1145/3862322.3863777](https://doi.org/10.1145/3862322.3863777)
> · **Supplement:** [`docs/SUPPLEMENT.pdf`](docs/SUPPLEMENT.pdf) ([markdown](docs/SUPPLEMENT.md))

TED-Trio makes a *frozen* commercial text-embedding space collaborative without touching
the embedder. For every item we mine its three most frequent sequence **predecessors**
(leak-safe: each user's held-out event is dropped first) and inject that directed
transition signal at two sites:

* **E (item side):** the predecessors' titles are appended to the item text
  ("Often bought after: ...") before the item is embedded once, offline, with OpenAI
  `text-embedding-3-large` (3072-d). The item index stays a single frozen matrix.
* **C (query side):** each history token of a small query tower is
  `[ e_i ; mean(e_j : j in P3(i)) ]` (6144-d), the item's enriched embedding concatenated
  with its predecessors' centroid.

Three 3.42M-parameter query towers (seeds 42/43/44, 4 query vectors each) are scored by
max inner product against the one frozen index and fused by per-query z-score averaging
(the "trio"); the optional **TED** upgrade replaces the hard next-item target with
transition-aware soft labels. On Yelp the trio is statically blended with a
**training-free arm** (whitened sentence-t5-xl content space + PPMI-SVD collaborative
space). No item embeddings are trained and no LLM runs at serving time.

This repository lets you rebuild every number in the paper from the public datasets:
data preparation, embedding, tower training, inner-world gates, the single real-test
looks, the masked rescores, all baselines (SASRec, SASRec-CE, TIGER, transition-count
heuristic), the matched-ensemble control, the 2x2 injection ablations, the paired
significance tests, the selector controls, and the efficiency benchmarks. **We do not
redistribute data, embeddings or checkpoints**; `results/` holds the *original* output
of every run in the paper so you can compare your reproduction against it field by field.

## Quick start

```bash
pip install -e ".[dev]"
make test              # unit tests: mining leak-safety, templates, centroids, z-fusion,
                       # masked/unmasked metrics, sign-flip test, free arm, heuristic (CPU, seconds)
make check-results     # every headline number in the paper == results/, and the six paired
                       # confirmatory tests + the §4.1 factorial inference re-run from the
                       # shipped per-user vectors (CPU, ~5-10 min; 254 checks)
make smoke             # synthetic data, fake offline embedder: EVERY pipeline stage end to end (CPU, ~1 h)
TT_DATA_DIR=/path/to/data make check-data      # validate your prepared datasets
TT_DATA_DIR=/path/to/data make quick           # every stage on 200 real users per dataset (CPU, ~2-3 h)
TT_DATA_DIR=/path/to/data make beauty-headline # Beauty towers -> gate -> look -> masked (GPU)
TT_DATA_DIR=/path/to/data make reproduce       # everything (GPU; section 5)
```

`scripts/reproduce.sh` is the driver behind the last four targets (`--help` for options):
it runs the pipeline of section 5 in order, per dataset and per stage
(`data embed towers gate look baselines tests ablation extras cross`), inside a workspace
(`outputs/full` or `outputs/quick/<dataset>`) so the shipped `results/` are never
overwritten; `python scripts/compare_results.py outputs/full/results results` then diffs
your numbers against the paper's. Full mode skips every step whose output exists (resume
after interruption) and stops at the first failure. `--quick` runs the *same scripts* on the
first 200 users of each dataset with 20 training steps, never calls the OpenAI API (prep
scripts run with `--embedder fake` in a separate scratch workspace; the rest uses your
existing item embeddings read-only), and writes a PASS/FAIL table to
`outputs/quick/STATUS-<datasets>.tsv`. Every embedding script takes
`--embedder openai|fake|st:<sentence-transformers model>`, so the pipeline also runs without
an OpenAI key (numbers then differ from the paper's).

---

## Contents

1. [Repository layout](#1-repository-layout)
2. [Environment](#2-environment)
3. [Data: obtaining the five datasets](#3-data-obtaining-the-five-datasets)
4. [The one-look ledger discipline (read before running a "look" script)](#4-the-one-look-ledger-discipline)
5. [End-to-end pipeline](#5-end-to-end-pipeline)
6. [Paper claim -> command -> expected number](#6-paper-claim---command---expected-number)
7. [What is and is not bit-reproducible](#7-what-is-and-is-not-bit-reproducible)
8. [Citation and license](#8-citation-and-license)

---

## 1. Repository layout

```
src/semplan/          the library: query towers + training (dense/), the single pinned
                      evaluation protocol (eval/protocol.py), paired significance
                      (eval/significance.py), whitened content/collab spaces (spaces.py)
scripts/              every runner behind a paper number (flat; shared helper modules such
                      as i41_common.py, i36_train_joint.py, close2_sasrec.py,
                      close3_serving_bench.py, simpleaug_selectors.py live here too and are
                      imported as siblings: no sys.path hacks needed)
baselines/tiger/      TIGER runbook + our dump / staging / latency helpers for the
                      (not vendored) LIGER reference pipeline
results/              ORIGINAL result JSONs + 67 per-user .npz vectors from the paper runs
ledger/               ledger.jsonl (append-only, 231 entries) + the frozen confirmatory
                      analysis plan (close2_prereg_draft.md) + matched-ensemble spec
docs/                 SUPPLEMENT.md/.pdf, MANIFEST.md (file -> ledger entry), data_checksums.md5
paper/                the camera-ready PDF
tests/                pytest unit tests (CPU, seconds); run in CI (.github/workflows/tests.yml)
Makefile              test / check-results / check-data / smoke / quick / beauty-headline / reproduce
data/  outputs/       created by you (git-ignored): datasets, embeddings, checkpoints
third_party/          created by you (git-ignored): the LIGER clone (TIGER baseline)
```

**Path convention.** Every script resolves paths from the repository root
(`ROOT = Path(__file__).resolve().parents[1]`), so commands can be run from anywhere,
but all commands below are written from the repo root. Inputs live in `data/`, towers and
caches in `outputs/artifacts/`, and every result is written to `results/`, i.e. **a
reproduction overwrites the shipped original in place**: `git diff results/` then shows
exactly which numbers moved, and `git checkout results/` restores the originals.
Script names keep the experiment ids used in the ledger (`i24_*` = Beauty trio,
`i30_*` = Toys, `i35_*` = Sports, `i50_*` = Instruments, `i51`/`*_yelp` = Yelp,
`i40_*` = TED, `i43_*` = masked rescore, `i44_*` = 2x2 ablation, `close2_*` =
baselines and paired tests, `ensctrl_*` = matched-ensemble control, `close3_*`/`ann_*` =
efficiency). Paths printed *inside* the shipped JSONs/ledger use the original project
layout: read `research/results/` as `results/` and `research/experiments/` as `scripts/`.

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10 (paper: 3.12)
pip install -e .                 # numpy scipy scikit-learn pandas faiss-cpu torch
                                 # transformers sentence-transformers requests tqdm
pip install -e ".[data]"         # + gdown (Instruments download), wget (LIGER import)
export OPENAI_API_KEY=...        # only needed for the embedding step (5.3); never stored
```

* **GPU.** Tower training, SASRec, TIGER and the TED gates expect one CUDA GPU; the paper
  used NVIDIA RTX 4090 cloud instances. One 20k-step tower takes ~0.5-1.2 h on an RTX 4090
  (73.8 min measured for a clean-box retrain of the Beauty s42 headline tower; TED towers
  ~2 h). The look/gate scorers run on CPU (20 min to ~2 h each on the original machines,
  see `runtime_sec` in each JSON; a slow 8-core CPU took 8.7 h for the Beauty look, so a
  big CPU is recommended). **Yelp full-corpus scoring needs >30 GB RAM**: run Yelp
  evaluations sequentially.
* **Versions.** Training/scoring boxes: torch 2.5.1+cu124, transformers 4.39.3. Serving
  benchmarks: Python 3.12.4, torch 2.12.0, faiss 1.13.2 on a 13th-gen Intel i5-1340P (the
  1M ANN benchmark on an Intel i7-13700 server); every benchmark JSON records its env.
  The TIGER baseline runs in LIGER's own environment (`baselines/tiger/TIGER_RUNBOOK.md`).
* **OpenAI key.** Only the embedding scripts read it, only from `$OPENAI_API_KEY`.
  Item metadata (never user data) is sent to `text-embedding-3-large` once per catalog.
  Volume: the paper-critical catalogs (5 datasets x {plain, enriched-real,
  enriched-inner}) are ~217k item texts, ~64M characters, i.e. roughly **16M tokens
  (about $2 at $0.13 per 1M tokens)**. The Beauty development studies add the i22b
  variants (3 x 12,101 texts) and the 10 augmentation-selector catalogs
  (~10 x 12,101 texts, plus the training-free selector probe), roughly another 20M
  tokens. Each catalog embeds in a few minutes (batches of 256 with a pause for rate
  limits).

## 3. Data: obtaining the five datasets

Nothing under `data/` is shipped. Every derived text/JSON file is md5-pinned in
[`docs/data_checksums.md5`](docs/data_checksums.md5); after step 5.2 run
`md5sum -c docs/data_checksums.md5` (lines for files you have not built yet just fail).

| dataset (paper name) | raw source (public) | how it is processed | our files |
|---|---|---|---|
| **Beauty, Toys and Games, Sports and Outdoors** (Amazon-2014 5-core, TIGER-lineage canonical snapshots) | McAuley Amazon-2014 per-category 5-core reviews + metadata on SNAP: `https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{Beauty,Toys_and_Games,Sports_and_Outdoors}_5.json.gz` and `.../meta_{Beauty,Toys_and_Games,Sports_and_Outdoors}.json.gz` | the unmodified LIGER preprocessing (`facebookresearch/liger` @ `b6ccc37`, `ID_generation/preprocessing/data_process.py`: 5-core, chronological, 1-indexed ids, item text = title/price/brand/categories), called by `scripts/prep_amazon2014.py` | `data/amazon_{beauty,toys,sports}_backbone/{Beauty,Toys_and_Games,Sports_and_Outdoors}.txt`, `*_item2attributes.json`, `data/amazon_meta/{beauty,toys,sports}_id2meta.json` |
| **Musical Instruments** (Amazon-2018, LC-Rec released snapshot) | LC-Rec authors' processed release (`RUCAIBox/LC-Rec` @ `e6f1b3b`, README "Dataset" Google Drive folder `https://drive.google.com/drive/folders/1RcJ2M1l5zWPHYuGd9l5Gibcs5w5aI3y6`, file `Instruments.zip`, id `19tsIxV_LbErh5FfYhT732M82y0V-3bxR`, md5 `21a8e3b8...`) | `scripts/i50_prep_instruments_data.py` (gdown download, md5 check, 1-indexing, same item-text template) | `data/amazon_instruments_backbone/Musical_Instruments.txt`, `data/amazon_meta/instruments_id2meta.json` |
| **Yelp** (LETTER in-tree release snapshot = S3Rec/DuoRec/BSARec snapshot) | sequences: `https://raw.githubusercontent.com/yehjin-shin/BSARec/c80bdc0f1a889b78fe9a27e2058389cc84caea84/src/data/Yelp.txt` (byte-identical fallback: `RUCAIBox/CIKM2020-S3Rec` @ `e6c774d`, `data/Yelp.txt`); item text: `https://raw.githubusercontent.com/HonghuiBao2000/LETTER/8d0154e28de37dbb6e24871c508ad8ddb1921cda/data/Yelp/Yelp.item.json` | `scripts/i51_prep_yelp_data.py` (md5-verified copy) + `scripts/build_yelp_meta.py` (LETTER's 0-indexed item file -> 1-indexed id2meta; LETTER's 30,431 sequences equal ours with item id + 1) | `data/yelp_backbone/Yelp.txt`, `data/yelp_meta/LETTER_Yelp.item.json`, `data/yelp_meta/yelp_id2meta.json` |

Dataset statistics (asserted by the prep scripts): Beauty 22,363 users / 12,101 items;
Toys 19,412 / 11,924; Sports 35,598 / 18,357; Instruments 24,772 / 9,922 / 206,153
interactions; Yelp 30,431 / 20,033 / 316,354. On Yelp, 2,027 of 30,431 test targets recur
in the user's own history, which is why Yelp is reported unmasked only.

Licensing: the Amazon review data, the LC-Rec release and the Yelp-derived files carry
their providers' terms (the Yelp Open Dataset license restricts redistribution), which is
why this repository ships download scripts and checksums instead of data.

## 4. The one-look ledger discipline

Every real-test number of ours in the paper is a **single, pre-declared evaluation**:

1. All selection (architecture, blend weights, gates) happens in the **inner world**: the
   `*_inner.txt` file drops each user's last event, so the canonical leave-one-out split on
   it has inner-val = original event -3 and inner **pseudo-test** = original event -2. The
   real test event is never read.
2. Before a real look, a `*-PREREG` entry stating config, bar and consequences was appended
   to `ledger/ledger.jsonl`; the look's result was booked as the matching `*-LOOK` entry;
   the configuration was never scored on test again. Failed gates and failed looks are
   booked as prominently as passes (e.g. `I39-GATE-FAIL`, `I40-LQTED-TOYS-REAL-LOOK`).
3. The six confirmatory baseline comparisons (trio vs. same-protocol TIGER and SASRec on
   Beauty/Toys/Sports) follow the frozen, SHA-stamped plan `ledger/close2_prereg_draft.md`
   (`CLOSE2-PREREG-FREEZE`, fixed after one disclosed retrospective TIGER-Beauty look);
   the matched-ensemble control follows `ledger/E_ENSCTRL.md`.

The look scripts enforce this mechanically: `i24_test_look.py`, `i30_real_look.py`,
`i35_real_look.py` and `i50_real_look_expand.py` write a `results/.<id>_taken` marker and
refuse to run again (the first two also require `I24_TEST_LOOK=1` / `I30_REAL_LOOK=1`);
`close2_analysis.py` refuses to overwrite an existing cell. **We ship no markers**, so a
reproduction can take each look once. To re-run a look whose output is already in
`results/` (e.g. a shipped cell), delete that one output file first. `docs/MANIFEST.md`
maps every shipped result file to the ledger entry that booked it.

## 5. End-to-end pipeline

`scripts/reproduce.sh --dataset <ds> --stages <stages>` runs exactly the commands below, in
this order, idempotently (see Quick start). They are spelled out here so every step can
also be run, inspected or re-run by hand.

Helper scripts added for this release (all others are the original runners, with paths
made repo-relative and an `--embedder` switch on the embedding scripts):
`prep_amazon2014.py`, `make_inner_split.py`, `build_yelp_meta.py` (data),
`embed_open_texts.py` (open-embedder check), `i44_factorial_from_npz.py`,
`check_data.py`, `check_results.py`, `compare_results.py` (verification),
`workspace.py`, `reproduce.sh`, `smoke_e2e.sh`, `make_synthetic_dataset.py`,
`make_synthetic_beams.py` (drivers / smoke tests) and, under `baselines/tiger/`,
`stage_expand_inputs.py` and `run_expand.py`. The mining / verbalization / centroid code
the prep scripts share now lives in `src/semplan/enrich.py` (verified to reproduce the
paper's enriched texts and input tables bit-for-bit), the embedders in
`src/semplan/embedders.py`, and the quick-mode hooks in `src/semplan/quick.py`.

Datasets: `beauty toys sports instruments yelp`. Seeds: 42 43 44. "inner" = inner world
(selection), "real" = real world (towers trained through each user's validation event,
`--include-val`, scored once on the real test event).

### 5.1 Raw data and backbones (CPU, minutes; downloads ~0.5 GB)

```bash
git clone https://github.com/facebookresearch/liger.git third_party/liger
git -C third_party/liger checkout b6ccc37af5ee623ddc1d1ead3490c31aaeaf4524

python scripts/prep_amazon2014.py --dataset beauty toys sports   # SNAP -> LIGER prep -> data/, md5-checked
python scripts/i50_prep_instruments_data.py                       # LC-Rec Instruments (gdown)
python scripts/i51_prep_yelp_data.py                              # S3Rec/BSARec Yelp.txt
python scripts/build_yelp_meta.py                                 # LETTER Yelp item text
```

### 5.2 Inner/outer splits

```bash
python scripts/make_inner_split.py --dataset beauty toys sports instruments yelp
```

(drops each user's last event; md5-checked. The Toys/Sports/Instruments/Yelp prep scripts in
5.3 would also create these with the identical rule and skip them if present.)

### 5.3 Predecessor mining + enrichment + OpenAI embedding + query-input tables

Each script mines directed predecessor counts on `ids[1:][:-1]` (user id and the world's
held-out final event dropped), appends "Often bought after: 'p1'; 'p2'; 'p3'." (Yelp:
"Often visited after"), embeds with `text-embedding-3-large`, and builds the
`[L2(own) ; L2(pred3 centroid)]` input matrix (items without predecessors fall back to
their own embedding). Every script below takes `--embedder openai` (default, the paper),
`--embedder st:<sentence-transformers model>` (open models, different numbers) or
`--embedder fake` (offline, smoke tests only), and skips outputs that already exist. Outputs: `*_openai-3-large_i22{inner,real}_embeddings.pt`
[N, 3072] (item index, E) and `*_openai-3-large_i24{inner,real}_input.pt` [N, 6144] (C).

```bash
# Beauty (three scripts)
python scripts/embed_openai_catalog.py        # plain catalog  Beauty_openai-3-large_embeddings.pt
python scripts/i22_enrich_embed_catalog.py    # E: i22inner / i22real
python scripts/i24_build_input_emb.py         # C: i24inner / i24real (no API)
# Toys, Sports, Instruments, Yelp (one script each: inner split + E + C)
python scripts/i30_prep_toys.py
python scripts/i35_prep_sports.py
python scripts/i50_prep_expand.py --dataset instruments
python scripts/i50_prep_expand.py --dataset yelp
# Plain catalogs + plain-centroid inputs for the 2x2 ablation (p = plain level)
for ds in beauty toys sports instruments yelp; do python scripts/i44_ablation_prep.py --dataset $ds; done
```

### 5.4 Training-free arm inputs (free arm, gates, Yelp blend)

```bash
python scripts/prep_amazon2014.py --dataset beauty --verify-only --t5   # Beauty: LIGER-style StandardScaler t5-xl
python scripts/i30_embed_t5_toys.py
python scripts/i35_embed_t5_sports.py
python scripts/i50_embed_t5_expand.py --dataset instruments
python scripts/i50_embed_t5_expand.py --dataset yelp
```

(`--model <sentence-transformers model>` on the three t5 scripts exists only so the quick
mode can exercise them with a small model; the paper uses sentence-t5-xl.)

The **collaborative** half of the free arm (PPMI over train-only windowed co-occurrence,
window 5, weight 1/dist, truncated SVD d=128) needs no separate step: the gates/looks
build it on first use with `semplan.spaces.build_collab_embeddings` and cache it as
`outputs/artifacts/collab_space_{dataset}_{inner,real}.npz` (Beauty real:
`collab_space_beauty.npz`).

### 5.5 Query towers (GPU; 3 seeds x 2 worlds x 5 datasets = 30 headline towers)

All towers: 2-layer transformer, d=256, 4 heads, FFN 512, max-len 20, 4 query vectors,
full-softmax CE at tau 0.05, AdamW lr 1e-3, batch 256, 20k steps, include-val.

```bash
T="python scripts/train_j5a.py --n-query 4 --steps 20000 --include-val"
D=data/amazon_beauty_backbone; P=$D/Beauty_openai-3-large
for s in 42 43 44; do
  $T --seed $s --n-items 12101 --seq $D/Beauty_inner.txt --item-emb ${P}_i22inner_embeddings.pt \
     --input-emb ${P}_i24inner_input.pt --out outputs/artifacts/i24_inner_openai_k4_s${s}_iv
  $T --seed $s --n-items 12101 --seq $D/Beauty.txt --item-emb ${P}_i22real_embeddings.pt \
     --input-emb ${P}_i24real_input.pt --out outputs/artifacts/i24_real_openai_k4_s${s}_iv
done
```

| dataset | `--n-items` | inner `--seq` / real `--seq` | prefix | inner / real `--out` (append `_s{seed}_iv`) |
|---|---|---|---|---|
| beauty | 12101 | `Beauty_inner.txt` / `Beauty.txt` | `data/amazon_beauty_backbone/Beauty_openai-3-large` | `i24_inner_openai_k4` / `i24_real_openai_k4` |
| toys | 11924 | `Toys_inner.txt` / `Toys_and_Games.txt` | `data/amazon_toys_backbone/Toys_openai-3-large` | `i30_toys_inner_openai_k4` / `i30_toys_real_openai_k4` |
| sports | 18357 | `Sports_inner.txt` / `Sports_and_Outdoors.txt` | `data/amazon_sports_backbone/Sports_openai-3-large` | `i35_sports_inner_openai_k4` / `i35_sports_real_openai_k4` |
| instruments | 9922 | `Instruments_inner.txt` / `Musical_Instruments.txt` | `data/amazon_instruments_backbone/Instruments_openai-3-large` | `i50_instruments_inner_openai_k4` / `i50_instruments_real_openai_k4` |
| yelp | 20033 | `Yelp_inner.txt` / `Yelp.txt` | `data/yelp_backbone/Yelp_openai-3-large` | `i51_yelp_inner_openai_k4` / `i51_yelp_real_openai_k4` |

Inner towers use `{prefix}_i22inner_embeddings.pt` + `{prefix}_i24inner_input.pt`; real towers
`{prefix}_i22real_embeddings.pt` + `{prefix}_i24real_input.pt`. The checkpoint
directory names matter: the gate/look scripts load `outputs/artifacts/<name>_s{seed}_iv/j5.pt`.

**TED towers** (`scripts/i40_train_tricks.py`, same architecture/recipe, soft labels
beta 0.1 / tau 0.1). Without overrides `--dataset {beauty,toys,sports}` trains on that
dataset's *inner* world; real-world and Instruments/Yelp towers pass all four of
`--seq --item-emb --input-emb --n-items` (the `--dataset` label is then ignored):

```bash
for s in 42 43 44; do
  # inner-world TED gates (Beauty: the gate FAILED, so no real Beauty TED towers exist)
  python scripts/i40_train_tricks.py --dataset beauty --seed $s --ted_beta 0.1 --ted_tau 0.1 --out-dir outputs/artifacts/i39_beauty_ted_s$s
  python scripts/i40_train_tricks.py --dataset toys   --seed $s --ted_beta 0.1 --ted_tau 0.1 --out-dir outputs/artifacts/i40_toys_ted_s$s
  python scripts/i40_train_tricks.py --dataset sports --seed $s --ted_beta 0.1 --ted_tau 0.1 --out-dir outputs/artifacts/i40_sports_ted_s$s
  # real-world TED towers (Toys, Sports)
  D=data/amazon_toys_backbone; P=$D/Toys_openai-3-large
  python scripts/i40_train_tricks.py --dataset toys --seed $s --ted_beta 0.1 --ted_tau 0.1 \
    --seq $D/Toys_and_Games.txt --item-emb ${P}_i22real_embeddings.pt --input-emb ${P}_i24real_input.pt \
    --n-items 11924 --out-dir outputs/artifacts/i40_toys_ted_REAL_s$s
  D=data/amazon_sports_backbone; P=$D/Sports_openai-3-large
  python scripts/i40_train_tricks.py --dataset sports --seed $s --ted_beta 0.1 --ted_tau 0.1 \
    --seq $D/Sports_and_Outdoors.txt --item-emb ${P}_i22real_embeddings.pt --input-emb ${P}_i24real_input.pt \
    --n-items 18357 --out-dir outputs/artifacts/i40_sports_ted_REAL_s$s
done
```

Further tower families (2x2 cells, TED transfer/scale, selector sweep) are listed with
the claims they back in section 6.

### 5.6 Inner-world gates (selection; zero test looks)

```bash
python scripts/i24_gate.py                              # -> results/i24_gate.json      (Beauty)
python scripts/i30_gate.py                              # -> results/i30_gate.json      (Toys)
python scripts/i35_gate.py                              # -> results/i35_gate.json      (Sports)
python scripts/i50_gate_expand.py --dataset instruments # -> results/i50_gate_instruments.json
python scripts/i50_gate_expand.py --dataset yelp        # -> results/i51_gate_yelp.json (selects w=0.9)
# TED inner gates (trio mean-z, same scorer as the deployed bar), e.g. Toys:
D=data/amazon_toys_backbone; P=$D/Toys_openai-3-large
python scripts/i40_gate.py --device cuda --seq $D/Toys_inner.txt --n-items 11924 \
  --item-emb ${P}_i22inner_embeddings.pt --input-emb ${P}_i24inner_input.pt \
  --ckpt outputs/artifacts/i40_toys_ted_s{42,43,44}/tower.pt --out results/i40_gate_toys_ted_trio.json
```

### 5.7 The single real looks (one each; see section 4)

```bash
I24_TEST_LOOK=1 python scripts/i24_test_look.py          # Beauty trio      -> results/i24_test_look.json (+ i24_test_per_user.npz)
I30_REAL_LOOK=1 python scripts/i30_real_look.py          # Toys trio        -> results/i30_real_look.json (+ i30_real_per_user.npz)
python scripts/i35_real_look.py                          # Sports trio      -> results/i35_real_look.json
python scripts/close2_sports_replay.py                   # Sports per-user vectors for the paired tests -> results/i35_real_per_user.npz
python scripts/i50_real_look_expand.py --dataset instruments   # -> results/i50_real_look_instruments.json
python scripts/i50_real_look_expand.py --dataset yelp          # -> results/i51_real_look_yelp.json (w=0.9 blend)
# TED trio looks: the i40 scorer fed the REAL files (its "pseudo_test" field is then the
# real test look; ledger CLARIFY-TED-NO-REAL-TEST-FLAG)
D=data/amazon_toys_backbone; P=$D/Toys_openai-3-large
python scripts/i40_gate.py --device cuda --seq $D/Toys_and_Games.txt --n-items 11924 \
  --item-emb ${P}_i22real_embeddings.pt --input-emb ${P}_i24real_input.pt \
  --ckpt outputs/artifacts/i40_toys_ted_REAL_s{42,43,44}/tower.pt --out results/i40_ted_real_look_toys.json
# Sports likewise (Sports_and_Outdoors.txt, 18357, i40_sports_ted_REAL_*) -> results/i40_ted_real_look_sports.json
```

### 5.8 Masked rescore (dense-lineage convention; no training, no selection)

Rescores the same frozen checkpoints with each user's train+val items excluded, after a
hard guard that the unmasked score reproduces the booked look exactly:

```bash
for ds in beauty toys sports toys_ted sports_ted; do python scripts/i43_masked_real.py --dataset $ds; done
```

### 5.9 Baselines

```bash
# SASRec: faithful 2018 recipe (BCE, 1 sampled negative) and the CE-loss variant
for ds in beauty toys sports; do for s in 42 43 44; do
  python scripts/close2_sasrec.py --dataset $ds --seed $s --loss bce --device cuda
  python scripts/close2_sasrec.py --dataset $ds --seed $s --loss ce  --device cuda
done; done
for ds in instruments yelp; do for s in 42 43 44; do
  python scripts/close2_sasrec.py --dataset $ds --seed $s --loss bce --device cuda
done; done
# zero-parameter transition-count heuristic (gamma selected in the inner world; CPU, ~5-10 min)
python scripts/i42_transition_only.py --dataset beauty --n-items 12101 \
  --seq-inner data/amazon_beauty_backbone/Beauty_inner.txt --seq data/amazon_beauty_backbone/Beauty.txt \
  --out results/i42_transition_only_beauty.json      # toys 11924 / sports 18357 likewise
# TIGER (LIGER reference pipeline, GPU, ~3-5 h per dataset/seed): see baselines/tiger/TIGER_RUNBOOK.md
python scripts/close2_tiger_score.py --dataset beauty --beams <beauty_tiger_beams_test.jsonl> --out results/close2_tiger_beauty_s42
```

### 5.10 Paired confirmatory tests (CPU, ~1 min per cell)

```bash
for ds in beauty toys sports; do
  python scripts/close2_analysis.py --dataset $ds --baseline tiger --baseline-npz results/close2_tiger_${ds}_s42_per_user.npz
done
# vs the best-test-seed SASRec (conservative in the baseline's favour): beauty s43, toys s44, sports s42
python scripts/close2_analysis.py --dataset beauty --baseline sasrec --baseline-npz results/close2_sasrec_beauty_s43_per_user.npz
python scripts/close2_analysis.py --dataset toys   --baseline sasrec --baseline-npz results/close2_sasrec_toys_s44_per_user.npz
python scripts/close2_analysis.py --dataset sports --baseline sasrec --baseline-npz results/close2_sasrec_sports_s42_per_user.npz
```

### 5.11 Matched-ensemble control (3 SASRec seeds, per-user z fusion, val-selected simplex weights)

```bash
for ds in beauty toys sports; do
  python scripts/ensctrl_sasrec_fusion.py --dataset $ds --loss ce  --device cuda
  python scripts/ensctrl_sasrec_fusion.py --dataset $ds --loss bce --device cuda
  python scripts/ensctrl_masked_rescore.py --dataset $ds --loss ce
  python scripts/ensctrl_masked_rescore.py --dataset $ds --loss bce
done
python scripts/ensctrl_sasrec_fusion.py --dataset yelp --loss bce --device cuda
python scripts/ensctrl_sasrec_fusion.py --dataset instruments --loss bce --device cuda
# (the shipped Instruments JSON was regenerated from the run's checkpoints under hard
#  guards by scripts/ensctrl_regen_instruments.py; ledger E-ENSCTRL-A-INSTRUMENTS)
```

Each fusion first re-trains the three seeds and asserts that every solo test score equals
the booked `close2_sasrec*` JSON before fusing.

### 5.12 2x2 injection ablation (E = enriched item side, C = centroid inputs)

| cell | item side (`--item-emb`) | `--input-emb` | towers |
|---|---|---|---|
| pp | `{prefix}_embeddings.pt` (plain) | none | new: `i44_{ds}_pp_s{seed}_iv` |
| pC | `{prefix}_embeddings.pt` | `{prefix}_i44plain_input.pt` | new: `i44_{ds}_pc_s{seed}_iv` |
| Ep | `{prefix}_i22real_embeddings.pt` | none | new `i44_{ds}_ep_s{seed}_iv` (Beauty: the existing `i22_real_openai_k4` trio) |
| EC | `{prefix}_i22real_embeddings.pt` | `{prefix}_i24real_input.pt` | the deployed real trio (re-used frozen) |

```bash
# train the new cells on the REAL world (same train_j5a recipe as 5.5), e.g. Toys pC:
for s in 42 43 44; do
  python scripts/train_j5a.py --n-query 4 --steps 20000 --include-val --seed $s --n-items 11924 \
    --seq data/amazon_toys_backbone/Toys_and_Games.txt \
    --item-emb data/amazon_toys_backbone/Toys_openai-3-large_embeddings.pt \
    --input-emb data/amazon_toys_backbone/Toys_openai-3-large_i44plain_input.pt \
    --out outputs/artifacts/i44_toys_pc_s${s}_iv
done
# evaluate each cell once (unmasked + masked + cold slice; POOL=200), e.g.:
python scripts/i44_eval_cell.py --seq data/amazon_toys_backbone/Toys_and_Games.txt --n-items 11924 \
  --item-emb data/amazon_toys_backbone/Toys_openai-3-large_embeddings.pt \
  --input-emb data/amazon_toys_backbone/Toys_openai-3-large_i44plain_input.pt \
  --ckpt outputs/artifacts/i44_toys_pc_s{42,43,44}_iv/j5.pt --tag pC --out results/i44_toys_pC.json
# paired per-user factorial inference (Sports; re-scores the 4 cells under exact-repro guards)
python scripts/i44_factorial_inference.py
python scripts/i44_factorial_from_npz.py     # statistics only, from the shipped per-user npz (CPU, ~1 min)
```

### 5.13 Single towers, seed soup, routing, enrichment variants, selector controls

```bash
# single towers + uniform weight soup (per dataset; Beauty shown)
python scripts/i45_seed_soup.py --seq data/amazon_beauty_backbone/Beauty.txt --n-items 12101 \
  --item-emb data/amazon_beauty_backbone/Beauty_openai-3-large_i22real_embeddings.pt \
  --input-emb data/amazon_beauty_backbone/Beauty_openai-3-large_i24real_input.pt \
  --ckpt outputs/artifacts/i24_real_openai_k4_s{42,43,44}_iv/j5.pt --tag beauty --out results/i45_beauty_seed_soup.json
# learned soft router + seed-disagreement diagnostic (inner world)
python scripts/i38_soft_router.py --dataset beauty instruments yelp
python scripts/i29_adaptive_k.py
# matched augmentation-selector sweep (Beauty inner; 6k-step towers, GPU)
python scripts/simpleaug_prep.py contentnn3 bm25_3 random3 popular3 upred3 translift3 ppmi3 ppr3 pred1 catbrand3
python scripts/simpleaug_box.py none pred3 pred1 ppr3 ppmi3 contentnn3 bm25_3 translift3 upred3 catbrand3 popular3 random3
python scripts/simpleaug_explore.py          # training-free proxy (supplement C)
```

Enrichment variants (§2.1: successors / top-5 / both, all <0.002) and the open-embedder
check (one leading open-source embedding model, +0.0075..+0.0106; set
`TT_OPEN_EMBEDDER` to the Hugging Face id of the sentence-transformers model to use): build the catalogs, train Beauty-inner
towers with `train_j5a.py` (no `--input-emb`), then score each tower with the P26
inner-validation scorer, which appends to `results/p26_inner_validation.json`:

```bash
python scripts/i22b_variant_embeds.py                       # succ / both / pred5 inner catalogs (OpenAI)
python scripts/embed_open_catalog.py                         # plain open-embedder catalog
python scripts/i25_dump_enriched_texts.py && python scripts/embed_open_texts.py \
  --texts data/amazon_beauty_backbone/i25_texts_inner_pred3.json \
  --out data/amazon_beauty_backbone/Beauty_openemb-large_i25_pred3_inner_embeddings.pt
# towers: --seq data/amazon_beauty_backbone/Beauty_inner.txt --item-emb <catalog> --out
#   outputs/artifacts/{i22b_<v>_inner_openai_k4, i25_inner_openemb_{plain,pred3}_k4}_s{seed}_iv
P26_CKPT=outputs/artifacts/i25_inner_openemb_pred3_k4_s42_iv/j5.pt \
P26_ITEM_EMB=data/amazon_beauty_backbone/Beauty_openemb-large_i25_pred3_inner_embeddings.pt \
P26_TAG=i25_openemb_pred3_s42_iv python scripts/p26_inner_validation.py
```

### 5.14 Efficiency

```bash
BENCH_THREADS=1 python scripts/close3_serving_bench.py           # B/T/S: params, latency, memory (needs the Beauty TIGER ckpt at outputs/artifacts/j1/tiger_ckpt_best.pt for its param count)
python scripts/close3_serving_bench_expand.py                    # Instruments / Yelp (Table 5)
BENCH_THREADS=1 python scripts/close3_serving_bench_yelp_blend.py # full Yelp blend
python scripts/i23_ann_replay.py                                 # serving-fidelity ANN replay
python scripts/ann_bench_deployed.py --corpus 1000000 --queries 2000   # Table 6, ~4 h single-thread, ~25 GB RAM
python baselines/tiger/tiger_latency_bench.py --device cuda      # TIGER context row (see runbook)
```

Latency numbers are hardware- and load-dependent; compare ratios, not absolute ms.

## 6. Paper claim -> command -> expected number

All expected values are copied verbatim from the shipped JSONs (key path in backticks).
Section numbers refer to the camera-ready paper.

### Table 1 (Amazon-2014 Beauty / Toys / Sports, HR@5) and §4 text

| claim | expected value (key) | artifact | produced by |
|---|---|---|---|
| Trio, unmasked: 0.0710 / 0.0804 / 0.0401 | `test.1.0.HR@5` = 0.071; `test.1.0.HR@5` = 0.0804; `arms.trio.HR@5` = 0.0401 | `i24_test_look.json`, `i30_real_look.json`, `i35_real_look.json` | 5.5 + 5.7 |
| Trio, masked rescore: 0.0824 / 0.0969 / 0.0464 | `masked.overall.HR@5`; guard `unmasked_reproduction.abs_diff` = 0.0 | `i43_masked_real_{beauty,toys,sports}.json` | 5.8 |
| TED trio unmasked: Toys 0.0838, Sports 0.0431 | `ensemble.pseudo_test.overall.HR@5` (real files) | `i40_ted_real_look_{toys,sports}.json` | 5.5 TED + 5.7 |
| TED trio masked: Toys 0.1009, Sports 0.0496 | `masked.overall.HR@5` | `i43_masked_real_{toys,sports}_ted.json` | 5.8 |
| Beauty TED gate FAIL 0.0797 < 0.0802 | `ensemble.pseudo_test.overall.HR@5` = 0.0797 | `i39_gate_beauty_ted_trio.json` (+ `i39_gate_beauty_ted_gpu.json`) | `i40_gate.py` on Beauty inner + `i39_beauty_ted_s*` towers |
| Second Toys config (longer-query, head-gated) 0.0794 < 0.0804 | `ensemble.pseudo_test.overall.HR@5` = 0.0794 | `i40_lqted_real_look_toys.json`, gate `i40_gate_toys_lqted_trio.json` | `i40_train_tricks.py ... --ted_beta 0.1 --ted_tau 0.1 --longq_L 3 --head_gate softmax` (inner, then real) + `i40_gate.py` |
| Transition heuristic unmasked 0.0617 / 0.0709 / 0.0343, masked 0.0648 / 0.0738 / 0.0353 | `real_test_unmasked.overall.HR@5`, `real_test_masked.overall.HR@5` | `i42_transition_only_{beauty,toys,sports}.json` | 5.9 |
| TIGER same-protocol 0.0453 / 0.0409 / 0.0275 | `overall.HR@5` | `close2_tiger_{beauty,toys,sports}_s42.json` | 5.9 + runbook |
| SASRec best seed (pre-declared) 0.0265 / 0.0347 / 0.0149 | `test.overall.HR@5` (max over seeds) | `close2_sasrec_{ds}_s{42,43,44}.json` | 5.9 |
| SASRec-CE best seed (descriptive) 0.0526 / 0.0587 / 0.0290 | `test.overall.HR@5` (max over seeds) | `close2_sasrec_ce_{ds}_s{42,43,44}.json` | 5.9 |
| SASRec-CE x3 matched fusion 0.0551 / 0.0634 / 0.0316 | `test.overall.HR@5` | `ensctrl_sasrec_ce_{ds}.json` | 5.11 |
| BCE-SASRec fused 0.0328 / 0.0429 / 0.0203; Instruments 0.0717; Yelp 0.0359 | `test.overall.HR@5` | `ensctrl_sasrec_{beauty,toys,sports,instruments,yelp}.json` | 5.11 |
| Control masked: CE 0.0689 / 0.0772 / 0.0379, BCE 0.0422 / 0.0541 / 0.0245 | `masked.overall.HR@5` | `ensctrl_masked_{ce,bce}_{ds}.json` | 5.11 |
| Trio vs TIGER: +0.0257 / +0.0396 / +0.0126, p = 1e-4 | `primary_HR@5.delta`, `primary_HR@5.p_sign_flip` = 9.999000099990002e-05 | `close2_cell_{ds}_tiger.json` | 5.10 |
| Trio vs best-seed SASRec: +0.0445 / +0.0457 / +0.0253, p = 1e-4 | same keys | `close2_cell_{ds}_sasrec.json` | 5.10 |
| Cold slice (unmasked) 0.0062 / 0.0236 / 0.0088; TED cold Toys 0.0372, Sports 0.0146 | `unmasked.cold.HR@5`; TED: cold block of the i43 `_ted` JSONs | `i44_{ds}_EC.json`, `i43_masked_real_{toys,sports}_ted.json` | 5.12, 5.8 |
| Masking worth +0.0114 / +0.0165 / +0.0063; repeat-target share 0.0 | masked minus unmasked; `repeat_target_share` = 0.0 | `i43_masked_real_*.json` | 5.8 |
| Single towers: unmasked 0.0671±.0005 / 0.0747±.0006 / 0.0383±.0004, masked 0.0788 / 0.0894 / 0.0439; uniform soup 0.0339 / 0.0478 / 0.0188 | `summary.{unmasked,masked}.HR@5.single_mean/single_sd`; `results.soup.unmasked.overall.HR@5` | `i45_{beauty,toys,sports}_seed_soup.json` | 5.13 |

### Table 2 (2x2 ablation, Beauty/Toys/Sports) and the §4.1 factorial inference

| claim | expected value (key) | artifact | produced by |
|---|---|---|---|
| pp / pC / Ep / EC, unmasked and masked HR@5 (e.g. Toys EC 0.0804 / 0.0969) | `unmasked.overall.HR@5`, `masked.overall.HR@5` | `i44_{beauty,toys,sports}_{pp,pC,Ep,EC}.json` | 5.12 |
| Sports factorial (n=35,598): E main +0.0044 [+0.0031,+0.0057] p=5e-5; C main +0.0013 [+0.0004,+0.0022] p=0.003; interaction -0.0022 [-0.0040,-0.0005] p=0.011 | `datasets.sports.effects.{E_main,C_main,interaction}.{estimate,ci95,p_signflip}` | `i44_factorial_inference.json` (+ `i44_factorial_per_user.npz`) | 5.12 |
| Beauty/Toys paired inference not recoverable (checkpoints not archived) | `datasets.{beauty,toys}.status` = `aborted_missing_artifacts` | same | disclosed |

### Table 3 (augmentation-selector controls, Beauty inner, masked HR@5)

ppr3 0.0814, pred3 0.0803, pred1 0.0760, ppmi3 0.0751, contentnn3 0.0749, bm25_3 0.0744,
translift3 0.0741, upred3 0.0740, none 0.0647, catbrand3 0.0597, popular3 0.0591,
random3 0.0587: `trio.masked.HR@5` in `results/simpleaug_box_<selector>.json` (5.13).

### §2 method details (development, inner world)

| claim | expected value (key) | artifact | produced by |
|---|---|---|---|
| Verbalization variants move HR@5 by < 0.002, none passes the 3-of-3 gate | pseudo-test HR@5 per seed for `i22b_{succ,both,pred5}_s{42,43,44}_iv` vs `i22_s*_iv` | `p26_inner_validation.json` (+ `i22b_variant_gate.json`, ledger `I22B-KILL`) | 5.13 |
| Open-embedder enrichment lift +0.0075..+0.0106 on 3/3 seeds | `towers.i25_openemb_{plain,pred3}_s*_iv` pseudo-test at the inner-selected w | `p26_inner_validation.json` (ledger `I25-OPEN-STACK-TRANSFER`) | 5.13 |
| n_q = 8 stack did not clear its gate | ledger `J8-*` entries | `ledger/ledger.jsonl` | historical (not re-run here) |

### Table 4 (Instruments / Yelp, HR@5) and Table 5 (their ablations)

| claim | expected value (key) | artifact | produced by |
|---|---|---|---|
| Instruments trio 0.0995 (HR@10 0.1254, N@5 0.0799, N@10 0.0882), free arm 0.0654, Δ +0.0342 | `arms.trio`, `arms.free`, significance block | `i50_real_look_instruments.json` (gate `i50_gate_instruments.json`) | 5.5-5.7 |
| Yelp deployed blend 0.0461 (HR@10 0.0648, N@5 0.034), free arm 0.0366, singles 0.0286 / 0.0284 / 0.0291 | `arms.trio`, `arms.free`, `arms.{42,43,44}` | `i51_real_look_yelp.json` (gate `i51_gate_yelp.json`, w=0.9) | 5.5-5.7 |
| Inner pseudo-test Instruments 0.1109, Yelp blend 0.0397 | gate JSONs | `i50_gate_instruments.json`, `i51_gate_yelp.json` | 5.6 |
| SASRec Instruments 0.0639 / 0.0629 / 0.0637, Yelp 0.0278 / 0.027 / 0.0279 | `test.overall.HR@5` | `close2_sasrec_{instruments,yelp}_s{42,43,44}.json` | 5.9 |
| TIGER repro Instruments 0.0886 / 0.0861 / 0.0901, Yelp 0.0198 / 0.0224 / 0.0218 (supplement A: HR@10/N@5/N@10) | `overall.*` | `close2_tiger_{instruments,yelp}_s{42,43,44}.json` | runbook |
| Ablation Instruments pp/pC/Ep/EC 0.0937 / 0.0963 / 0.0967 / 0.0995, masked 0.0989 / 0.1020 / 0.1046 / 0.1066; Yelp 0.0173 / 0.0200 / 0.0266 | `unmasked.overall.HR@5`, `masked.overall.HR@5` | `i44_instruments_{pp,pC,Ep,EC}.json`, `i44_yelp_{pp,pC,Ep}.json` | 5.12 (prefixes `Instruments_openai-3-large`, `Yelp_openai-3-large`; EC = the `i50_instruments_real` trio) |

### Table 5/6 and §6 efficiency

| claim | expected value (key) | artifact | produced by |
|---|---|---|---|
| Instruments 22.9 ms, Yelp 35.2 ms/query; 388 / 743 MB | `datasets.{instruments,yelp}.end_to_end_batch1_ms.trio_batch1_ms` = 22.8995... / 35.2488...; `datasets.*.memory_mb.total_serving_mb_trio` | `serving_bench_expand.json` | 5.14 |
| B/T/S one tower 84 ms, trio 198 ms (1 thread), 465 MB | `end_to_end_batch1_ms.{single,trio}_batch1_ms` = 84.12... / 197.79... | `serving_bench.json` (+ `serving_bench_multithread.json`) | 5.14 |
| Full Yelp blend 104.4 ms vs trio 84.6 ms; free arm +69 MB | `end_to_end_blend_batch1_ms.median_ms`, `components_batch1_ms.*`, `memory_mb.*` | `serving_bench_yelp_blend.json` | 5.14 |
| TIGER 739 ms/query (co-tenant upper bound; idle rerun 48.9 ms GPU / 130.4 ms CPU) | latency fields | `close2_tiger_latency.json`, `close2_tiger_latency_reconstructed_{gpu,cpu}.json` | runbook §6 |
| Serving-fidelity ANN replay: exact 0.069 = served 0.069 | `exact_reference.HR@5`, `served.HR@5` | `i23_ann_replay.json` | 5.14 |
| 1M items: exact 1291.7 ms; IVF-Flat np16 0.952 @ 161.5 ms; IVF-PQ64 np64 0.959 @ 145.0 ms in 126 MB; HNSW32 ef128 0.955 @ 132.7 ms | `exact_singlethread_batch1.ms_p50`, `rows[*]`, `ivf_pq_mem_MB`, `distractor_calibration` | `ann_bench_deployed_1m.json` | 5.14 |

### Supplement D-F (inner world, zero test looks)

| claim | artifact | produced by |
|---|---|---|
| Router: Instruments always-trio 0.1109 / hard 0.1104 / soft 0.1029 / oracle 0.1269; Yelp soft 0.0393 vs 0.0272 / 0.0309, oracle 0.0509; Beauty hard 0.0787 / soft 0.0711 | `i38_inner.json` (`datasets.*.pseudo_test.*`) | `i38_soft_router.py` |
| Seed-disagreement AUC 0.73 | `i29_adaptive_k.json` | `i29_adaptive_k.py` |
| TED inner gates: Beauty -0.0005, Toys +0.0041, Sports +0.0041, Instruments +0.0005, Yelp +0.0048 | `i39_gate_beauty_ted_trio.json`, `i40_gate_{toys,sports}_ted_trio.json`, `i46_gate_instruments_ted_trio.json`, `i47_gate_yelp_ted_trio.json`; same-scorer anchors `i46_gate_instruments_i50_trio.json`, `i47_gate_yelp_i51_trio.json` | TED towers (`i46ted_instruments_inner_s*`, `i47ted_yelp_inner_s*`: `--seq/--item-emb/--input-emb/--n-items` = the dataset's inner files) + `i40_gate.py` |
| Yelp TED-in-blend 0.0398 vs 0.0397 | `i47_blend_yelp_ted.json` | `i50_gate_expand.py --dataset yelp --ckpt-prefix outputs/artifacts/i47ted_yelp_inner_s --out results/i47_blend_yelp_ted.json` |
| Scale: 13.1M M-tower Beauty 0.0463, Toys 0.0638; with TED 0.0578 / 0.0724; Sports M+TED 0.0313; 59.9M L-tower | `i18_size_score.json`, `i18hard_gate_toys_m.json`, `i18ted_gate_{beauty,toys,sports}_m.json` | M = `--d 512 --layers 4 --heads 8 --d-ff 1024` (`train_j5a.py`, Beauty inner, `i18_size_m_s42_iv`) / `--d 512 --layers 4 --heads 8 --d_ff 1024` (`i40_train_tricks.py`, +TED); L = `--d 1024 --layers 6 --heads 16 --d-ff 2048` (lr 1e-3 and 5e-4); scored with `i18_size_score.py` / `i40_gate.py` |
| MiniLM trunk + TED 0.0778 < 0.0802 | `i48_gate.json` | `i37_train_lmq.py --seed {42,43,44} --ted_beta 0.1 --ted_tau 0.1 --out-dir outputs/artifacts/i48_lmqted_s{seed}` + `i37_gate.py --ckpt-prefix outputs/artifacts/i48_lmqted_s --out results/i48_gate.json` |
| Yelp cold slice: free 0.0424 vs trio 0.0254; injection cells pp 0.0169 / pC 0.0085 / Ep 0.0 | `i51_real_look_yelp.json`, `i44_yelp_*.json` (cold blocks) | 5.7, 5.12 |

## 7. What is and is not bit-reproducible

### Verified on the released tree (CPU only, no GPU, no API calls)

| check | command | outcome |
|---|---|---|
| unit tests | `pytest -q` | 33 passed (+4 real-data tests with `TT_DATA_DIR`: 37 passed) |
| paper numbers == `results/`, paired tests + factorial re-run from the shipped npz | `python scripts/check_results.py` | 254 / 254 checks pass; all six confirmatory cells recomputed **bit-identically** (every delta, CI, p) and every §4.1 estimate/CI/p identical |
| dataset validation (counts, 5-core, inner split, id2meta, embedding shapes, sha256) | `TT_DATA_DIR=... python scripts/check_data.py` | 135 / 135 checks pass on the paper's prepared data |
| Amazon-2014 prep from the SNAP files via the pinned LIGER code | `python scripts/prep_amazon2014.py --dataset toys` | `Toys_and_Games.txt`, id2meta, item2attributes byte-identical |
| Yelp sequences + item text, Beauty inner split | `i51_prep_yelp_data.py`, `build_yelp_meta.py`, `make_inner_split.py` | byte-identical (md5) |
| E/C construction (`semplan.enrich`) | `TT_DATA_DIR=... pytest tests/test_real_data.py` | Beauty enriched texts and the Beauty/Toys input tables identical to the paper's files |
| matched-ensemble masked rescore from the archived SASRec checkpoints | `ensctrl_masked_rescore.py --dataset beauty --loss ce` / `--dataset toys --loss bce` | unmasked guard passes; JSONs identical to `results/` (masked 0.0689, 0.0541) |
| transition-count heuristic | `i42_transition_only.py` | selected gamma identical everywhere; Beauty 0.0617 / 0.0649, Toys 0.0708 / 0.0740, Sports 0.0343 / 0.0353 (unmasked / masked) vs paper 0.0617 / 0.0648, 0.0709 / 0.0738, 0.0343 / 0.0353: the heuristic has exact score ties whose order depends on the numpy version (+-0.0002) |
| every pipeline stage on 200 real users per dataset | `make quick` | one clean run, 280 / 280 steps PASS (Beauty 82, Toys 53, Sports 54, Instruments 42, Yelp 43, cross-dataset 6) |
| every pipeline stage on synthetic data with the fake embedder | `make smoke` | 82 / 82 steps PASS (+1 skipped: t5 matrix pre-generated) |

### Expected sources of deviation when rebuilding from scratch

* **OpenAI embeddings are not bit-deterministic** across calls/time, and the endpoint may
  change; the paper's vectors were queried in August 2026 and archived, but are not
  redistributed. A re-embedded catalog gives close, not identical, numbers.
* **Tower retraining is bit-identical only on the same GPU class and software stack**:
  clean-box RTX 4090 retrains of a headline tower (`train_j5a.py`) and a TED tower
  (`i40_train_tricks.py`) matched the shipped checkpoints with max abs weight difference
  0.0 (ledger `REPRO-RETRAIN-VARIANCE`, `REPRO-TEDVAR`) under torch 2.5.1+cu124 /
  transformers 4.39.3; a torch/transformers change alone moved an LM-trunk tower by ~0.002
  HR@5 (`I48-CROSSENV-CHECK`). SASRec retrains on CPU are deterministic (the matched-ensemble
  scripts assert it).
* The free arm's collaborative space uses scipy's truncated SVD; it is rebuilt on first use.
* Latencies depend on hardware and load (all benchmark JSONs record machine, threads and
  load average).

### Not reproducible from this repository alone (disclosed)

* **TIGER Instruments/Yelp driver.** The paper's runs used an on-box copy of LIGER's
  `run.py` with preprocessing replaced by existence checks, plus a staging step; neither
  was archived. `baselines/tiger/run_expand.py` and `stage_expand_inputs.py` reconstruct
  them from the recorded recipe (unmodified `run.main`, StandardScaler t5-xl as a CUDA
  tensor) but have not been re-run end to end. The TIGER training itself was not re-run for
  this release; our scorer is exercised on synthetic beams in `make quick`/`make smoke`.
* **The I37 plain MiniLM-trunk gate value (0.0756, supplement E)** has no archived result
  JSON (only the TED variant `i48_gate.json` does); `i37_gate.py` regenerates it from
  `i37_train_lmq.py` towers.
* **The open-embedder pred3 catalog** (`Beauty_openemb-large_i25_pred3_inner_embeddings.pt`) was embedded
  ad hoc on a GPU box; `scripts/embed_open_texts.py` reconstructs that step with the same
  encoder settings as `embed_open_catalog.py` (not exercised in `make quick`: it downloads
  a 1.3 GB model).
* **Beauty/Toys 2x2 per-user inference** (the checkpoints were never archived; disclosed in
  the paper).
* `scripts/ensctrl_regen_instruments.py` regenerates one booked JSON from its run log and
  only makes sense against the original checkpoints; it is not part of the pipeline.

## 8. Citation and license

```bibtex
@inproceedings{magnani2027tedtrio,
  title     = {{TED-Trio}: Injecting Directed Transition Signal into Frozen Text-Embedding
               Retrievers for Sequential Recommendation},
  author    = {Magnani, Alessandro and Xie, Min and Tsai, Ming-Hen and Jiang, Jyun-Yu},
  booktitle = {Proceedings of the 20th ACM International Conference on Web Search and
               Data Mining (WSDM '27)},
  year      = {2027},
  month     = feb,
  address   = {Hong Kong},
  publisher = {ACM},
  doi       = {10.1145/3862322.3863777},
  url       = {https://doi.org/10.1145/3862322.3863777},
}
```

Code: MIT (see `LICENSE`). `results/` and `ledger/` are released under the same terms.
Third-party code is not vendored: LIGER (CC BY-NC 4.0) is cloned by the user for the TIGER
baseline and data preparation. Datasets remain under their providers' terms.
