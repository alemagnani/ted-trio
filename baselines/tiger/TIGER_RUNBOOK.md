# TIGER baseline runbook (same-protocol re-runs, all five datasets)

Our TIGER rows (paper Table 1 "TIGER (ours, same-protocol)" and Table 3 "TIGER (our
repro)") are trained with the **unmodified** LIGER reference pipeline
(facebookresearch/liger), which contains the TIGER implementation. LIGER is licensed
CC BY-NC 4.0, so it is **not vendored** here: clone it at the pinned commit.

```bash
git clone https://github.com/facebookresearch/liger.git third_party/liger
git -C third_party/liger checkout b6ccc37af5ee623ddc1d1ead3490c31aaeaf4524
```

Everything in this directory is ours (MIT):

| file | role |
|---|---|
| `j1_dump.py` | runs inside the LIGER clone after training; dumps test-split TIGER beams (BeamRecord JSONL) |
| `stage_expand_inputs.py` | Instruments/Yelp only: stages our backbone files into `processed/` (reconstruction, see below) |
| `run_expand.py` | Instruments/Yelp only: LIGER `run.py` (executed unchanged as `__main__`) with `preprocessing()` replaced by existence checks (reconstruction) |
| `tiger_latency_bench.py` | batch-1 beam-20 decode latency (Table 5 context row; reconstruction validated against the booked artifact) |

Scoring always happens in this repo, with `scripts/close2_tiger_score.py`, through the same
`semplan.eval.protocol` code that scores our models.

## 0. Dataset constants

| dataset | LIGER `dataset.name` | LIGER dir (`{type}_{name}`) | n_items | test users | seeds | experiment_id |
|---|---|---|---|---|---|---|
| beauty | `Beauty` | `Amazon_Beauty` | 12101 | 22363 | 42 | `j1_tiger` |
| toys | `Toys_and_Games` | `Amazon_Toys_and_Games` | 11924 | 19412 | 42 | `close2_tiger` |
| sports | `Sports_and_Outdoors` | `Amazon_Sports_and_Outdoors` | 18357 | 35598 | 42 | `close2_tiger` |
| instruments | `Musical_Instruments` | `Amazon_Musical_Instruments` | 9922 | 24772 | 42, 43, 44 | `close2_tiger` |
| yelp | `Yelp` | `Amazon_Yelp` | 20033 | 30431 | 42, 43, 44 | `close2_tiger` |

Content model everywhere: **`sentence-t5-xl`** (not the LIGER default `sentence-t5-xxl`),
so all TIGER rows share one SID family. The `experiment_id` values are what `j1_dump.py`
expects when it looks for checkpoints (`results/tiger/{dir}/{experiment_id}_seed_{seed}`).

## 1. Environment (GPU box)

LIGER's own environment (`third_party/liger/env.yml`), i.e. Python 3.10/3.11 with

```bash
pip install wandb wget sentence-transformers==2.7.0 omegaconf==2.3.0 \
    transformers==4.39.3 hydra-core==1.3.2 scikit-learn
# torch: the paper's TIGER boxes ran torch 2.6.0+cu126 (training) and 2.5.1+cu124 (latency)
export WANDB_MODE=offline HYDRA_FULL_ERROR=1
```

## 2. Beauty / Toys / Sports (Amazon-2014)

LIGER's `run.py` runs its own preprocessing (`ID_generation/preprocessing/data_process.py`:
downloads the SNAP 5-core reviews + metadata and builds `processed/{Name}.txt`,
`{Name}_title_price_brand_categories_amazon_id2meta.json`, `{Name}_item2attributes.json`),
embeds items with sentence-t5-xl (StandardScaler), fits the RQ-VAE semantic IDs
(codebook 256 x 3 + collision level = 4 tokens) and trains TIGER (T5, d_model 128,
6+6 layers, batch 256, 200k steps, best-val checkpoint).

```bash
cd third_party/liger
python run.py dataset=amazon dataset.name=Beauty dataset.content_model=sentence-t5-xl \
  seed=42 device_id=0 method=base test_method=tiger experiment_id=j1_tiger logging.mode=disabled
python run.py dataset=amazon dataset.name=Toys_and_Games dataset.content_model=sentence-t5-xl \
  seed=42 device_id=0 method=base test_method=tiger experiment_id=close2_tiger logging.mode=disabled
python run.py dataset=amazon dataset.name=Sports_and_Outdoors dataset.content_model=sentence-t5-xl \
  seed=42 device_id=0 method=base test_method=tiger experiment_id=close2_tiger logging.mode=disabled
```

**Hard guard (protocol alignment).** The scorer pairs TIGER outputs with our split user by
user, so LIGER's `processed/{Name}.txt` must be byte-identical to our backbone file:

```bash
md5sum ID_generation/preprocessing/processed/{Beauty,Toys_and_Games,Sports_and_Outdoors}.txt
# must equal the values in docs/data_checksums.md5 (cb036353..., b139ecac..., 9691c051...)
```

(`scripts/prep_amazon2014.py` builds our backbones by calling this same LIGER preprocessing,
so the two always agree when the SNAP files are unchanged.)

## 3. Instruments / Yelp (seeds 42, 43, 44)

LIGER's preprocessing only knows Amazon-2014 Beauty/Toys/Sports. For the two transfer
datasets the paper's runs pre-seeded `processed/` with our backbone files and replaced
`preprocessing()` by existence checks. The on-box driver used for that was not archived;
`stage_expand_inputs.py` + `run_expand.py` reconstruct it without editing vendor code
(LIGER's `run.py` is executed unchanged). Key detail recorded at the time: the seeded
t5-xl matrix must be **StandardScaler-transformed** (as LIGER does for its own datasets)
and stored as a **CUDA tensor**.

```bash
# from this repo's root (needs data/ prepared, see README steps 1-4)
python baselines/tiger/stage_expand_inputs.py --dataset instruments yelp
cp baselines/tiger/run_expand.py third_party/liger/
cd third_party/liger
for seed in 42 43 44; do
  for name in Musical_Instruments Yelp; do
    python run_expand.py dataset=amazon dataset.name=$name dataset.content_model=sentence-t5-xl \
      seed=$seed device_id=0 method=base test_method=tiger experiment_id=close2_tiger \
      logging.mode=disabled
  done
done
```

## 4. Dump test-split beams (inside the LIGER clone)

```bash
cp baselines/tiger/j1_dump.py third_party/liger/ && cd third_party/liger
J1_DUMP_DIR=dumps/s42 python j1_dump.py --dataset beauty      --which tiger --split test
J1_DUMP_DIR=dumps/s42 python j1_dump.py --dataset toys        --which tiger --split test
J1_DUMP_DIR=dumps/s42 python j1_dump.py --dataset sports      --which tiger --split test
for s in 42 43 44; do
  J1_DUMP_DIR=dumps/s$s python j1_dump.py --dataset instruments --which tiger --split test --seed $s
  J1_DUMP_DIR=dumps/s$s python j1_dump.py --dataset yelp        --which tiger --split test --seed $s
done
```

Beams: 20 unconstrained beams, `max_new_tokens=4`, per-token log-probs via
`compute_transition_scores` (transformers 4.39.3). Invalid semantic-ID sequences are
written with `valid: false` and skipped by the scorer (Beauty 0.11% of beams). Eval order
is seen test users first, then cold users; the scorer re-derives that order from
`protocol.build` and asserts each row's `target_item`. The `--seed` flag is a release
addition: the paper's seed-43/44 dumps used per-seed copies of this script in which exactly
the SID-pickle and checkpoint-directory seed constants were changed (the `seed` field of
`results/close2_tiger_{instruments,yelp}_s4{3,4}.json` was corrected after scoring; see the
`seed_note` inside those JSONs).

## 5. Score (this repo, CPU, minutes)

```bash
python scripts/close2_tiger_score.py --dataset beauty \
  --beams third_party/liger/dumps/s42/beauty_tiger_beams_test.jsonl \
  --out results/close2_tiger_beauty_s42
# toys / sports likewise (s42); instruments / yelp for --seed 42 43 44:
python scripts/close2_tiger_score.py --dataset instruments --seed 43 \
  --beams third_party/liger/dumps/s43/instruments_tiger_beams_test.jsonl \
  --out results/close2_tiger_instruments_s43
```

Writes `{out}.json` (HR/NDCG@{5,10}, overall/seen/cold) and `{out}_per_user.npz`
(`hit5, dcg5, hit10, dcg10, cold_mask` in `split.test` order), directly pairable with the
trio per-user vectors for the sign-flip tests (`scripts/close2_analysis.py`).

Expected (verbatim from `results/`): Beauty 0.0453 (published 0.0454), Toys 0.0409
(published 0.0521; under-reproduced, disclosed), Sports 0.0275 (published 0.0264);
Instruments 0.0886 / 0.0861 / 0.0901; Yelp 0.0198 / 0.0224 / 0.0218.

## 6. Latency (Table 5 context row)

```bash
python baselines/tiger/tiger_latency_bench.py --device cuda --n-queries 200 \
  --ckpt <.../results/tiger/Amazon_Beauty/j1_tiger_seed_42/results/ckpt_best.pt> \
  --sids third_party/liger/ID_generation/ID/Beauty_sentence-t5-xl_42.pkl \
  --seq data/amazon_beauty_backbone/Beauty.txt \
  --id2meta third_party/liger/ID_generation/preprocessing/processed/Beauty_title_price_brand_categories_amazon_id2meta.json
```

The paper's 739 ms/query (GPU) / 1.7 s (CPU) were measured on a shared, co-tenant box and
are upper bounds (`results/close2_tiger_latency.json`, co-tenancy disclosed inside); the
validated idle-RTX-4090 reruns of the same decode path are 48.9 ms GPU / 130.4 ms CPU
single-thread (`results/close2_tiger_latency_reconstructed_{gpu,cpu}.json`). Quote both.

## Compute

RQ-VAE ~0.5 h per dataset/seed; TIGER 200k steps ~2-3 h (Beauty/Toys/Instruments) to
~4-5 h (Sports/Yelp) on one RTX 4090; beam dump ~0.3-0.5 h. Beauty's original J1 job
(LIGER + TIGER + dumps) took ~26 GPU-h; the Instruments/Yelp 3-seed sweep ~20 GPU-h.
