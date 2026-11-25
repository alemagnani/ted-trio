#!/usr/bin/env bash
# TED-Trio end-to-end driver.
#
#   scripts/reproduce.sh [--quick] [--dataset beauty|toys|sports|instruments|yelp|all ...]
#                        [--stages LIST] [--users N] [--ws DIR] [--data-src DIR]
#                        [--embedder openai|fake|st:MODEL] [--device cuda|cpu] [--keep-going]
#                        [--resume]   (quick mode: keep the workspace, skip finished steps)
#
# FULL mode (default): the real pipeline, run inside a workspace (default outputs/full)
#   whose data/ is a link to the dataset directory (--data-src, $TT_DATA_DIR, default data/)
#   and whose results/ is fresh, so the shipped results/ stay untouched for comparison
#   (scripts/compare_results.py <ws>/results results). Every step is idempotent: it is
#   skipped when its output already exists in the workspace. Stops at the first failure.
#
# QUICK mode (--quick): the SAME scripts on the first --users users (default 200) of every
#   sequence file, real item embeddings (read-only links to --data-src), 20 training steps,
#   1-2 SASRec epochs, a 20k-item ANN corpus; booked-number guards only warn (TT_QUICK=1).
#   Embedding/prep scripts run in a separate "prep" workspace with --embedder fake. Each
#   dataset gets outputs/quick/<dataset>; cross-dataset steps run in outputs/quick/_all.
#   A per-step PASS/FAIL table is written to <ws>/STATUS-<datasets>.tsv. CPU only.
#
# Stages (default: all, in this order):
#   data      raw download / backbone prep, inner split, dataset validation
#   embed     enrichment + item embedding (OpenAI) + query-input tables + t5 free-arm side
#   towers    headline inner + real trios (and TED towers on Toys/Sports)
#   gate      inner-world gates
#   look      the single real looks, Sports per-user replay, masked rescore
#   baselines SASRec (BCE, CE), transition-count heuristic, TIGER scoring
#   tests     paired confirmatory tests, matched-ensemble control (+ masked)
#   ablation  2x2 injection cells (train + eval)
#   extras    seed soup, router, TED transfer gates, selector controls, i22b/openemb, scale probes
#   cross     factorial inference, serving benchmarks, ANN replay, 1M ANN benchmark
#
# GPU time per dataset (one RTX 4090; CPU-only scorers in brackets):
#   towers    6 towers x ~0.5-1.2 h (+6 TED towers x ~2 h on Toys/Sports)
#   gate/look [0.5-2 h each on a 16-core CPU]; masked rescore [1-11 h CPU]
#   baselines SASRec 6 runs x 5-75 min; TIGER see baselines/tiger/TIGER_RUNBOOK.md (3-5 h/seed)
#   ablation  6-9 towers x ~0.5-1.2 h + 3-4 cell evals [~1 h CPU each]
#   extras    Beauty selector sweep 30 short towers (~5 min each), TED gates 15 towers x ~2 h
#   cross     factorial [~4.5 h CPU], 1M ANN bench [~4 h CPU, ~25 GB RAM]
#
# "Reproduce the Beauty headline from existing item embeddings" (data/ already holds
#  Beauty.txt, Beauty_inner.txt, the i22/i24 embedding+input files and the t5 matrix):
#   scripts/reproduce.sh --dataset beauty --stages towers,gate,look
set -uo pipefail

# Run from a private copy so that editing this file during a long run cannot corrupt it.
if [[ -z "${TT_REPRO_REEXEC:-}" ]]; then
  _repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  _copy="$(mktemp "${TMPDIR:-/tmp}/reproduce.XXXXXX.sh")"
  cp "${BASH_SOURCE[0]}" "$_copy"
  TT_REPRO_REEXEC=1 TT_REPO="$_repo" exec bash "$_copy" "$@"
fi
REPO="$TT_REPO"; trap 'rm -f "$0"' EXIT
PY="${PYTHON:-python}"
MODE=full; DATASETS=(); STAGES=""; USERS=200; WS_BASE=""; KEEP_GOING=0; RESUME=0
DATA_SRC="${TT_DATA_DIR:-$REPO/data}"; EMBEDDER=openai; DEVICE=""
ALL_DS=(beauty toys sports instruments yelp)
ALL_STAGES="data embed towers gate look baselines tests ablation extras cross"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE=quick; shift ;;
    --dataset) shift; while [[ $# -gt 0 && "$1" != --* ]]; do
                 if [[ "$1" == all ]]; then DATASETS+=("${ALL_DS[@]}"); else DATASETS+=("$1"); fi; shift; done ;;
    --stages) STAGES="${2//,/ }"; shift 2 ;;
    --users) USERS="$2"; shift 2 ;;
    --ws) WS_BASE="$2"; shift 2 ;;
    --data-src) DATA_SRC="$2"; shift 2 ;;
    --embedder) EMBEDDER="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --keep-going) KEEP_GOING=1; shift ;;
    --resume) RESUME=1; shift ;;
    -h|--help) sed -n '2,44p' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=("${ALL_DS[@]}")
[[ -z "$STAGES" ]] && STAGES="$ALL_STAGES"
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

if [[ $MODE == quick ]]; then
  WS_BASE="${WS_BASE:-$REPO/outputs/quick}"; KEEP_GOING=1
  export TT_QUICK=1 TT_MAX_STEPS=20 TT_MAX_EPOCHS=2
  # be a good neighbour on shared machines; override by exporting these yourself
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
  STEPS=20; SEEDS="42 43 44"; EMBEDDER=fake; DEVICE="${DEVICE:-cpu}"
  ANN_ARGS="--corpus 20000 --queries 20"; LM_STEPS=5
  T5_ARGS="--model sentence-transformers/all-MiniLM-L6-v2"   # small stand-in, prep workspace only
else
  WS_BASE="${WS_BASE:-$REPO/outputs/full}"
  STEPS=20000; SEEDS="42 43 44"; DEVICE="${DEVICE:-cuda}"
  ANN_ARGS="--corpus 1000000 --queries 2000"; LM_STEPS=20000; T5_ARGS=""
fi
mkdir -p "$WS_BASE"; WS_BASE="$(cd "$WS_BASE" && pwd)"   # absolute (links between workspaces)
RUN_ID="$(IFS=-; echo "${DATASETS[*]}")"; [[ $RESUME -eq 1 ]] && RUN_ID="$RUN_ID-resume-$(date +%H%M%S)"
STATUS="$WS_BASE/STATUS-$RUN_ID.tsv"; : > "$STATUS.tmp"

# ------------------------------------------------------------------ dataset table
declare -A NI BB SEQ INNER PFX TOW META
NI=([beauty]=12101 [toys]=11924 [sports]=18357 [instruments]=9922 [yelp]=20033)
BB=([beauty]=amazon_beauty_backbone [toys]=amazon_toys_backbone [sports]=amazon_sports_backbone
    [instruments]=amazon_instruments_backbone [yelp]=yelp_backbone)
SEQ=([beauty]=Beauty.txt [toys]=Toys_and_Games.txt [sports]=Sports_and_Outdoors.txt
     [instruments]=Musical_Instruments.txt [yelp]=Yelp.txt)
INNER=([beauty]=Beauty_inner.txt [toys]=Toys_inner.txt [sports]=Sports_inner.txt
       [instruments]=Instruments_inner.txt [yelp]=Yelp_inner.txt)
PFX=([beauty]=Beauty [toys]=Toys [sports]=Sports [instruments]=Instruments [yelp]=Yelp)
TOW=([beauty]=i24 [toys]=i30_toys [sports]=i35_sports [instruments]=i50_instruments [yelp]=i51_yelp)

# ------------------------------------------------------------------ step runner
WS=""; DS=""
run() {  # run <step-name> <output-that-marks-done|-> <command...>
  local name="$1" creates="$2"; shift 2
  local log="$WS/logs/$name.log"; mkdir -p "$WS/logs"
  if [[ ( $MODE == full || $RESUME -eq 1 ) && "$creates" != - && -e "$WS/$creates" ]]; then
    printf '%s\t%s\tSKIP\t(exists: %s)\n' "${DS:-all}" "$name" "$creates" | tee -a "$STATUS.tmp"; return 0
  fi
  local t0=$SECONDS
  ( cd "$WS" && "$@" ) > "$log" 2>&1
  local rc=$? dt=$((SECONDS - t0))
  if [[ $rc -eq 0 ]]; then
    printf '%s\t%s\tPASS\t%ss\n' "${DS:-all}" "$name" "$dt" | tee -a "$STATUS.tmp"
  else
    printf '%s\t%s\tFAIL(rc=%s)\t%ss\t%s\n' "${DS:-all}" "$name" "$rc" "$dt" "$log" | tee -a "$STATUS.tmp"
    tail -5 "$log" | sed 's/^/    | /'
    [[ $KEEP_GOING -eq 1 ]] || { cp "$STATUS.tmp" "$STATUS"; exit 1; }
  fi
}

train() {  # train <outdir> <seq> <item-emb> [input-emb] [extra args...]
  local out="$1" seq="$2" item="$3" inp="${4:-}"; shift 4 || shift $#
  local ia=(); [[ -n "$inp" && "$inp" != none ]] && ia=(--input-emb "$inp")
  run "train_$(basename "$out")" "$out/j5.pt" $PY scripts/train_j5a.py --n-query 4 --steps $STEPS \
      --include-val --device "$DEVICE" --seed "$(sed -E 's/.*_s([0-9]+)(_iv)?$/\1/' <<<"$out")" --n-items "${NI[$DS]}" --seq "$seq" \
      --item-emb "$item" "${ia[@]}" --out "$out" "$@"
}

best_seed() {  # best test HR@5 seed among close2_sasrec[_ce]_<ds>_s*.json in the workspace
  ( cd "$WS" && $PY - "$1" <<'EOF'
import json, sys
tag = sys.argv[1]
best = max((json.load(open(f"results/{tag}_s{s}.json"))["test"]["overall"]["HR@5"], s)
           for s in (42, 43, 44))
print(best[1])
EOF
  )
}

# ------------------------------------------------------------------ per-dataset chain
dataset_chain() {
  DS="$1"
  local D="data/${BB[$DS]}" P="data/${BB[$DS]}/${PFX[$DS]}_openai-3-large" T="${TOW[$DS]}"
  local N="${NI[$DS]}" S="${SEQ[$DS]}" I="${INNER[$DS]}"
  if [[ $MODE == quick ]]; then
    WS="$WS_BASE/$DS"; [[ $RESUME -eq 1 ]] || rm -rf "$WS"
    $PY "$REPO/scripts/workspace.py" --mode quick --dataset "$DS" --users "$USERS" \
        --data-src "$DATA_SRC" --ws "$WS" > /dev/null || exit 1
  else
    WS="$WS_BASE"
    [[ -d "$WS/scripts" ]] || $PY "$REPO/scripts/workspace.py" --mode full --data-src "$DATA_SRC" --ws "$WS" > /dev/null
  fi

  # ---------------- data
  if has_stage data; then
    if [[ $MODE == full ]]; then
      case $DS in
        beauty|toys|sports) run prep_amazon2014 "$D/$S" $PY scripts/prep_amazon2014.py --dataset $DS ;;
        instruments) run prep_instruments "$D/$S" $PY scripts/i50_prep_instruments_data.py ;;
        yelp) run prep_yelp "$D/$S" $PY scripts/i51_prep_yelp_data.py
              run yelp_meta data/yelp_meta/yelp_id2meta.json $PY scripts/build_yelp_meta.py ;;
      esac
      run inner_split "$D/$I" $PY scripts/make_inner_split.py --dataset $DS
      run check_data - $PY scripts/check_data.py --dataset $DS --no-embeddings
    else
      run check_data - $PY scripts/check_data.py --dataset $DS --data-dir "$DATA_SRC"
    fi
  fi

  # ---------------- embed (quick: separate prep workspace, fake embedder, never the real files)
  if has_stage embed; then
    local SAVE_WS="$WS"
    if [[ $MODE == quick ]]; then
      WS="$WS_BASE/_prep_$DS"; rm -rf "$WS"
      $PY "$REPO/scripts/workspace.py" --mode prep --dataset "$DS" --users "$USERS" \
          --data-src "$DATA_SRC" --ws "$WS" > /dev/null || exit 1
      case $DS in
        beauty|toys|sports) run prep_amazon2014_verify - $PY scripts/prep_amazon2014.py --dataset $DS --verify-only ;;
        instruments) run prep_instruments_verify - $PY scripts/i50_prep_instruments_data.py ;;
        yelp) run prep_yelp_verify - $PY scripts/i51_prep_yelp_data.py
              run yelp_meta - $PY scripts/build_yelp_meta.py ;;
      esac
      run inner_split - $PY scripts/make_inner_split.py --dataset $DS
    fi
    case $DS in
      beauty)
        run embed_plain "${P}_embeddings.pt" $PY scripts/embed_openai_catalog.py --embedder "$EMBEDDER"
        run enrich_E "${P}_i22real_embeddings.pt" $PY scripts/i22_enrich_embed_catalog.py --embedder "$EMBEDDER"
        run centroid_C "${P}_i24real_input.pt" $PY scripts/i24_build_input_emb.py
        [[ $MODE == full ]] && run t5_free_arm "$D/Beauty_sentence-t5-xl_embeddings.pt" $PY scripts/prep_amazon2014.py --dataset beauty --verify-only --t5 ;;
      toys)   run prep_E_C "${P}_i24real_input.pt" $PY scripts/i30_prep_toys.py --embedder "$EMBEDDER"
              run t5_free_arm "$D/Toys_sentence-t5-xl_embeddings.pt" $PY scripts/i30_embed_t5_toys.py $T5_ARGS ;;
      sports) run prep_E_C "${P}_i24real_input.pt" $PY scripts/i35_prep_sports.py --embedder "$EMBEDDER"
              run t5_free_arm "$D/Sports_sentence-t5-xl_embeddings.pt" $PY scripts/i35_embed_t5_sports.py $T5_ARGS ;;
      instruments|yelp)
              run prep_E_C "${P}_i24real_input.pt" $PY scripts/i50_prep_expand.py --dataset $DS --embedder "$EMBEDDER"
              run t5_free_arm "$D/${PFX[$DS]}_sentence-t5-xl_embeddings.pt" $PY scripts/i50_embed_t5_expand.py --dataset $DS $T5_ARGS ;;
    esac
    run plain_p_inputs "${P}_i44plain_input.pt" $PY scripts/i44_ablation_prep.py --dataset $DS --embedder "$EMBEDDER"
    WS="$SAVE_WS"
  fi

  # ---------------- towers
  if has_stage towers; then
    for s in $SEEDS; do
      train "outputs/artifacts/${T}_inner_openai_k4_s${s}_iv" "$D/$I" "${P}_i22inner_embeddings.pt" "${P}_i24inner_input.pt"
      train "outputs/artifacts/${T}_real_openai_k4_s${s}_iv" "$D/$S" "${P}_i22real_embeddings.pt" "${P}_i24real_input.pt"
    done
    if [[ $DS == toys || $DS == sports ]]; then
      for s in $SEEDS; do
        run "ted_inner_s$s" "outputs/artifacts/i40_${DS}_ted_s$s/tower.pt" $PY scripts/i40_train_tricks.py --dataset $DS \
            --seed $s --steps $STEPS --device "$DEVICE" --ted_beta 0.1 --ted_tau 0.1 --out-dir "outputs/artifacts/i40_${DS}_ted_s$s"
        run "ted_real_s$s" "outputs/artifacts/i40_${DS}_ted_REAL_s$s/tower.pt" $PY scripts/i40_train_tricks.py --dataset $DS \
            --seed $s --steps $STEPS --device "$DEVICE" --ted_beta 0.1 --ted_tau 0.1 --seq "$D/$S" \
            --item-emb "${P}_i22real_embeddings.pt" --input-emb "${P}_i24real_input.pt" --n-items $N \
            --out-dir "outputs/artifacts/i40_${DS}_ted_REAL_s$s"
      done
    fi
  fi

  # ---------------- gates
  if has_stage gate; then
    case $DS in
      beauty) run gate results/i24_gate.json $PY scripts/i24_gate.py ;;
      toys)   run gate results/i30_gate.json $PY scripts/i30_gate.py ;;
      sports) run gate results/i35_gate.json $PY scripts/i35_gate.py ;;
      instruments) run gate results/i50_gate_instruments.json $PY scripts/i50_gate_expand.py --dataset instruments ;;
      yelp)   run gate results/i51_gate_yelp.json $PY scripts/i50_gate_expand.py --dataset yelp ;;
    esac
    if [[ $DS == toys || $DS == sports ]]; then
      run ted_gate "results/i40_gate_${DS}_ted_trio.json" $PY scripts/i40_gate.py --device "$DEVICE" --seq "$D/$I" --n-items $N \
          --item-emb "${P}_i22inner_embeddings.pt" --input-emb "${P}_i24inner_input.pt" \
          --ckpt outputs/artifacts/i40_${DS}_ted_s{42,43,44}/tower.pt --out "results/i40_gate_${DS}_ted_trio.json"
    fi
  fi

  # ---------------- the single real looks + masked rescore
  if has_stage look; then
    case $DS in
      beauty) run look results/i24_test_look.json env I24_TEST_LOOK=1 $PY scripts/i24_test_look.py ;;
      toys)   run look results/i30_real_look.json env I30_REAL_LOOK=1 $PY scripts/i30_real_look.py ;;
      sports) run look results/i35_real_look.json $PY scripts/i35_real_look.py
              run sports_replay results/i35_real_per_user.npz $PY scripts/close2_sports_replay.py ;;
      instruments) run look results/i50_real_look_instruments.json $PY scripts/i50_real_look_expand.py --dataset instruments ;;
      yelp)   run look results/i51_real_look_yelp.json $PY scripts/i50_real_look_expand.py --dataset yelp ;;
    esac
    if [[ $DS == toys || $DS == sports ]]; then
      run ted_look "results/i40_ted_real_look_${DS}.json" $PY scripts/i40_gate.py --device "$DEVICE" --seq "$D/$S" --n-items $N \
          --item-emb "${P}_i22real_embeddings.pt" --input-emb "${P}_i24real_input.pt" \
          --ckpt outputs/artifacts/i40_${DS}_ted_REAL_s{42,43,44}/tower.pt --out "results/i40_ted_real_look_${DS}.json"
    fi
    case $DS in
      beauty|toys|sports)
        run masked "results/i43_masked_real_${DS}.json" $PY scripts/i43_masked_real.py --dataset $DS ;;
    esac
    if [[ $DS == toys || $DS == sports ]]; then
      run masked_ted "results/i43_masked_real_${DS}_ted.json" $PY scripts/i43_masked_real.py --dataset ${DS}_ted
    fi
  fi

  # ---------------- baselines
  if has_stage baselines; then
    local losses="bce"; [[ $DS == beauty || $DS == toys || $DS == sports ]] && losses="bce ce"
    for loss in $losses; do for s in $SEEDS; do
      local tag="close2_sasrec_${DS}_s$s"; [[ $loss == ce ]] && tag="close2_sasrec_ce_${DS}_s$s"
      run "sasrec_${loss}_s$s" "results/$tag.json" $PY scripts/close2_sasrec.py --dataset $DS --seed $s --loss $loss --device "$DEVICE"
    done; done
    case $DS in beauty|toys|sports)
      run heuristic "results/i42_transition_only_${DS}.json" $PY scripts/i42_transition_only.py --dataset $DS --n-items $N \
          --seq-inner "$D/$I" --seq "$D/$S" --out "results/i42_transition_only_${DS}.json" ;;
    esac
    local tseeds="42"; [[ $DS == instruments || $DS == yelp ]] && tseeds="$SEEDS"
    for s in $tseeds; do
      local beams="${TIGER_BEAMS_DIR:-third_party/liger/dumps}/s$s/${DS}_tiger_beams_test.jsonl"
      if [[ $MODE == quick ]]; then
        beams="outputs/tiger/${DS}_s${s}_beams.jsonl"
        run "tiger_synthetic_beams_s$s" - $PY scripts/make_synthetic_beams.py --dataset $DS --seed $s --out "$beams"
      fi
      if [[ -e "$WS/$beams" ]]; then
        run "tiger_score_s$s" "results/close2_tiger_${DS}_s$s.json" $PY scripts/close2_tiger_score.py --dataset $DS --seed $s \
            --beams "$beams" --out "results/close2_tiger_${DS}_s$s"
      else
        printf '%s\t%s\tSKIP\t(no TIGER beams at %s; see baselines/tiger/TIGER_RUNBOOK.md)\n' "$DS" "tiger_score_s$s" "$beams" | tee -a "$STATUS.tmp"
      fi
    done
  fi

  # ---------------- paired tests + matched-ensemble control
  if has_stage tests; then
    case $DS in beauty|toys|sports)
      run paired_vs_tiger "results/close2_cell_${DS}_tiger.json" $PY scripts/close2_analysis.py --dataset $DS --baseline tiger \
          --baseline-npz "results/close2_tiger_${DS}_s42_per_user.npz"
      local bs; bs=$(best_seed "close2_sasrec_${DS}")
      run paired_vs_sasrec "results/close2_cell_${DS}_sasrec.json" $PY scripts/close2_analysis.py --dataset $DS --baseline sasrec \
          --baseline-npz "results/close2_sasrec_${DS}_s${bs}_per_user.npz"
      for loss in ce bce; do
        local pre="ensctrl_sasrec_ce"; [[ $loss == bce ]] && pre="ensctrl_sasrec"
        run "ensctrl_$loss" "results/${pre}_${DS}.json" $PY scripts/ensctrl_sasrec_fusion.py --dataset $DS --loss $loss --device "$DEVICE"
        run "ensctrl_masked_$loss" "results/ensctrl_masked_${loss}_${DS}.json" $PY scripts/ensctrl_masked_rescore.py --dataset $DS --loss $loss
      done ;;
    instruments|yelp)
      run ensctrl_bce "results/ensctrl_sasrec_${DS}.json" $PY scripts/ensctrl_sasrec_fusion.py --dataset $DS --loss bce --device "$DEVICE" ;;
    esac
  fi

  # ---------------- 2x2 injection ablation
  if has_stage ablation; then
    for s in $SEEDS; do
      train "outputs/artifacts/i44_${DS}_pp_s${s}_iv" "$D/$S" "${P}_embeddings.pt" none
      train "outputs/artifacts/i44_${DS}_pc_s${s}_iv" "$D/$S" "${P}_embeddings.pt" "${P}_i44plain_input.pt"
      if [[ $DS == beauty ]]; then
        train "outputs/artifacts/i22_real_openai_k4_s${s}_iv" "$D/$S" "${P}_i22real_embeddings.pt" none
      else
        train "outputs/artifacts/i44_${DS}_ep_s${s}_iv" "$D/$S" "${P}_i22real_embeddings.pt" none
      fi
    done
    local ep="i44_${DS}_ep"; [[ $DS == beauty ]] && ep="i22_real_openai_k4"
    local ec="${T}_real_openai_k4"
    cell() {  # cell <tag> <tower-prefix> <item-emb> [input-emb]
      local ia=(); [[ -n "${4:-}" ]] && ia=(--input-emb "$4")
      run "cell_$1" "results/i44_${DS}_$1.json" $PY scripts/i44_eval_cell.py --seq "$D/$S" --n-items $N --item-emb "$3" "${ia[@]}" \
          --ckpt outputs/artifacts/${2}_s{42,43,44}_iv/j5.pt --tag "$1" --out "results/i44_${DS}_$1.json"
    }
    cell pp "i44_${DS}_pp" "${P}_embeddings.pt"
    cell pC "i44_${DS}_pc" "${P}_embeddings.pt" "${P}_i44plain_input.pt"
    cell Ep "$ep" "${P}_i22real_embeddings.pt"
    [[ $DS != yelp ]] && cell EC "$ec" "${P}_i22real_embeddings.pt" "${P}_i24real_input.pt"
  fi

  # ---------------- extras (development / supplement)
  if has_stage extras; then
    case $DS in beauty|toys|sports)
      run seed_soup "results/i45_${DS}_seed_soup.json" $PY scripts/i45_seed_soup.py --seq "$D/$S" --n-items $N \
          --item-emb "${P}_i22real_embeddings.pt" --input-emb "${P}_i24real_input.pt" \
          --ckpt outputs/artifacts/${T}_real_openai_k4_s{42,43,44}_iv/j5.pt --tag $DS --out "results/i45_${DS}_seed_soup.json" ;;
    esac
    case $DS in beauty|instruments|yelp)
      run router "results/i38_${DS}_inner_per_user.npz" $PY scripts/i38_soft_router.py --dataset $DS ;;
    esac
    # TED inner gates on the remaining datasets (Toys/Sports done in the gate stage)
    case $DS in beauty|instruments|yelp)
      local tag="i39_beauty_ted" gout="results/i39_gate_beauty_ted_trio.json"
      [[ $DS == instruments ]] && { tag="i46ted_instruments_inner"; gout="results/i46_gate_instruments_ted_trio.json"; }
      [[ $DS == yelp ]] && { tag="i47ted_yelp_inner"; gout="results/i47_gate_yelp_ted_trio.json"; }
      for s in $SEEDS; do
        run "ted_inner_s$s" "outputs/artifacts/${tag}_s$s/tower.pt" $PY scripts/i40_train_tricks.py --dataset beauty --seed $s \
            --steps $STEPS --device "$DEVICE" --ted_beta 0.1 --ted_tau 0.1 --seq "$D/$I" \
            --item-emb "${P}_i22inner_embeddings.pt" --input-emb "${P}_i24inner_input.pt" --n-items $N \
            --out-dir "outputs/artifacts/${tag}_s$s"
      done
      run ted_gate "$gout" $PY scripts/i40_gate.py --device "$DEVICE" --seq "$D/$I" --n-items $N \
          --item-emb "${P}_i22inner_embeddings.pt" --input-emb "${P}_i24inner_input.pt" \
          --ckpt outputs/artifacts/${tag}_s{42,43,44}/tower.pt --out "$gout" ;;
    esac
    if [[ $DS == yelp ]]; then
      run ted_blend results/i47_blend_yelp_ted.json $PY scripts/i50_gate_expand.py --dataset yelp \
          --ckpt-prefix outputs/artifacts/i47ted_yelp_inner_s --out results/i47_blend_yelp_ted.json
    fi
    if [[ $DS == beauty ]]; then
      run adaptive_k results/i29_adaptive_k.json $PY scripts/i29_adaptive_k.py
      local SEL="contentnn3 bm25_3 random3 popular3 upred3 translift3 ppmi3 ppr3 pred1 catbrand3"
      run selector_prep - $PY scripts/simpleaug_prep.py --embedder "$EMBEDDER" $SEL
      run selector_towers results/simpleaug_box_ppr3.json $PY scripts/simpleaug_box.py none pred3 $SEL
      run selector_trainfree_probe results/simpleaug_explore_beauty.json $PY scripts/simpleaug_explore.py --embedder "$EMBEDDER"
      run i22b_variants "${P}_i22b_pred5_inner_embeddings.pt" $PY scripts/i22b_variant_embeds.py --embedder "$EMBEDDER"
      for v in succ both pred5; do for s in $SEEDS; do
        train "outputs/artifacts/i22b_${v}_inner_openai_k4_s${s}_iv" "$D/$I" "${P}_i22b_${v}_inner_embeddings.pt" none
        run "p26_i22b_${v}_s$s" - env P26_CKPT="outputs/artifacts/i22b_${v}_inner_openai_k4_s${s}_iv/j5.pt" \
            P26_ITEM_EMB="${P}_i22b_${v}_inner_embeddings.pt" P26_TAG="i22b_${v}_s${s}_iv" $PY scripts/p26_inner_validation.py
      done; done
      for spec in "m:512:4:8:1024:1e-3" "l:1024:6:16:2048:1e-3" "l5:1024:6:16:2048:5e-4"; do
        IFS=: read nm d l h f lr <<<"$spec"
        train "outputs/artifacts/i18_size_${nm}_s42_iv" "$D/$I" "${P}_i22inner_embeddings.pt" "${P}_i24inner_input.pt" \
            --d $d --layers $l --heads $h --d-ff $f --lr $lr
      done
      run size_score results/i18_size_score.json $PY scripts/i18_size_score.py
      for s in $SEEDS; do
        run "minilm_ted_s$s" "outputs/artifacts/i48_lmqted_s$s/lmq.pt" $PY scripts/i37_train_lmq.py --seed $s --steps $LM_STEPS \
            --device "$DEVICE" --ted_beta 0.1 --ted_tau 0.1 --out-dir "outputs/artifacts/i48_lmqted_s$s"
      done
      run minilm_gate results/i48_gate.json $PY scripts/i37_gate.py --ckpt-prefix outputs/artifacts/i48_lmqted_s --out results/i48_gate.json
      run ann_replay results/i23_ann_replay.json $PY scripts/i23_ann_replay.py
    fi
  fi
}

# ------------------------------------------------------------------ cross-dataset steps
cross_chain() {
  DS=""
  if [[ $MODE == quick ]]; then
    WS="$WS_BASE/_all"; rm -rf "$WS"
    $PY "$REPO/scripts/workspace.py" --mode quick --dataset "${DATASETS[@]}" --users "$USERS" \
        --data-src "$DATA_SRC" --ws "$WS" > /dev/null || exit 1
    for d in "${DATASETS[@]}"; do   # gather the per-dataset towers + results
      [[ -d "$WS_BASE/$d" ]] || continue
      for a in "$WS_BASE/$d"/outputs/artifacts/*; do ln -sfn "$a" "$WS/outputs/artifacts/$(basename "$a")"; done
      cp -n "$WS_BASE/$d"/results/* "$WS/results/" 2>/dev/null
    done
  else
    WS="$WS_BASE"
  fi
  have() { local d; for d in "$@"; do [[ " ${DATASETS[*]} " == *" $d "* ]] || return 1; done; }
  if have beauty || have toys || have sports; then
    run factorial results/i44_factorial_inference.json $PY scripts/i44_factorial_inference.py
    [[ -e "$WS/results/i44_factorial_per_user.npz" ]] && run factorial_from_npz - $PY scripts/i44_factorial_from_npz.py
  fi
  have beauty toys sports && run serving_bench results/serving_bench.json env BENCH_THREADS=1 $PY scripts/close3_serving_bench.py
  have instruments yelp && run serving_bench_expand results/serving_bench_expand.json $PY scripts/close3_serving_bench_expand.py
  have yelp && run serving_bench_yelp_blend results/serving_bench_yelp_blend.json env BENCH_THREADS=1 $PY scripts/close3_serving_bench_yelp_blend.py
  have beauty && run ann_bench results/ann_bench_deployed_1m.json $PY scripts/ann_bench_deployed.py $ANN_ARGS --out results/ann_bench_deployed_1m.json
  true
}

for d in "${DATASETS[@]}"; do
  dataset_chain "$d"
done
has_stage cross && cross_chain
mv "$STATUS.tmp" "$STATUS"
echo; echo "== $(grep -c PASS "$STATUS") PASS / $(grep -c FAIL "$STATUS") FAIL / $(grep -c SKIP "$STATUS") SKIP  -> $STATUS"
