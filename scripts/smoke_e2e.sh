#!/usr/bin/env bash
# Synthetic end-to-end smoke test (CPU, no data download, no API key, no GPU).
#
#   scripts/smoke_e2e.sh [dataset ...]        (default: beauty)
#
# 1. scripts/make_synthetic_dataset.py writes a tiny synthetic dataset (300 users, the real
#    catalog size) in the real data layout under outputs/smoke/data;
# 2. the REAL prep scripts build the enriched item catalogs (E), the centroid input tables
#    (C) and the plain catalogs with the offline fake embedder (--embedder fake);
# 3. the REAL pipeline runs on it in quick mode (20 training steps, 2 SASRec epochs):
#    towers -> gate -> real look -> masked rescore -> baselines (SASRec, heuristic,
#    TIGER scoring of synthetic beams) -> paired tests -> matched-ensemble control ->
#    2x2 cells -> extras (soup, router, TED, selectors, i22b, scale, MiniLM) -> cross
#    (factorial, serving benchmarks, ANN benchmark on a 20k-item corpus).
# Status table: outputs/smoke/run/STATUS-<dataset>.tsv. Numbers are meaningless by construction.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DS=("$@"); [[ ${#DS[@]} -eq 0 ]] && DS=(beauty)
SYN="$REPO/outputs/smoke/data"
rm -rf "$REPO/outputs/smoke"
$PY "$REPO/scripts/make_synthetic_dataset.py" --dataset "${DS[@]}" --users 300 --out "$SYN" || exit 1
PYTHON="$PY" bash "$REPO/scripts/reproduce.sh" --dataset "${DS[@]}" --stages embed \
    --data-src "$SYN" --ws "$REPO/outputs/smoke/prep" --embedder fake || exit 1
PYTHON="$PY" bash "$REPO/scripts/reproduce.sh" --quick --dataset "${DS[@]}" --users 300 \
    --data-src "$SYN" --ws "$REPO/outputs/smoke/run" \
    --stages towers,gate,look,baselines,tests,ablation,extras,cross
