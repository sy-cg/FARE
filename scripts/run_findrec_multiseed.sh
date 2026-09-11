#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# Run FindRec on the three IP&M experiment datasets with three random seeds.
#
# Usage:
#   bash scripts/run_findrec_multiseed.sh
#
# Useful overrides:
#   DATASETS="Video_Games Musical_Instruments Baby_Products" bash scripts/run_findrec_multiseed.sh
#   SEEDS="2024 2025 2026" bash scripts/run_findrec_multiseed.sh
#   RUN_TAG=ipm_findrec_3seed bash scripts/run_findrec_multiseed.sh
#   SMOKE=1 bash scripts/run_findrec_multiseed.sh
#   DRY_RUN=1 bash scripts/run_findrec_multiseed.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_STR="${DATASETS:-Video_Games Musical_Instruments Baby_Products}"
SEEDS_STR="${SEEDS:-2024 2025 2026}"
CONFIG="${CONFIG:-configs/findrec_3090.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SMOKE="${SMOKE:-0}"

read -r -a DATASET_LIST <<< "$DATASETS_STR"
read -r -a SEED_LIST <<< "$SEEDS_STR"

run_cmd() {
  echo
  echo "[CMD] $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

if [[ ! -f "$CONFIG" ]]; then
  echo "[ERROR] Missing config: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "scripts/run_findrec.py" ]]; then
  echo "[ERROR] Missing script: scripts/run_findrec.py" >&2
  exit 1
fi

echo "========== Run FindRec multi-seed experiments =========="
echo "root:          $ROOT_DIR"
echo "datasets:      ${DATASET_LIST[*]}"
echo "seeds:         ${SEED_LIST[*]}"
echo "config:        $CONFIG"
echo "python:        $PYTHON_BIN"
echo "run_tag:       $RUN_TAG"
echo "skip_existing: $SKIP_EXISTING"
echo "smoke:         $SMOKE"
echo

for dataset in "${DATASET_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    run_id="findrec_${dataset}_seed${seed}_${RUN_TAG}"
    out_dir="results/${dataset}/findrec/${run_id}"

    if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/best_model.pt" ]]; then
      echo "[SKIP] Existing checkpoint: ${out_dir}/best_model.pt"
      continue
    fi

    args=(
      "$PYTHON_BIN" scripts/run_findrec.py
      --dataset "$dataset"
      --config "$CONFIG"
      --run_id "$run_id"
      --seed "$seed"
    )

    if [[ "$SMOKE" == "1" ]]; then
      args+=(--epochs "${SMOKE_EPOCHS:-2}" --eval_every 1 --patience 2)
    fi
    if [[ -n "${EPOCHS:-}" ]]; then
      args+=(--epochs "$EPOCHS")
    fi
    if [[ -n "${BATCH_SIZE:-}" ]]; then
      args+=(--batch_size "$BATCH_SIZE")
    fi
    if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
      args+=(--eval_batch_size "$EVAL_BATCH_SIZE")
    fi
    if [[ -n "${LR:-}" ]]; then
      args+=(--lr "$LR")
    fi

    run_cmd "${args[@]}"
  done
done

echo
echo "========== FindRec multi-seed runs finished =========="
echo "RUN_TAG=$RUN_TAG"
