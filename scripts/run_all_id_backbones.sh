#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# scripts/run_all_id_backbones.sh
#
# Run ID-only sequential backbones on all target datasets:
#   - SASRec-ID
#   - GRU4Rec-ID
#   - BERT4Rec-ID
#
# Default datasets:
#   Video_Games Musical_Instruments Baby_Products
#
# Usage:
#   bash scripts/run_all_id_backbones.sh
#
# Useful overrides:
#   DATASETS="Video_Games Musical_Instruments Baby_Products" bash scripts/run_all_id_backbones.sh
#   BACKBONES="sasrec gru4rec bert4rec" bash scripts/run_all_id_backbones.sh
#   RUN_TAG=main_20260519 bash scripts/run_all_id_backbones.sh
#   DRY_RUN=1 bash scripts/run_all_id_backbones.sh
#
# Smoke test:
#   SMOKE=1 bash scripts/run_all_id_backbones.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_STR="${DATASETS:-Video_Games Musical_Instruments Baby_Products}"
BACKBONES_STR="${BACKBONES:-sasrec gru4rec bert4rec}"
RESULT_ROOT="${RESULT_ROOT:-results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SMOKE="${SMOKE:-0}"

read -r -a DATASET_LIST <<< "$DATASETS_STR"
read -r -a BACKBONE_LIST <<< "$BACKBONES_STR"

config_for_backbone() {
  case "$1" in
    sasrec) echo "configs/sasrec_3090.yaml" ;;
    gru4rec) echo "configs/gru4rec_3090.yaml" ;;
    bert4rec) echo "configs/bert4rec_3090.yaml" ;;
    *) echo "configs/${1}_3090.yaml" ;;
  esac
}

run_name_for_backbone() {
  case "$1" in
    sasrec) echo "sasrec_id" ;;
    gru4rec) echo "gru4rec_id" ;;
    bert4rec) echo "bert4rec_id" ;;
    *) echo "${1}_id" ;;
  esac
}

run_cmd() {
  echo
  echo "[CMD] $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

echo "========== Run all ID backbones =========="
echo "root:          $ROOT_DIR"
echo "datasets:      ${DATASET_LIST[*]}"
echo "backbones:     ${BACKBONE_LIST[*]}"
echo "run_tag:       $RUN_TAG"
echo "result_root:   $RESULT_ROOT"
echo "skip_existing: $SKIP_EXISTING"
echo "smoke:         $SMOKE"
echo

for dataset in "${DATASET_LIST[@]}"; do
  for backbone in "${BACKBONE_LIST[@]}"; do
    config="$(config_for_backbone "$backbone")"
    run_name="$(run_name_for_backbone "$backbone")"
    run_id="${run_name}_${dataset}_${RUN_TAG}"
    out_dir="${RESULT_ROOT}/${dataset}/${run_name}/${run_id}"

    if [[ ! -f "$config" ]]; then
      echo "[ERROR] Missing config: $config" >&2
      exit 1
    fi

    if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/best_model.pt" ]]; then
      echo "[SKIP] Existing checkpoint: ${out_dir}/best_model.pt"
      continue
    fi

    args=(
      python scripts/run_id_backbone.py
      --model "$backbone"
      --dataset "$dataset"
      --config "$config"
      --run_id "$run_id"
    )

    if [[ "$SMOKE" == "1" ]]; then
      args+=(--epochs "${SMOKE_EPOCHS:-2}" --eval_every 1 --patience 2)
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
echo "========== ID backbone runs finished =========="
echo "RUN_TAG=$RUN_TAG"
