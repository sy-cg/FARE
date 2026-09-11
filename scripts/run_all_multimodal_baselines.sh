#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# scripts/run_all_multimodal_baselines.sh
#
# Run multimodal recommendation baselines:
#   - VBPR
#   - BM3
#   - FREEDOM
#   - LATTICE
#   - FindRec
#
# Usage:
#   bash scripts/run_all_multimodal_baselines.sh
#
# Useful overrides:
#   DATASETS="Video_Games Musical_Instruments Baby_Products" bash scripts/run_all_multimodal_baselines.sh
#   MODELS="vbpr bm3 freedom lattice findrec" bash scripts/run_all_multimodal_baselines.sh
#   RUN_TAG=main_20260519 bash scripts/run_all_multimodal_baselines.sh
#   SMOKE=1 bash scripts/run_all_multimodal_baselines.sh
#   DRY_RUN=1 bash scripts/run_all_multimodal_baselines.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_STR="${DATASETS:-Video_Games Musical_Instruments Baby_Products}"
MODELS_STR="${MODELS:-vbpr bm3 freedom lattice findrec}"
RESULT_ROOT="${RESULT_ROOT:-results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SMOKE="${SMOKE:-0}"

read -r -a DATASET_LIST <<< "$DATASETS_STR"
read -r -a MODEL_LIST <<< "$MODELS_STR"

script_for_model() {
  case "$1" in
    vbpr) echo "scripts/run_vbpr.py" ;;
    bm3) echo "scripts/run_bm3.py" ;;
    freedom) echo "scripts/run_freedom.py" ;;
    lattice) echo "scripts/run_lattice.py" ;;
    findrec) echo "scripts/run_findrec.py" ;;
    *) echo "scripts/run_${1}.py" ;;
  esac
}

config_for_model() {
  case "$1" in
    vbpr) echo "configs/vbpr_3090.yaml" ;;
    bm3) echo "configs/bm3_3090.yaml" ;;
    freedom) echo "configs/freedom_3090.yaml" ;;
    lattice) echo "configs/lattice_3090.yaml" ;;
    findrec) echo "configs/findrec_3090.yaml" ;;
    *) echo "configs/${1}_3090.yaml" ;;
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

echo "========== Run all multimodal baselines =========="
echo "root:          $ROOT_DIR"
echo "datasets:      ${DATASET_LIST[*]}"
echo "models:        ${MODEL_LIST[*]}"
echo "run_tag:       $RUN_TAG"
echo "result_root:   $RESULT_ROOT"
echo "skip_existing: $SKIP_EXISTING"
echo "smoke:         $SMOKE"
echo

for dataset in "${DATASET_LIST[@]}"; do
  for model in "${MODEL_LIST[@]}"; do
    script="$(script_for_model "$model")"
    config="$(config_for_model "$model")"
    run_id="${model}_${dataset}_${RUN_TAG}"
    out_dir="${RESULT_ROOT}/${dataset}/${model}/${run_id}"

    if [[ ! -f "$script" ]]; then
      echo "[ERROR] Missing script: $script" >&2
      exit 1
    fi
    if [[ ! -f "$config" ]]; then
      echo "[ERROR] Missing config: $config" >&2
      exit 1
    fi

    if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/best_model.pt" ]]; then
      echo "[SKIP] Existing checkpoint: ${out_dir}/best_model.pt"
      continue
    fi

    args=(
      python "$script"
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
echo "========== Multimodal baseline runs finished =========="
echo "RUN_TAG=$RUN_TAG"
