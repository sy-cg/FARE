#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# scripts/run_all_modality_debias.sh
#
# Run ModalityDebias variants:
#   - VBPR + ModalityDebias
#   - BM3 + ModalityDebias
#   - FREEDOM + ModalityDebias
#   - LATTICE + ModalityDebias
#   - LateFusion-SASRec + ModalityDebias
#   - LateFusion-GRU4Rec + ModalityDebias
#   - LateFusion-BERT4Rec + ModalityDebias
#
# Requirements:
#   1. For VBPR/BM3/FREEDOM/LATTICE + ModalityDebias:
#        corresponding base best_model.pt should already exist.
#   2. For LateFusion-* + ModalityDebias:
#        ID backbone best_model.pt should already exist.
#
# Usage:
#   bash scripts/run_all_modality_debias.sh
#
# Useful overrides:
#   BASE_MODELS="vbpr bm3 freedom lattice latefusion_sasrec latefusion_gru4rec latefusion_bert4rec" bash ...
#   FREEZE_BASE=1 bash scripts/run_all_modality_debias.sh
#   DRY_RUN=1 bash scripts/run_all_modality_debias.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_STR="${DATASETS:-Video_Games Musical_Instruments Baby_Products}"
BASE_MODELS_STR="${BASE_MODELS:-vbpr bm3 freedom lattice latefusion_sasrec latefusion_gru4rec latefusion_bert4rec}"
RESULT_ROOT="${RESULT_ROOT:-results}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SMOKE="${SMOKE:-0}"
FREEZE_BASE="${FREEZE_BASE:-1}"

SCRIPT="${SCRIPT:-scripts/run_modality_debias.py}"
CONFIG="${CONFIG:-configs/modality_debias_3090.yaml}"

read -r -a DATASET_LIST <<< "$DATASETS_STR"
read -r -a BASE_MODEL_LIST <<< "$BASE_MODELS_STR"

run_name_for_backbone() {
  case "$1" in
    sasrec) echo "sasrec_id" ;;
    gru4rec) echo "gru4rec_id" ;;
    bert4rec) echo "bert4rec_id" ;;
    *) echo "${1}_id" ;;
  esac
}

parent_for_base_model() {
  local base="$1"
  case "$base" in
    vbpr) echo "vbpr" ;;
    bm3) echo "bm3" ;;
    freedom) echo "freedom" ;;
    lattice) echo "lattice" ;;
    *) echo "" ;;
  esac
}

backbone_from_latefusion() {
  local base="$1"
  case "$base" in
    latefusion_sasrec) echo "sasrec" ;;
    latefusion_gru4rec) echo "gru4rec" ;;
    latefusion_bert4rec) echo "bert4rec" ;;
    *) echo "" ;;
  esac
}

find_latest_run_dir() {
  local parent="$1"
  if [[ ! -d "$parent" ]]; then
    return 1
  fi
  find "$parent" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR==1 {sub(/^[^ ]+ /, ""); print; exit}'
}

find_latest_checkpoint() {
  local parent="$1"
  local run_dir
  run_dir="$(find_latest_run_dir "$parent" || true)"
  if [[ -n "$run_dir" && -f "$run_dir/best_model.pt" ]]; then
    echo "$run_dir/best_model.pt"
    return 0
  fi
  return 1
}

script_accepts_arg() {
  local script="$1"
  local arg="$2"
  python "$script" --help 2>/dev/null | grep -q -- "$arg"
}

run_cmd() {
  echo
  echo "[CMD] $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing $SCRIPT" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "[ERROR] Missing $CONFIG" >&2; exit 1; }

echo "========== Run all ModalityDebias variants =========="
echo "root:          $ROOT_DIR"
echo "datasets:      ${DATASET_LIST[*]}"
echo "base_models:   ${BASE_MODEL_LIST[*]}"
echo "run_tag:       $RUN_TAG"
echo "freeze_base:   $FREEZE_BASE"
echo "result_root:   $RESULT_ROOT"
echo

for dataset in "${DATASET_LIST[@]}"; do
  for base_model in "${BASE_MODEL_LIST[@]}"; do
    run_id="${base_model}_modality_debias_${dataset}_${RUN_TAG}"
    out_dir="${RESULT_ROOT}/${dataset}/${base_model}_modality_debias/${run_id}"

    if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/best_model.pt" ]]; then
      echo "[SKIP] Existing ModalityDebias checkpoint: ${out_dir}/best_model.pt"
      continue
    fi

    args=(
      python "$SCRIPT"
      --dataset "$dataset"
      --base_model "$base_model"
      --config "$CONFIG"
      --run_id "$run_id"
    )

    # Multimodal base model checkpoints.
    parent="$(parent_for_base_model "$base_model")"
    if [[ -n "$parent" ]]; then
      ckpt="$(find_latest_checkpoint "${RESULT_ROOT}/${dataset}/${parent}" || true)"
      if [[ -z "$ckpt" ]]; then
        echo "[WARN] Missing base checkpoint for ${dataset}/${base_model}. Expected under ${RESULT_ROOT}/${dataset}/${parent}. Skipping."
        continue
      fi
      if script_accepts_arg "$SCRIPT" "--init_base_checkpoint"; then
        args+=(--init_base_checkpoint "$ckpt")
      else
        echo "[WARN] $SCRIPT does not support --init_base_checkpoint. It will train base from scratch."
      fi
      if [[ "$FREEZE_BASE" == "1" ]] && script_accepts_arg "$SCRIPT" "--freeze_base"; then
        args+=(--freeze_base)
      fi
    fi

    # LateFusion-* uses ID backbone checkpoint.
    backbone="$(backbone_from_latefusion "$base_model")"
    if [[ -n "$backbone" ]]; then
      id_run_name="$(run_name_for_backbone "$backbone")"
      ckpt="$(find_latest_checkpoint "${RESULT_ROOT}/${dataset}/${id_run_name}" || true)"
      if [[ -z "$ckpt" ]]; then
        echo "[WARN] Missing ID checkpoint for ${dataset}/${backbone}. Expected under ${RESULT_ROOT}/${dataset}/${id_run_name}. Skipping."
        continue
      fi
      args+=(--init_backbone_checkpoint "$ckpt")
    fi

    if [[ "$SMOKE" == "1" ]]; then
      args+=(--epochs "${SMOKE_EPOCHS:-2}" --eval_every 1 --patience 2)
    fi

    if [[ -n "${BATCH_SIZE:-}" ]]; then
      args+=(--batch_size "$BATCH_SIZE")
    fi
    if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
      args+=(--eval_batch_size "$EVAL_BATCH_SIZE")
    fi

    run_cmd "${args[@]}"
  done
done

echo
echo "========== ModalityDebias runs finished =========="
echo "RUN_TAG=$RUN_TAG"
