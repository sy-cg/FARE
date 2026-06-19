#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# scripts/run_all_fare.sh
#
# Run FARE on the configured datasets and ID backbones with the final
# Fair-SDR-MR-Exposure experiment policy migrated to FARE.
#
# FARE requires:
#   1. an ID-backbone checkpoint for initialization;
#   2. a reference Top-K .npz file for exposure-aware reweighting.
#
# By default both are read from the latest run under:
#   results/<Dataset>/<backbone>_id/
#
# Usage:
#   bash scripts/run_all_fare.sh
#
# Useful overrides:
#   DATASETS="Video_Games Baby_Products" BACKBONES="sasrec" bash scripts/run_all_fare.sh
#   RESULT_ROOT=results RUN_TAG=paper_v1 bash scripts/run_all_fare.sh
#   EXPOSURE_TOPK_SPLIT=val bash scripts/run_all_fare.sh
#   FAIR_REC_REWEIGHT_GROUPS=popularity_group FAIR_REC_REWEIGHT_WEIGHT=0.20 bash scripts/run_all_fare.sh
#   DRY_RUN=1 bash scripts/run_all_fare.sh
#   SMOKE=1 bash scripts/run_all_fare.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_STR="${DATASETS:-Video_Games Musical_Instruments Baby_Products}"
BACKBONES_STR="${BACKBONES:-sasrec gru4rec bert4rec}"

RESULT_ROOT="${RESULT_ROOT:-results}"
RUN_TAG="${RUN_TAG:-v1}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SMOKE="${SMOKE:-0}"
FREEZE_ID_BACKBONE="${FREEZE_ID_BACKBONE:-1}"
EXPOSURE_TOPK_SPLIT="${EXPOSURE_TOPK_SPLIT:-val}"
ALLOW_TEST_EXPOSURE_TOPK="${ALLOW_TEST_EXPOSURE_TOPK:-0}"

SCRIPT="${SCRIPT:-scripts/run_fare.py}"
CONFIG="${CONFIG:-}"
DEFAULT_CONFIG="${DEFAULT_CONFIG:-configs/fare_3090.yaml}"
GRU1_CONFIG="${GRU1_CONFIG:-configs/fare_3090_gru1.yaml}"

FARE_GROUPS="${FARE_GROUPS:-popularity_group,text_quality_group,vision_quality_group,category_proxy_group,brand_store_proxy_group,multimodal_cluster_proxy_group}"

# Final Fair-SDR-MR-Exposure constants migrated to FARE.
LR="${LR:-0.0003}"
RESIDUAL_SCORE_WEIGHT="${RESIDUAL_SCORE_WEIGHT:-1.0}"
FAIR_WEIGHT_INIT="${FAIR_WEIGHT_INIT:-0.01}"
MAX_FAIR_WEIGHT="${MAX_FAIR_WEIGHT:-0.1}"
FAIR_REC_EXPOSURE_K="${FAIR_REC_EXPOSURE_K:-10}"
FAIR_REC_EXPOSURE_TARGET="${FAIR_REC_EXPOSURE_TARGET:-catalog}"
FAIR_REC_REWEIGHT_MIN="${FAIR_REC_REWEIGHT_MIN:-0.5}"
FAIR_REC_REWEIGHT_MAX="${FAIR_REC_REWEIGHT_MAX:-2.0}"

read -r -a DATASET_LIST <<< "$DATASETS_STR"
read -r -a BACKBONE_LIST <<< "$BACKBONES_STR"

run_name_for_backbone() {
  case "$1" in
    sasrec) echo "sasrec_id" ;;
    gru4rec) echo "gru4rec_id" ;;
    bert4rec) echo "bert4rec_id" ;;
    *) echo "${1}_id" ;;
  esac
}

group_to_tag() {
  local group="$1"
  if [[ "$group" == "popularity_group" ]]; then
    echo "pop"
  elif [[ "$group" == "multimodal_cluster_proxy_group" ]]; then
    echo "cluster"
  else
    echo "$group" | sed 's/_group//g' | sed 's/_proxy//g'
  fi
}

value_to_tag() {
  local value="$1"
  echo "${value/./p}"
}

clip_to_tag() {
  local min_value="${1/./}"
  local max_value="${2/./}"
  echo "clip${min_value}_${max_value}"
}

fare_selected_exposure_policy() {
  local dataset="$1"
  case "$dataset" in
    Baby_Products) echo "popularity_group 0.20" ;;
    Musical_Instruments) echo "multimodal_cluster_proxy_group 0.20" ;;
    Video_Games) echo "popularity_group 0.20" ;;
    *) echo "popularity_group 0.20" ;;
  esac
}

fare_config_for_backbone() {
  local backbone="$1"
  if [[ -n "$CONFIG" ]]; then
    echo "$CONFIG"
    return 0
  fi
  if [[ "$backbone" == "gru4rec" ]]; then
    echo "$GRU1_CONFIG"
  else
    echo "$DEFAULT_CONFIG"
  fi
}

fare_run_id() {
  local dataset="$1"
  local backbone="$2"
  local group="$3"
  local gamma="$4"
  local group_tag
  local gamma_tag
  local clip_tag
  group_tag="$(group_to_tag "$group")"
  gamma_tag="$(value_to_tag "$gamma")"
  clip_tag="$(clip_to_tag "$FAIR_REC_REWEIGHT_MIN" "$FAIR_REC_REWEIGHT_MAX")"

  case "$backbone" in
    sasrec)
      echo "fare_exposure_tune_${dataset}_sasrec_${group_tag}_g${gamma_tag}_k${FAIR_REC_EXPOSURE_K}_${FAIR_REC_EXPOSURE_TARGET}_${clip_tag}_${RUN_TAG}"
      ;;
    gru4rec)
      echo "fare_exposure_fairfirst_gru1_${dataset}_gru4rec_${group_tag}_g${gamma_tag}_k${FAIR_REC_EXPOSURE_K}_${FAIR_REC_EXPOSURE_TARGET}_${clip_tag}_${RUN_TAG}"
      ;;
    bert4rec)
      echo "fare_exposure_fairfirst_${dataset}_bert4rec_${group_tag}_g${gamma_tag}_k${FAIR_REC_EXPOSURE_K}_${FAIR_REC_EXPOSURE_TARGET}_${clip_tag}_${RUN_TAG}"
      ;;
    *)
      echo "fare_exposure_fairfirst_${dataset}_${backbone}_${group_tag}_g${gamma_tag}_k${FAIR_REC_EXPOSURE_K}_${FAIR_REC_EXPOSURE_TARGET}_${clip_tag}_${RUN_TAG}"
      ;;
  esac
}

find_latest_run_dir() {
  local parent="$1"
  [[ -d "$parent" ]] || return 1
  find "$parent" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR==1 {sub(/^[^ ]+ /, ""); print; exit}'
}

find_latest_file() {
  local parent="$1"
  local file_name="$2"
  local run_dir
  run_dir="$(find_latest_run_dir "$parent" || true)"
  if [[ -n "$run_dir" && -f "$run_dir/$file_name" ]]; then
    echo "$run_dir/$file_name"
    return 0
  fi
  return 1
}

run_cmd() {
  echo
  echo "[CMD] $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

check_output_exists() {
  local dataset="$1"
  local run_id="$2"
  [[ -f "${RESULT_ROOT}/${dataset}/fare/${run_id}/best_model.pt" ]]
}

run_fare_one() {
  local dataset="$1"
  local backbone="$2"
  local run_id="$3"
  local ckpt="$4"
  local topk_path="$5"

  local config
  config="$(fare_config_for_backbone "$backbone")"
  [[ -f "$config" ]] || { echo "[ERROR] Missing config $config" >&2; exit 1; }

  local policy_group
  local policy_gamma
  read -r policy_group policy_gamma < <(fare_selected_exposure_policy "$dataset")

  local reweight_groups="${FAIR_REC_REWEIGHT_GROUPS:-$policy_group}"
  local reweight_weight="${FAIR_REC_REWEIGHT_WEIGHT:-$policy_gamma}"

  local cmd=(
    python "$SCRIPT"
    --dataset "$dataset"
    --config "$config"
    --backbone "$backbone"
    --init_backbone_checkpoint "$ckpt"
    --fair_rec_exposure_topk_path "$topk_path"
    --run_id "$run_id"
    --method_name "FARE"
    --groups "$FARE_GROUPS"
    --lr "$LR"
    --fair_weight_init "$FAIR_WEIGHT_INIT"
    --max_fair_weight "$MAX_FAIR_WEIGHT"
    --residual_score_weight "$RESIDUAL_SCORE_WEIGHT"
    --fair_rec_reweight_weight "$reweight_weight"
    --fair_rec_reweight_groups "$reweight_groups"
    --fair_rec_exposure_target "$FAIR_REC_EXPOSURE_TARGET"
    --fair_rec_exposure_k "$FAIR_REC_EXPOSURE_K"
    --fair_rec_reweight_min "$FAIR_REC_REWEIGHT_MIN"
    --fair_rec_reweight_max "$FAIR_REC_REWEIGHT_MAX"
  )

  if [[ "$FREEZE_ID_BACKBONE" == "1" ]]; then
    cmd+=(--freeze_id_backbone)
  fi

  if [[ "$SMOKE" == "1" ]]; then
    cmd+=(--epochs "${SMOKE_EPOCHS:-2}" --eval_every 1 --patience 2)
  fi

  if [[ -n "${BATCH_SIZE:-}" ]]; then
    cmd+=(--batch_size "$BATCH_SIZE")
  fi
  if [[ -n "${EVAL_BATCH_SIZE:-}" ]]; then
    cmd+=(--eval_batch_size "$EVAL_BATCH_SIZE")
  fi
  if [[ -n "${NUM_WORKERS:-}" ]]; then
    cmd+=(--num_workers "$NUM_WORKERS")
  fi

  if [[ "$ALLOW_TEST_EXPOSURE_TOPK" == "1" ]]; then
    cmd+=(--allow_test_exposure_topk)
  fi

  echo "[PARAM] dataset=${dataset} backbone=${backbone} run_id=${run_id}" >&2
  echo "[PARAM] config=${config}" >&2
  echo "[PARAM] ckpt=${ckpt}" >&2
  echo "[PARAM] exposure_topk=${topk_path}" >&2
  echo "[PARAM] lr=${LR} fair_weight_init=${FAIR_WEIGHT_INIT} max_fair_weight=${MAX_FAIR_WEIGHT} residual=${RESIDUAL_SCORE_WEIGHT}" >&2
  echo "[PARAM] reweight=${reweight_weight} reweight_groups=${reweight_groups} target=${FAIR_REC_EXPOSURE_TARGET} k=${FAIR_REC_EXPOSURE_K} clip=[${FAIR_REC_REWEIGHT_MIN},${FAIR_REC_REWEIGHT_MAX}]" >&2

  run_cmd "${cmd[@]}"
}

[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing $SCRIPT" >&2; exit 1; }
[[ -f "$DEFAULT_CONFIG" ]] || { echo "[ERROR] Missing $DEFAULT_CONFIG" >&2; exit 1; }
if [[ -n "$CONFIG" && ! -f "$CONFIG" ]]; then
  echo "[ERROR] Missing $CONFIG" >&2
  exit 1
fi

if [[ "$EXPOSURE_TOPK_SPLIT" == "test" && "$ALLOW_TEST_EXPOSURE_TOPK" != "1" ]]; then
  echo "[ERROR] EXPOSURE_TOPK_SPLIT=test would leak test exposure into FARE training." >&2
  echo "[ERROR] Use val/train exposure, or set ALLOW_TEST_EXPOSURE_TOPK=1 only for diagnostics." >&2
  exit 1
fi

echo "========== Run FARE experiments =========="
echo "root:          $ROOT_DIR"
echo "datasets:      ${DATASET_LIST[*]}"
echo "backbones:     ${BACKBONE_LIST[*]}"
echo "result_root:   $RESULT_ROOT"
echo "run_tag:       $RUN_TAG"
echo "topk_split:    $EXPOSURE_TOPK_SPLIT"
echo "freeze_id:     $FREEZE_ID_BACKBONE"
echo "skip_existing: $SKIP_EXISTING"
echo "dry_run:       $DRY_RUN"
echo "config:        ${CONFIG:-per-backbone default}"
echo "default_config:$DEFAULT_CONFIG"
echo "gru1_config:   $GRU1_CONFIG"
echo "groups:        $FARE_GROUPS"
echo "reweight:      SDR final policy unless FAIR_REC_REWEIGHT_GROUPS/WEIGHT override"
echo "lr:            $LR"
echo "fair_weight:   init=$FAIR_WEIGHT_INIT max=$MAX_FAIR_WEIGHT residual=$RESIDUAL_SCORE_WEIGHT"
echo "exposure:      target=$FAIR_REC_EXPOSURE_TARGET k=$FAIR_REC_EXPOSURE_K clip=[$FAIR_REC_REWEIGHT_MIN,$FAIR_REC_REWEIGHT_MAX]"
echo

for dataset in "${DATASET_LIST[@]}"; do
  for backbone in "${BACKBONE_LIST[@]}"; do
    id_run_name="$(run_name_for_backbone "$backbone")"
    id_parent="${RESULT_ROOT}/${dataset}/${id_run_name}"

    ckpt="$(find_latest_file "$id_parent" "best_model.pt" || true)"
    topk="$(find_latest_file "$id_parent" "topk_${EXPOSURE_TOPK_SPLIT}.npz" || true)"

    if [[ -z "$ckpt" ]]; then
      echo "[WARN] Missing ID checkpoint for ${dataset}/${backbone}; skipping FARE."
      continue
    fi
    if [[ -z "$topk" ]]; then
      echo "[WARN] Missing exposure Top-K for ${dataset}/${backbone}; expected topk_${EXPOSURE_TOPK_SPLIT}.npz under ${id_parent}; skipping FARE."
      continue
    fi

    read -r selected_group selected_gamma < <(fare_selected_exposure_policy "$dataset")
    reweight_group_for_id="${FAIR_REC_REWEIGHT_GROUPS:-$selected_group}"
    reweight_gamma_for_id="${FAIR_REC_REWEIGHT_WEIGHT:-$selected_gamma}"
    run_id="$(fare_run_id "$dataset" "$backbone" "$reweight_group_for_id" "$reweight_gamma_for_id")"

    if [[ "$SKIP_EXISTING" == "1" ]] && check_output_exists "$dataset" "$run_id"; then
      echo "[SKIP] Existing checkpoint: ${RESULT_ROOT}/${dataset}/fare/${run_id}/best_model.pt"
      continue
    fi

    run_fare_one "$dataset" "$backbone" "$run_id" "$ckpt" "$topk"
  done
done

echo
echo "========== FARE runs finished =========="
echo "RUN_TAG=$RUN_TAG"
