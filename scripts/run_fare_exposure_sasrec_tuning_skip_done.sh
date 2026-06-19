#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_STR="${DATASETS:-Baby_Products Musical_Instruments Video_Games}"
EXPOSURE_GROUPS_STR="${EXPOSURE_GROUPS:-popularity_group multimodal_cluster_proxy_group}"
GAMMAS_STR="${GAMMAS:-0.05 0.10 0.20}"

SCRIPT="${SCRIPT:-scripts/run_fare.py}"
CONFIG="${CONFIG:-configs/fare_3090.yaml}"
RESULT_ROOT="${RESULT_ROOT:-results}"

LR="${LR:-0.0003}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"

RESIDUAL_SCORE_WEIGHT="${RESIDUAL_SCORE_WEIGHT:-1.0}"
FAIR_WEIGHT_INIT="${FAIR_WEIGHT_INIT:-0.01}"
MAX_FAIR_WEIGHT="${MAX_FAIR_WEIGHT:-0.1}"

EXPOSURE_K="${EXPOSURE_K:-10}"
EXPOSURE_TARGET="${EXPOSURE_TARGET:-catalog}"
REWEIGHT_MIN="${REWEIGHT_MIN:-0.5}"
REWEIGHT_MAX="${REWEIGHT_MAX:-2.0}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SMOKE="${SMOKE:-0}"

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a EXPOSURE_GROUPS <<< "$EXPOSURE_GROUPS_STR"
read -r -a GAMMAS <<< "$GAMMAS_STR"

find_sasrec_run_dir() {
  local dataset="$1"
  local base="${RESULT_ROOT}/${dataset}/sasrec_id"

  if [[ ! -d "$base" ]]; then
    echo ""
    return 1
  fi

  local dirs
  dirs=$(find "$base" -mindepth 1 -maxdepth 1 -type d | sort -r)

  for d in $dirs; do
    if [[ -f "${d}/best_model.pt" && -f "${d}/topk_val.npz" ]]; then
      echo "$d"
      return 0
    fi
  done

  echo ""
  return 1
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

run_cmd() {
  echo
  echo "[CMD] $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

already_done() {
  local dataset="$1"
  local run_id="$2"
  [[ -f "${RESULT_ROOT}/${dataset}/fare/${run_id}/metrics_summary.json" ]]
}

[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing $SCRIPT" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "[ERROR] Missing $CONFIG" >&2; exit 1; }

for dataset in "${DATASETS[@]}"; do
  RUN_DIR=$(find_sasrec_run_dir "$dataset")

  if [[ -z "$RUN_DIR" ]]; then
    echo "[ERROR] Cannot find SASRec-ID run dir with both best_model.pt and topk_val.npz for ${dataset}" >&2
    echo "[ERROR] Please generate topk_val.npz first." >&2
    exit 1
  fi

  CKPT="${RUN_DIR}/best_model.pt"
  TOPK_VAL="${RUN_DIR}/topk_val.npz"

  echo "============================================================"
  echo "Dataset: ${dataset}"
  echo "Checkpoint: ${CKPT}"
  echo "Validation exposure: ${TOPK_VAL}"
  echo "============================================================"

  for group in "${EXPOSURE_GROUPS[@]}"; do
    group_tag=$(group_to_tag "$group")

    for gamma in "${GAMMAS[@]}"; do
      gamma_tag=$(value_to_tag "$gamma")
      clip_tag=$(clip_to_tag "$REWEIGHT_MIN" "$REWEIGHT_MAX")
      RUN_ID="fare_exposure_tune_${dataset}_sasrec_${group_tag}_g${gamma_tag}_k${EXPOSURE_K}_${EXPOSURE_TARGET}_${clip_tag}_v1"

      if [[ "$SKIP_EXISTING" == "1" ]] && already_done "$dataset" "$RUN_ID"; then
        echo "[SKIP] ${dataset} ${group_tag} gamma=${gamma} already exists."
        continue
      fi

      cmd=(
        python "$SCRIPT"
        --config "$CONFIG"
        --dataset "$dataset"
        --backbone sasrec
        --init_backbone_checkpoint "$CKPT"
        --freeze_id_backbone
        --run_id "$RUN_ID"
        --method_name FARE
        --lr "$LR"
        --fair_weight_init "$FAIR_WEIGHT_INIT"
        --max_fair_weight "$MAX_FAIR_WEIGHT"
        --residual_score_weight "$RESIDUAL_SCORE_WEIGHT"
        --fair_rec_reweight_weight "$gamma"
        --fair_rec_reweight_groups "$group"
        --fair_rec_exposure_topk_path "$TOPK_VAL"
        --fair_rec_exposure_target "$EXPOSURE_TARGET"
        --fair_rec_exposure_k "$EXPOSURE_K"
        --fair_rec_reweight_min "$REWEIGHT_MIN"
        --fair_rec_reweight_max "$REWEIGHT_MAX"
        --batch_size "$BATCH_SIZE"
        --eval_batch_size "$EVAL_BATCH_SIZE"
      )

      if [[ "$SMOKE" == "1" ]]; then
        cmd+=(--epochs "${SMOKE_EPOCHS:-2}" --eval_every 1 --patience 2)
      fi

      echo ""
      echo "========== Running ${RUN_ID} =========="
      echo "dataset=${dataset}"
      echo "group=${group}"
      echo "gamma=${gamma}"
      echo "checkpoint=${CKPT}"
      echo "topk_val=${TOPK_VAL}"

      run_cmd "${cmd[@]}"
    done
  done
done

echo "========== FARE Exposure SASRec tuning finished =========="
