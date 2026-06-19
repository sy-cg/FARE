#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_STR="${DATASETS:-Baby_Products Musical_Instruments Video_Games}"
BACKBONES_STR="${BACKBONES:-gru4rec bert4rec}"

declare -A BEST_GROUP
declare -A BEST_GAMMA

BEST_GROUP["Baby_Products"]="popularity_group"
BEST_GAMMA["Baby_Products"]="0.20"

BEST_GROUP["Musical_Instruments"]="multimodal_cluster_proxy_group"
BEST_GAMMA["Musical_Instruments"]="0.20"

BEST_GROUP["Video_Games"]="popularity_group"
BEST_GAMMA["Video_Games"]="0.20"

SCRIPT="${SCRIPT:-scripts/run_fare.py}"
DEFAULT_CONFIG="${DEFAULT_CONFIG:-configs/fare_3090.yaml}"
GRU1_CONFIG="${GRU1_CONFIG:-configs/fare_3090_gru1.yaml}"
CONFIG="${CONFIG:-}"
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
read -r -a BACKBONES <<< "$BACKBONES_STR"

find_id_run_dir() {
  local dataset="$1"
  local backbone="$2"
  local base="${RESULT_ROOT}/${dataset}/${backbone}_id"

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

config_for_backbone() {
  local backbone="$1"
  if [[ -n "$CONFIG" ]]; then
    echo "$CONFIG"
  elif [[ "$backbone" == "gru4rec" ]]; then
    echo "$GRU1_CONFIG"
  else
    echo "$DEFAULT_CONFIG"
  fi
}

run_id_for_backbone() {
  local dataset="$1"
  local backbone="$2"
  local group_tag="$3"
  local gamma_tag="$4"
  local clip_tag="$5"
  if [[ "$backbone" == "gru4rec" ]]; then
    echo "fare_exposure_fairfirst_gru1_${dataset}_${backbone}_${group_tag}_g${gamma_tag}_k${EXPOSURE_K}_${EXPOSURE_TARGET}_${clip_tag}_v1"
  else
    echo "fare_exposure_fairfirst_${dataset}_${backbone}_${group_tag}_g${gamma_tag}_k${EXPOSURE_K}_${EXPOSURE_TARGET}_${clip_tag}_v1"
  fi
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
[[ -f "$DEFAULT_CONFIG" ]] || { echo "[ERROR] Missing $DEFAULT_CONFIG" >&2; exit 1; }
[[ -f "$GRU1_CONFIG" ]] || { echo "[ERROR] Missing $GRU1_CONFIG" >&2; exit 1; }
if [[ -n "$CONFIG" && ! -f "$CONFIG" ]]; then
  echo "[ERROR] Missing $CONFIG" >&2
  exit 1
fi

for dataset in "${DATASETS[@]}"; do
  group="${BEST_GROUP[$dataset]}"
  gamma="${BEST_GAMMA[$dataset]}"
  group_tag=$(group_to_tag "$group")
  gamma_tag=$(value_to_tag "$gamma")
  clip_tag=$(clip_to_tag "$REWEIGHT_MIN" "$REWEIGHT_MAX")

  for backbone in "${BACKBONES[@]}"; do
    RUN_DIR=$(find_id_run_dir "$dataset" "$backbone")

    if [[ -z "$RUN_DIR" ]]; then
      echo "[ERROR] Cannot find ${backbone}-ID run dir with both best_model.pt and topk_val.npz for ${dataset}" >&2
      echo "[ERROR] Please generate ${backbone}-ID topk_val.npz first." >&2
      exit 1
    fi

    config=$(config_for_backbone "$backbone")
    [[ -f "$config" ]] || { echo "[ERROR] Missing $config" >&2; exit 1; }

    CKPT="${RUN_DIR}/best_model.pt"
    TOPK_VAL="${RUN_DIR}/topk_val.npz"
    RUN_ID=$(run_id_for_backbone "$dataset" "$backbone" "$group_tag" "$gamma_tag" "$clip_tag")

    if [[ "$SKIP_EXISTING" == "1" ]] && already_done "$dataset" "$RUN_ID"; then
      echo "[SKIP] ${dataset} ${backbone} ${group_tag} gamma=${gamma} already exists."
      continue
    fi

    cmd=(
      python "$SCRIPT"
      --config "$config"
      --dataset "$dataset"
      --backbone "$backbone"
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
    echo "backbone=${backbone}"
    echo "config=${config}"
    echo "group=${group}"
    echo "gamma=${gamma}"
    echo "checkpoint=${CKPT}"
    echo "topk_val=${TOPK_VAL}"

    run_cmd "${cmd[@]}"
  done
done

echo "========== FARE Exposure fairness-first transfer finished =========="
