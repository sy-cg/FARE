#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# scripts/run_all_fairness_baselines.sh
#
# Run fairness baselines on ID backbones:
#   - Adv-SASRec / Adv-GRU4Rec / Adv-BERT4Rec
#   - SASRec/GRU4Rec/BERT4Rec + PopReweight
#   - SASRec/GRU4Rec/BERT4Rec + FairRerank
#
# Requirements:
#   - ID backbone runs should already exist for Adv initialization and
#     FairRerank source top-k.
#
# Usage:
#   bash scripts/run_all_fairness_baselines.sh
#
# Useful overrides:
#   RUN_ADV=1 RUN_POP=1 RUN_RERANK=1 bash scripts/run_all_fairness_baselines.sh
#   RUN_ADV=0 bash scripts/run_all_fairness_baselines.sh
#   INIT_ADV_FROM_ID=0 bash scripts/run_all_fairness_baselines.sh
#   CANDIDATE_K=100 TOP_K=20 bash scripts/run_all_fairness_baselines.sh
#   DRY_RUN=1 bash scripts/run_all_fairness_baselines.sh
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

RUN_ADV="${RUN_ADV:-1}"
RUN_POP="${RUN_POP:-1}"
RUN_RERANK="${RUN_RERANK:-1}"
INIT_ADV_FROM_ID="${INIT_ADV_FROM_ID:-1}"

CANDIDATE_K="${CANDIDATE_K:-100}"
TOP_K="${TOP_K:-20}"
RERANK_SPLIT="${RERANK_SPLIT:-all}"

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

detect_adv_script() {
  local candidates=(
    "${ADV_SCRIPT:-}"
    "scripts/run_adv_backbone.py"
    "scripts/run_adv_baselines.py"
    "scripts/run_adversarial_backbone.py"
    "scripts/run_adv_id_backbone.py"
  )
  for s in "${candidates[@]}"; do
    [[ -n "$s" && -f "$s" ]] && echo "$s" && return 0
  done
  return 1
}

detect_adv_config() {
  local backbone="${1:-}"
  local candidates=(
    "${ADV_CONFIG:-}"
    "configs/adv_${backbone}_3090.yaml"
    "configs/adv_backbone_3090.yaml"
    "configs/adv_3090.yaml"
    "configs/adversarial_3090.yaml"
  )
  for c in "${candidates[@]}"; do
    [[ -n "$c" && -f "$c" ]] && echo "$c" && return 0
  done
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

echo "========== Run all fairness baselines =========="
echo "root:          $ROOT_DIR"
echo "datasets:      ${DATASET_LIST[*]}"
echo "backbones:     ${BACKBONE_LIST[*]}"
echo "run_tag:       $RUN_TAG"
echo "run_adv:       $RUN_ADV"
echo "run_pop:       $RUN_POP"
echo "run_rerank:    $RUN_RERANK"
echo "init_adv_id:   $INIT_ADV_FROM_ID"
echo "result_root:   $RESULT_ROOT"
echo

# ----------------------------------------------------------
# Adv-* baselines
# ----------------------------------------------------------
if [[ "$RUN_ADV" == "1" ]]; then
  adv_script="$(detect_adv_script || true)"
  if [[ -z "${adv_script:-}" ]]; then
    echo "[WARN] Adv baseline script not found. Skipping Adv-*."
    echo "       Expected one of scripts/run_adv_backbone.py, scripts/run_adv_baselines.py, etc."
  else
    echo "[INFO] Adv script: $adv_script"

    for dataset in "${DATASET_LIST[@]}"; do
      for backbone in "${BACKBONE_LIST[@]}"; do
        adv_config="$(detect_adv_config "$backbone" || true)"
        if [[ -z "${adv_config:-}" ]]; then
          echo "[WARN] Adv config not found for backbone=${backbone}. Skipping ${dataset}/${backbone}."
          continue
        fi

        echo "[INFO] Adv config for ${backbone}: $adv_config"

        run_id="adv_${backbone}_${dataset}_${RUN_TAG}"
        out_dir="${RESULT_ROOT}/${dataset}/adv_${backbone}/${run_id}"

        if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/best_model.pt" ]]; then
          echo "[SKIP] Existing Adv checkpoint: ${out_dir}/best_model.pt"
          continue
        fi

        args=(python "$adv_script" --dataset "$dataset" --config "$adv_config" --run_id "$run_id")

        if script_accepts_arg "$adv_script" "--backbone"; then
          args+=(--backbone "$backbone")
        elif script_accepts_arg "$adv_script" "--model"; then
          args+=(--model "$backbone")
        else
          echo "[WARN] $adv_script has neither --backbone nor --model; using ADV_EXTRA_ARGS only."
        fi

        if [[ "$INIT_ADV_FROM_ID" == "1" ]]; then
          id_run_name="$(run_name_for_backbone "$backbone")"
          init_ckpt="$(find_latest_checkpoint "${RESULT_ROOT}/${dataset}/${id_run_name}" || true)"
          if [[ -z "$init_ckpt" ]]; then
            echo "[WARN] Missing ID checkpoint for Adv ${dataset}/${backbone}: ${RESULT_ROOT}/${dataset}/${id_run_name}. Skipping."
            continue
          fi
          if script_accepts_arg "$adv_script" "--init_backbone_checkpoint"; then
            args+=(--init_backbone_checkpoint "$init_ckpt")
          else
            echo "[WARN] $adv_script does not support --init_backbone_checkpoint. Adv will train from scratch."
          fi
        fi

        if [[ "$SMOKE" == "1" ]]; then
          args+=(--epochs "${SMOKE_EPOCHS:-2}" --eval_every 1 --patience 2)
        fi

        if [[ -n "${ADV_EXTRA_ARGS:-}" ]]; then
          # shellcheck disable=SC2206
          extra=( $ADV_EXTRA_ARGS )
          args+=("${extra[@]}")
        fi

        run_cmd "${args[@]}"
      done
    done
  fi
fi

# ----------------------------------------------------------
# PopReweight baselines
# ----------------------------------------------------------
if [[ "$RUN_POP" == "1" ]]; then
  pop_script="scripts/run_pop_reweight.py"
  pop_config="${POP_CONFIG:-configs/pop_reweight_3090.yaml}"

  [[ -f "$pop_script" ]] || { echo "[ERROR] Missing $pop_script" >&2; exit 1; }
  [[ -f "$pop_config" ]] || { echo "[ERROR] Missing $pop_config" >&2; exit 1; }

  for dataset in "${DATASET_LIST[@]}"; do
    for backbone in "${BACKBONE_LIST[@]}"; do
      run_id="${backbone}_pop_reweight_${dataset}_${RUN_TAG}"
      out_dir="${RESULT_ROOT}/${dataset}/${backbone}_pop_reweight/${run_id}"

      if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/best_model.pt" ]]; then
        echo "[SKIP] Existing PopReweight checkpoint: ${out_dir}/best_model.pt"
        continue
      fi

      args=(python "$pop_script" --dataset "$dataset" --config "$pop_config" --run_id "$run_id")
      if script_accepts_arg "$pop_script" "--backbone"; then
        args+=(--backbone "$backbone")
      else
        args+=(--model "$backbone")
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
fi

# ----------------------------------------------------------
# FairRerank baselines
# ----------------------------------------------------------
if [[ "$RUN_RERANK" == "1" ]]; then
  rerank_script="scripts/run_reranker_fair.py"
  rerank_config="${RERANK_CONFIG:-configs/reranker_fair_3090.yaml}"

  [[ -f "$rerank_script" ]] || { echo "[ERROR] Missing $rerank_script" >&2; exit 1; }
  [[ -f "$rerank_config" ]] || { echo "[ERROR] Missing $rerank_config" >&2; exit 1; }

  for dataset in "${DATASET_LIST[@]}"; do
    for backbone in "${BACKBONE_LIST[@]}"; do
      run_name="$(run_name_for_backbone "$backbone")"
      source_parent="${RESULT_ROOT}/${dataset}/${run_name}"
      source_run_dir="$(find_latest_run_dir "$source_parent" || true)"

      if [[ -z "$source_run_dir" || ! -f "${source_run_dir}/topk_test.npz" ]]; then
        echo "[WARN] Missing source topk_test.npz for ${dataset}/${backbone}: $source_parent. Skipping FairRerank."
        continue
      fi

      run_id="${backbone}_fair_rerank_${dataset}_${RUN_TAG}"
      out_dir="${RESULT_ROOT}/${dataset}/${backbone}_fair_rerank/${run_id}"

      if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/topk_test.npz" ]]; then
        echo "[SKIP] Existing FairRerank topk: ${out_dir}/topk_test.npz"
        continue
      fi

      args=(
        python "$rerank_script"
        --dataset "$dataset"
        --config "$rerank_config"
        --backbone "$backbone"
        --source_run_dir "$source_run_dir"
        --split "$RERANK_SPLIT"
        --candidate_k "$CANDIDATE_K"
        --top_k "$TOP_K"
        --run_id "$run_id"
      )

      if [[ -n "${RERANK_EXTRA_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        extra=( $RERANK_EXTRA_ARGS )
        args+=("${extra[@]}")
      fi

      run_cmd "${args[@]}"
    done
  done
fi

echo
echo "========== Fairness baseline runs finished =========="
echo "RUN_TAG=$RUN_TAG"
