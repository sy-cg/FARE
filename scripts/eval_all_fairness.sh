#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# scripts/eval_all_fairness.sh
#
# Evaluate fairness/diversity for every run directory that contains topk files.
#
# Default:
#   - scans results/<Dataset>/<ModelName>/<RunId>/topk_test.npz
#     or test_ranking.npz
#   - runs scripts/eval_fairness.py --append_global
#
# Usage:
#   bash scripts/eval_all_fairness.sh
#
# Useful overrides:
#   SPLITS="test" bash scripts/eval_all_fairness.sh
#   SPLITS="val test" bash scripts/eval_all_fairness.sh
#   RESET_FAIRNESS=1 bash scripts/eval_all_fairness.sh
#   DRY_RUN=1 bash scripts/eval_all_fairness.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS_STR="${DATASETS:-Video_Games Musical_Instruments Baby_Products}"
RESULT_ROOT="${RESULT_ROOT:-results}"
SPLITS_STR="${SPLITS:-test}"
FAIRNESS_GROUPS="${FAIRNESS_GROUPS:-all}"
APPEND_GLOBAL="${APPEND_GLOBAL:-1}"
RESET_FAIRNESS="${RESET_FAIRNESS:-0}"
DRY_RUN="${DRY_RUN:-0}"
K_LIST="${K_LIST:-5,10,20}"

SCRIPT="${SCRIPT:-scripts/eval_fairness.py}"
FAIRNESS_CSV="${FAIRNESS_CSV:-${RESULT_ROOT}/fairness.csv}"

read -r -a DATASET_LIST <<< "$DATASETS_STR"
read -r -a SPLIT_LIST <<< "$SPLITS_STR"

run_cmd() {
  echo
  echo "[CMD] $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

script_accepts_arg() {
  local script="$1"
  local arg="$2"
  python "$script" --help 2>/dev/null | grep -q -- "$arg"
}

[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing $SCRIPT" >&2; exit 1; }

if [[ "$RESET_FAIRNESS" == "1" && -f "$FAIRNESS_CSV" ]]; then
  backup="${FAIRNESS_CSV}.bak.$(date +%Y%m%d_%H%M%S)"
  echo "[INFO] Backing up existing fairness CSV: $backup"
  cp "$FAIRNESS_CSV" "$backup"
  rm -f "$FAIRNESS_CSV"
fi

echo "========== Evaluate all fairness metrics =========="
echo "root:          $ROOT_DIR"
echo "datasets:      ${DATASET_LIST[*]}"
echo "splits:        ${SPLIT_LIST[*]}"
echo "groups:        $FAIRNESS_GROUPS"
echo "result_root:   $RESULT_ROOT"
echo "append_global: $APPEND_GLOBAL"
echo "fairness_csv:  $FAIRNESS_CSV"
echo

for dataset in "${DATASET_LIST[@]}"; do
  dataset_dir="${RESULT_ROOT}/${dataset}"
  if [[ ! -d "$dataset_dir" ]]; then
    echo "[WARN] Missing dataset result dir: $dataset_dir"
    continue
  fi

  while IFS= read -r -d '' run_dir; do
    for split in "${SPLIT_LIST[@]}"; do
      topk_file="${run_dir}/topk_${split}.npz"
      ranking_file="${run_dir}/${split}_ranking.npz"
      if [[ ! -f "$topk_file" && ! -f "$ranking_file" ]]; then
        continue
      fi

      args=(
        python "$SCRIPT"
        --dataset "$dataset"
        --run_dir "$run_dir"
        --split "$split"
        --groups "$FAIRNESS_GROUPS"
      )

      if [[ "$APPEND_GLOBAL" == "1" ]]; then
        args+=(--append_global)
      fi

      # Support either --ks or --k if your eval_fairness.py exposes it.
      if script_accepts_arg "$SCRIPT" "--ks"; then
        args+=(--ks "$K_LIST")
      fi

      run_cmd "${args[@]}"
    done
  done < <(find "$dataset_dir" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z)
done

echo
echo "========== Fairness evaluation finished =========="
echo "fairness_csv: $FAIRNESS_CSV"
