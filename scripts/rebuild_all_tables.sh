#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# scripts/rebuild_all_tables.sh
#
# Rebuild all result tables for k = 5, 10, 20.
#
# Outputs:
#   results/tables/overall_k5.csv
#   results/tables/overall_k10.csv
#   results/tables/overall_k20.csv
#
#   results/tables/compact_comparison_k5.csv/.md
#   results/tables/compact_comparison_k10.csv/.md
#   results/tables/compact_comparison_k20.csv/.md
#
#   results/tables/accuracy_fairness_table_k5.csv/.md
#   results/tables/accuracy_fairness_table_k10.csv/.md
#   results/tables/accuracy_fairness_table_k20.csv/.md
#
# Usage:
#   bash scripts/rebuild_all_tables.sh
#
# Useful overrides:
#   K_LIST="5 10 20" bash scripts/rebuild_all_tables.sh
#   SPLIT=test bash scripts/rebuild_all_tables.sh
#   DATASETS="Video_Games,Musical_Instruments,Baby_Products" bash scripts/rebuild_all_tables.sh
#   DISPLAY_NAMES=1 bash scripts/rebuild_all_tables.sh
#   DRY_RUN=1 bash scripts/rebuild_all_tables.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULT_ROOT="${RESULT_ROOT:-results}"
TABLE_DIR="${TABLE_DIR:-${RESULT_ROOT}/tables}"
K_LIST_STR="${K_LIST:-5 10 20}"
SPLIT="${SPLIT:-test}"
DATASETS_CSV="${DATASETS_CSV:-Video_Games,Musical_Instruments,Baby_Products}"
BASELINE_METHOD="${BASELINE_METHOD:-SASRec-ID}"
ADD_DELTA="${ADD_DELTA:-1}"
DELTA_BY_BACKBONE="${DELTA_BY_BACKBONE:-1}"
DISPLAY_NAMES="${DISPLAY_NAMES:-0}"
DRY_RUN="${DRY_RUN:-0}"
DECIMALS="${DECIMALS:-6}"

APPEND_SCRIPT="${APPEND_SCRIPT:-scripts/append_id_backbone_results.py}"
COMPACT_SCRIPT="${COMPACT_SCRIPT:-scripts/build_compact_table.py}"
ACC_FAIR_SCRIPT="${ACC_FAIR_SCRIPT:-scripts/build_accuracy_fairness_table.py}"

FAIRNESS_CSV="${FAIRNESS_CSV:-${RESULT_ROOT}/fairness.csv}"

read -r -a K_LIST_ARRAY <<< "$K_LIST_STR"

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

[[ -f "$APPEND_SCRIPT" ]] || { echo "[ERROR] Missing $APPEND_SCRIPT" >&2; exit 1; }
[[ -f "$COMPACT_SCRIPT" ]] || { echo "[ERROR] Missing $COMPACT_SCRIPT" >&2; exit 1; }
[[ -f "$ACC_FAIR_SCRIPT" ]] || { echo "[ERROR] Missing $ACC_FAIR_SCRIPT" >&2; exit 1; }
[[ -f "$FAIRNESS_CSV" ]] || { echo "[ERROR] Missing fairness CSV: $FAIRNESS_CSV. Run scripts/eval_all_fairness.sh first." >&2; exit 1; }

mkdir -p "$TABLE_DIR"

echo "========== Rebuild all tables =========="
echo "root:              $ROOT_DIR"
echo "result_root:       $RESULT_ROOT"
echo "table_dir:         $TABLE_DIR"
echo "k_list:            ${K_LIST_ARRAY[*]}"
echo "split:             $SPLIT"
echo "datasets:          $DATASETS_CSV"
echo "baseline_method:   $BASELINE_METHOD"
echo "add_delta:         $ADD_DELTA"
echo "delta_by_backbone: $DELTA_BY_BACKBONE"
echo "display_names:     $DISPLAY_NAMES"
echo

for k in "${K_LIST_ARRAY[@]}"; do
  overall_csv="${TABLE_DIR}/overall_k${k}.csv"
  compact_csv="${TABLE_DIR}/compact_comparison_k${k}.csv"
  compact_md="${TABLE_DIR}/compact_comparison_k${k}.md"
  accfair_csv="${TABLE_DIR}/accuracy_fairness_table_k${k}.csv"
  accfair_md="${TABLE_DIR}/accuracy_fairness_table_k${k}.md"

  # --------------------------------------------------------
  # 1. Rebuild overall accuracy table for this k
  # --------------------------------------------------------
  append_args=(
    python "$APPEND_SCRIPT"
    --results_root "$RESULT_ROOT"
    --output "$overall_csv"
    --split "$SPLIT"
    --k "$k"
  )

  if script_accepts_arg "$APPEND_SCRIPT" "--dedup"; then
    append_args+=(--dedup)
  fi
  if script_accepts_arg "$APPEND_SCRIPT" "--backup"; then
    append_args+=(--backup)
  fi

  run_cmd "${append_args[@]}"

  # --------------------------------------------------------
  # 2. Rebuild compact fairness/diversity table for this k
  # --------------------------------------------------------
  compact_args=(
    python "$COMPACT_SCRIPT"
    --input "$FAIRNESS_CSV"
    --output "$compact_csv"
    --markdown "$compact_md"
    --dataset "$DATASETS_CSV"
    --split "$SPLIT"
    --k "$k"
    --decimals "$DECIMALS"
  )

  if [[ "$ADD_DELTA" == "1" ]]; then
    compact_args+=(--add_delta --baseline_method "$BASELINE_METHOD")
  fi
  if [[ "$DELTA_BY_BACKBONE" == "1" ]] && script_accepts_arg "$COMPACT_SCRIPT" "--delta_by_backbone"; then
    compact_args+=(--delta_by_backbone)
  fi
  if [[ "$DISPLAY_NAMES" == "1" ]]; then
    compact_args+=(--display_names)
  fi

  run_cmd "${compact_args[@]}"

  # --------------------------------------------------------
  # 3. Rebuild merged accuracy+fairness table for this k
  # --------------------------------------------------------
  accfair_args=(
    python "$ACC_FAIR_SCRIPT"
    --overall "$overall_csv"
    --fairness "$FAIRNESS_CSV"
    --output "$accfair_csv"
    --markdown "$accfair_md"
    --dataset "$DATASETS_CSV"
    --split "$SPLIT"
    --k "$k"
    --decimals "$DECIMALS"
  )

  if [[ "$ADD_DELTA" == "1" ]]; then
    accfair_args+=(--add_delta --baseline_method "$BASELINE_METHOD")
  fi
  if [[ "$DELTA_BY_BACKBONE" == "1" ]] && script_accepts_arg "$ACC_FAIR_SCRIPT" "--delta_by_backbone"; then
    accfair_args+=(--delta_by_backbone)
  fi
  if [[ "$DISPLAY_NAMES" == "1" ]]; then
    accfair_args+=(--display_names)
  fi

  run_cmd "${accfair_args[@]}"
done

# Convenience latest pointers.
if [[ "$DRY_RUN" != "1" ]]; then
  cp -f "${TABLE_DIR}/overall_k10.csv" "${RESULT_ROOT}/overall.csv" 2>/dev/null || true
  cp -f "${TABLE_DIR}/compact_comparison_k10.csv" "${RESULT_ROOT}/compact_comparison.csv" 2>/dev/null || true
  cp -f "${TABLE_DIR}/compact_comparison_k10.md" "${RESULT_ROOT}/compact_comparison.md" 2>/dev/null || true
  cp -f "${TABLE_DIR}/accuracy_fairness_table_k10.csv" "${RESULT_ROOT}/accuracy_fairness_table.csv" 2>/dev/null || true
  cp -f "${TABLE_DIR}/accuracy_fairness_table_k10.md" "${RESULT_ROOT}/accuracy_fairness_table.md" 2>/dev/null || true
fi

echo
echo "========== Tables rebuilt =========="
echo "table_dir: $TABLE_DIR"
echo "Default convenience copies use k=10:"
echo "  ${RESULT_ROOT}/overall.csv"
echo "  ${RESULT_ROOT}/compact_comparison.csv"
echo "  ${RESULT_ROOT}/accuracy_fairness_table.csv"
