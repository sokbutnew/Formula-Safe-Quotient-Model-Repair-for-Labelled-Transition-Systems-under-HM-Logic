#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${RESULTS_ROOT:-results/add_delete_run}"
OUTPUT_RESULTS_ROOT="${OUTPUT_RESULTS_ROOT:-${RESULTS_ROOT}}"
PREPARED_DIR="${PREPARED_DIR:-results/add_delete_prepared}"
STAGE3_SCRIPT_LIST="${STAGE3_SCRIPT_LIST:-}"
FORCE_MATERIALIZE="${FORCE_MATERIALIZE:-0}"
LIMIT="${MATERIALIZE_LIMIT:-0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
STAGE3_CASE_START_EVERY="${STAGE3_CASE_START_EVERY:-1}"
TARGET_STATE="${TARGET_STATE:--1}"
STAGE3_CEX_ITERS="${STAGE3_CEX_ITERS:-16}"
STAGE3_CEX_LIFT_MODE="${STAGE3_CEX_LIFT_MODE:-closure}"
STAGE3_CEX_BATCH_SIZE="${STAGE3_CEX_BATCH_SIZE:-512}"
STAGE3_CASE_PROGRESS_EVERY="${STAGE3_CASE_PROGRESS_EVERY:-100}"
STAGE3_MAX_CASE_SECONDS="${STAGE3_MAX_CASE_SECONDS:-0}"
STAGE3_TRIM_MEMORY_EVERY_CASE="${STAGE3_TRIM_MEMORY_EVERY_CASE:-1}"

ARGS=(
  --results-root "${RESULTS_ROOT}"
  --output-results-root "${OUTPUT_RESULTS_ROOT}"
  --prepared-dir "${PREPARED_DIR}"
  --limit "${LIMIT}"
  --progress-every "${PROGRESS_EVERY}"
  --case-start-every "${STAGE3_CASE_START_EVERY}"
  --target-state "${TARGET_STATE}"
  --cex-lift-iters "${STAGE3_CEX_ITERS}"
  --cex-lift-mode "${STAGE3_CEX_LIFT_MODE}"
  --cex-batch-size "${STAGE3_CEX_BATCH_SIZE}"
  --case-progress-every "${STAGE3_CASE_PROGRESS_EVERY}"
  --max-case-seconds "${STAGE3_MAX_CASE_SECONDS}"
  --trim-memory-every-case "${STAGE3_TRIM_MEMORY_EVERY_CASE}"
)

if [[ -n "${STAGE3_SCRIPT_LIST}" ]]; then
  ARGS+=(--script-list "${STAGE3_SCRIPT_LIST}")
fi

if [[ "${FORCE_MATERIALIZE}" == "1" ]]; then
  ARGS+=(--force)
fi

python -m svbr.experiments.materialize_repaired_aut "${ARGS[@]}"
