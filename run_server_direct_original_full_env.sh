#!/usr/bin/env bash
set -euo pipefail

# Supplement an existing Results6/Results7 comparison with the full direct
# original-LTS baseline. Stage 1 and the quotient-based main method are reused.

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PREPARED_DIR="${PREPARED_DIR:-results6/add_delete_prepared}"
COMPARE_ROOT="${COMPARE_ROOT:-results6/linear_ranker_ablation}"
CONTEXTUAL_NAME="${CONTEXTUAL_NAME:-lightweight_contextual_linear}"
DIRECT_ORIGINAL_NAME="${DIRECT_ORIGINAL_NAME:-direct_original_contextual_full}"
CONTEXTUAL_RANKER_MODEL="${CONTEXTUAL_RANKER_MODEL:-models/add_delete_ranker_lightweight_contextual_linear.pt}"

DEVICE="${DEVICE:-cuda}"
STRICT_DEVICE="${STRICT_DEVICE:-1}"
V_SIZES="${V_SIZES:-0,1,3,5}"
FORMULA_LIMIT="${FORMULA_LIMIT:-0}"
TARGET_STATE="${TARGET_STATE:--1}"
FAIR_MAX_CASE_SECONDS="${FAIR_MAX_CASE_SECONDS:-300}"

BEAM_WIDTH="${BEAM_WIDTH:-4}"
MAX_ITERS="${MAX_ITERS:-16}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-64}"
CANDIDATE_STATE_LIMIT="${CANDIDATE_STATE_LIMIT:-128}"
STATE_SCAN_LIMIT="${STATE_SCAN_LIMIT:-5000}"
SEARCH_STRATEGY="${SEARCH_STRATEGY:-beam}"
MINIMAL_LAYER_WIDTH="${MINIMAL_LAYER_WIDTH:-2048}"
MINIMAL_SEEN_LIMIT="${MINIMAL_SEEN_LIMIT:-500000}"
DYNAMIC_BUDGET_ROUNDS="${DYNAMIC_BUDGET_ROUNDS:-0}"
DYNAMIC_MAX_ITERS="${DYNAMIC_MAX_ITERS:-512}"
DYNAMIC_MAX_BEAM_WIDTH="${DYNAMIC_MAX_BEAM_WIDTH:-256}"
DYNAMIC_MAX_CANDIDATE_LIMIT="${DYNAMIC_MAX_CANDIDATE_LIMIT:-0}"
DYNAMIC_MAX_CANDIDATE_STATE_LIMIT="${DYNAMIC_MAX_CANDIDATE_STATE_LIMIT:-0}"
DYNAMIC_MAX_STATE_SCAN_LIMIT="${DYNAMIC_MAX_STATE_SCAN_LIMIT:-0}"
DYNAMIC_MAX_MINIMAL_LAYER_WIDTH="${DYNAMIC_MAX_MINIMAL_LAYER_WIDTH:-32768}"
DYNAMIC_MAX_MINIMAL_SEEN_LIMIT="${DYNAMIC_MAX_MINIMAL_SEEN_LIMIT:-500000}"
DYNAMIC_FINAL_SEARCH_STRATEGY="${DYNAMIC_FINAL_SEARCH_STRATEGY:-neural_guided_minimal}"
DRIFT_MODE="${DRIFT_MODE:-estimate}"
EXACT_DRIFT_MAX_TRANSITIONS="${EXACT_DRIFT_MAX_TRANSITIONS:-200000}"

DIRECT_ORIGINAL_LIMIT="${DIRECT_ORIGINAL_LIMIT:-0}"
DIRECT_ORIGINAL_MAX_STATES="${DIRECT_ORIGINAL_MAX_STATES:-0}"
DIRECT_ORIGINAL_MAX_TRANSITIONS="${DIRECT_ORIGINAL_MAX_TRANSITIONS:-0}"
DIRECT_ORIGINAL_RESUME="${DIRECT_ORIGINAL_RESUME:-1}"
DIRECT_ORIGINAL_SHARD_COUNT="${DIRECT_ORIGINAL_SHARD_COUNT:-1}"
DIRECT_ORIGINAL_SHARD_INDEX="${DIRECT_ORIGINAL_SHARD_INDEX:-0}"
DIRECT_ORIGINAL_PROGRESS_EVERY="${DIRECT_ORIGINAL_PROGRESS_EVERY:-1}"
RUN_DIRECT_STAGE2="${RUN_DIRECT_STAGE2:-1}"

if [[ -z "${DIRECT_ORIGINAL_FINALIZE+x}" ]]; then
  if [[ "${DIRECT_ORIGINAL_SHARD_COUNT}" == "1" ]]; then
    DIRECT_ORIGINAL_FINALIZE=1
  else
    DIRECT_ORIGINAL_FINALIZE=0
  fi
fi

FORCE_MATERIALIZE="${FORCE_MATERIALIZE:-0}"
MATERIALIZE_LIMIT="${MATERIALIZE_LIMIT:-0}"
STAGE3_PROGRESS_EVERY="${STAGE3_PROGRESS_EVERY:-100}"
STAGE3_CASE_START_EVERY="${STAGE3_CASE_START_EVERY:-1}"
STAGE3_CEX_ITERS="${STAGE3_CEX_ITERS:-16}"
STAGE3_CEX_LIFT_MODE="${STAGE3_CEX_LIFT_MODE:-closure}"
STAGE3_CEX_BATCH_SIZE="${STAGE3_CEX_BATCH_SIZE:-512}"
STAGE3_CASE_PROGRESS_EVERY="${STAGE3_CASE_PROGRESS_EVERY:-100}"
STAGE3_MAX_CASE_SECONDS="${STAGE3_MAX_CASE_SECONDS:-0}"
STAGE3_TRIM_MEMORY_EVERY_CASE="${STAGE3_TRIM_MEMORY_EVERY_CASE:-1}"

if [[ "${STRICT_DEVICE}" != "1" || "${DEVICE}" != cuda* ]]; then
  echo "ERROR: direct full baseline requires DEVICE=cuda* and STRICT_DEVICE=1." >&2
  exit 2
fi
if [[ ! -f "${PREPARED_DIR}/manifest.json" ]]; then
  echo "ERROR: prepared manifest not found: ${PREPARED_DIR}/manifest.json" >&2
  echo "Run Stage 1 first or set PREPARED_DIR to the existing prepared directory." >&2
  exit 2
fi
if [[ ! -d "${COMPARE_ROOT}/${CONTEXTUAL_NAME}" ]]; then
  echo "ERROR: main-method results not found: ${COMPARE_ROOT}/${CONTEXTUAL_NAME}" >&2
  echo "Run the quotient-based main method first or set COMPARE_ROOT correctly." >&2
  exit 2
fi

mkdir -p "${COMPARE_ROOT}" models

echo "== Full direct original-LTS supplemental baseline =="
echo "Prepared reuse: ${PREPARED_DIR}"
echo "Compare root:   ${COMPARE_ROOT}"
echo "Direct output:  ${COMPARE_ROOT}/${DIRECT_ORIGINAL_NAME}"
echo "Fair per-case:  ${FAIR_MAX_CASE_SECONDS}s"
echo "Full workload:  limit=${DIRECT_ORIGINAL_LIMIT} max_states=${DIRECT_ORIGINAL_MAX_STATES} max_transitions=${DIRECT_ORIGINAL_MAX_TRANSITIONS}"
echo "Resume:         ${DIRECT_ORIGINAL_RESUME}"
echo "Shard:          ${DIRECT_ORIGINAL_SHARD_INDEX}/${DIRECT_ORIGINAL_SHARD_COUNT}"
echo "Finalize:       ${DIRECT_ORIGINAL_FINALIZE}"

DIRECT_RESUME_ARGS=(--no-resume)
if [[ "${DIRECT_ORIGINAL_RESUME}" == "1" ]]; then
  DIRECT_RESUME_ARGS=(--resume)
fi

if [[ "${RUN_DIRECT_STAGE2}" == "1" ]]; then
  python -m svbr.experiments.direct_original_repair \
    --prepared-dir "${PREPARED_DIR}" \
    --results-root "${COMPARE_ROOT}/${DIRECT_ORIGINAL_NAME}" \
    --ranker neural \
    --ranker-model "${CONTEXTUAL_RANKER_MODEL}" \
    --device "${DEVICE}" \
    --strict-device \
    --v-sizes "${V_SIZES}" \
    --v-selection formula_safe \
    --formula-limit "${FORMULA_LIMIT}" \
    --limit "${DIRECT_ORIGINAL_LIMIT}" \
    --max-original-states "${DIRECT_ORIGINAL_MAX_STATES}" \
    --max-original-transitions "${DIRECT_ORIGINAL_MAX_TRANSITIONS}" \
    --target-state "${TARGET_STATE}" \
    --beam-width "${BEAM_WIDTH}" \
    --max-iters "${MAX_ITERS}" \
    --candidate-limit "${CANDIDATE_LIMIT}" \
    --candidate-state-limit "${CANDIDATE_STATE_LIMIT}" \
    --state-scan-limit "${STATE_SCAN_LIMIT}" \
    --search-strategy "${SEARCH_STRATEGY}" \
    --minimal-layer-width "${MINIMAL_LAYER_WIDTH}" \
    --minimal-seen-limit "${MINIMAL_SEEN_LIMIT}" \
    --dynamic-repair-budget \
    --dynamic-budget-rounds "${DYNAMIC_BUDGET_ROUNDS}" \
    --dynamic-max-iters "${DYNAMIC_MAX_ITERS}" \
    --dynamic-max-beam-width "${DYNAMIC_MAX_BEAM_WIDTH}" \
    --dynamic-max-candidate-limit "${DYNAMIC_MAX_CANDIDATE_LIMIT}" \
    --dynamic-max-candidate-state-limit "${DYNAMIC_MAX_CANDIDATE_STATE_LIMIT}" \
    --dynamic-max-state-scan-limit "${DYNAMIC_MAX_STATE_SCAN_LIMIT}" \
    --dynamic-max-minimal-layer-width "${DYNAMIC_MAX_MINIMAL_LAYER_WIDTH}" \
    --dynamic-max-minimal-seen-limit "${DYNAMIC_MAX_MINIMAL_SEEN_LIMIT}" \
    --dynamic-final-search-strategy "${DYNAMIC_FINAL_SEARCH_STRATEGY}" \
    --max-case-seconds "${FAIR_MAX_CASE_SECONDS}" \
    --drift-mode "${DRIFT_MODE}" \
    --exact-drift-max-transitions "${EXACT_DRIFT_MAX_TRANSITIONS}" \
    --progress-every "${DIRECT_ORIGINAL_PROGRESS_EVERY}" \
    --shard-count "${DIRECT_ORIGINAL_SHARD_COUNT}" \
    --shard-index "${DIRECT_ORIGINAL_SHARD_INDEX}" \
    "${DIRECT_RESUME_ARGS[@]}"
fi

if [[ "${DIRECT_ORIGINAL_FINALIZE}" == "1" ]]; then
  echo "== Stage 3: materialize direct-original scripts and verify original LTS =="
  PREPARED_DIR="${PREPARED_DIR}" RESULTS_ROOT="${COMPARE_ROOT}/${DIRECT_ORIGINAL_NAME}" \
  OUTPUT_RESULTS_ROOT="${COMPARE_ROOT}/${DIRECT_ORIGINAL_NAME}" TARGET_STATE="${TARGET_STATE}" \
  FORCE_MATERIALIZE="${FORCE_MATERIALIZE}" MATERIALIZE_LIMIT="${MATERIALIZE_LIMIT}" \
  STAGE3_CASE_START_EVERY="${STAGE3_CASE_START_EVERY}" STAGE3_CEX_ITERS="${STAGE3_CEX_ITERS}" \
  STAGE3_CEX_LIFT_MODE="${STAGE3_CEX_LIFT_MODE}" STAGE3_CEX_BATCH_SIZE="${STAGE3_CEX_BATCH_SIZE}" \
  STAGE3_CASE_PROGRESS_EVERY="${STAGE3_CASE_PROGRESS_EVERY}" STAGE3_MAX_CASE_SECONDS="${STAGE3_MAX_CASE_SECONDS}" \
  STAGE3_TRIM_MEMORY_EVERY_CASE="${STAGE3_TRIM_MEMORY_EVERY_CASE}" PROGRESS_EVERY="${STAGE3_PROGRESS_EVERY}" \
  bash scripts/materialize_repaired_aut.sh

  echo "== Compare direct-original full baseline with quotient main method =="
  FAIR_MAX_CASE_SECONDS="${FAIR_MAX_CASE_SECONDS}" python -m svbr.experiments.compare_rankers \
    --results-root "${COMPARE_ROOT}" \
    --left "${DIRECT_ORIGINAL_NAME}" \
    --right "${CONTEXTUAL_NAME}" \
    --output-prefix "direct_original_full_vs_quotient_contextual" \
    --pairing semantic

  FAIR_MAX_CASE_SECONDS="${FAIR_MAX_CASE_SECONDS}" python -m svbr.experiments.ablation_report \
    --results-root "${COMPARE_ROOT}" \
    --prepared-dir "${PREPARED_DIR}" \
    --output-prefix reviewer_ablation
else
  echo "== Finalize skipped =="
  echo "After every shard is complete, run:"
  echo "RUN_DIRECT_STAGE2=0 DIRECT_ORIGINAL_FINALIZE=1 bash run_server_direct_original_full_env.sh"
fi

echo "== Direct original-LTS supplemental baseline complete =="
