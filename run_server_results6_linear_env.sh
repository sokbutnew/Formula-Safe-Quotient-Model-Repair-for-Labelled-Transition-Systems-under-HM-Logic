#!/usr/bin/env bash
set -euo pipefail

# Experiment 6 end-to-end comparison.
# Stage 1 prepares LTS' and formulas once. Stage 2/3 then run:
#   1) heuristic_baseline: deterministic heuristic candidate order
#   2) random_baseline: random candidate order
#   3) legacy8_ablation: older 8-feature linear ranker
#   4) fixed_budget_contextual: 27-feature ranker without dynamic budget
#   5) unsafe_v_contextual: no formula-safe filtering
#   6) add_only_contextual/delete_only_contextual: grammar ablations
#   7) direct_original_contextual_full: full direct original-LTS repair baseline
#   8) lightweight_contextual_linear: experiment-6 27-feature linear ranker
#
# This script intentionally does not run MLP/GNN.

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

AUT_DIR="${AUT_DIR:-/root/sj-tmp/data/download}"
PREPARED_DIR="${PREPARED_DIR:-results6/add_delete_prepared}"
COMPARE_ROOT="${COMPARE_ROOT:-results6/linear_ranker_ablation}"
LIMIT="${LIMIT:-50}"
MAX_STATES="${MAX_STATES:-1000000}"
MAX_TRANSITIONS="${MAX_TRANSITIONS:-5000000}"
DEVICE="${DEVICE:-cuda}"
STRICT_DEVICE="${STRICT_DEVICE:-1}"
QUOTIENT_DEVICE="${QUOTIENT_DEVICE:-cuda}"
TARGET_STATE="${TARGET_STATE:--1}"

V_SIZES="${V_SIZES:-0,1,3,5}"
FORMULAS_PER_MODEL="${FORMULAS_PER_MODEL:-30}"
KNOWN_FORMULA_COUNT="${KNOWN_FORMULA_COUNT:-20}"
MIXED_FORMULA_COUNT="${MIXED_FORMULA_COUNT:-10}"
EASY_FORMULA_COUNT="${EASY_FORMULA_COUNT:-5}"
MEDIUM_FORMULA_COUNT="${MEDIUM_FORMULA_COUNT:-10}"
HARD_FORMULA_COUNT="${HARD_FORMULA_COUNT:-15}"
FORMULA_MIN_ACTIONS="${FORMULA_MIN_ACTIONS:-5}"
FORMULA_MAX_ACTIONS="${FORMULA_MAX_ACTIONS:-10}"
MIN_UNSATISFIED_FORMULAS="${MIN_UNSATISFIED_FORMULAS:-30}"
FORMULA_SEED="${FORMULA_SEED:-13}"
FORMULA_LIMIT="${FORMULA_LIMIT:-0}"

EXPERIMENT_PROFILE="${EXPERIMENT_PROFILE:-ranker-add-delete}"
CONTEXTUAL_NAME="${CONTEXTUAL_NAME:-lightweight_contextual_linear}"
CONTEXTUAL_RANKER_MODEL="${CONTEXTUAL_RANKER_MODEL:-models/add_delete_ranker_lightweight_contextual_linear.pt}"
LEGACY8_NAME="${LEGACY8_NAME:-legacy8_ablation}"
LEGACY8_RANKER_MODEL="${LEGACY8_RANKER_MODEL:-models/add_delete_ranker_linear_legacy8.pt}"
HEURISTIC_NAME="${HEURISTIC_NAME:-heuristic_baseline}"
RANDOM_NAME="${RANDOM_NAME:-random_baseline}"
FIXED_BUDGET_NAME="${FIXED_BUDGET_NAME:-fixed_budget_contextual}"
UNSAFE_V_NAME="${UNSAFE_V_NAME:-unsafe_v_contextual}"
ADD_ONLY_NAME="${ADD_ONLY_NAME:-add_only_contextual}"
DELETE_ONLY_NAME="${DELETE_ONLY_NAME:-delete_only_contextual}"
DIRECT_ORIGINAL_NAME="${DIRECT_ORIGINAL_NAME:-direct_original_contextual_full}"

RANKER_TRAIN_SAMPLES="${RANKER_TRAIN_SAMPLES:-20000}"
RANKER_TRAIN_FORMULA_LIMIT="${RANKER_TRAIN_FORMULA_LIMIT:-30}"
RANKER_TRAIN_CANDIDATE_LIMIT="${RANKER_TRAIN_CANDIDATE_LIMIT:-128}"
RANKER_EPOCHS="${RANKER_EPOCHS:-40}"
RANKER_LR="${RANKER_LR:-0.003}"
FORCE_RANKER_TRAIN="${FORCE_RANKER_TRAIN:-0}"

BEAM_WIDTH="${BEAM_WIDTH:-4}"
MAX_ITERS="${MAX_ITERS:-16}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-64}"
CANDIDATE_STATE_LIMIT="${CANDIDATE_STATE_LIMIT:-128}"
STATE_SCAN_LIMIT="${STATE_SCAN_LIMIT:-5000}"
SEARCH_STRATEGY="${SEARCH_STRATEGY:-beam}"
MINIMAL_LAYER_WIDTH="${MINIMAL_LAYER_WIDTH:-2048}"
MINIMAL_SEEN_LIMIT="${MINIMAL_SEEN_LIMIT:-500000}"
FAIR_MAX_CASE_SECONDS="${FAIR_MAX_CASE_SECONDS:-300}"
STAGE2_MAX_CASE_SECONDS="${STAGE2_MAX_CASE_SECONDS:-${FAIR_MAX_CASE_SECONDS}}"
DYNAMIC_REPAIR_BUDGET="${DYNAMIC_REPAIR_BUDGET:-1}"
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
STAGE2_MAX_STATES="${STAGE2_MAX_STATES:-${MAX_STATES}}"
STAGE2_MAX_TRANSITIONS="${STAGE2_MAX_TRANSITIONS:-${MAX_TRANSITIONS}}"
STAGE2_PROGRESS_EVERY="${STAGE2_PROGRESS_EVERY:-1}"
STAGE2_CASE_PROGRESS_EVERY="${STAGE2_CASE_PROGRESS_EVERY:-0}"
TRIM_MEMORY_EVERY_CASE="${TRIM_MEMORY_EVERY_CASE:-1}"
CACHE_QUOTIENT_MODELS="${CACHE_QUOTIENT_MODELS:-0}"

RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE3="${RUN_STAGE3:-1}"
RUN_HEURISTIC_BASELINE="${RUN_HEURISTIC_BASELINE:-1}"
RUN_RANDOM_BASELINE="${RUN_RANDOM_BASELINE:-1}"
RUN_LEGACY8_ABLATION="${RUN_LEGACY8_ABLATION:-1}"
RUN_FIXED_BUDGET_ABLATION="${RUN_FIXED_BUDGET_ABLATION:-1}"
RUN_UNSAFE_V_ABLATION="${RUN_UNSAFE_V_ABLATION:-1}"
RUN_ADD_ONLY_ABLATION="${RUN_ADD_ONLY_ABLATION:-1}"
RUN_DELETE_ONLY_ABLATION="${RUN_DELETE_ONLY_ABLATION:-1}"
RUN_DIRECT_ORIGINAL="${RUN_DIRECT_ORIGINAL:-${RUN_DIRECT_ORIGINAL_SMALL:-1}}"
RUN_CONTEXTUAL_LINEAR="${RUN_CONTEXTUAL_LINEAR:-1}"
DIRECT_ORIGINAL_LIMIT="${DIRECT_ORIGINAL_LIMIT:-0}"
DIRECT_ORIGINAL_MAX_STATES="${DIRECT_ORIGINAL_MAX_STATES:-0}"
DIRECT_ORIGINAL_MAX_TRANSITIONS="${DIRECT_ORIGINAL_MAX_TRANSITIONS:-0}"
DIRECT_ORIGINAL_RESUME="${DIRECT_ORIGINAL_RESUME:-1}"
DIRECT_ORIGINAL_SHARD_COUNT="${DIRECT_ORIGINAL_SHARD_COUNT:-1}"
DIRECT_ORIGINAL_SHARD_INDEX="${DIRECT_ORIGINAL_SHARD_INDEX:-0}"
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

if [[ "${STRICT_DEVICE}" != "1" ]]; then
  echo "ERROR: results6 linear comparison requires STRICT_DEVICE=1. Current STRICT_DEVICE=${STRICT_DEVICE}" >&2
  exit 2
fi
if [[ "${DEVICE}" != cuda* ]]; then
  echo "ERROR: Stage 2 requires DEVICE=cuda*. Current DEVICE=${DEVICE}" >&2
  exit 2
fi
if [[ "${QUOTIENT_DEVICE}" != cuda* ]]; then
  echo "ERROR: Stage 1 quotient requires QUOTIENT_DEVICE=cuda*. Current QUOTIENT_DEVICE=${QUOTIENT_DEVICE}" >&2
  exit 2
fi
if [[ ! -d "${AUT_DIR}" ]]; then
  echo "ERROR: AUT_DIR not found: ${AUT_DIR}" >&2
  exit 2
fi

chmod +x run_add_delete_all.sh run_server_gpu_all_env.sh run_server_nnunet_env.sh run_server_dual_compare_env.sh run_server_results6_linear_env.sh scripts/*.sh
mkdir -p "${COMPARE_ROOT}" models

echo "== Results6 linear-ranker baseline/ablation comparison =="
echo "Prepared once: ${PREPARED_DIR}"
echo "Compare root:  ${COMPARE_ROOT}"
echo "Baseline:      ${HEURISTIC_NAME}"
echo "Random:        ${RANDOM_NAME}"
echo "Ablation:      ${LEGACY8_NAME} (linear/legacy_v3, 8 features)"
echo "Fixed budget:  ${FIXED_BUDGET_NAME}"
echo "Unsafe V:      ${UNSAFE_V_NAME}"
echo "Repair modes:  ${ADD_ONLY_NAME}, ${DELETE_ONLY_NAME}"
echo "Direct full:   ${DIRECT_ORIGINAL_NAME} limit=${DIRECT_ORIGINAL_LIMIT:-0} shard=${DIRECT_ORIGINAL_SHARD_INDEX}/${DIRECT_ORIGINAL_SHARD_COUNT}"
echo "Main method:   ${CONTEXTUAL_NAME} (linear/current, 27 features)"
echo "Profile:       ${EXPERIMENT_PROFILE}"
echo "Fair per-case: max_seconds=${STAGE2_MAX_CASE_SECONDS}; verifier calls are measured, not capped"

python - <<'PY' | tee "${COMPARE_ROOT}/env_info.txt"
import sys
print("python =", sys.executable)
try:
    import torch
except ModuleNotFoundError:
    raise SystemExit("ERROR: torch is not installed.")
print("torch =", torch.__version__)
print("cuda_available =", torch.cuda.is_available())
print("cuda_version =", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available and STRICT_DEVICE=1.")
print("cuda_device =", torch.cuda.get_device_name(0))
PY

if [[ "${RUN_STAGE1}" == "1" || ! -f "${PREPARED_DIR}/manifest.json" ]]; then
  echo "== Stage 1: prepare formulas and strong-V quotient LTS' once =="
  AUT_DIR="${AUT_DIR}" PREPARED_DIR="${PREPARED_DIR}" LIMIT="${LIMIT}" \
  MAX_STATES="${MAX_STATES}" MAX_TRANSITIONS="${MAX_TRANSITIONS}" \
  QUOTIENT_DEVICE="${QUOTIENT_DEVICE}" STRICT_DEVICE="${STRICT_DEVICE}" \
  TARGET_STATE="${TARGET_STATE}" \
  V_SIZES="${V_SIZES}" \
  FORMULAS_PER_MODEL="${FORMULAS_PER_MODEL}" KNOWN_FORMULA_COUNT="${KNOWN_FORMULA_COUNT}" \
  MIXED_FORMULA_COUNT="${MIXED_FORMULA_COUNT}" FORMULA_MIN_ACTIONS="${FORMULA_MIN_ACTIONS}" \
  EASY_FORMULA_COUNT="${EASY_FORMULA_COUNT}" MEDIUM_FORMULA_COUNT="${MEDIUM_FORMULA_COUNT}" \
  HARD_FORMULA_COUNT="${HARD_FORMULA_COUNT}" FORMULA_MAX_ACTIONS="${FORMULA_MAX_ACTIONS}" \
  MIN_UNSATISFIED_FORMULAS="${MIN_UNSATISFIED_FORMULAS}" FORMULA_SEED="${FORMULA_SEED}" \
  bash scripts/run_add_delete_stage1_prepare_gpu.sh
else
  echo "== Stage 1 skipped: using existing ${PREPARED_DIR}/manifest.json =="
fi

run_stage2() {
  local name="$1"
  local ranker_model="$2"
  local linear_feature_set="$3"
  local include_neural="$4"
  local include_heuristic="$5"
  local default_ranker="$6"
  local include_random="${7:-0}"
  local dynamic_budget="${8:-${DYNAMIC_REPAIR_BUDGET}}"
  local profile="${9:-${EXPERIMENT_PROFILE}}"
  local v_selection="${10:-formula_safe}"
  local repair_mode_filter="${11:-all}"
  local budget_profile="${12:-fair}"
  local max_case_seconds="${STAGE2_MAX_CASE_SECONDS}"
  local dynamic_budget_rounds="${DYNAMIC_BUDGET_ROUNDS}"
  local dynamic_max_iters="${DYNAMIC_MAX_ITERS}"
  local dynamic_max_beam_width="${DYNAMIC_MAX_BEAM_WIDTH}"
  local dynamic_max_candidate_limit="${DYNAMIC_MAX_CANDIDATE_LIMIT}"
  local dynamic_max_candidate_state_limit="${DYNAMIC_MAX_CANDIDATE_STATE_LIMIT}"
  local dynamic_max_state_scan_limit="${DYNAMIC_MAX_STATE_SCAN_LIMIT}"
  local dynamic_max_minimal_layer_width="${DYNAMIC_MAX_MINIMAL_LAYER_WIDTH}"
  local dynamic_max_minimal_seen_limit="${DYNAMIC_MAX_MINIMAL_SEEN_LIMIT}"
  local dynamic_final_search_strategy="${DYNAMIC_FINAL_SEARCH_STRATEGY}"
  local out_dir="${COMPARE_ROOT}/${name}"

  echo "== Stage 2: ${name} (${budget_profile} budget, max_seconds=${max_case_seconds}, verifier_calls=measured_only) =="
  PREPARED_DIR="${PREPARED_DIR}" RESULTS_ROOT="${out_dir}" DEVICE="${DEVICE}" STRICT_DEVICE="${STRICT_DEVICE}" \
  RANKER_MODEL="${ranker_model}" V_SIZES="${V_SIZES}" FORMULA_LIMIT="${FORMULA_LIMIT}" TARGET_STATE="${TARGET_STATE}" \
  RANKER_TRAIN_SAMPLES="${RANKER_TRAIN_SAMPLES}" RANKER_TRAIN_FORMULA_LIMIT="${RANKER_TRAIN_FORMULA_LIMIT}" \
  RANKER_TRAIN_CANDIDATE_LIMIT="${RANKER_TRAIN_CANDIDATE_LIMIT}" RANKER_EPOCHS="${RANKER_EPOCHS}" RANKER_LR="${RANKER_LR}" \
  RANKER_ARCHITECTURE="linear" LINEAR_FEATURE_SET="${linear_feature_set}" \
  DEFAULT_RANKER="${default_ranker}" INCLUDE_NEURAL="${include_neural}" INCLUDE_HEURISTIC_COMPARISON="${include_heuristic}" \
  INCLUDE_RANDOM_COMPARISON="${include_random}" V_SELECTION="${v_selection}" REPAIR_MODE_FILTER="${repair_mode_filter}" \
  FORCE_RANKER_TRAIN="${FORCE_RANKER_TRAIN}" \
  BEAM_WIDTH="${BEAM_WIDTH}" MAX_ITERS="${MAX_ITERS}" CANDIDATE_LIMIT="${CANDIDATE_LIMIT}" \
  CANDIDATE_STATE_LIMIT="${CANDIDATE_STATE_LIMIT}" STATE_SCAN_LIMIT="${STATE_SCAN_LIMIT}" SEARCH_STRATEGY="${SEARCH_STRATEGY}" \
  EXPERIMENT_PROFILE="${profile}" MINIMAL_LAYER_WIDTH="${MINIMAL_LAYER_WIDTH}" MINIMAL_SEEN_LIMIT="${MINIMAL_SEEN_LIMIT}" \
  MAX_CASE_SECONDS="${max_case_seconds}" \
  DYNAMIC_REPAIR_BUDGET="${dynamic_budget}" DYNAMIC_BUDGET_ROUNDS="${dynamic_budget_rounds}" \
  DYNAMIC_MAX_ITERS="${dynamic_max_iters}" DYNAMIC_MAX_BEAM_WIDTH="${dynamic_max_beam_width}" \
  DYNAMIC_MAX_CANDIDATE_LIMIT="${dynamic_max_candidate_limit}" \
  DYNAMIC_MAX_CANDIDATE_STATE_LIMIT="${dynamic_max_candidate_state_limit}" \
  DYNAMIC_MAX_STATE_SCAN_LIMIT="${dynamic_max_state_scan_limit}" \
  DYNAMIC_MAX_MINIMAL_LAYER_WIDTH="${dynamic_max_minimal_layer_width}" \
  DYNAMIC_MAX_MINIMAL_SEEN_LIMIT="${dynamic_max_minimal_seen_limit}" \
  DYNAMIC_FINAL_SEARCH_STRATEGY="${dynamic_final_search_strategy}" \
  DRIFT_MODE="${DRIFT_MODE}" EXACT_DRIFT_MAX_TRANSITIONS="${EXACT_DRIFT_MAX_TRANSITIONS}" \
  STAGE2_MAX_STATES="${STAGE2_MAX_STATES}" STAGE2_MAX_TRANSITIONS="${STAGE2_MAX_TRANSITIONS}" \
  STAGE2_PROGRESS_EVERY="${STAGE2_PROGRESS_EVERY}" STAGE2_CASE_PROGRESS_EVERY="${STAGE2_CASE_PROGRESS_EVERY}" \
  TRIM_MEMORY_EVERY_CASE="${TRIM_MEMORY_EVERY_CASE}" CACHE_QUOTIENT_MODELS="${CACHE_QUOTIENT_MODELS}" \
  bash scripts/run_add_delete_stage2_gpu.sh
}

run_stage3() {
  local name="$1"
  local out_dir="${COMPARE_ROOT}/${name}"
  echo "== Stage 3: ${name} materialize writeback and verify original LTS =="
  PREPARED_DIR="${PREPARED_DIR}" RESULTS_ROOT="${out_dir}" OUTPUT_RESULTS_ROOT="${out_dir}" \
  TARGET_STATE="${TARGET_STATE}" FORCE_MATERIALIZE="${FORCE_MATERIALIZE}" MATERIALIZE_LIMIT="${MATERIALIZE_LIMIT}" \
  STAGE3_CASE_START_EVERY="${STAGE3_CASE_START_EVERY}" STAGE3_CEX_ITERS="${STAGE3_CEX_ITERS}" \
  STAGE3_CEX_LIFT_MODE="${STAGE3_CEX_LIFT_MODE}" STAGE3_CEX_BATCH_SIZE="${STAGE3_CEX_BATCH_SIZE}" \
  STAGE3_CASE_PROGRESS_EVERY="${STAGE3_CASE_PROGRESS_EVERY}" STAGE3_MAX_CASE_SECONDS="${STAGE3_MAX_CASE_SECONDS}" \
  STAGE3_TRIM_MEMORY_EVERY_CASE="${STAGE3_TRIM_MEMORY_EVERY_CASE}" PROGRESS_EVERY="${STAGE3_PROGRESS_EVERY}" \
  bash scripts/materialize_repaired_aut.sh
}

if [[ "${RUN_CONTEXTUAL_LINEAR}" == "1" ]]; then
  run_stage2 "${CONTEXTUAL_NAME}" "${CONTEXTUAL_RANKER_MODEL}" "current" "1" "0" "neural" "0" "${DYNAMIC_REPAIR_BUDGET}" "${EXPERIMENT_PROFILE}" "formula_safe" "all" "fair"
else
  echo "== Stage 2 skipped: ${CONTEXTUAL_NAME} =="
fi

if [[ "${RUN_HEURISTIC_BASELINE}" == "1" ]]; then
  run_stage2 "${HEURISTIC_NAME}" "models/unused_heuristic_baseline.pt" "current" "0" "1" "heuristic" "0" "${DYNAMIC_REPAIR_BUDGET}" "${EXPERIMENT_PROFILE}" "formula_safe" "all" "fair"
else
  echo "== Stage 2 skipped: ${HEURISTIC_NAME} =="
fi

if [[ "${RUN_RANDOM_BASELINE}" == "1" ]]; then
  run_stage2 "${RANDOM_NAME}" "models/unused_random_baseline.pt" "current" "0" "0" "random" "1" "${DYNAMIC_REPAIR_BUDGET}" "${EXPERIMENT_PROFILE}" "formula_safe" "all" "fair"
else
  echo "== Stage 2 skipped: ${RANDOM_NAME} =="
fi

if [[ "${RUN_LEGACY8_ABLATION}" == "1" ]]; then
  run_stage2 "${LEGACY8_NAME}" "${LEGACY8_RANKER_MODEL}" "legacy_v3" "1" "0" "neural" "0" "${DYNAMIC_REPAIR_BUDGET}" "${EXPERIMENT_PROFILE}" "formula_safe" "all" "fair"
else
  echo "== Stage 2 skipped: ${LEGACY8_NAME} =="
fi

if [[ "${RUN_FIXED_BUDGET_ABLATION}" == "1" ]]; then
  run_stage2 "${FIXED_BUDGET_NAME}" "${CONTEXTUAL_RANKER_MODEL}" "current" "1" "0" "neural" "0" "0" "${EXPERIMENT_PROFILE}" "formula_safe" "all" "fair"
else
  echo "== Stage 2 skipped: ${FIXED_BUDGET_NAME} =="
fi

if [[ "${RUN_UNSAFE_V_ABLATION}" == "1" ]]; then
  run_stage2 "${UNSAFE_V_NAME}" "${CONTEXTUAL_RANKER_MODEL}" "current" "1" "0" "neural" "0" "${DYNAMIC_REPAIR_BUDGET}" "${EXPERIMENT_PROFILE}" "unsafe" "all" "fair"
else
  echo "== Stage 2 skipped: ${UNSAFE_V_NAME} =="
fi

if [[ "${RUN_ADD_ONLY_ABLATION}" == "1" ]]; then
  run_stage2 "${ADD_ONLY_NAME}" "${CONTEXTUAL_RANKER_MODEL}" "current" "1" "0" "neural" "0" "${DYNAMIC_REPAIR_BUDGET}" "repair-mode-ablation" "formula_safe" "add-only" "fair"
else
  echo "== Stage 2 skipped: ${ADD_ONLY_NAME} =="
fi

if [[ "${RUN_DELETE_ONLY_ABLATION}" == "1" ]]; then
  run_stage2 "${DELETE_ONLY_NAME}" "${CONTEXTUAL_RANKER_MODEL}" "current" "1" "0" "neural" "0" "${DYNAMIC_REPAIR_BUDGET}" "repair-mode-ablation" "formula_safe" "delete-only" "fair"
else
  echo "== Stage 2 skipped: ${DELETE_ONLY_NAME} =="
fi

if [[ "${RUN_DIRECT_ORIGINAL}" == "1" ]]; then
  echo "== Direct original-LTS full baseline: ${DIRECT_ORIGINAL_NAME} =="
  DIRECT_RESUME_ARGS=(--no-resume)
  if [[ "${DIRECT_ORIGINAL_RESUME}" == "1" ]]; then
    DIRECT_RESUME_ARGS=(--resume)
  fi
  PREPARED_DIR="${PREPARED_DIR}" RESULTS_ROOT="${COMPARE_ROOT}/${DIRECT_ORIGINAL_NAME}" \
  DEVICE="${DEVICE}" STRICT_DEVICE="${STRICT_DEVICE}" RANKER_MODEL="${CONTEXTUAL_RANKER_MODEL}" \
  V_SIZES="${V_SIZES}" FORMULA_LIMIT="${FORMULA_LIMIT}" TARGET_STATE="${TARGET_STATE}" \
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
    --max-case-seconds "${STAGE2_MAX_CASE_SECONDS}" \
    --drift-mode "${DRIFT_MODE}" \
    --exact-drift-max-transitions "${EXACT_DRIFT_MAX_TRANSITIONS}" \
    --progress-every "${STAGE2_PROGRESS_EVERY}" \
    --shard-count "${DIRECT_ORIGINAL_SHARD_COUNT}" \
    --shard-index "${DIRECT_ORIGINAL_SHARD_INDEX}" \
    "${DIRECT_RESUME_ARGS[@]}"
else
  echo "== Direct original baseline skipped: ${DIRECT_ORIGINAL_NAME} =="
fi

if [[ "${RUN_STAGE3}" == "1" ]]; then
  if [[ "${RUN_HEURISTIC_BASELINE}" == "1" ]]; then
    run_stage3 "${HEURISTIC_NAME}"
  fi
  if [[ "${RUN_RANDOM_BASELINE}" == "1" ]]; then
    run_stage3 "${RANDOM_NAME}"
  fi
  if [[ "${RUN_LEGACY8_ABLATION}" == "1" ]]; then
    run_stage3 "${LEGACY8_NAME}"
  fi
  if [[ "${RUN_FIXED_BUDGET_ABLATION}" == "1" ]]; then
    run_stage3 "${FIXED_BUDGET_NAME}"
  fi
  if [[ "${RUN_UNSAFE_V_ABLATION}" == "1" ]]; then
    run_stage3 "${UNSAFE_V_NAME}"
  fi
  if [[ "${RUN_ADD_ONLY_ABLATION}" == "1" ]]; then
    run_stage3 "${ADD_ONLY_NAME}"
  fi
  if [[ "${RUN_DELETE_ONLY_ABLATION}" == "1" ]]; then
    run_stage3 "${DELETE_ONLY_NAME}"
  fi
  if [[ "${RUN_CONTEXTUAL_LINEAR}" == "1" ]]; then
    run_stage3 "${CONTEXTUAL_NAME}"
  fi
  if [[ "${RUN_DIRECT_ORIGINAL}" == "1" ]]; then
    run_stage3 "${DIRECT_ORIGINAL_NAME}"
  fi
else
  echo "== Stage 3 skipped: RUN_STAGE3=${RUN_STAGE3} =="
fi

echo "== Compare Stage 2 + Stage 3 metrics =="
if [[ "${RUN_HEURISTIC_BASELINE}" == "1" && "${RUN_CONTEXTUAL_LINEAR}" == "1" ]]; then
  python -m svbr.experiments.compare_rankers \
    --results-root "${COMPARE_ROOT}" \
    --left "${HEURISTIC_NAME}" \
    --right "${CONTEXTUAL_NAME}" \
    --output-prefix "baseline_vs_contextual_linear" \
    --pairing semantic
fi
if [[ "${RUN_RANDOM_BASELINE}" == "1" && "${RUN_CONTEXTUAL_LINEAR}" == "1" ]]; then
  python -m svbr.experiments.compare_rankers --results-root "${COMPARE_ROOT}" --left "${RANDOM_NAME}" --right "${CONTEXTUAL_NAME}" --output-prefix "random_vs_contextual_linear" --pairing semantic
fi
if [[ "${RUN_LEGACY8_ABLATION}" == "1" && "${RUN_CONTEXTUAL_LINEAR}" == "1" ]]; then
  python -m svbr.experiments.compare_rankers \
    --results-root "${COMPARE_ROOT}" \
    --left "${LEGACY8_NAME}" \
    --right "${CONTEXTUAL_NAME}" \
    --output-prefix "legacy8_ablation_vs_contextual_linear" \
    --pairing semantic
fi
if [[ "${RUN_FIXED_BUDGET_ABLATION}" == "1" && "${RUN_CONTEXTUAL_LINEAR}" == "1" ]]; then
  python -m svbr.experiments.compare_rankers --results-root "${COMPARE_ROOT}" --left "${FIXED_BUDGET_NAME}" --right "${CONTEXTUAL_NAME}" --output-prefix "fixed_budget_vs_contextual_linear" --pairing semantic
fi
if [[ "${RUN_UNSAFE_V_ABLATION}" == "1" && "${RUN_CONTEXTUAL_LINEAR}" == "1" ]]; then
  python -m svbr.experiments.compare_rankers --results-root "${COMPARE_ROOT}" --left "${UNSAFE_V_NAME}" --right "${CONTEXTUAL_NAME}" --output-prefix "unsafe_v_vs_formula_safe_contextual" --pairing semantic
fi
if [[ "${RUN_ADD_ONLY_ABLATION}" == "1" && "${RUN_CONTEXTUAL_LINEAR}" == "1" ]]; then
  python -m svbr.experiments.compare_rankers --results-root "${COMPARE_ROOT}" --left "${ADD_ONLY_NAME}" --right "${CONTEXTUAL_NAME}" --output-prefix "add_only_vs_add_delete_contextual" --pairing semantic
fi
if [[ "${RUN_DELETE_ONLY_ABLATION}" == "1" && "${RUN_CONTEXTUAL_LINEAR}" == "1" ]]; then
  python -m svbr.experiments.compare_rankers --results-root "${COMPARE_ROOT}" --left "${DELETE_ONLY_NAME}" --right "${CONTEXTUAL_NAME}" --output-prefix "delete_only_vs_add_delete_contextual" --pairing semantic
fi
if [[ "${RUN_DIRECT_ORIGINAL}" == "1" && "${RUN_CONTEXTUAL_LINEAR}" == "1" ]]; then
  python -m svbr.experiments.compare_rankers --results-root "${COMPARE_ROOT}" --left "${DIRECT_ORIGINAL_NAME}" --right "${CONTEXTUAL_NAME}" --output-prefix "direct_original_full_vs_quotient_contextual" --pairing semantic
fi
if [[ "${RUN_HEURISTIC_BASELINE}" == "1" && "${RUN_LEGACY8_ABLATION}" == "1" ]]; then
  python -m svbr.experiments.compare_rankers \
    --results-root "${COMPARE_ROOT}" \
    --left "${HEURISTIC_NAME}" \
    --right "${LEGACY8_NAME}" \
    --output-prefix "baseline_vs_legacy8_ablation" \
    --pairing semantic
fi

python -m svbr.experiments.ablation_report \
  --results-root "${COMPARE_ROOT}" \
  --prepared-dir "${PREPARED_DIR}" \
  --output-prefix reviewer_ablation

echo "== Results6 linear comparison complete =="
echo "Prepared:         ${PREPARED_DIR}"
echo "Baseline results: ${COMPARE_ROOT}/${HEURISTIC_NAME}"
echo "Random results:   ${COMPARE_ROOT}/${RANDOM_NAME}"
echo "Ablation results: ${COMPARE_ROOT}/${LEGACY8_NAME}"
echo "Fixed results:    ${COMPARE_ROOT}/${FIXED_BUDGET_NAME}"
echo "Unsafe results:   ${COMPARE_ROOT}/${UNSAFE_V_NAME}"
echo "Mode results:     ${COMPARE_ROOT}/${ADD_ONLY_NAME}, ${COMPARE_ROOT}/${DELETE_ONLY_NAME}"
echo "Direct results:   ${COMPARE_ROOT}/${DIRECT_ORIGINAL_NAME}"
echo "Main results:     ${COMPARE_ROOT}/${CONTEXTUAL_NAME}"
echo "Reports:"
echo "  ${COMPARE_ROOT}/baseline_vs_contextual_linear_report.md"
echo "  ${COMPARE_ROOT}/legacy8_ablation_vs_contextual_linear_report.md"
echo "  ${COMPARE_ROOT}/baseline_vs_legacy8_ablation_report.md"
echo "  ${COMPARE_ROOT}/reviewer_ablation_report.md"
