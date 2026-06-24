#!/usr/bin/env bash
set -euo pipefail

PREPARED_DIR="${PREPARED_DIR:-results/add_delete_prepared}"
RESULTS_ROOT="${RESULTS_ROOT:-results/add_delete_run}"
DEVICE="${DEVICE:-cuda}"
STRICT_DEVICE="${STRICT_DEVICE:-1}"
RANKER_MODEL="${RANKER_MODEL:-models/add_delete_ranker.pt}"
FORMULA_LIMIT="${FORMULA_LIMIT:-0}"
V_SIZES="${V_SIZES:-0,1,3,5}"
RANKER_TRAIN_SAMPLES="${RANKER_TRAIN_SAMPLES:-20000}"
RANKER_TRAIN_FORMULA_LIMIT="${RANKER_TRAIN_FORMULA_LIMIT:-30}"
RANKER_TRAIN_CANDIDATE_LIMIT="${RANKER_TRAIN_CANDIDATE_LIMIT:-128}"
RANKER_EPOCHS="${RANKER_EPOCHS:-40}"
RANKER_LR="${RANKER_LR:-0.003}"
RANKER_ARCHITECTURE="${RANKER_ARCHITECTURE:-linear}"
GNN_GRAPH_MODE="${GNN_GRAPH_MODE:-dynamic}"
LINEAR_FEATURE_SET="${LINEAR_FEATURE_SET:-current}"
RANKER_HIDDEN_DIM="${RANKER_HIDDEN_DIM:-64}"
RANKER_HIDDEN_LAYERS="${RANKER_HIDDEN_LAYERS:-2}"
NEURAL_PREFILTER_MULTIPLIER="${NEURAL_PREFILTER_MULTIPLIER:-4}"
NEURAL_PREFILTER_LIMIT="${NEURAL_PREFILTER_LIMIT:-512}"
NEURAL_LINEAR_BLEND="${NEURAL_LINEAR_BLEND:-0.35}"
NEURAL_VERIFY_FRONTIER_ONLY="${NEURAL_VERIFY_FRONTIER_ONLY:-1}"
NEURAL_VERIFY_TOP_K="${NEURAL_VERIFY_TOP_K:-0}"
NEURAL_CEGIS_RETRAIN="${NEURAL_CEGIS_RETRAIN:-0}"
NEURAL_CEGIS_ATTEMPTS="${NEURAL_CEGIS_ATTEMPTS:-0}"
NEURAL_CEGIS_EPOCHS="${NEURAL_CEGIS_EPOCHS:-4}"
NEURAL_CEGIS_LR="${NEURAL_CEGIS_LR:-0.001}"
NEURAL_CEGIS_CANDIDATE_LIMIT="${NEURAL_CEGIS_CANDIDATE_LIMIT:-256}"
NEURAL_CEGIS_ORACLE_MODEL="${NEURAL_CEGIS_ORACLE_MODEL:-}"
NEURAL_CEGIS_ADOPT_ORACLE="${NEURAL_CEGIS_ADOPT_ORACLE:-0}"
NEURAL_RESCUE_LINEAR="${NEURAL_RESCUE_LINEAR:-0}"
NEURAL_RESCUE_LINEAR_MODEL="${NEURAL_RESCUE_LINEAR_MODEL:-}"
DEFAULT_RANKER="${DEFAULT_RANKER:-neural}"
INCLUDE_HEURISTIC_COMPARISON="${INCLUDE_HEURISTIC_COMPARISON:-0}"
INCLUDE_NEURAL="${INCLUDE_NEURAL:-1}"
INCLUDE_RANDOM_COMPARISON="${INCLUDE_RANDOM_COMPARISON:-0}"
FORCE_RANKER_TRAIN="${FORCE_RANKER_TRAIN:-0}"
BEAM_WIDTH="${BEAM_WIDTH:-4}"
MAX_ITERS="${MAX_ITERS:-16}"
CANDIDATE_LIMIT="${CANDIDATE_LIMIT:-64}"
CANDIDATE_STATE_LIMIT="${CANDIDATE_STATE_LIMIT:-128}"
STATE_SCAN_LIMIT="${STATE_SCAN_LIMIT:-5000}"
SEARCH_STRATEGY="${SEARCH_STRATEGY:-beam}"
EXPERIMENT_PROFILE="${EXPERIMENT_PROFILE:-add-delete-only}"
V_SELECTION="${V_SELECTION:-formula_safe}"
REPAIR_MODE_FILTER="${REPAIR_MODE_FILTER:-all}"
MINIMAL_LAYER_WIDTH="${MINIMAL_LAYER_WIDTH:-2048}"
MINIMAL_SEEN_LIMIT="${MINIMAL_SEEN_LIMIT:-500000}"
MAX_CASE_SECONDS="${MAX_CASE_SECONDS:-0}"
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
STAGE2_MAX_STATES="${STAGE2_MAX_STATES:-1000000}"
STAGE2_MAX_TRANSITIONS="${STAGE2_MAX_TRANSITIONS:-5000000}"
STAGE2_PROGRESS_EVERY="${STAGE2_PROGRESS_EVERY:-1}"
STAGE2_CASE_PROGRESS_EVERY="${STAGE2_CASE_PROGRESS_EVERY:-0}"
TRIM_MEMORY_EVERY_CASE="${TRIM_MEMORY_EVERY_CASE:-1}"
CACHE_QUOTIENT_MODELS="${CACHE_QUOTIENT_MODELS:-0}"
TARGET_STATE="${TARGET_STATE:--1}"

if [[ "${STRICT_DEVICE}" != "1" ]]; then
  echo "ERROR: GPU Stage 2 requires STRICT_DEVICE=1 so CUDA failures cannot silently fall back to CPU. Current STRICT_DEVICE=${STRICT_DEVICE}" >&2
  exit 2
fi
if [[ "${DEVICE}" != cuda* ]]; then
  echo "ERROR: GPU Stage 2 requires DEVICE=cuda*. Current DEVICE=${DEVICE}" >&2
  exit 2
fi

ARGS=(
  --prepared-dir "${PREPARED_DIR}"
  --results-root "${RESULTS_ROOT}"
  --device "${DEVICE}"
  --ranker-model "${RANKER_MODEL}"
  --formula-limit "${FORMULA_LIMIT}"
  --v-sizes "${V_SIZES}"
  --ranker-train-samples "${RANKER_TRAIN_SAMPLES}"
  --ranker-train-formula-limit "${RANKER_TRAIN_FORMULA_LIMIT}"
  --ranker-train-candidate-limit "${RANKER_TRAIN_CANDIDATE_LIMIT}"
  --ranker-epochs "${RANKER_EPOCHS}"
  --ranker-lr "${RANKER_LR}"
  --ranker-architecture "${RANKER_ARCHITECTURE}"
  --gnn-graph-mode "${GNN_GRAPH_MODE}"
  --linear-feature-set "${LINEAR_FEATURE_SET}"
  --ranker-hidden-dim "${RANKER_HIDDEN_DIM}"
  --ranker-hidden-layers "${RANKER_HIDDEN_LAYERS}"
  --neural-prefilter-multiplier "${NEURAL_PREFILTER_MULTIPLIER}"
  --neural-prefilter-limit "${NEURAL_PREFILTER_LIMIT}"
  --neural-linear-blend "${NEURAL_LINEAR_BLEND}"
  --neural-verify-top-k "${NEURAL_VERIFY_TOP_K}"
  --neural-cegis-attempts "${NEURAL_CEGIS_ATTEMPTS}"
  --neural-cegis-epochs "${NEURAL_CEGIS_EPOCHS}"
  --neural-cegis-lr "${NEURAL_CEGIS_LR}"
  --neural-cegis-candidate-limit "${NEURAL_CEGIS_CANDIDATE_LIMIT}"
  --neural-cegis-oracle-model "${NEURAL_CEGIS_ORACLE_MODEL}"
  --neural-rescue-linear-model "${NEURAL_RESCUE_LINEAR_MODEL}"
  --default-ranker "${DEFAULT_RANKER}"
  --v-selection "${V_SELECTION}"
  --repair-mode-filter "${REPAIR_MODE_FILTER}"
  --beam-width "${BEAM_WIDTH}"
  --max-iters "${MAX_ITERS}"
  --candidate-limit "${CANDIDATE_LIMIT}"
  --candidate-state-limit "${CANDIDATE_STATE_LIMIT}"
  --state-scan-limit "${STATE_SCAN_LIMIT}"
  --search-strategy "${SEARCH_STRATEGY}"
  --experiment-profile "${EXPERIMENT_PROFILE}"
  --minimal-layer-width "${MINIMAL_LAYER_WIDTH}"
  --minimal-seen-limit "${MINIMAL_SEEN_LIMIT}"
  --max-case-seconds "${MAX_CASE_SECONDS}"
  --dynamic-budget-rounds "${DYNAMIC_BUDGET_ROUNDS}"
  --dynamic-max-iters "${DYNAMIC_MAX_ITERS}"
  --dynamic-max-beam-width "${DYNAMIC_MAX_BEAM_WIDTH}"
  --dynamic-max-candidate-limit "${DYNAMIC_MAX_CANDIDATE_LIMIT}"
  --dynamic-max-candidate-state-limit "${DYNAMIC_MAX_CANDIDATE_STATE_LIMIT}"
  --dynamic-max-state-scan-limit "${DYNAMIC_MAX_STATE_SCAN_LIMIT}"
  --dynamic-max-minimal-layer-width "${DYNAMIC_MAX_MINIMAL_LAYER_WIDTH}"
  --dynamic-max-minimal-seen-limit "${DYNAMIC_MAX_MINIMAL_SEEN_LIMIT}"
  --dynamic-final-search-strategy "${DYNAMIC_FINAL_SEARCH_STRATEGY}"
  --drift-mode "${DRIFT_MODE}"
  --exact-drift-max-transitions "${EXACT_DRIFT_MAX_TRANSITIONS}"
  --stage2-max-states "${STAGE2_MAX_STATES}"
  --stage2-max-transitions "${STAGE2_MAX_TRANSITIONS}"
  --progress-every "${STAGE2_PROGRESS_EVERY}"
  --case-progress-every "${STAGE2_CASE_PROGRESS_EVERY}"
  --target-state "${TARGET_STATE}"
)

if [[ "${STRICT_DEVICE}" == "1" ]]; then
  ARGS+=(--strict-device)
fi
if [[ "${FORCE_RANKER_TRAIN}" == "1" ]]; then
  ARGS+=(--force-ranker-train)
fi
if [[ "${NEURAL_CEGIS_RETRAIN}" == "1" ]]; then
  ARGS+=(--neural-cegis-retrain)
else
  ARGS+=(--no-neural-cegis-retrain)
fi
if [[ "${NEURAL_VERIFY_FRONTIER_ONLY}" == "1" ]]; then
  ARGS+=(--neural-verify-frontier-only)
else
  ARGS+=(--no-neural-verify-frontier-only)
fi
if [[ "${NEURAL_CEGIS_ADOPT_ORACLE}" == "1" ]]; then
  ARGS+=(--neural-cegis-adopt-oracle)
else
  ARGS+=(--no-neural-cegis-adopt-oracle)
fi
if [[ "${NEURAL_RESCUE_LINEAR}" == "1" ]]; then
  ARGS+=(--neural-rescue-linear)
else
  ARGS+=(--no-neural-rescue-linear)
fi
if [[ "${DYNAMIC_REPAIR_BUDGET}" == "1" ]]; then
  ARGS+=(--dynamic-repair-budget)
else
  ARGS+=(--no-dynamic-repair-budget)
fi
if [[ "${TRIM_MEMORY_EVERY_CASE}" == "1" ]]; then
  ARGS+=(--trim-memory-every-case)
else
  ARGS+=(--no-trim-memory-every-case)
fi
if [[ "${CACHE_QUOTIENT_MODELS}" == "1" ]]; then
  ARGS+=(--cache-quotient-models)
fi
if [[ "${INCLUDE_HEURISTIC_COMPARISON}" == "1" ]]; then
  ARGS+=(--include-heuristic-comparison)
else
  ARGS+=(--no-include-heuristic-comparison)
fi
if [[ "${INCLUDE_RANDOM_COMPARISON}" == "1" ]]; then
  ARGS+=(--include-random-comparison)
else
  ARGS+=(--no-include-random-comparison)
fi
if [[ "${INCLUDE_NEURAL}" == "1" ]]; then
  ARGS+=(--include-neural)
else
  ARGS+=(--no-include-neural)
fi

python -m svbr.experiments.add_delete_prepared_run "${ARGS[@]}"
