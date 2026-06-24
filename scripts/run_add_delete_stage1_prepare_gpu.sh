#!/usr/bin/env bash
set -euo pipefail

AUT_DIR="${AUT_DIR:-/root/sj-tmp/data/download}"
PREPARED_DIR="${PREPARED_DIR:-results/add_delete_prepared_gpu_all}"
LIMIT="${LIMIT:-50}"
MAX_STATES="${MAX_STATES:-1000000}"
MAX_TRANSITIONS="${MAX_TRANSITIONS:-5000000}"
V_SIZES="${V_SIZES:-0,1,3,5}"
V_POLICY="${V_POLICY:-least-frequent}"
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
QUOTIENT_DEVICE="${QUOTIENT_DEVICE:-cuda}"
STRICT_DEVICE="${STRICT_DEVICE:-1}"
TARGET_STATE="${TARGET_STATE:--1}"

echo "Stage 1 GPU-all attempt: strong-V quotient tensor operations use QUOTIENT_DEVICE=${QUOTIENT_DEVICE}."
if [[ "${STRICT_DEVICE}" != "1" ]]; then
  echo "ERROR: GPU Stage 1 requires STRICT_DEVICE=1 so CUDA failures cannot silently fall back to CPU. Current STRICT_DEVICE=${STRICT_DEVICE}" >&2
  exit 2
fi
if [[ "${QUOTIENT_DEVICE}" != cuda* ]]; then
  echo "ERROR: GPU Stage 1 requires QUOTIENT_DEVICE=cuda*. Current QUOTIENT_DEVICE=${QUOTIENT_DEVICE}" >&2
  exit 2
fi
ARGS=(
  --aut-dir "${AUT_DIR}"
  --recursive
  --prepared-dir "${PREPARED_DIR}"
  --limit "${LIMIT}"
  --max-states "${MAX_STATES}"
  --max-transitions "${MAX_TRANSITIONS}"
  --v-sizes "${V_SIZES}"
  --v-policy "${V_POLICY}"
  --formulas-per-model "${FORMULAS_PER_MODEL}"
  --known-formula-count "${KNOWN_FORMULA_COUNT}"
  --mixed-formula-count "${MIXED_FORMULA_COUNT}"
  --easy-formula-count "${EASY_FORMULA_COUNT}"
  --medium-formula-count "${MEDIUM_FORMULA_COUNT}"
  --hard-formula-count "${HARD_FORMULA_COUNT}"
  --formula-min-actions "${FORMULA_MIN_ACTIONS}"
  --formula-max-actions "${FORMULA_MAX_ACTIONS}"
  --min-unsatisfied-formulas "${MIN_UNSATISFIED_FORMULAS}"
  --formula-seed "${FORMULA_SEED}"
  --target-state "${TARGET_STATE}"
  --quotient-backend torch
  --quotient-device "${QUOTIENT_DEVICE}"
)

if [[ "${STRICT_DEVICE}" == "1" ]]; then
  ARGS+=(--strict-quotient-device)
fi

python -m svbr.experiments.add_delete_prepare "${ARGS[@]}"
