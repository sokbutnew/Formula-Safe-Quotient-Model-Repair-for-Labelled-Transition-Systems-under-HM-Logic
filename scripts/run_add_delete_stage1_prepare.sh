#!/usr/bin/env bash
set -euo pipefail

AUT_DIR="${AUT_DIR:-/root/sj-tmp/data/download}"
PREPARED_DIR="${PREPARED_DIR:-results/add_delete_prepared}"
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
TARGET_STATE="${TARGET_STATE:--1}"

echo "Stage 1 runs on CPU only; CUDA_VISIBLE_DEVICES is hidden for the prepare process."
CUDA_VISIBLE_DEVICES="" python -m svbr.experiments.add_delete_prepare \
  --aut-dir "${AUT_DIR}" \
  --recursive \
  --prepared-dir "${PREPARED_DIR}" \
  --limit "${LIMIT}" \
  --max-states "${MAX_STATES}" \
  --max-transitions "${MAX_TRANSITIONS}" \
  --v-sizes "${V_SIZES}" \
  --v-policy "${V_POLICY}" \
  --formulas-per-model "${FORMULAS_PER_MODEL}" \
  --known-formula-count "${KNOWN_FORMULA_COUNT}" \
  --mixed-formula-count "${MIXED_FORMULA_COUNT}" \
  --easy-formula-count "${EASY_FORMULA_COUNT}" \
  --medium-formula-count "${MEDIUM_FORMULA_COUNT}" \
  --hard-formula-count "${HARD_FORMULA_COUNT}" \
  --formula-min-actions "${FORMULA_MIN_ACTIONS}" \
  --formula-max-actions "${FORMULA_MAX_ACTIONS}" \
  --min-unsatisfied-formulas "${MIN_UNSATISFIED_FORMULAS}" \
  --formula-seed "${FORMULA_SEED}" \
  --target-state "${TARGET_STATE}"
