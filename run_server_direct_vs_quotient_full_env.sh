#!/usr/bin/env bash
set -euo pipefail

# Reviewer-facing two-method comparison:
#   1) lightweight_contextual_linear: formula-safe quotient LTS' repair
#   2) direct_original_contextual_full: repair directly on the original LTS
#
# Stage 1 runs once. Stage 2 and Stage 3 run only for these two methods.

cd "$(dirname "$0")"

export PREPARED_DIR="${PREPARED_DIR:-results8/add_delete_prepared}"
export COMPARE_ROOT="${COMPARE_ROOT:-results8/direct_vs_quotient_full}"

export RUN_HEURISTIC_BASELINE=0
export RUN_RANDOM_BASELINE=0
export RUN_LEGACY8_ABLATION=0
export RUN_FIXED_BUDGET_ABLATION=0
export RUN_UNSAFE_V_ABLATION=0
export RUN_ADD_ONLY_ABLATION=0
export RUN_DELETE_ONLY_ABLATION=0
export RUN_CONTEXTUAL_LINEAR=1
export RUN_DIRECT_ORIGINAL=1

echo "== Two-method full comparison =="
echo "Prepared: ${PREPARED_DIR}"
echo "Results:  ${COMPARE_ROOT}"
echo "Methods:  lightweight_contextual_linear, direct_original_contextual_full"

exec bash run_server_results6_linear_env.sh
