#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible entry point.
# The old dual script compared contextual linear ranker with MLP/GNN probes.
# For experiment 6 we no longer run MLP/GNN; this command now delegates to the
# linear-ranker baseline/ablation runner.

cd "$(dirname "$0")"
echo "run_server_dual_compare_env.sh now runs experiment-6 linear baseline/ablation only."
exec bash run_server_results6_linear_env.sh
