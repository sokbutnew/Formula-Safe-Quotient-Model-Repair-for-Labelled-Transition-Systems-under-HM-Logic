# Server Bundle

Run the GPU-all version on the server:

```bash
chmod +x run_server_nnunet_env.sh run_server_gpu_all_env.sh run_server_dual_compare_env.sh run_server_results6_linear_env.sh run_server_direct_original_full_env.sh run_server_direct_vs_quotient_full_env.sh run_add_delete_all.sh scripts/*.sh
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_gpu_all_env.sh
```

Run only the quotient main method and the full direct-original baseline:

```bash
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_direct_vs_quotient_full_env.sh
```

Supplement an existing comparison with the full direct original-LTS baseline only:

```bash
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_direct_original_full_env.sh
```

Run the experiment-6 linear-ranker baseline/ablation comparison:

```bash
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_results6_linear_env.sh
```

The old dual script name is kept as a compatibility wrapper and now runs the same experiment-6 linear comparison:

```bash
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_dual_compare_env.sh
```

The older CPU+GPU runner is still included for comparison:

```bash
bash run_server_nnunet_env.sh
```

This bundle is configured for an environment that already has CUDA PyTorch installed,
for example torch 2.11.0+cu13 in nnunet_env. Do not reinstall torch unless you
intentionally want to replace the current CUDA build.

Default forgotten-action sizes: |V|=0,1,3,5.
Default server AUT_DIR: `/root/sj-tmp/data/download`.
Override it at runtime with `AUT_DIR=/path/to/aut_dir` if your dataset is stored elsewhere.
V is selected per formula, and formula actions are excluded from V.
Default formula suite per LTS: easy=5, medium=10, hard=15;
20 formulas use only existing LTS actions and 10 mix existing/missing actions.
Mixed formulas randomize where missing actions appear.
All 30 positive formulas are generated to be initially unsatisfied.
Stage 1 keeps regenerating formulas for the current LTS until the quota is met.
Stage 1 rejects target formulas that are logically unsatisfiable in every LTS.

Three-stage pipeline:

1. Stage 1 writes formulas plus strong-V quotient LTS' files; the GPU-all runner uses torch/CUDA for quotient tensor work.
2. Stage 2 trains/uses the neural ranker on GPU, repairs quotient LTS' into LTS'', and checks HML on LTS''.
3. Stage 3 writes successful block-level repair operations back to original LTS files and verifies them.
   LTS'' block-level edits are the authoritative write-back templates.
   Counterexamples only prioritize concrete representatives inside the template blocks.
   Stage 3 generates concrete edits only from Stage 2 quotient templates.
   By default Stage 3 uses counterexample-guided closure lifting in one output directory.
   Existential (<>) paths keep one concrete repair edge when one path suffices.
   Universal ([]) obligations keep adding only concrete branches exposed by Stage 3 counterexamples.
   Counterexample-guided lifting starts at STAGE3_CEX_ITERS=16.
   If verification still fails, closure mode doubles the budget up to the original LTS transition count.
   Independent universal counterexample edits are batched with STAGE3_CEX_BATCH_SIZE=512.
   Stage 3 prints stage3-case before each edit script by default: STAGE3_CASE_START_EVERY=1.
   Long lift cases print stage3-lift progress every STAGE3_CASE_PROGRESS_EVERY=100 concrete edits.
   Stage 3 always keeps minimal required template instances; it does not fill whole blocks.
   STAGE3_MAX_CASE_SECONDS=0 by default, so slow cases are not abandoned for timeout.
   Existing repaired_aut files are skipped by default: FORCE_MATERIALIZE=0.
   Stage 3 trims process memory after each case by default: STAGE3_TRIM_MEMORY_EVERY_CASE=1.
   Unresolved Stage 3 cases are written to materialize_repaired_aut_unresolved.csv
   and stage3_unresolved_scripts.txt, which can be retried with STAGE3_SCRIPT_LIST.
   Set STAGE3_COMPARE_CEX=1 only when you explicitly want an extra single-edge/no-cex comparison output.

The GPU-all attempt keeps the same three-stage logic, but uses the torch/CUDA
quotient backend in Stage 1. AUT parsing, Python beam search/HML recursion,
and final file materialization remain CPU/Python work because they are not tensor kernels.
The GPU-all runner requires STRICT_DEVICE=1, DEVICE=cuda*, and QUOTIENT_DEVICE=cuda*.
If CUDA is unavailable it stops instead of silently falling back to CPU.
Its main Stage 2 experiment groups use DEFAULT_RANKER=neural by default.
The stable neural ranker is contextual linear by default: RANKER_ARCHITECTURE=linear, LINEAR_FEATURE_SET=current.
Stage 2 uses fast beam search by default: SEARCH_STRATEGY=beam.
DYNAMIC_REPAIR_BUDGET=1 retries failed cases with geometrically wider search/candidate budgets.
DYNAMIC_BUDGET_ROUNDS=0 keeps growing failed cases until the configured safety ceilings are saturated.
The final retry can use DYNAMIC_FINAL_SEARCH_STRATEGY=neural_guided_minimal,
and exact HML verification remains the only success criterion.
Old prepared manifests with logically unsatisfiable target formulas are rejected before dynamic search; rerun Stage 1.
run_server_results6_linear_env.sh prepares Stage 1 once, then runs Stage 2 and Stage 3
for `heuristic_baseline`, `random_baseline`, `legacy8_ablation`,
`fixed_budget_contextual`, `unsafe_v_contextual`, `add_only_contextual`,
`delete_only_contextual`, `direct_original_contextual_full`, and
`lightweight_contextual_linear`, followed by paired comparison reports.
MLP/GNN probes are intentionally not run in the experiment-6 script.
Stage 3 follows the current counterexample path downward and keeps one concrete add representative per step.
The CPU heuristic comparison group is skipped by default: INCLUDE_HEURISTIC_COMPARISON=0.

All three stages print progress bars.
Stage 2 skips target formulas that are already satisfied by quotient LTS'.

Outputs:

- `results/add_delete_run/*/runs.csv`
- `results/add_delete_run/*/errors.csv`
- `results/add_delete_run/*/repaired_aut/*.aut`
- `results/add_delete_run/*/edit_scripts/*.json`
- `results/add_delete_run/sf_vs_no_sf_summary.csv`
- `results/add_delete_run/repair_mode_summary.csv`
- `results/add_delete_run/postprocess_summary.csv`
- `results/add_delete_run/ranker_summary.csv`
- `results/add_delete_run/ranker_training.json`
- `results/add_delete_run/*_by_formula.csv`
- `results/add_delete_run/materialize_repaired_aut.csv`
- `results/add_delete_run/*/writeback_operations/*.json`
- `results6/linear_ranker_ablation/baseline_vs_contextual_linear_report.md`
- `results6/linear_ranker_ablation/random_vs_contextual_linear_report.md`
- `results6/linear_ranker_ablation/legacy8_ablation_vs_contextual_linear_report.md`
- `results6/linear_ranker_ablation/fixed_budget_vs_contextual_linear_report.md`
- `results6/linear_ranker_ablation/unsafe_v_vs_formula_safe_contextual_report.md`
- `results6/linear_ranker_ablation/add_only_vs_add_delete_contextual_report.md`
- `results6/linear_ranker_ablation/delete_only_vs_add_delete_contextual_report.md`
- `results6/linear_ranker_ablation/direct_original_full_vs_quotient_contextual_report.md`
- `results6/linear_ranker_ablation/reviewer_ablation_report.md`
- `results6/linear_ranker_ablation/reviewer_ablation_quotient_stats.csv`
- `results6/linear_ranker_ablation/reviewer_ablation_worst_cases.csv`
- `results6/linear_ranker_ablation/*_overall.csv`
- `results6/linear_ranker_ablation/*_by_stratum.csv`
- `results6/linear_ranker_ablation/*_case_deltas.csv`
- `results6/linear_ranker_ablation/*_paired_summary.csv`

Stage-2-only ranker comparison after Stage 1:

```bash
PREPARED_DIR=results/add_delete_prepared_gpu_all RESULTS_ROOT=results/ranker_compare_stage2 DEVICE=cuda STRICT_DEVICE=1 bash scripts/run_ranker_compare_stage2_gpu.sh
```

The Stage-2-only helper compares `legacy8_ablation` against
`lightweight_contextual_linear`; the end-to-end results6 script is preferred
when you need Stage 3 materialization metrics in the same report.

This writes legacy8_ablation_vs_contextual_linear_stage2_*.csv/md.

Stage 3 can also be run directly after Stage 2:

```bash
PREPARED_DIR=results/add_delete_prepared RESULTS_ROOT=results/add_delete_run bash scripts/materialize_repaired_aut.sh
PREPARED_DIR=results/add_delete_prepared RESULTS_ROOT=results/add_delete_run STAGE3_CEX_LIFT_MODE=closure STAGE3_CEX_ITERS=16 bash scripts/materialize_repaired_aut.sh
# In closure mode this is the initial budget; Stage 3 doubles it until verified or the original LTS transition count is reached.
```

Included data: NO
AUT files copied: 0
Small test fixtures copied: 16
