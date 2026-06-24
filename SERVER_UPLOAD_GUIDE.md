# Strong-Forgetting Add/Delete Server Guide

This bundle is for the uploaded MD task:

```text
Strong-Forgetting-Priority Add--Delete Repair for Labelled Transition Systems
```

It is not the old VLTS neural HML certificate project.

## Build Upload Bundle

With data included:

```bash
python scripts/make_server_bundle.py --include-data --clean
```

Without data:

```bash
python scripts/make_server_bundle.py --clean
```

Upload:

```text
server_bundle/svbr_strong_forgetting/
```

If data is not bundled, put `.aut` files on the server under:

```text
/root/sj-tmp/data/download/
```

## Minimal Upload List

```text
run_add_delete_all.sh
run_server_nnunet_env.sh
run_server_gpu_all_env.sh
run_server_dual_compare_env.sh
run_server_results6_linear_env.sh
README.md
SERVER_UPLOAD_GUIDE.md
requirements.txt
pyproject.toml
svbr/
scripts/make_server_bundle.py
scripts/run_add_delete_stage1_prepare.sh
scripts/run_add_delete_stage1_prepare_gpu.sh
scripts/run_add_delete_stage2_gpu.sh
tests/
/root/sj-tmp/data/download/*.aut
```

Do not upload old project scripts, cached results, `__pycache__`, or stale `server_bundle` contents.

## Server Environment

Your server already has `nnunet_env` with `torch 2.11.0+cu13`. Do not reinstall torch unless you intentionally want to replace that CUDA build.

```bash
cd svbr_strong_forgetting
chmod +x run_server_nnunet_env.sh run_server_gpu_all_env.sh run_server_dual_compare_env.sh run_server_results6_linear_env.sh run_server_direct_original_full_env.sh run_server_direct_vs_quotient_full_env.sh run_add_delete_all.sh scripts/*.sh
bash run_server_nnunet_env.sh
```

GPU-all attempt version:

```bash
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_gpu_all_env.sh
```

The GPU-all runner refuses `STRICT_DEVICE=0`, `DEVICE=cpu`, or `QUOTIENT_DEVICE=cpu`. It uses the torch/CUDA quotient backend in Stage 1 and the neural ranker on CUDA for the main Stage 2 experiment groups by default (`DEFAULT_RANKER=neural`). If CUDA is unavailable it stops instead of silently falling back to CPU. It skips the CPU heuristic comparison by default (`INCLUDE_HEURISTIC_COMPARISON=0`); set it to `1` only when you explicitly need that control group.

Experiment-6 linear-ranker baseline/ablation comparison:

```bash
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_results6_linear_env.sh
```

This prepares Stage 1 once, then runs Stage 2 and Stage 3 for reviewer-facing controls under `results6/linear_ranker_ablation`: `heuristic_baseline`, `random_baseline`, `legacy8_ablation` (8-feature linear ranker), `fixed_budget_contextual`, `unsafe_v_contextual`, `add_only_contextual`, `delete_only_contextual`, full `direct_original_contextual_full`, and `lightweight_contextual_linear` (27-feature contextual linear ranker). The direct-original baseline now uses the complete workload and the same fair per-case cap. It supports resume and deterministic sharding because direct search on the original LTS is expensive. It also writes `reviewer_ablation_report.md`, quotient-reduction stats, worst-case tables, and paired CSV/Markdown comparisons. MLP/GNN probes are intentionally not run.

For the focused paper comparison, run only the two methods that matter:

```bash
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 \
bash run_server_direct_vs_quotient_full_env.sh
```

This creates `results8/direct_vs_quotient_full`, prepares Stage 1 once, and runs only:

```text
lightweight_contextual_linear
direct_original_contextual_full
```

Stage 2 uses the same formula workload and `FAIR_MAX_CASE_SECONDS=300` cap for both methods. Stage 3 verifies both repaired original LTS outputs.

All compared Stage 2 methods use the same fair per-case wall-clock cap by default (`FAIR_MAX_CASE_SECONDS=300`, i.e. 5 minutes). Verifier calls are recorded as an efficiency metric, but they are not capped in the fair comparison runner. Cases that exceed the common time budget are reported as failures/timeouts for that method. The comparison reports include per-method success/verified rates, timeout counts/rates, average/median/p90 time, verifier-call statistics, average/median edit counts, Stage 3 write-back verification rate, end-to-end verification rate, Stage 3 concrete edge expansion, and worst-case tables.

If Stage 1 and `lightweight_contextual_linear` already finished, supplement only the full direct-original baseline:

```bash
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 \
PREPARED_DIR=results6/add_delete_prepared \
COMPARE_ROOT=results6/linear_ranker_ablation \
bash run_server_direct_original_full_env.sh
```

The direct-original run resumes by default. Re-running the same command skips completed case IDs and continues appending to the existing shard CSV.

For a long run, deterministic shards can be executed independently. Use separate terminal sessions or jobs:

```bash
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 DIRECT_ORIGINAL_SHARD_COUNT=4 DIRECT_ORIGINAL_SHARD_INDEX=0 bash run_server_direct_original_full_env.sh
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 DIRECT_ORIGINAL_SHARD_COUNT=4 DIRECT_ORIGINAL_SHARD_INDEX=1 bash run_server_direct_original_full_env.sh
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 DIRECT_ORIGINAL_SHARD_COUNT=4 DIRECT_ORIGINAL_SHARD_INDEX=2 bash run_server_direct_original_full_env.sh
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 DIRECT_ORIGINAL_SHARD_COUNT=4 DIRECT_ORIGINAL_SHARD_INDEX=3 bash run_server_direct_original_full_env.sh
```

After all shards finish, materialize and regenerate the comparison report once:

```bash
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 \
DIRECT_ORIGINAL_SHARD_COUNT=4 RUN_DIRECT_STAGE2=0 DIRECT_ORIGINAL_FINALIZE=1 \
bash run_server_direct_original_full_env.sh
```

Start with one shard when memory is the main concern. Multiple shards trade higher CPU RAM usage for shorter wall-clock time.

Defaults:

```text
AUT_DIR=/root/sj-tmp/data/download
PREPARED_DIR=results/add_delete_prepared
RESULTS_ROOT=results/add_delete_run
DEVICE=cuda
CUDA_VISIBLE_DEVICES=0
STRICT_DEVICE=1
LIMIT=50
MAX_STATES=1000000
MAX_TRANSITIONS=5000000
V_SIZES=0,1,3,5
FORMULAS_PER_MODEL=30
MIN_UNSATISFIED_FORMULAS=30
RANKER_TRAIN_FORMULA_LIMIT=30
BEAM_WIDTH=2
MAX_ITERS=8
CANDIDATE_LIMIT=16
CANDIDATE_STATE_LIMIT=64
STATE_SCAN_LIMIT=1000
DRIFT_MODE=estimate
TRIM_MEMORY_EVERY_CASE=1
CACHE_QUOTIENT_MODELS=0
```

## Three Stages

Stage 1: prepare formulas and strong-V quotient `LTS'` files.

```bash
AUT_DIR=/root/sj-tmp/data/download PREPARED_DIR=results/add_delete_prepared LIMIT=50 bash scripts/run_add_delete_stage1_prepare.sh
```

Stage 1 does all model-dependent preparation:

- parses selected `.aut` files;
- prepares strong-V quotients for `|V|=0,1,3,5` by default;
- generates 30 formula cases per LTS by default;
- difficulty split: easy=5, medium=10, hard=15;
- uses 20 formulas with only LTS-existing actions;
- uses 10 formulas mixing existing actions and generated missing actions;
- randomizes the missing-action position in mixed formulas, so the first target action is not always missing;
- guarantees all 30 positive formulas are initially unsatisfied per LTS by default;
- does not move to the next LTS until the current LTS has reached the unsatisfied-formula quota;
- rejects target formulas that are logically unsatisfiable in every LTS, including contradictions exposed only after negation is pushed through `<>`/`[]`;
- builds easy / medium / hard formulas with 5-10 modal action occurrences;
- precomputes formula-specific strong-V quotient files;
- excludes every action that appears in the checked formula from that formula's `V`.

Stage 2: train/use the neural ranker, repair `LTS'` into `LTS''`, and run HML checks on `LTS''`.

```bash
PREPARED_DIR=results/add_delete_prepared RESULTS_ROOT=results/add_delete_run DEVICE=cuda bash scripts/run_add_delete_stage2_gpu.sh
```

Stage 2 loads the prepared manifest and quotient files. The repair surface is the quotient `LTS'`: quotient blocks are treated as states, repair operations are block-level add/delete edges, and HML checking is performed on repaired `LTS''`.
The neural ranker is trained if needed, then initialized once and reused for all runs in that process. The stable GPU runner uses a lightweight contextual linear neural ranker (`RANKER_ARCHITECTURE=linear`, `LINEAR_FEATURE_SET=current`) with fast beam search (`SEARCH_STRATEGY=beam`). It includes formula-depth, active-branch-kind, required-action, counterexample-path, next-subformula, and edit-count features without enabling the slower MLP/GNN paths. If a case fails under the base budget, `DYNAMIC_REPAIR_BUDGET=1` retries it with geometrically wider search and candidate budgets. `DYNAMIC_BUDGET_ROUNDS=0` continues until the configured safety ceilings are saturated; the final retry can switch to `DYNAMIC_FINAL_SEARCH_STRATEGY=neural_guided_minimal`. Exact HML verification remains the only success criterion.
When `DEVICE=cuda`, neural-ranker training and neural candidate scoring batches run on GPU. Formula checking, graph search, and AUT/quotient bookkeeping remain CPU-side graph algorithms.
The ranker checkpoint records a signature of the prepared manifest and training settings; if you rerun stage 1 with different data, formulas, or `V_SIZES`, stage 2 retrains automatically instead of reusing a stale smoke-test checkpoint.
Old prepared manifests with logically unsatisfiable target formulas are rejected before dynamic search; rerun Stage 1 so those formulas are regenerated.
Target formulas that are already satisfied by `LTS'` are skipped by default and written to `skipped_initially_satisfied.csv`, so main repair metrics only count formulas that actually require repair on the repair surface.
By default, Stage 2 writes edit scripts only and does not materialize full repaired AUT files during search. This keeps memory and I/O low.
For memory safety on large AUT files, Stage 2 defaults to estimated quotient drift during search and applies `STAGE2_MAX_TRANSITIONS=5000000` to quotient transitions unless you override that limit.
Stage 2 also releases memory after every formula case by default (`TRIM_MEMORY_EVERY_CASE=1`) and does not keep quotient repair models cached between cases (`CACHE_QUOTIENT_MODELS=0`). On Linux it additionally hints that consumed AUT/quotient files can be dropped from page cache after loading, which helps container memory graphs avoid growing just because many pickle files were streamed. To see formula-level progress and current RSS, run with `STAGE2_CASE_PROGRESS_EVERY=1`.

Stage 3: materialize successful block-level repair operations back into the original LTS, then verify the original LTS result:

```bash
PREPARED_DIR=results/add_delete_prepared RESULTS_ROOT=results/add_delete_run bash scripts/materialize_repaired_aut.sh
PREPARED_DIR=results/add_delete_prepared RESULTS_ROOT=results/add_delete_run STAGE3_CEX_LIFT_MODE=closure STAGE3_CEX_ITERS=16 bash scripts/materialize_repaired_aut.sh
```

The materializer reads one edit script and one original AUT at a time, lifts quotient-block operations back to original states, writes `repaired_aut/*.aut`, verifies the target HML formula on the materialized original LTS, then releases memory before moving to the next script.

For Stage 3 lifting, the repaired quotient `LTS''` edit script is the authoritative template: concrete writes are generated from a Stage 2 source block, action label, and destination block. Counterexamples only prioritize concrete representative states inside those blocks. `<a>phi` keeps one concrete `a`-successor repair path when one witness is sufficient. `[a]phi` processes only the concrete `a`-successor branches still required to satisfy `phi`; it does not fill every state in a template block. The box operator covers successors matching action `a`, not unrelated action labels.
By default, the one-command runner runs one Stage 3 output with strict counterexample-guided closure lifting. Concrete adds follow the current counterexample path downward and keep only one suitable destination representative per step instead of collecting many same-level candidates. The default budget starts at `STAGE3_CEX_ITERS=16`; if verification still fails, Stage 3 automatically tries `32`, `64`, and so on, up to the transition count of the current original LTS. Independent universal edits are batched with `STAGE3_CEX_BATCH_SIZE=512`. `STAGE3_MAX_CASE_SECONDS=0` by default, so Stage 3 does not abandon slow cases for timeout when the goal is verified materialization.

Each written Stage 3 case adds `writeback_operations/*.json`. The file separately lists the Stage 2 `LTS' -> LTS''` block operations and the Stage 3 concrete operations written into the original LTS.

All three stages print progress bars:

- `stage1-model` / `stage1-quotient`
- `stage2-ranker-data` / `stage2-ranker-epoch` / `stage2-repair`
- `stage3-case` / `stage3-materialize`

The GPU-all attempt keeps the same three-stage logic, but Stage 1 uses the
torch/CUDA quotient backend. AUT parsing, Python beam search/HML recursion, and
Stage 3 file materialization remain CPU/Python work because they are not tensor
kernels.

For a quick smoke run:

```bash
AUT_DIR=/root/sj-tmp/data/download LIMIT=1 MAX_STATES=1000 MAX_TRANSITIONS=5000 FORMULA_LIMIT=2 RESULTS_ROOT=results/smoke bash run_add_delete_all.sh
```

For a full run:

```bash
AUT_DIR=/root/sj-tmp/data/download LIMIT=50 RESULTS_ROOT=results/add_delete_run bash run_add_delete_all.sh
```

## Main Outputs

```text
results/add_delete_prepared/
  manifest.json
  selection.json
  quotients/*.pkl

results/add_delete_run/
  no_sf_add_delete_V*/runs.csv
  soft_sf_add_delete_V*/runs.csv
  strict_then_escalate_add_delete_V*/runs.csv
  pos_add-only_V*/runs.csv
  pos_delete-only_V*/runs.csv
  pos_add-delete_V*/runs.csv
  neg_exist_add-only_V*/runs.csv
  neg_exist_delete-only_V*/runs.csv
  neg_exist_add-delete_V*/runs.csv
  neg_univ_add-only_V*/runs.csv
  neg_univ_delete-only_V*/runs.csv
  neg_univ_add-delete_V*/runs.csv
  post_off_V*/runs.csv
  post_on_V*/runs.csv
  ranker_heuristic_V*/runs.csv   # only when INCLUDE_HEURISTIC_COMPARISON=1
  ranker_neural_V*/runs.csv
  */skipped_initially_satisfied.csv
  ranker_training.json
  *_summary.csv
  *_summary_by_formula.csv
  */writeback_operations/*.json

results6/linear_ranker_ablation/
  heuristic_baseline/*/runs.csv
  heuristic_baseline/materialize_repaired_aut.csv
  random_baseline/*/runs.csv
  legacy8_ablation/*/runs.csv
  legacy8_ablation/materialize_repaired_aut.csv
  fixed_budget_contextual/*/runs.csv
  unsafe_v_contextual/*/runs.csv
  add_only_contextual/*/runs.csv
  delete_only_contextual/*/runs.csv
  direct_original_contextual_full/*/runs.csv
  lightweight_contextual_linear/*/runs.csv
  lightweight_contextual_linear/materialize_repaired_aut.csv
  reviewer_ablation_report.md
  reviewer_ablation_quotient_stats.csv
  reviewer_ablation_worst_cases.csv
  baseline_vs_contextual_linear_report.md
  random_vs_contextual_linear_report.md
  legacy8_ablation_vs_contextual_linear_report.md
  fixed_budget_vs_contextual_linear_report.md
  unsafe_v_vs_formula_safe_contextual_report.md
  add_only_vs_add_delete_contextual_report.md
  delete_only_vs_add_delete_contextual_report.md
  direct_original_full_vs_quotient_contextual_report.md
  *_overall.csv
  *_by_stratum.csv
  *_case_deltas.csv
  *_paired_summary.csv
```

Each `runs.csv` now includes V metadata, formula metadata, and evaluation metrics: `V_requested_size`, `V_size`, `V_size_note`, `V_source`, `formula_id`, `formula_difficulty`, `formula_source`, `formula_target_action_in_lts`, modal action count, missing action count, success/verification flags, edit counts, non-`V` edit counts, quotient drift, verifier calls, and elapsed time. In the three-stage pipeline, `formula_actions_in_V` should always be empty and `target_action_in_V` should always be `NO`.

If an LTS has fewer than the requested number of real actions, `V_requested_size` keeps the experimental setting and `V_size` records the actual number selected. For example, a two-action LTS under `|V|=5` will have `V_requested_size=5`, `V_size=2`, and `V_size_note="required |V|=5, actual |V|=2"`.
