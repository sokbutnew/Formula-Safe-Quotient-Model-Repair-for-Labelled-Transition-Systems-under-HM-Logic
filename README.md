# Strong-Forgetting-Priority Add/Delete Repair Experiments

This repository is organized for the uploaded MD experiment:

```text
Strong-Forgetting-Priority Add--Delete Repair for Labelled Transition Systems
```

It runs fixed-state LTS repair over CADP/Aldebaran `.aut` files and writes auditable repair outputs:

```text
runs.csv
errors.csv
summary.json
repaired_aut/
edit_scripts/
writeback_operations/
logs/
```

## Main Command

```bash
python -m svbr.experiments.add_delete_run --help
```

## One-Command Server Run

```bash
chmod +x run_server_nnunet_env.sh run_server_gpu_all_env.sh run_server_dual_compare_env.sh run_server_results6_linear_env.sh run_add_delete_all.sh scripts/*.sh
bash run_server_nnunet_env.sh
```

GPU-all attempt version:

```bash
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_gpu_all_env.sh
```

The GPU-all runner requires `STRICT_DEVICE=1`, `DEVICE=cuda*`, and `QUOTIENT_DEVICE=cuda*`; if CUDA is unavailable it stops instead of silently falling back to CPU. Its main experiment groups use the neural ranker by default (`DEFAULT_RANKER=neural`) and skip the CPU heuristic comparison by default (`INCLUDE_HEURISTIC_COMPARISON=0`). Set `INCLUDE_HEURISTIC_COMPARISON=1` only when you explicitly want that control group.

Experiment-6 linear-ranker baseline/ablation comparison, with Stage 1 prepared once and Stage 2/Stage 3 run for all three controls:

```bash
DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_results6_linear_env.sh
```

See [SERVER_UPLOAD_GUIDE.md](SERVER_UPLOAD_GUIDE.md) for the upload list, one-command run, and expected outputs.

## Three Stages

Stage 1 parses each selected `.aut`, generates a per-model formula suite, and precomputes strong-V quotient files. In the normal runner this is the CPU prepare stage; in the GPU-all runner the quotient projection/unique tensor work uses torch/CUDA. It outputs `LTS'` quotient pickles plus formulas:

- default forgotten-action sizes are `|V|=0,1,3,5`.
- if an LTS has fewer actions than a requested `|V|`, the run records both requested and actual sizes.
- 30 formulas per LTS by default.
- difficulty split: easy=5, medium=10, hard=15.
- 20 formulas use only actions that occur in that LTS.
- 10 formulas mix existing actions with generated missing actions.
- mixed formulas randomize where the missing action appears, so the first target action is not always missing.
- all 30 positive formulas are initially unsatisfied and therefore need repair.
- Stage 1 keeps regenerating formulas for the current LTS until the unsatisfied quota is met before moving to the next LTS.
- Stage 1 rejects HML targets that are logically unsatisfiable in every LTS, for example modal contradictions such as `<a>true & [a]false` after negation/NNF rewriting.
- each formula has 5-10 modal action occurrences with `&`, `|`, `<>`, and `[]`.
- `V` is selected per formula, and every action that appears in that formula is excluded from `V`.
- easy / medium / hard difficulty metadata is stored in `manifest.json`.

Stage 2 consumes only the prepared manifest and quotient files. The neural ranker is trained from prepared `LTS'` repair candidates when needed, loaded once in the Python process, and reused across all repair runs. The stable GPU runner uses a lightweight contextual linear neural ranker (`RANKER_ARCHITECTURE=linear`, `LINEAR_FEATURE_SET=current`) with fast beam search (`SEARCH_STRATEGY=beam`). Its features include formula modal depth, the active formula-branch kinds, required-action matching, counterexample-path membership, whether the destination satisfies the next subformula, current edit count, and remaining modal depth. If a case fails under the base budget, `DYNAMIC_REPAIR_BUDGET=1` retries it with geometrically wider search and candidate budgets. `DYNAMIC_BUDGET_ROUNDS=0` keeps growing failed cases until the configured safety ceilings are saturated; the final retry can switch to `DYNAMIC_FINAL_SEARCH_STRATEGY=neural_guided_minimal`. Exact HML verification remains the only acceptance criterion. Frontiers and seen-key sets are bounded to control RSS during retries. The checkpoint includes a prepared-manifest signature to avoid reusing a stale ranker.
The signature includes formula texts and initial-satisfaction flags, so changing formula generation retrains the ranker instead of reusing a stale checkpoint.
Old prepared manifests containing logically unsatisfiable target formulas are rejected before search; rerun Stage 1 after updating the code.
By default, Stage 2 skips any target formula already satisfied by `LTS'` and records it in `skipped_initially_satisfied.csv`.
The server scripts run Stage 1 as a CPU-only prepare step and reserve CUDA/GPU checks for Stage 2.
With `DEVICE=cuda`, Stage 2 trains the neural ranker and scores neural-ranker candidate batches on GPU; formula checking and graph search still run on CPU.
Stage 2 repairs and checks `LTS'`, producing `LTS''` logically, but writes block-level edit scripts by default instead of full repaired AUT files.

Ranker comparison after Stage 1, Stage 2 only:

```bash
PREPARED_DIR=results/add_delete_prepared_gpu_all \
RESULTS_ROOT=results/ranker_compare_stage2 \
DEVICE=cuda STRICT_DEVICE=1 \
bash scripts/run_ranker_compare_stage2_gpu.sh
```

The preferred end-to-end comparison is `run_server_results6_linear_env.sh`: it runs Stage 1 once, then compares `heuristic_baseline`, `random_baseline`, `legacy8_ablation` (the older 8-feature linear ranker), `fixed_budget_contextual`, `unsafe_v_contextual`, `add_only_contextual`, `delete_only_contextual`, the full `direct_original_contextual_full` baseline, and `lightweight_contextual_linear` (the experiment-6 27-feature contextual linear ranker). The full direct-original baseline uses the same workload and fair per-case cap as the quotient method. It supports resume and deterministic sharding because direct original-LTS search is intentionally expensive. To supplement an existing experiment without rerunning Stage 1 or the quotient method, run `run_server_direct_original_full_env.sh`. The old `run_server_dual_compare_env.sh` name is kept as a compatibility wrapper. MLP/GNN probes are intentionally not run.

For a focused two-method paper comparison, run `run_server_direct_vs_quotient_full_env.sh`. It prepares Stage 1 once and executes only `lightweight_contextual_linear` and `direct_original_contextual_full`, writing isolated outputs under `results8/direct_vs_quotient_full`.
All compared Stage 2 methods use the same fair per-case wall-clock cap by default (`FAIR_MAX_CASE_SECONDS=300`, i.e. 5 minutes). Verifier calls are recorded as an efficiency metric, but they are not capped in the fair comparison runner. Cases that exceed the common time budget are reported as failures/timeouts for that method. The comparison reports include per-method success/verified rates, timeout counts/rates, average/median/p90 time, verifier-call statistics, average/median edit counts, Stage 3 write-back verification rate, end-to-end verification rate, Stage 3 concrete edge expansion, and worst-case tables.

Stage 3 materializes successful block-level edit scripts back into the original LTS files one at a time, then verifies the target HML formula on the materialized original LTS. The repaired quotient `LTS''` edit script is the authoritative template: each concrete write-back edge is generated from the template's source block, action label, and destination block. Counterexamples only prioritize which concrete representative states inside those template blocks are instantiated first; Stage 3 does not invent new block-level add/delete operations. Concrete add lifting follows the current counterexample path downward and selects one satisfying destination representative per repair step instead of retaining many same-level block states. An existential obligation (`<a>phi`) keeps one concrete `a`-successor repair path when one path suffices. A universal obligation (`[a]phi`) processes only the concrete `a`-successor branches still required to satisfy `phi`, batching independent required edits; it does not fill every state in a template block. `[]` ranges over all matching successors of its action label, not over unrelated LTS action labels. Each materialized case writes `writeback_operations/*.json`, listing both the Stage 2 `LTS' -> LTS''` block edits and the Stage 3 original-LTS concrete edits. The default counterexample budget starts at `STAGE3_CEX_ITERS=16`; closure lifting doubles it when verification still fails, up to the original LTS transition count. `STAGE3_MAX_CASE_SECONDS=0` by default, so Stage 3 does not abandon slow cases for timeout when the goal is verified materialization.

Each stage prints an ASCII progress bar.

The GPU-all attempt keeps the same three-stage logic, but Stage 1 uses the
torch/CUDA quotient backend. AUT parsing, Python beam search/HML recursion, and
Stage 3 file materialization remain CPU/Python work because they are not tensor
kernels.

## Experiment Groups

- A: `no_sf` / `soft_sf` / `strict_then_escalate`
- B: formula-safe `V` sizes, where formula actions are never forgotten
- C: `add-only` / `delete-only` / `add-delete`
- D: post-processing off vs on
- E: heuristic/random/8-feature/fixed-budget/unsafe-V/add-only/delete-only/direct-original baselines vs the 27-feature contextual linear ranker

## Smoke Test

```bash
python -m unittest discover -s tests
python -m svbr.experiments.add_delete_run \
  --aut-dir data/hml_deadlock.aut \
  --out-dir results/smoke \
  --task positive \
  --formula '<z>true' \
  --repair-mode add-only \
  --sf-setting strict_then_escalate \
  --V z \
  --limit 1
```
