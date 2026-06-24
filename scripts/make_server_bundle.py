from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REQUIRED_FILES = [
    ".gitignore",
    "LICENSE",
    "README.md",
    "SERVER_UPLOAD_GUIDE.md",
    "requirements.txt",
    "pyproject.toml",
    "run_add_delete_all.sh",
    "run_server_nnunet_env.sh",
    "run_server_gpu_all_env.sh",
    "run_server_dual_compare_env.sh",
    "run_server_results6_linear_env.sh",
    "run_server_direct_original_full_env.sh",
    "run_server_direct_vs_quotient_full_env.sh",
]

REQUIRED_SCRIPT_FILES = [
    "scripts/make_server_bundle.py",
    "scripts/materialize_repaired_aut.sh",
    "scripts/run_add_delete_stage1_prepare.sh",
    "scripts/run_add_delete_stage1_prepare_gpu.sh",
    "scripts/run_add_delete_stage2_gpu.sh",
    "scripts/run_ranker_compare_stage2_gpu.sh",
]

REQUIRED_DIRS = [
    "svbr",
    "tests",
]


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def copy_aut_data(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        raise SystemExit(f"Data directory not found: {src_dir}")
    copied = 0
    for path in sorted(src_dir.rglob("*.aut")):
        relative = path.relative_to(src_dir)
        copy_file(path, dst_dir / relative)
        copied += 1
    return copied


def copy_test_fixtures(root: Path, bundle_dir: Path) -> int:
    copied = 0
    data_dir = root / "data"
    for path in sorted(data_dir.glob("*.aut")):
        copy_file(path, bundle_dir / "data" / path.name)
        copied += 1
    generated_dir = data_dir / "generated"
    if generated_dir.exists():
        for path in sorted(generated_dir.glob("*.aut")):
            copy_file(path, bundle_dir / "data" / "generated" / path.name)
            copied += 1
    return copied


def set_stage3_cex_default(bundle_dir: Path, default_iters: int) -> None:
    targets = [
        bundle_dir / "run_add_delete_all.sh",
        bundle_dir / "run_server_nnunet_env.sh",
        bundle_dir / "run_server_gpu_all_env.sh",
        bundle_dir / "run_server_dual_compare_env.sh",
        bundle_dir / "run_server_results6_linear_env.sh",
        bundle_dir / "run_server_direct_original_full_env.sh",
        bundle_dir / "run_server_direct_vs_quotient_full_env.sh",
        bundle_dir / "scripts" / "materialize_repaired_aut.sh",
    ]
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace(
            'STAGE3_CEX_ITERS="${STAGE3_CEX_ITERS:-16}"',
            f'STAGE3_CEX_ITERS="${{STAGE3_CEX_ITERS:-{default_iters}}}"',
        )
        text = text.replace(
            'STAGE3_CEX_ITERS="${STAGE3_CEX_ITERS:-0}"',
            f'STAGE3_CEX_ITERS="${{STAGE3_CEX_ITERS:-{default_iters}}}"',
        )
        path.write_text(text, encoding="utf-8", newline="\n")


def set_server_aut_dir(bundle_dir: Path, server_aut_dir: str) -> None:
    targets = [
        bundle_dir / "run_add_delete_all.sh",
        bundle_dir / "run_server_nnunet_env.sh",
        bundle_dir / "run_server_gpu_all_env.sh",
        bundle_dir / "run_server_dual_compare_env.sh",
        bundle_dir / "run_server_results6_linear_env.sh",
        bundle_dir / "run_server_direct_original_full_env.sh",
        bundle_dir / "run_server_direct_vs_quotient_full_env.sh",
        bundle_dir / "scripts" / "run_add_delete_stage1_prepare.sh",
        bundle_dir / "scripts" / "run_add_delete_stage1_prepare_gpu.sh",
        bundle_dir / "SERVER_UPLOAD_GUIDE.md",
    ]
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        text = text.replace("/root/sj-tmp/data/download", server_aut_dir)
        path.write_text(text, encoding="utf-8", newline="\n")


def normalize_shell_line_endings(bundle_dir: Path) -> None:
    for path in [*bundle_dir.glob("*.sh"), *bundle_dir.glob("scripts/*.sh")]:
        data = path.read_bytes()
        fixed = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        if fixed != data:
            path.write_bytes(fixed)


def write_manifest(bundle_dir: Path, data_count: int, fixture_count: int, include_data: bool, stage3_cex_iters_default: int, server_aut_dir: str) -> None:
    text = [
        "# Server Bundle",
        "",
        "Run the GPU-all version on the server:",
        "",
        "```bash",
        "chmod +x run_server_nnunet_env.sh run_server_gpu_all_env.sh run_server_dual_compare_env.sh run_server_results6_linear_env.sh run_server_direct_original_full_env.sh run_server_direct_vs_quotient_full_env.sh run_add_delete_all.sh scripts/*.sh",
        "DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_gpu_all_env.sh",
        "```",
        "",
        "Run only the quotient main method and the full direct-original baseline:",
        "",
        "```bash",
        "DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_direct_vs_quotient_full_env.sh",
        "```",
        "",
        "Supplement an existing comparison with the full direct original-LTS baseline only:",
        "",
        "```bash",
        "DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_direct_original_full_env.sh",
        "```",
        "",
        "Run the experiment-6 linear-ranker baseline/ablation comparison:",
        "",
        "```bash",
        "DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_results6_linear_env.sh",
        "```",
        "",
        "The old dual script name is kept as a compatibility wrapper and now runs the same experiment-6 linear comparison:",
        "",
        "```bash",
        "DEVICE=cuda STRICT_DEVICE=1 CUDA_VISIBLE_DEVICES=0 bash run_server_dual_compare_env.sh",
        "```",
        "",
        "The older CPU+GPU runner is still included for comparison:",
        "",
        "```bash",
        "bash run_server_nnunet_env.sh",
        "```",
        "",
        "This bundle is configured for an environment that already has CUDA PyTorch installed,",
        "for example torch 2.11.0+cu13 in nnunet_env. Do not reinstall torch unless you",
        "intentionally want to replace the current CUDA build.",
        "",
        "Default forgotten-action sizes: |V|=0,1,3,5.",
        f"Default server AUT_DIR: `{server_aut_dir}`.",
        "Override it at runtime with `AUT_DIR=/path/to/aut_dir` if your dataset is stored elsewhere.",
        "V is selected per formula, and formula actions are excluded from V.",
        "Default formula suite per LTS: easy=5, medium=10, hard=15;",
        "20 formulas use only existing LTS actions and 10 mix existing/missing actions.",
        "Mixed formulas randomize where missing actions appear.",
        "All 30 positive formulas are generated to be initially unsatisfied.",
        "Stage 1 keeps regenerating formulas for the current LTS until the quota is met.",
        "Stage 1 rejects target formulas that are logically unsatisfiable in every LTS.",
        "",
        "Three-stage pipeline:",
        "",
        "1. Stage 1 writes formulas plus strong-V quotient LTS' files; the GPU-all runner uses torch/CUDA for quotient tensor work.",
        "2. Stage 2 trains/uses the neural ranker on GPU, repairs quotient LTS' into LTS'', and checks HML on LTS''.",
        "3. Stage 3 writes successful block-level repair operations back to original LTS files and verifies them.",
        "   LTS'' block-level edits are the authoritative write-back templates.",
        "   Counterexamples only prioritize concrete representatives inside the template blocks.",
        "   Stage 3 generates concrete edits only from Stage 2 quotient templates.",
        "   By default Stage 3 uses counterexample-guided closure lifting in one output directory.",
        "   Existential (<>) paths keep one concrete repair edge when one path suffices.",
        "   Universal ([]) obligations keep adding only concrete branches exposed by Stage 3 counterexamples.",
        f"   Counterexample-guided lifting starts at STAGE3_CEX_ITERS={stage3_cex_iters_default}.",
        "   If verification still fails, closure mode doubles the budget up to the original LTS transition count.",
        "   Independent universal counterexample edits are batched with STAGE3_CEX_BATCH_SIZE=512.",
        "   Stage 3 prints stage3-case before each edit script by default: STAGE3_CASE_START_EVERY=1.",
        "   Long lift cases print stage3-lift progress every STAGE3_CASE_PROGRESS_EVERY=100 concrete edits.",
        "   Stage 3 always keeps minimal required template instances; it does not fill whole blocks.",
        "   STAGE3_MAX_CASE_SECONDS=0 by default, so slow cases are not abandoned for timeout.",
        "   Existing repaired_aut files are skipped by default: FORCE_MATERIALIZE=0.",
        "   Stage 3 trims process memory after each case by default: STAGE3_TRIM_MEMORY_EVERY_CASE=1.",
        "   Unresolved Stage 3 cases are written to materialize_repaired_aut_unresolved.csv",
        "   and stage3_unresolved_scripts.txt, which can be retried with STAGE3_SCRIPT_LIST.",
        "   Set STAGE3_COMPARE_CEX=1 only when you explicitly want an extra single-edge/no-cex comparison output.",
        "",
        "The GPU-all attempt keeps the same three-stage logic, but uses the torch/CUDA",
        "quotient backend in Stage 1. AUT parsing, Python beam search/HML recursion,",
        "and final file materialization remain CPU/Python work because they are not tensor kernels.",
        "The GPU-all runner requires STRICT_DEVICE=1, DEVICE=cuda*, and QUOTIENT_DEVICE=cuda*.",
        "If CUDA is unavailable it stops instead of silently falling back to CPU.",
        "Its main Stage 2 experiment groups use DEFAULT_RANKER=neural by default.",
        "The stable neural ranker is contextual linear by default: RANKER_ARCHITECTURE=linear, LINEAR_FEATURE_SET=current.",
        "Stage 2 uses fast beam search by default: SEARCH_STRATEGY=beam.",
        "DYNAMIC_REPAIR_BUDGET=1 retries failed cases with geometrically wider search/candidate budgets.",
        "DYNAMIC_BUDGET_ROUNDS=0 keeps growing failed cases until the configured safety ceilings are saturated.",
        "The final retry can use DYNAMIC_FINAL_SEARCH_STRATEGY=neural_guided_minimal,",
        "and exact HML verification remains the only success criterion.",
        "Old prepared manifests with logically unsatisfiable target formulas are rejected before dynamic search; rerun Stage 1.",
        "run_server_results6_linear_env.sh prepares Stage 1 once, then runs Stage 2 and Stage 3",
        "for `heuristic_baseline`, `random_baseline`, `legacy8_ablation`,",
        "`fixed_budget_contextual`, `unsafe_v_contextual`, `add_only_contextual`,",
        "`delete_only_contextual`, `direct_original_contextual_full`, and",
        "`lightweight_contextual_linear`, followed by paired comparison reports.",
        "MLP/GNN probes are intentionally not run in the experiment-6 script.",
        "Stage 3 follows the current counterexample path downward and keeps one concrete add representative per step.",
        "The CPU heuristic comparison group is skipped by default: INCLUDE_HEURISTIC_COMPARISON=0.",
        "",
        "All three stages print progress bars.",
        "Stage 2 skips target formulas that are already satisfied by quotient LTS'.",
        "",
        "Outputs:",
        "",
        "- `results/add_delete_run/*/runs.csv`",
        "- `results/add_delete_run/*/errors.csv`",
        "- `results/add_delete_run/*/repaired_aut/*.aut`",
        "- `results/add_delete_run/*/edit_scripts/*.json`",
        "- `results/add_delete_run/sf_vs_no_sf_summary.csv`",
        "- `results/add_delete_run/repair_mode_summary.csv`",
        "- `results/add_delete_run/postprocess_summary.csv`",
        "- `results/add_delete_run/ranker_summary.csv`",
        "- `results/add_delete_run/ranker_training.json`",
        "- `results/add_delete_run/*_by_formula.csv`",
        "- `results/add_delete_run/materialize_repaired_aut.csv`",
        "- `results/add_delete_run/*/writeback_operations/*.json`",
        "- `results6/linear_ranker_ablation/baseline_vs_contextual_linear_report.md`",
        "- `results6/linear_ranker_ablation/random_vs_contextual_linear_report.md`",
        "- `results6/linear_ranker_ablation/legacy8_ablation_vs_contextual_linear_report.md`",
        "- `results6/linear_ranker_ablation/fixed_budget_vs_contextual_linear_report.md`",
        "- `results6/linear_ranker_ablation/unsafe_v_vs_formula_safe_contextual_report.md`",
        "- `results6/linear_ranker_ablation/add_only_vs_add_delete_contextual_report.md`",
        "- `results6/linear_ranker_ablation/delete_only_vs_add_delete_contextual_report.md`",
        "- `results6/linear_ranker_ablation/direct_original_full_vs_quotient_contextual_report.md`",
        "- `results6/linear_ranker_ablation/reviewer_ablation_report.md`",
        "- `results6/linear_ranker_ablation/reviewer_ablation_quotient_stats.csv`",
        "- `results6/linear_ranker_ablation/reviewer_ablation_worst_cases.csv`",
        "- `results6/linear_ranker_ablation/*_overall.csv`",
        "- `results6/linear_ranker_ablation/*_by_stratum.csv`",
        "- `results6/linear_ranker_ablation/*_case_deltas.csv`",
        "- `results6/linear_ranker_ablation/*_paired_summary.csv`",
        "",
        "Stage-2-only ranker comparison after Stage 1:",
        "",
        "```bash",
        "PREPARED_DIR=results/add_delete_prepared_gpu_all RESULTS_ROOT=results/ranker_compare_stage2 DEVICE=cuda STRICT_DEVICE=1 bash scripts/run_ranker_compare_stage2_gpu.sh",
        "```",
        "",
        "The Stage-2-only helper compares `legacy8_ablation` against",
        "`lightweight_contextual_linear`; the end-to-end results6 script is preferred",
        "when you need Stage 3 materialization metrics in the same report.",
        "",
        "This writes legacy8_ablation_vs_contextual_linear_stage2_*.csv/md.",
        "",
        "Stage 3 can also be run directly after Stage 2:",
        "",
        "```bash",
        "PREPARED_DIR=results/add_delete_prepared RESULTS_ROOT=results/add_delete_run bash scripts/materialize_repaired_aut.sh",
        "PREPARED_DIR=results/add_delete_prepared RESULTS_ROOT=results/add_delete_run STAGE3_CEX_LIFT_MODE=closure STAGE3_CEX_ITERS=16 bash scripts/materialize_repaired_aut.sh",
        "# In closure mode this is the initial budget; Stage 3 doubles it until verified or the original LTS transition count is reached.",
        "```",
        "",
        f"Included data: {'YES' if include_data else 'NO'}",
        f"AUT files copied: {data_count}",
        f"Small test fixtures copied: {fixture_count}",
        "",
    ]
    (bundle_dir / "SERVER_BUNDLE_README.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean upload bundle for the strong-forgetting add/delete repair experiment")
    parser.add_argument("--out-dir", default="server_bundle/svbr_strong_forgetting")
    parser.add_argument("--include-data", action="store_true")
    parser.add_argument("--data-dir", default="data/dowanload", help="local data directory copied when --include-data is set")
    parser.add_argument("--server-aut-dir", default="/root/sj-tmp/data/download", help="default AUT_DIR written into bundled server scripts")
    parser.add_argument("--clean", action="store_true", help="remove --out-dir before creating the bundle")
    parser.add_argument("--stage3-cex-iters-default", type=int, default=16, help="default STAGE3_CEX_ITERS written into copied runner scripts")
    args = parser.parse_args()

    root = Path.cwd()
    bundle_dir = Path(args.out_dir)
    if args.clean and bundle_dir.exists():
        resolved = bundle_dir.resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            raise SystemExit(f"Refusing to remove outside workspace: {resolved}")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    for relative in REQUIRED_FILES:
        copy_file(root / relative, bundle_dir / relative)
    for relative in REQUIRED_SCRIPT_FILES:
        copy_file(root / relative, bundle_dir / relative)
    for relative in REQUIRED_DIRS:
        copy_tree(root / relative, bundle_dir / relative)

    fixture_count = copy_test_fixtures(root, bundle_dir)
    data_count = 0
    if args.include_data:
        data_count = copy_aut_data(root / args.data_dir, bundle_dir / args.data_dir)

    set_stage3_cex_default(bundle_dir, args.stage3_cex_iters_default)
    set_server_aut_dir(bundle_dir, args.server_aut_dir)
    normalize_shell_line_endings(bundle_dir)
    write_manifest(bundle_dir, data_count, fixture_count, args.include_data, args.stage3_cex_iters_default, args.server_aut_dir)
    print(f"Bundle: {bundle_dir}")
    print(f"AUT files copied: {data_count}")
    print(f"Small test fixtures copied: {fixture_count}")


if __name__ == "__main__":
    main()
