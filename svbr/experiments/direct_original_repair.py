from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path

from svbr.core import HMLParser
from svbr.experiments.add_delete_run import RUN_FIELDS, yes_no
from svbr.experiments.add_delete_prepared_run import (
    ExperimentSpec,
    current_rss_mb,
    find_v_meta,
    formula_actions,
    formula_texts,
    parse_int_list,
    release_runtime_memory,
    selected_formula_cases,
    target_state_for_model,
)
from svbr.experiments.progress import print_progress
from svbr.repair import RepairConfig
from svbr.repair.add_delete import (
    RepairLTS,
    first_modal_action,
    make_ranker,
    run_repair,
    verify_formula,
)


ERROR_FIELDS = ["case_id", "model_path", "error"]


def read_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class CsvSink:
    def __init__(self, path: Path, fieldnames: list[str], append: bool = False):
        self.path = path
        self.fieldnames = fieldnames
        self.append = append
        self.handle = None
        self.writer = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.append or not self.path.exists() or self.path.stat().st_size == 0
        self.handle = self.path.open("a" if self.append else "w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames, extrasaction="ignore")
        if write_header:
            self.writer.writeheader()
        return self

    def writerow(self, row: dict) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.close()


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def completed_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row.get("case_id", "")) for row in csv.DictReader(handle) if row.get("case_id")}


def shard_accepts(case_id: str, shard_index: int, shard_count: int) -> bool:
    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count == shard_index


def direct_config(args) -> RepairConfig:
    return RepairConfig(
        repair_mode="add-delete",
        sf_setting="strict_then_escalate",
        ranker=args.ranker,
        model_path=args.ranker_model if args.ranker == "neural" else "",
        ranker_device=args.device,
        strict_ranker_device=args.strict_device,
        candidate_limit=args.candidate_limit,
        candidate_state_limit=args.candidate_state_limit,
        state_scan_limit=args.state_scan_limit,
        beam_width=args.beam_width,
        max_iters=args.max_iters,
        search_strategy=args.search_strategy,
        minimal_layer_width=args.minimal_layer_width,
        minimal_seen_limit=args.minimal_seen_limit,
        dynamic_budget=args.dynamic_repair_budget,
        dynamic_budget_rounds=args.dynamic_budget_rounds,
        dynamic_max_iters=args.dynamic_max_iters,
        dynamic_max_beam_width=args.dynamic_max_beam_width,
        dynamic_max_candidate_limit=args.dynamic_max_candidate_limit,
        dynamic_max_candidate_state_limit=args.dynamic_max_candidate_state_limit,
        dynamic_max_state_scan_limit=args.dynamic_max_state_scan_limit,
        dynamic_max_minimal_layer_width=args.dynamic_max_minimal_layer_width,
        dynamic_max_minimal_seen_limit=args.dynamic_max_minimal_seen_limit,
        dynamic_final_search_strategy=args.dynamic_final_search_strategy,
        max_case_seconds=args.max_case_seconds,
        drift_mode=args.drift_mode,
        exact_drift_max_transitions=args.exact_drift_max_transitions,
        seed=args.seed,
    )


def write_direct_script(
    path: Path,
    case_id: str,
    model_meta: dict,
    formula_id: str,
    target_state: int,
    spec: ExperimentSpec,
    v_meta: dict,
    target_text: str,
    psi_text: str,
    result,
) -> None:
    payload = result.edit_script_json()
    dump_json(
        path,
        {
            "case_id": case_id,
            "repair_surface": "original_lts",
            "model_path": model_meta["model_path"],
            "original_initial": model_meta.get("initial", ""),
            "original_target_state": target_state,
            "original_states": model_meta.get("states", ""),
            "original_transitions": model_meta.get("transitions", ""),
            "task_type": spec.task,
            "repair_mode": spec.repair_mode,
            "sf_setting": spec.sf_setting,
            "ranker": spec.ranker,
            "V_selection": spec.v_selection,
            "V_requested_size": v_meta.get("requested_size", len(v_meta["v_actions"])),
            "V_size": len(v_meta["v_actions"]),
            "V_source": v_meta.get("source", ""),
            "V_label": v_meta.get("v_label", ""),
            "formula_id": formula_id,
            "formula_kind": spec.formula_kind,
            "V_actions": v_meta["v_actions"],
            "target_formula": target_text,
            "psi": psi_text,
            "original_lts_operations": {
                "surface": "original_lts_states",
                "adds": payload["adds"],
                "dels": payload["dels"],
            },
            "result": payload,
        },
    )


def row_for_direct_result(
    model_meta: dict,
    formula_case: dict,
    formula_meta: dict,
    spec: ExperimentSpec,
    v_meta: dict,
    target_text: str,
    psi_text: str,
    result,
    out_dir: Path,
    case_id: str,
    model: RepairLTS,
    formula,
    initial_satisfied: bool,
    target_state: int,
) -> dict:
    script_path = out_dir / "edit_scripts" / f"{case_id}.json"
    write_direct_script(script_path, case_id, model_meta, formula_case["formula_id"], target_state, spec, v_meta, target_text, psi_text, result)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs" / f"{case_id}.log").write_text(result.message + "\n", encoding="utf-8")

    v_actions = set(v_meta["v_actions"])
    action_set = formula_actions(formula)
    actions_in_v = sorted(action_set & v_actions)
    first_action = first_modal_action(formula)
    metrics = result.actual_metrics
    return {
        "case_id": case_id,
        "model_path": model_meta["model_path"],
        "target_state": target_state,
        "task_type": spec.task,
        "repair_mode": spec.repair_mode,
        "sf_setting": spec.sf_setting,
        "ranker": spec.ranker,
        "ranker_architecture": "linear" if spec.ranker == "neural" else spec.ranker,
        "gnn_graph_mode": "",
        "search_strategy": "direct_original_" + spec.ranker,
        "V_requested_size": v_meta.get("requested_size", len(v_meta["v_actions"])),
        "V_size": len(v_meta["v_actions"]),
        "V_size_note": f"direct original repair; required |V|={v_meta.get('requested_size', len(v_meta['v_actions']))}, actual |V|={len(v_meta['v_actions'])}",
        "V_selection": spec.v_selection,
        "V_source": v_meta.get("source", ""),
        "V_label": v_meta.get("v_label", ""),
        "formula_id": formula_case["formula_id"],
        "formula_kind": spec.formula_kind,
        "formula_difficulty": formula_case["difficulty"],
        "formula_source": formula_case["source"],
        "formula_modal_action_count": formula_meta.get("modal_action_count", formula.modal_action_count()),
        "formula_known_action_count": formula_meta.get("known_action_count", ""),
        "formula_missing_action_count": formula_meta.get("missing_action_count", ""),
        "formula_uses_missing_actions": yes_no(bool(formula_meta.get("uses_missing_actions", False))),
        "formula_initial_satisfied": yes_no(initial_satisfied),
        "formula_first_action": first_action,
        "formula_target_action_in_lts": yes_no(bool(first_action) and first_action in model.actions),
        "formula_actions": json.dumps(formula_meta.get("formula_actions", sorted(action_set)), ensure_ascii=False),
        "formula_actions_in_V": json.dumps(actions_in_v, ensure_ascii=False),
        "all_formula_actions_in_V": yes_no(bool(action_set) and action_set.issubset(v_actions)),
        "any_formula_action_in_V": yes_no(bool(actions_in_v)),
        "V_actions": ":".join(v_meta["v_actions"]),
        "target_action_in_V": yes_no(bool(first_action) and first_action in v_actions),
        "formula": target_text if spec.task == "positive" else "",
        "psi": psi_text,
        "target_formula": target_text,
        "states": model.state_count,
        "transitions": model.transition_count,
        "actions": len(model.actions),
        "success": yes_no(result.success),
        "verified": yes_no(result.verified),
        "add_edges": metrics.add_edges,
        "del_edges": metrics.del_edges,
        "nonV_add_edges": metrics.non_v_add_edges,
        "nonV_del_edges": metrics.non_v_del_edges,
        "quotient_drift": metrics.quotient_drift,
        "raw_cost": f"{result.raw_metrics.cost:.6f}",
        "actual_cost": f"{result.actual_metrics.cost:.6f}",
        "verifier_calls": result.verifier_calls,
        "cex_iters": result.cex_iters,
        "post_removed_add": result.post_removed_add,
        "post_restored_del": result.post_restored_del,
        "elapsed_ms": f"{result.elapsed_ms:.3f}",
        "stage": result.stage,
        "message": result.message,
        "repaired_path": "",
        "edit_script_path": str(script_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct original-LTS repair baseline from a prepared manifest")
    parser.add_argument("--prepared-dir", default="results6/add_delete_prepared")
    parser.add_argument("--results-root", default="results6/direct_original_contextual_full")
    parser.add_argument("--suite-name", default="direct_original_add_delete", help="Output suite directory below --results-root")
    parser.add_argument("--ranker", choices=["heuristic", "neural", "random"], default="neural")
    parser.add_argument("--ranker-model", default="models/add_delete_ranker_lightweight_contextual_linear.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strict-device", action="store_true")
    parser.add_argument("--v-sizes", default="0,1,3,5")
    parser.add_argument("--v-selection", choices=["formula_safe", "unsafe"], default="formula_safe")
    parser.add_argument("--formula-limit", type=int, default=0)
    parser.add_argument("--formula-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0, help="Maximum attempted direct repair cases; 0 runs the full workload")
    parser.add_argument("--max-original-states", type=int, default=0, help="Skip larger original LTS models; 0 disables filtering")
    parser.add_argument("--max-original-transitions", type=int, default=0, help="Skip larger original LTS models; 0 disables filtering")
    parser.add_argument("--target-state", type=int, default=-1)
    parser.add_argument("--skip-initially-satisfied", action="store_true")
    parser.add_argument("--no-skip-initially-satisfied", dest="skip_initially_satisfied", action="store_false")
    parser.set_defaults(skip_initially_satisfied=True)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=16)
    parser.add_argument("--candidate-limit", type=int, default=64)
    parser.add_argument("--candidate-state-limit", type=int, default=128)
    parser.add_argument("--state-scan-limit", type=int, default=5000)
    parser.add_argument("--search-strategy", choices=["beam", "neural_guided_minimal"], default="beam")
    parser.add_argument("--minimal-layer-width", type=int, default=2048)
    parser.add_argument("--minimal-seen-limit", type=int, default=500000)
    parser.add_argument("--dynamic-repair-budget", action="store_true")
    parser.add_argument("--no-dynamic-repair-budget", dest="dynamic_repair_budget", action="store_false")
    parser.set_defaults(dynamic_repair_budget=True)
    parser.add_argument("--dynamic-budget-rounds", type=int, default=1)
    parser.add_argument("--dynamic-max-iters", type=int, default=64)
    parser.add_argument("--dynamic-max-beam-width", type=int, default=32)
    parser.add_argument("--dynamic-max-candidate-limit", type=int, default=256)
    parser.add_argument("--dynamic-max-candidate-state-limit", type=int, default=512)
    parser.add_argument("--dynamic-max-state-scan-limit", type=int, default=10000)
    parser.add_argument("--dynamic-max-minimal-layer-width", type=int, default=8192)
    parser.add_argument("--dynamic-max-minimal-seen-limit", type=int, default=100000)
    parser.add_argument("--dynamic-final-search-strategy", default="beam")
    parser.add_argument("--max-case-seconds", type=float, default=300.0)
    parser.add_argument("--drift-mode", choices=["estimate", "exact"], default="estimate")
    parser.add_argument("--exact-drift-max-transitions", type=int, default=200000)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--trim-memory-every-case", action="store_true")
    parser.add_argument("--no-trim-memory-every-case", dest="trim_memory_every_case", action="store_false")
    parser.set_defaults(trim_memory_every_case=True)
    parser.add_argument("--resume", action="store_true", help="Append to an existing shard CSV and skip completed case IDs")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--shard-count", type=int, default=1, help="Split the deterministic case stream into this many independent shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index for this process")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.shard_count <= 0:
        parser.error("--shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        parser.error("--shard-index must satisfy 0 <= index < shard-count")

    prepared_dir = Path(args.prepared_dir)
    manifest = read_manifest(prepared_dir / "manifest.json")
    suite_name = args.suite_name
    if args.shard_count > 1:
        suite_name += f"_shard_{args.shard_index:03d}_of_{args.shard_count:03d}"
    out_dir = Path(args.results_root) / suite_name
    out_dir.mkdir(parents=True, exist_ok=True)
    config = direct_config(args)
    ranker = make_ranker(config)
    v_sizes = parse_int_list(args.v_sizes)
    formula_kinds = ["positive", "negative_existential", "negative_universal"]
    total_limit = max(0, args.limit)
    completed = completed_case_ids(out_dir / "runs.csv") if args.resume else set()
    resumed_runs = len(completed)
    runs = resumed_runs
    new_runs = 0
    errors = 0
    skipped_by_shard = 0
    skipped_by_size = 0
    skipped_completed = 0

    print(
        "Direct original baseline start: "
        f"root={Path(args.results_root)} suite={suite_name} resume={args.resume} completed={resumed_runs} "
        f"limit={total_limit or 'all'} shard={args.shard_index}/{args.shard_count}"
    )
    with CsvSink(out_dir / "runs.csv", RUN_FIELDS, append=args.resume) as runs_sink, CsvSink(out_dir / "errors.csv", ERROR_FIELDS, append=args.resume) as errors_sink:
        for model_meta in manifest.get("models", []):
            if total_limit and runs >= total_limit:
                break
            states = int(model_meta.get("states", 0) or 0)
            transitions = int(model_meta.get("transitions", 0) or 0)
            if args.max_original_states > 0 and states > args.max_original_states:
                skipped_by_size += 1
                continue
            if args.max_original_transitions > 0 and transitions > args.max_original_transitions:
                skipped_by_size += 1
                continue
            base_model = RepairLTS.from_aut(model_meta["model_path"])
            target_state = target_state_for_model(model_meta, args)
            if target_state != base_model.initial:
                base_model = RepairLTS(target_state, base_model.state_count, base_model.edges)
            model_meta = dict(model_meta)
            model_meta["effective_target_state"] = target_state
            try:
                for formula_case in selected_formula_cases(model_meta, args):
                    for formula_kind in formula_kinds:
                        for v_size in v_sizes:
                            if total_limit and runs >= total_limit:
                                break
                            spec = ExperimentSpec(
                                "direct_original",
                                f"direct_{formula_kind}_V{v_size}",
                                "positive" if formula_kind == "positive" else "negative",
                                "add-delete",
                                "strict_then_escalate",
                                formula_kind=formula_kind,
                                v_size=v_size,
                                v_selection=args.v_selection,
                                ranker=args.ranker,
                            )
                            case_id = f"{model_meta['model_id']}_{formula_case['formula_id']}_{spec.out_name}"
                            if not shard_accepts(case_id, args.shard_index, args.shard_count):
                                skipped_by_shard += 1
                                continue
                            if case_id in completed:
                                skipped_completed += 1
                                continue
                            try:
                                v_meta = find_v_meta(model_meta, spec, formula_case)
                                target_text, psi_text, formula_meta = formula_texts(spec, formula_case)
                                formula = HMLParser.parse(target_text)
                                initial_satisfied, _checker = verify_formula(base_model, formula)
                                if args.skip_initially_satisfied and initial_satisfied:
                                    continue
                                result = run_repair(
                                    base_model,
                                    formula,
                                    set(v_meta["v_actions"]),
                                    config,
                                    original_quotient=None,
                                    ranker=ranker,
                                    case_id=case_id,
                                )
                                row = row_for_direct_result(
                                    model_meta,
                                    formula_case,
                                    formula_meta,
                                    spec,
                                    v_meta,
                                    target_text,
                                    psi_text,
                                    result,
                                    out_dir,
                                    case_id,
                                    base_model,
                                    formula,
                                    initial_satisfied,
                                    target_state,
                                )
                                runs_sink.writerow(row)
                                runs += 1
                                new_runs += 1
                                if args.progress_every > 0 and (runs == total_limit or runs % args.progress_every == 0):
                                    print_progress("direct-original", runs, total_limit or runs, new=new_runs, current=case_id, rss_mb=current_rss_mb())
                            except Exception as exc:
                                errors += 1
                                errors_sink.writerow({"case_id": case_id, "model_path": model_meta.get("model_path", ""), "error": repr(exc)})
                            finally:
                                if args.trim_memory_every_case:
                                    release_runtime_memory(args.device, trim_process=True)
                    if total_limit and runs >= total_limit:
                        break
            finally:
                base_model._adjacency_cache = None
                base_model._edge_adjacency_cache = None
                base_model._actions_cache = None
                base_model._degree_cache = None
                gc.collect()
    print(
        "Direct original baseline complete: "
        f"runs={runs} resumed={resumed_runs} new={new_runs} errors={errors} "
        f"skipped_completed={skipped_completed} skipped_by_shard={skipped_by_shard} "
        f"skipped_models_by_size={skipped_by_size} root={Path(args.results_root)}"
    )


if __name__ == "__main__":
    main()
