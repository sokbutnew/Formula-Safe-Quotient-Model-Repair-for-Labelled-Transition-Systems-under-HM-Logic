from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path

from svbr.core import HMLParser, parse_aut_header
from svbr.repair import CostConfig, RepairConfig, RepairLTS
from svbr.repair.add_delete import choose_v_actions, dump_json, first_modal_action, formula_actions, parse_v_actions, run_repair
from svbr.repair.add_delete import make_ranker


RUN_FIELDS = [
    "case_id",
    "model_path",
    "target_state",
    "task_type",
    "repair_mode",
    "sf_setting",
    "ranker",
    "ranker_architecture",
    "gnn_graph_mode",
    "search_strategy",
    "V_requested_size",
    "V_size",
    "V_size_note",
    "V_selection",
    "V_source",
    "V_label",
    "formula_id",
    "formula_kind",
    "formula_difficulty",
    "formula_source",
    "formula_modal_action_count",
    "formula_known_action_count",
    "formula_missing_action_count",
    "formula_uses_missing_actions",
    "formula_initial_satisfied",
    "formula_first_action",
    "formula_target_action_in_lts",
    "formula_actions",
    "formula_actions_in_V",
    "all_formula_actions_in_V",
    "any_formula_action_in_V",
    "V_actions",
    "target_action_in_V",
    "formula",
    "psi",
    "target_formula",
    "states",
    "transitions",
    "actions",
    "success",
    "verified",
    "add_edges",
    "del_edges",
    "nonV_add_edges",
    "nonV_del_edges",
    "quotient_drift",
    "raw_cost",
    "actual_cost",
    "verifier_calls",
    "cex_iters",
    "post_removed_add",
    "post_restored_del",
    "elapsed_ms",
    "stage",
    "message",
    "repaired_path",
    "edit_script_path",
]


ERROR_FIELDS = ["case_id", "model_path", "error"]


def discover_aut_files(aut_dir: str | Path, recursive: bool) -> list[Path]:
    root = Path(aut_dir)
    if root.is_file():
        return [root] if root.suffix == ".aut" else []
    pattern = "**/*.aut" if recursive else "*.aut"
    return sorted(root.glob(pattern))


def read_header(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
        initial, transitions, states = parse_aut_header(handle.readline())
    return {"path": path, "initial": initial, "transitions": transitions, "states": states}


def select_inputs(args) -> tuple[list[dict], list[dict]]:
    all_rows = []
    selected = []
    for path in discover_aut_files(args.aut_dir, args.recursive):
        try:
            row = read_header(path)
        except Exception as exc:
            all_rows.append({"path": path, "selected": False, "reason": f"header_error:{exc}", "states": "", "transitions": ""})
            continue
        row["selected"] = True
        row["reason"] = ""
        if row["states"] > args.max_states:
            row["selected"] = False
            row["reason"] = f"states>{args.max_states}"
        elif row["transitions"] > args.max_transitions:
            row["selected"] = False
            row["reason"] = f"transitions>{args.max_transitions}"
        all_rows.append(row)
    selected = [row for row in all_rows if row.get("selected")]
    selected.sort(key=lambda item: (item["states"], item["transitions"], str(item["path"])))
    if args.limit > 0:
        selected = selected[: args.limit]
    return all_rows, selected


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def target_formula_text(args) -> tuple[str, str]:
    if args.task == "positive":
        if not args.formula:
            raise ValueError("--task positive requires --formula")
        return args.formula, ""
    if not args.target_formula:
        if not args.psi:
            raise ValueError("--task negative requires --psi or --target-formula")
        return f"!({args.psi})", args.psi
    return args.target_formula, args.psi or ""


def env_info() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def make_config(args) -> RepairConfig:
    costs = CostConfig(
        w_add=args.w_add,
        w_del=args.w_del,
        lambda_add_non_v=args.lambda_add_nonV,
        lambda_del_non_v=args.lambda_del_nonV,
        quotient_weight=args.quotient_weight,
    )
    return RepairConfig(
        repair_mode=args.repair_mode,
        sf_setting=args.sf_setting,
        ranker=args.ranker,
        gnn_graph_mode=args.gnn_graph_mode,
        model_path=args.model_path,
        ranker_device=args.ranker_device,
        strict_ranker_device=args.strict_ranker_device,
        beam_width=args.beam_width,
        max_iters=args.max_iters,
        candidate_limit=args.candidate_limit,
        candidate_state_limit=args.candidate_state_limit,
        state_scan_limit=args.state_scan_limit,
        max_quotient_drift=args.max_quotient_drift,
        postprocess=args.postprocess == "on",
        include_partition_drift=args.partition_drift == "on",
        seed=args.seed,
        costs=costs,
    )


def run_case(item: dict, index: int, args, config: RepairConfig, out_dir: Path, ranker=None) -> dict:
    model_path = item["path"]
    model = RepairLTS.from_aut(model_path)
    target_state = model.initial if args.target_state < 0 else args.target_state
    if not 0 <= target_state < model.state_count:
        raise ValueError(f"{model_path}: target_state {target_state} is outside 0..{model.state_count - 1}")
    repair_model = RepairLTS(target_state, model.state_count, model.edges)
    target_text, psi_text = target_formula_text(args)
    target_formula = HMLParser.parse(target_text)
    action_set = formula_actions(target_formula)
    v_actions = choose_v_actions(model, args.V, args.v_size, args.v_policy, excluded=action_set)
    result = run_repair(repair_model, target_formula, v_actions, config, ranker=ranker)

    case_id = f"{model_path.stem}_{index}_{args.task}_{args.repair_mode}_{args.sf_setting}"
    repaired_path = ""
    script_path = out_dir / "edit_scripts" / f"{case_id}.json"
    script_payload = {
        "case_id": case_id,
        "model_path": str(model_path),
        "original_initial": model.initial,
        "original_target_state": target_state,
        "task_type": args.task,
        "repair_mode": args.repair_mode,
        "sf_setting": args.sf_setting,
        "ranker": args.ranker,
        "V_actions": sorted(v_actions),
        "target_formula": target_text,
        "result": result.edit_script_json(),
    }
    dump_json(script_path, script_payload)
    if result.success:
        repaired_path = str(out_dir / "repaired_aut" / f"{case_id}.aut")
        result.final_model.write_aut(repaired_path)

    log_path = out_dir / "logs" / f"{case_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.message + "\n", encoding="utf-8")

    first_action = first_modal_action(target_formula)
    actions_in_v = sorted(action_set & v_actions)
    metrics = result.actual_metrics
    row = {
        "case_id": case_id,
        "model_path": str(model_path),
        "target_state": target_state,
        "task_type": args.task,
        "repair_mode": args.repair_mode,
        "sf_setting": args.sf_setting,
        "ranker": args.ranker,
        "gnn_graph_mode": args.gnn_graph_mode,
        "V_requested_size": len(parse_v_actions(args.V)) if args.V else args.v_size,
        "V_size": len(v_actions),
        "V_size_note": f"required |V|={len(parse_v_actions(args.V)) if args.V else args.v_size}, actual |V|={len(v_actions)} after excluding formula actions",
        "V_source": "formula_safe_explicit" if args.V else "formula_safe_v_size",
        "V_label": "cli",
        "formula_id": "",
        "formula_kind": args.task,
        "formula_difficulty": "",
        "formula_source": "manual_cli",
        "formula_modal_action_count": target_formula.modal_action_count(),
        "formula_known_action_count": "",
        "formula_missing_action_count": "",
        "formula_uses_missing_actions": "",
        "formula_initial_satisfied": "",
        "formula_first_action": first_action,
        "formula_target_action_in_lts": yes_no(first_action in model.actions),
        "formula_actions": json.dumps(sorted(action_set), ensure_ascii=False),
        "formula_actions_in_V": json.dumps(actions_in_v, ensure_ascii=False),
        "all_formula_actions_in_V": yes_no(bool(action_set) and action_set.issubset(v_actions)),
        "any_formula_action_in_V": yes_no(bool(actions_in_v)),
        "V_actions": ":".join(sorted(v_actions)),
        "target_action_in_V": yes_no(bool(first_action) and first_action in v_actions),
        "formula": args.formula or "",
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
        "repaired_path": repaired_path,
        "edit_script_path": str(script_path),
    }
    return row


def summarize(rows: list[dict], errors: list[dict]) -> dict:
    successes = [row for row in rows if row["success"] == "YES"]
    verified = [row for row in rows if row["verified"] == "YES"]
    def avg(field: str) -> float:
        values = [float(row[field]) for row in rows if row.get(field, "") != ""]
        return sum(values) / len(values) if values else 0.0

    return {
        "runs": len(rows),
        "errors": len(errors),
        "successes": len(successes),
        "verified": len(verified),
        "success_rate": len(successes) / len(rows) if rows else 0.0,
        "verified_rate": len(verified) / len(rows) if rows else 0.0,
        "avg_actual_cost": avg("actual_cost"),
        "avg_nonV_edits": avg("nonV_add_edges") + avg("nonV_del_edges"),
        "avg_quotient_drift": avg("quotient_drift"),
        "avg_verifier_calls": avg("verifier_calls"),
        "avg_elapsed_ms": avg("elapsed_ms"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strong-forgetting-priority add/delete LTS repair runner")
    parser.add_argument("--aut-dir", default="data", help="AUT file or directory containing AUT files")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover .aut files under --aut-dir")
    parser.add_argument("--out-dir", default="results/add_delete_run")
    parser.add_argument("--task", choices=["positive", "negative"], default="positive")
    parser.add_argument("--formula", default="")
    parser.add_argument("--psi", default="")
    parser.add_argument("--target-formula", default="")
    parser.add_argument("--repair-mode", choices=["add-only", "delete-only", "add-delete"], default="add-delete")
    parser.add_argument("--sf-setting", choices=["no_sf", "soft_sf", "strict_then_escalate"], default="strict_then_escalate")
    parser.add_argument("--V", default="", help="Explicit forgotten actions separated by ':' or ','")
    parser.add_argument("--v-size", type=int, default=0)
    parser.add_argument("--v-policy", "--v-size-policy", choices=["least-frequent", "most-frequent", "deterministic", "non-internal-first"], default="least-frequent")
    parser.add_argument("--ranker", choices=["heuristic", "neural"], default="heuristic")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--ranker-device", default="cpu")
    parser.add_argument("--strict-ranker-device", action="store_true")
    parser.add_argument("--gnn-graph-mode", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target-state", type=int, default=-1, help="Original LTS state to repair/check; -1 means AUT initial state")
    parser.add_argument("--max-states", type=int, default=100000)
    parser.add_argument("--max-transitions", type=int, default=1000000)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-iters", type=int, default=16)
    parser.add_argument("--candidate-limit", type=int, default=64)
    parser.add_argument("--candidate-state-limit", type=int, default=256)
    parser.add_argument("--state-scan-limit", type=int, default=5000)
    parser.add_argument("--max-quotient-drift", type=int, default=1000000000)
    parser.add_argument("--postprocess", choices=["on", "off"], default="on")
    parser.add_argument("--partition-drift", choices=["on", "off"], default="off")
    parser.add_argument("--w-add", type=float, default=1.0)
    parser.add_argument("--w-del", type=float, default=1.0)
    parser.add_argument("--lambda-add-nonV", dest="lambda_add_nonV", type=float, default=5.0)
    parser.add_argument("--lambda-del-nonV", dest="lambda_del_nonV", type=float, default=5.0)
    parser.add_argument("--quotient-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "env_info.json", env_info())

    all_inputs, selected = select_inputs(args)
    selection_rows = [
        {
            "model_path": str(row["path"]),
            "states": row.get("states", ""),
            "transitions": row.get("transitions", ""),
            "selected": yes_no(bool(row.get("selected")) and row in selected),
            "reason": row.get("reason", ""),
        }
        for row in all_inputs
    ]
    write_csv(out_dir / "selection.csv", selection_rows, ["model_path", "states", "transitions", "selected", "reason"])

    config = make_config(args)
    ranker = make_ranker(config)
    rows = []
    errors = []
    for index, item in enumerate(selected):
        case_id = f"{item['path'].stem}_{index}_{args.task}_{args.repair_mode}_{args.sf_setting}"
        try:
            rows.append(run_case(item, index, args, config, out_dir, ranker=ranker))
        except Exception as exc:
            errors.append({"case_id": case_id, "model_path": str(item["path"]), "error": repr(exc)})
            print(f"[ERROR] {case_id}: {exc}", file=sys.stderr)

    write_csv(out_dir / "runs.csv", rows, RUN_FIELDS)
    write_csv(out_dir / "errors.csv", errors, ERROR_FIELDS)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summarize(rows, errors), handle, ensure_ascii=False, indent=2)

    print(f"Selected: {len(selected)}")
    print(f"Runs: {len(rows)}")
    print(f"Errors: {len(errors)}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
