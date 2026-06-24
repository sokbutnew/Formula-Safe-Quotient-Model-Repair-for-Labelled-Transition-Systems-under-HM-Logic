from __future__ import annotations

import argparse
import csv
import os
import statistics
from pathlib import Path


STRICT_CASE_KEY_FIELDS = [
    "model_path",
    "target_state",
    "task_type",
    "repair_mode",
    "sf_setting",
    "formula_id",
    "formula_kind",
    "V_requested_size",
    "V_label",
    "target_formula",
]

SEMANTIC_CASE_KEY_FIELDS = [
    "model_path",
    "target_state",
    "task_type",
    "formula_id",
    "formula_kind",
    "V_requested_size",
    "target_formula",
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def yes(value: str) -> bool:
    return str(value).upper() == "YES"


def is_timeout(row: dict) -> bool:
    text = " ".join(
        str(row.get(field, ""))
        for field in ("stage", "message", "status", "verification_error", "error")
    ).lower()
    return "case time cap exhausted" in text or "timeout" in text


def f(row: dict, field: str, default: float = 0.0) -> float:
    try:
        value = row.get(field, "")
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_key(row: dict, fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in fields)


def load_runs(root: Path, experiment: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((root / experiment).glob("*/runs.csv")):
        suite_name = path.parent.name
        for row in read_csv(path):
            row = dict(row)
            row["experiment"] = experiment
            row["suite_dir"] = suite_name
            rows.append(row)
    return rows


def load_stage3(root: Path, experiment: str) -> list[dict]:
    return read_csv(root / experiment / "materialize_repaired_aut.csv")


def by_case_id(rows: list[dict]) -> dict[str, dict]:
    return {str(row.get("case_id", "")): row for row in rows if row.get("case_id")}


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_experiment(name: str, rows: list[dict], stage3_rows: list[dict] | None = None) -> dict:
    stage3_rows = stage3_rows or []
    stage3 = by_case_id(stage3_rows)
    successes = [row for row in rows if yes(row.get("success", ""))]
    verified = [row for row in rows if yes(row.get("verified", ""))]
    timeouts = [row for row in rows if is_timeout(row)]
    failed = [row for row in rows if not yes(row.get("success", ""))]
    materialized = [row for row in stage3_rows if yes(row.get("materialized_verified", ""))]
    written = [row for row in stage3_rows if row.get("status", "") in {"written", "skipped_exists"}]
    stage3_timeouts = [row for row in stage3_rows if is_timeout(row)]
    rescue_verified = [
        row
        for row in rows
        if yes(row.get("verified", ""))
        and ("linear_rescue" in str(row.get("stage", "")) or "Linear rescue fallback" in str(row.get("message", "")))
    ]
    stage3_errors = [row for row in stage3_rows if row.get("status", "") == "error"]
    stage3_unresolved = [
        row
        for row in stage3_rows
        if row.get("status", "") == "error" or (row.get("status", "") in {"written", "skipped_exists"} and not yes(row.get("materialized_verified", "")))
    ]
    end_to_end = [
        row
        for row in rows
        if yes(row.get("verified", "")) and yes(stage3.get(str(row.get("case_id", "")), {}).get("materialized_verified", ""))
    ]
    materialized_pairs = [
        (row, stage3[str(row.get("case_id", ""))])
        for row in rows
        if str(row.get("case_id", "")) in stage3 and yes(stage3[str(row.get("case_id", ""))].get("materialized_verified", ""))
    ]
    successful_edit_counts = [f(row, "add_edges") + f(row, "del_edges") for row in successes]
    lifted_edit_counts = [f(item, "lifted_add_edges") + f(item, "lifted_del_edges") for item in materialized]
    expansion_ratios = [
        (f(stage3_row, "lifted_add_edges") + f(stage3_row, "lifted_del_edges")) / max(1.0, f(stage2_row, "add_edges") + f(stage2_row, "del_edges"))
        for stage2_row, stage3_row in materialized_pairs
    ]
    return {
        "experiment": name,
        "N": len(rows),
        "success": len(successes),
        "failed": len(failed),
        "verified": len(verified),
        "timeout": len(timeouts),
        "linear_rescue_verified": len(rescue_verified),
        "success_rate": f"{len(successes) / len(rows):.6f}" if rows else "0.000000",
        "verified_rate": f"{len(verified) / len(rows):.6f}" if rows else "0.000000",
        "timeout_rate": f"{len(timeouts) / len(rows):.6f}" if rows else "0.000000",
        "linear_rescue_verified_rate": f"{len(rescue_verified) / len(rows):.6f}" if rows else "0.000000",
        "stage3_cases": len(stage3_rows),
        "stage3_written": len(written),
        "stage3_verified": len(materialized),
        "stage3_timeout": len(stage3_timeouts),
        "stage3_verified_rate": f"{len(materialized) / len(stage3_rows):.6f}" if stage3_rows else "0.000000",
        "stage3_verified_over_stage2_verified_rate": f"{len(materialized) / len(verified):.6f}" if verified else "0.000000",
        "stage3_verified_over_written_rate": f"{len(materialized) / len(written):.6f}" if written else "0.000000",
        "end_to_end_verified": len(end_to_end),
        "end_to_end_verified_rate": f"{len(end_to_end) / len(rows):.6f}" if rows else "0.000000",
        "stage3_errors": len(stage3_errors),
        "stage3_unresolved": len(stage3_unresolved),
        "avg_verifier_calls": f"{mean([f(row, 'verifier_calls') for row in rows]):.6f}",
        "median_verifier_calls": f"{median([f(row, 'verifier_calls') for row in rows]):.6f}",
        "p90_verifier_calls": f"{percentile([f(row, 'verifier_calls') for row in rows], 0.90):.6f}",
        "max_verifier_calls": f"{max([f(row, 'verifier_calls') for row in rows], default=0.0):.6f}",
        "avg_elapsed_ms": f"{mean([f(row, 'elapsed_ms') for row in rows]):.6f}",
        "median_elapsed_ms": f"{median([f(row, 'elapsed_ms') for row in rows]):.6f}",
        "p90_elapsed_ms": f"{percentile([f(row, 'elapsed_ms') for row in rows], 0.90):.6f}",
        "max_elapsed_ms": f"{max([f(row, 'elapsed_ms') for row in rows], default=0.0):.6f}",
        "avg_elapsed_s": f"{mean([f(row, 'elapsed_ms') for row in rows]) / 1000.0:.6f}",
        "median_elapsed_s": f"{median([f(row, 'elapsed_ms') for row in rows]) / 1000.0:.6f}",
        "p90_elapsed_s": f"{percentile([f(row, 'elapsed_ms') for row in rows], 0.90) / 1000.0:.6f}",
        "max_elapsed_s": f"{max([f(row, 'elapsed_ms') for row in rows], default=0.0) / 1000.0:.6f}",
        "avg_cost_success": f"{mean([f(row, 'actual_cost') for row in successes]):.6f}",
        "avg_edits_success": f"{mean([f(row, 'add_edges') + f(row, 'del_edges') for row in successes]):.6f}",
        "median_edits_success": f"{median(successful_edit_counts):.6f}",
        "max_edits_success": f"{max(successful_edit_counts, default=0.0):.6f}",
        "avg_nonV_edits_success": f"{mean([f(row, 'nonV_add_edges') + f(row, 'nonV_del_edges') for row in successes]):.6f}",
        "avg_quotient_drift_success": f"{mean([f(row, 'quotient_drift') for row in successes]):.6f}",
        "avg_cex_iters": f"{mean([f(row, 'cex_iters') for row in rows]):.6f}",
        "avg_stage3_lifted_add_edges_verified": f"{mean([f(row, 'lifted_add_edges') for row in materialized]):.6f}",
        "avg_stage3_lifted_del_edges_verified": f"{mean([f(row, 'lifted_del_edges') for row in materialized]):.6f}",
        "avg_stage3_lifted_total_edges_verified": f"{mean([f(row, 'lifted_add_edges') + f(row, 'lifted_del_edges') for row in materialized]):.6f}",
        "median_stage3_lifted_total_edges_verified": f"{median([f(row, 'lifted_add_edges') + f(row, 'lifted_del_edges') for row in materialized]):.6f}",
        "max_stage3_lifted_total_edges_verified": f"{max(lifted_edit_counts, default=0.0):.6f}",
        "avg_stage3_expansion_ratio_verified": f"{mean(expansion_ratios):.6f}",
        "median_stage3_expansion_ratio_verified": f"{median(expansion_ratios):.6f}",
        "max_stage3_expansion_ratio_verified": f"{max(expansion_ratios, default=0.0):.6f}",
        "avg_stage3_lifting_iters": f"{mean([f(row, 'lifting_iters') for row in stage3_rows]):.6f}",
        "median_stage3_lifting_iters": f"{median([f(row, 'lifting_iters') for row in stage3_rows]):.6f}",
        "avg_stage3_elapsed_ms": f"{mean([f(row, 'elapsed_ms') for row in stage3_rows]):.6f}",
        "median_stage3_elapsed_ms": f"{median([f(row, 'elapsed_ms') for row in stage3_rows]):.6f}",
        "avg_stage3_elapsed_s": f"{mean([f(row, 'elapsed_ms') for row in stage3_rows]) / 1000.0:.6f}",
        "median_stage3_elapsed_s": f"{median([f(row, 'elapsed_ms') for row in stage3_rows]) / 1000.0:.6f}",
        "max_stage3_elapsed_ms": f"{max([f(row, 'elapsed_ms') for row in stage3_rows], default=0.0):.6f}",
        "max_stage3_elapsed_s": f"{max([f(row, 'elapsed_ms') for row in stage3_rows], default=0.0) / 1000.0:.6f}",
        "max_stage3_rss_mb": f"{max([f(row, 'rss_mb') for row in stage3_rows], default=0.0):.6f}",
    }


def summarize_by_stratum(name: str, rows: list[dict], stage3_rows: list[dict] | None = None) -> list[dict]:
    stage3 = by_case_id(stage3_rows or [])
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (
            row.get("formula_kind", ""),
            row.get("formula_difficulty", ""),
            str(row.get("V_requested_size", "")),
        )
        groups.setdefault(key, []).append(row)
    out = []
    for key, group_rows in sorted(groups.items()):
        group_stage3 = [stage3[row.get("case_id", "")] for row in group_rows if row.get("case_id", "") in stage3]
        summary = summarize_experiment(name, group_rows, group_stage3)
        summary.update(
            {
                "formula_kind": key[0],
                "formula_difficulty": key[1],
                "V_requested_size": key[2],
            }
        )
        out.append(summary)
    return out


def paired_rows(
    left_name: str,
    left_rows: list[dict],
    right_name: str,
    right_rows: list[dict],
    left_stage3_rows: list[dict] | None = None,
    right_stage3_rows: list[dict] | None = None,
    key_fields: list[str] | None = None,
) -> list[dict]:
    fields = key_fields or STRICT_CASE_KEY_FIELDS
    left_by_key = {stable_key(row, fields): row for row in left_rows}
    right_by_key = {stable_key(row, fields): row for row in right_rows}
    left_stage3 = by_case_id(left_stage3_rows or [])
    right_stage3 = by_case_id(right_stage3_rows or [])
    rows = []
    for key in sorted(set(left_by_key) & set(right_by_key)):
        left = left_by_key[key]
        right = right_by_key[key]
        left_mat = left_stage3.get(str(left.get("case_id", "")), {})
        right_mat = right_stage3.get(str(right.get("case_id", "")), {})
        left_success = yes(left.get("success", ""))
        right_success = yes(right.get("success", ""))
        left_verified = yes(left.get("verified", ""))
        right_verified = yes(right.get("verified", ""))
        left_materialized = yes(left_mat.get("materialized_verified", ""))
        right_materialized = yes(right_mat.get("materialized_verified", ""))
        delta_calls = f(right, "verifier_calls") - f(left, "verifier_calls")
        delta_elapsed = f(right, "elapsed_ms") - f(left, "elapsed_ms")
        delta_cost = f(right, "actual_cost") - f(left, "actual_cost")
        delta_edits = (f(right, "add_edges") + f(right, "del_edges")) - (f(left, "add_edges") + f(left, "del_edges"))
        delta_lifted_edges = (f(right_mat, "lifted_add_edges") + f(right_mat, "lifted_del_edges")) - (f(left_mat, "lifted_add_edges") + f(left_mat, "lifted_del_edges"))
        delta_lifting_iters = f(right_mat, "lifting_iters") - f(left_mat, "lifting_iters")
        if right_verified and not left_verified:
            outcome = f"{right_name}_only_verified"
        elif left_verified and not right_verified:
            outcome = f"{left_name}_only_verified"
        elif right_success and not left_success:
            outcome = f"{right_name}_only_success"
        elif left_success and not right_success:
            outcome = f"{left_name}_only_success"
        elif right_success and left_success:
            outcome = "both_success"
        else:
            outcome = "both_failed"
        rows.append(
            {
                "comparison_key": "|".join(key),
                "pairing_fields": ",".join(fields),
                "model_path": left.get("model_path", ""),
                "target_state": left.get("target_state", ""),
                "task_type": left.get("task_type", ""),
                "formula_kind": left.get("formula_kind", ""),
                "formula_difficulty": left.get("formula_difficulty", ""),
                "formula_source": left.get("formula_source", ""),
                "formula_id": left.get("formula_id", ""),
                "V_requested_size": left.get("V_requested_size", ""),
                "V_size": left.get("V_size", ""),
                "outcome": outcome,
                f"{left_name}_success": left.get("success", ""),
                f"{right_name}_success": right.get("success", ""),
                f"{left_name}_verified": left.get("verified", ""),
                f"{right_name}_verified": right.get("verified", ""),
                f"{left_name}_timeout": "YES" if is_timeout(left) else "NO",
                f"{right_name}_timeout": "YES" if is_timeout(right) else "NO",
                f"{left_name}_materialized_verified": left_mat.get("materialized_verified", ""),
                f"{right_name}_materialized_verified": right_mat.get("materialized_verified", ""),
                f"{left_name}_stage3_status": left_mat.get("status", ""),
                f"{right_name}_stage3_status": right_mat.get("status", ""),
                f"{left_name}_calls": left.get("verifier_calls", ""),
                f"{right_name}_calls": right.get("verifier_calls", ""),
                "delta_calls_right_minus_left": f"{delta_calls:.6f}",
                f"{left_name}_elapsed_ms": left.get("elapsed_ms", ""),
                f"{right_name}_elapsed_ms": right.get("elapsed_ms", ""),
                "delta_elapsed_ms_right_minus_left": f"{delta_elapsed:.6f}",
                f"{left_name}_actual_cost": left.get("actual_cost", ""),
                f"{right_name}_actual_cost": right.get("actual_cost", ""),
                "delta_cost_right_minus_left": f"{delta_cost:.6f}",
                f"{left_name}_edits": f"{f(left, 'add_edges') + f(left, 'del_edges'):.6f}",
                f"{right_name}_edits": f"{f(right, 'add_edges') + f(right, 'del_edges'):.6f}",
                "delta_edits_right_minus_left": f"{delta_edits:.6f}",
                f"{left_name}_stage3_lifted_edges": f"{f(left_mat, 'lifted_add_edges') + f(left_mat, 'lifted_del_edges'):.6f}",
                f"{right_name}_stage3_lifted_edges": f"{f(right_mat, 'lifted_add_edges') + f(right_mat, 'lifted_del_edges'):.6f}",
                "delta_stage3_lifted_edges_right_minus_left": f"{delta_lifted_edges:.6f}",
                f"{left_name}_stage3_lifting_iters": left_mat.get("lifting_iters", ""),
                f"{right_name}_stage3_lifting_iters": right_mat.get("lifting_iters", ""),
                "delta_stage3_lifting_iters_right_minus_left": f"{delta_lifting_iters:.6f}",
                f"{left_name}_message": left.get("message", ""),
                f"{right_name}_message": right.get("message", ""),
            }
        )
    return rows


def pair_summary(left_name: str, right_name: str, rows: list[dict]) -> dict:
    both_success = [row for row in rows if row["outcome"] == "both_success"]
    same_outcome = [row for row in rows if row["outcome"] in {"both_success", "both_failed"}]
    right_only = [row for row in rows if row["outcome"] in {f"{right_name}_only_verified", f"{right_name}_only_success"}]
    left_only = [row for row in rows if row["outcome"] in {f"{left_name}_only_verified", f"{left_name}_only_success"}]
    both_materialized = [row for row in rows if yes(row.get(f"{left_name}_materialized_verified", "")) and yes(row.get(f"{right_name}_materialized_verified", ""))]
    right_stage3_only = [row for row in rows if yes(row.get(f"{right_name}_materialized_verified", "")) and not yes(row.get(f"{left_name}_materialized_verified", ""))]
    left_stage3_only = [row for row in rows if yes(row.get(f"{left_name}_materialized_verified", "")) and not yes(row.get(f"{right_name}_materialized_verified", ""))]
    right_timeout = [row for row in rows if yes(row.get(f"{right_name}_timeout", ""))]
    left_timeout = [row for row in rows if yes(row.get(f"{left_name}_timeout", ""))]
    right_only_timeout = [row for row in right_timeout if not yes(row.get(f"{left_name}_timeout", ""))]
    left_only_timeout = [row for row in left_timeout if not yes(row.get(f"{right_name}_timeout", ""))]
    deltas_calls = [float(row["delta_calls_right_minus_left"]) for row in same_outcome]
    deltas_elapsed = [float(row["delta_elapsed_ms_right_minus_left"]) for row in same_outcome]
    deltas_cost = [float(row["delta_cost_right_minus_left"]) for row in both_success]
    deltas_stage3_edges = [float(row["delta_stage3_lifted_edges_right_minus_left"]) for row in both_materialized]
    deltas_stage3_iters = [float(row["delta_stage3_lifting_iters_right_minus_left"]) for row in both_materialized]
    return {
        "paired_cases": len(rows),
        f"{right_name}_only_success_or_verified": len(right_only),
        f"{left_name}_only_success_or_verified": len(left_only),
        "both_success": len(both_success),
        "both_failed": sum(1 for row in rows if row["outcome"] == "both_failed"),
        "both_stage3_materialized_verified": len(both_materialized),
        f"{right_name}_only_stage3_materialized_verified": len(right_stage3_only),
        f"{left_name}_only_stage3_materialized_verified": len(left_stage3_only),
        f"{right_name}_timeouts": len(right_timeout),
        f"{left_name}_timeouts": len(left_timeout),
        f"{right_name}_only_timeouts": len(right_only_timeout),
        f"{left_name}_only_timeouts": len(left_only_timeout),
        "avg_delta_calls_right_minus_left_same_outcome": f"{mean(deltas_calls):.6f}",
        "median_delta_calls_right_minus_left_same_outcome": f"{median(deltas_calls):.6f}",
        "avg_delta_elapsed_ms_right_minus_left_same_outcome": f"{mean(deltas_elapsed):.6f}",
        "median_delta_elapsed_ms_right_minus_left_same_outcome": f"{median(deltas_elapsed):.6f}",
        "avg_delta_cost_right_minus_left_both_success": f"{mean(deltas_cost):.6f}",
        "median_delta_cost_right_minus_left_both_success": f"{median(deltas_cost):.6f}",
        "avg_delta_stage3_lifted_edges_right_minus_left_both_materialized": f"{mean(deltas_stage3_edges):.6f}",
        "median_delta_stage3_lifted_edges_right_minus_left_both_materialized": f"{median(deltas_stage3_edges):.6f}",
        "avg_delta_stage3_lifting_iters_right_minus_left_both_materialized": f"{mean(deltas_stage3_iters):.6f}",
        "median_delta_stage3_lifting_iters_right_minus_left_both_materialized": f"{median(deltas_stage3_iters):.6f}",
    }


def write_markdown(path: Path, left_name: str, right_name: str, summaries: list[dict], paired_summary: dict, fair_max_case_seconds: str = "") -> None:
    lines = [
        "# Ranker Comparison Report",
        "",
        f"Left baseline: `{left_name}`. Right candidate: `{right_name}`.",
        "",
        f"Fair Stage 2 cap: `{fair_max_case_seconds}s/case` for every compared method. Verifier calls are measured, not capped.",
        "",
        "Primary comparison order:",
        "",
        "1. Higher verified/success rate on the same attempted add-delete cases.",
        "2. Fewer verifier calls for paired cases with the same success outcome.",
        "3. Lower elapsed time with the same success outcome.",
        "4. Lower successful-repair cost and fewer edits among cases both solve.",
        "",
        "## Overall",
        "",
        "| experiment | N | success_rate | verified_rate | timeout | timeout_rate | stage3_writeback | end_to_end | avg_s | median_s | p90_s | avg_calls | p90_calls | avg_edits | median_edits | avg_stage3_edges | avg_expansion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['experiment']} | {summary['N']} | {summary['success_rate']} | {summary['verified_rate']} | "
            f"{summary['timeout']} | {summary['timeout_rate']} | "
            f"{summary['stage3_verified_over_stage2_verified_rate']} | {summary['end_to_end_verified_rate']} | "
            f"{summary['avg_elapsed_s']} | {summary['median_elapsed_s']} | {summary['p90_elapsed_s']} | "
            f"{summary['avg_verifier_calls']} | {summary['p90_verifier_calls']} | "
            f"{summary['avg_edits_success']} | {summary['median_edits_success']} | "
            f"{summary['avg_stage3_lifted_total_edges_verified']} | "
            f"{summary['avg_stage3_expansion_ratio_verified']} |"
        )
    lines.extend(
        [
            "",
            "## Paired Delta",
            "",
            f"- Paired cases: {paired_summary['paired_cases']}",
            f"- {right_name} only success/verified: {paired_summary[f'{right_name}_only_success_or_verified']}",
            f"- {left_name} only success/verified: {paired_summary[f'{left_name}_only_success_or_verified']}",
            f"- Both success: {paired_summary['both_success']}",
            f"- Both failed: {paired_summary['both_failed']}",
            f"- {right_name} timeouts: {paired_summary[f'{right_name}_timeouts']}",
            f"- {left_name} timeouts: {paired_summary[f'{left_name}_timeouts']}",
            f"- {right_name} only timeouts: {paired_summary[f'{right_name}_only_timeouts']}",
            f"- {left_name} only timeouts: {paired_summary[f'{left_name}_only_timeouts']}",
            f"- Both Stage 3 materialized verified: {paired_summary['both_stage3_materialized_verified']}",
            f"- {right_name} only Stage 3 materialized verified: {paired_summary[f'{right_name}_only_stage3_materialized_verified']}",
            f"- {left_name} only Stage 3 materialized verified: {paired_summary[f'{left_name}_only_stage3_materialized_verified']}",
            f"- Avg delta calls ({right_name} - {left_name}) on same-outcome cases: {paired_summary['avg_delta_calls_right_minus_left_same_outcome']}",
            f"- Avg delta elapsed ms ({right_name} - {left_name}) on same-outcome cases: {paired_summary['avg_delta_elapsed_ms_right_minus_left_same_outcome']}",
            f"- Avg delta cost ({right_name} - {left_name}) on both-success cases: {paired_summary['avg_delta_cost_right_minus_left_both_success']}",
            f"- Avg delta Stage 3 lifted edges ({right_name} - {left_name}) on both-materialized cases: {paired_summary['avg_delta_stage3_lifted_edges_right_minus_left_both_materialized']}",
            f"- Avg delta Stage 3 lifting iters ({right_name} - {left_name}) on both-materialized cases: {paired_summary['avg_delta_stage3_lifting_iters_right_minus_left_both_materialized']}",
            "",
            "Negative deltas mean the right-side candidate used fewer calls/time/cost.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two Stage 2 ranker experiment outputs")
    parser.add_argument("--results-root", default="results/ranker_compare_stage2")
    parser.add_argument("--left", default="linear_v3")
    parser.add_argument("--right", default="gnn")
    parser.add_argument("--output-prefix", default="ranker_compare")
    parser.add_argument("--pairing", choices=["strict", "semantic"], default="strict")
    parser.add_argument("--fair-max-case-seconds", default=os.environ.get("FAIR_MAX_CASE_SECONDS", "300"))
    args = parser.parse_args()

    root = Path(args.results_root)
    left_rows = load_runs(root, args.left)
    right_rows = load_runs(root, args.right)
    left_stage3_rows = load_stage3(root, args.left)
    right_stage3_rows = load_stage3(root, args.right)

    summaries = [summarize_experiment(args.left, left_rows, left_stage3_rows), summarize_experiment(args.right, right_rows, right_stage3_rows)]
    summary_fields = list(summaries[0].keys()) if summaries else []
    overall_path = root / f"{args.output_prefix}_overall.csv"
    by_stratum_path = root / f"{args.output_prefix}_by_stratum.csv"
    deltas_path = root / f"{args.output_prefix}_case_deltas.csv"
    paired_path = root / f"{args.output_prefix}_paired_summary.csv"
    report_path = root / f"{args.output_prefix}_report.md"
    write_csv(overall_path, summaries, summary_fields)

    stratum_rows = summarize_by_stratum(args.left, left_rows, left_stage3_rows) + summarize_by_stratum(args.right, right_rows, right_stage3_rows)
    if stratum_rows:
        write_csv(by_stratum_path, stratum_rows, list(stratum_rows[0].keys()))

    key_fields = STRICT_CASE_KEY_FIELDS if args.pairing == "strict" else SEMANTIC_CASE_KEY_FIELDS
    pairs = paired_rows(args.left, left_rows, args.right, right_rows, left_stage3_rows, right_stage3_rows, key_fields=key_fields)
    if pairs:
        write_csv(deltas_path, pairs, list(pairs[0].keys()))
    paired_summary = pair_summary(args.left, args.right, pairs)
    write_csv(paired_path, [paired_summary], list(paired_summary.keys()))
    write_markdown(report_path, args.left, args.right, summaries, paired_summary, args.fair_max_case_seconds)

    print(f"Overall summary: {overall_path}")
    print(f"Paired deltas:   {deltas_path}")
    print(f"Report:          {report_path}")


if __name__ == "__main__":
    main()
