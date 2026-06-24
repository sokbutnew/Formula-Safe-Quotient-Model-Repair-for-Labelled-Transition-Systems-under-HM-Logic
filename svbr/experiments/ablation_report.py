from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from svbr.experiments.compare_rankers import f, is_timeout, load_runs, load_stage3, summarize_experiment


DEFAULT_EXPERIMENTS = [
    "heuristic_baseline",
    "random_baseline",
    "legacy8_ablation",
    "fixed_budget_contextual",
    "unsafe_v_contextual",
    "add_only_contextual",
    "delete_only_contextual",
    "lightweight_contextual_linear",
    "direct_original_contextual_full",
]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def quotient_stats(manifest: dict) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for model in manifest.get("models", []):
        original_states = float(model.get("states", 0) or 0)
        original_transitions = float(model.get("transitions", 0) or 0)
        for v_meta in model.get("v_sets", []):
            key = (str(v_meta.get("source", "")), str(v_meta.get("requested_size", "")))
            groups.setdefault(key, []).append(
                {
                    "original_states": original_states,
                    "original_transitions": original_transitions,
                    "quotient_states": float(v_meta.get("quotient_states", 0) or 0),
                    "quotient_transitions": float(v_meta.get("quotient_transitions", 0) or 0),
                    "quotient_time_ms": float(v_meta.get("quotient_time_ms", 0) or 0),
                }
            )
    rows = []
    for (source, requested_size), items in sorted(groups.items()):
        state_ratios = [item["quotient_states"] / item["original_states"] for item in items if item["original_states"] > 0]
        transition_ratios = [item["quotient_transitions"] / item["original_transitions"] for item in items if item["original_transitions"] > 0]
        rows.append(
            {
                "V_source": source,
                "V_requested_size": requested_size,
                "quotient_count": len(items),
                "avg_original_states": f"{mean([item['original_states'] for item in items]):.6f}",
                "avg_original_transitions": f"{mean([item['original_transitions'] for item in items]):.6f}",
                "avg_quotient_states": f"{mean([item['quotient_states'] for item in items]):.6f}",
                "avg_quotient_transitions": f"{mean([item['quotient_transitions'] for item in items]):.6f}",
                "avg_state_ratio_q_over_original": f"{mean(state_ratios):.6f}",
                "avg_transition_ratio_q_over_original": f"{mean(transition_ratios):.6f}",
                "avg_quotient_time_ms": f"{mean([item['quotient_time_ms'] for item in items]):.6f}",
                "max_quotient_time_ms": f"{max([item['quotient_time_ms'] for item in items], default=0.0):.6f}",
            }
        )
    return rows


def worst_case_rows(results_root: Path, experiments: list[str], per_experiment: int) -> list[dict]:
    rows = []
    for name in experiments:
        runs = load_runs(results_root, name)
        stage3 = {row.get("case_id", ""): row for row in load_stage3(results_root, name)}
        for row in runs:
            stage3_row = stage3.get(row.get("case_id", ""), {})
            stage2_edits = f(row, "add_edges") + f(row, "del_edges")
            lifted = f(stage3_row, "lifted_add_edges") + f(stage3_row, "lifted_del_edges")
            rows.append(
                {
                    "experiment": name,
                    "case_id": row.get("case_id", ""),
                    "model_path": row.get("model_path", ""),
                    "formula_id": row.get("formula_id", ""),
                    "formula_kind": row.get("formula_kind", ""),
                    "formula_difficulty": row.get("formula_difficulty", ""),
                    "V_requested_size": row.get("V_requested_size", ""),
                    "V_source": row.get("V_source", ""),
                    "any_formula_action_in_V": row.get("any_formula_action_in_V", ""),
                    "success": row.get("success", ""),
                    "verified": row.get("verified", ""),
                    "timeout": "YES" if is_timeout(row) else "NO",
                    "stage3_verified": stage3_row.get("materialized_verified", ""),
                    "verifier_calls": row.get("verifier_calls", ""),
                    "elapsed_ms": row.get("elapsed_ms", ""),
                    "stage2_edits": f"{stage2_edits:.6f}",
                    "stage3_lifted_edges": f"{lifted:.6f}",
                    "stage3_expansion_ratio": f"{lifted / max(1.0, stage2_edits):.6f}",
                    "stage3_elapsed_ms": stage3_row.get("elapsed_ms", ""),
                    "message": row.get("message", ""),
                }
            )
    selected = []
    for metric in ["verifier_calls", "elapsed_ms", "stage3_lifted_edges", "stage3_expansion_ratio"]:
        selected.extend(sorted(rows, key=lambda row: f(row, metric), reverse=True)[:per_experiment])
    dedup = {}
    for row in selected:
        dedup[(row["experiment"], row["case_id"])] = row
    return list(dedup.values())


def write_markdown(path: Path, summaries: list[dict], quotient_rows: list[dict], worst_rows: list[dict], fair_max_case_seconds: str = "") -> None:
    lines = [
        "# Reviewer Ablation And Baseline Report",
        "",
        "This report is generated from the experiment roots and is intended to answer the main reviewer concerns: formula-safe forgetting necessity, learned-ranker benefit, fixed-budget sensitivity, add/delete necessity, quotient reduction, direct-original baseline feasibility, and materialization expansion.",
        "",
        f"Fair Stage 2 setting: every compared method uses the same `{fair_max_case_seconds}s/case` wall-clock cap. Verifier calls are reported as an efficiency metric but are not capped.",
        "",
        "## Overall",
        "",
        "| experiment | N | success | verified | timeout | timeout rate | stage3 writeback | end-to-end | avg s | median s | p90 s | avg calls | p90 calls | avg edits | median edits | max edits | avg stage3 edges | avg expansion | max expansion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['experiment']} | {row['N']} | {row['success_rate']} | {row['verified_rate']} | "
            f"{row['timeout']} | {row['timeout_rate']} | {row['stage3_verified_over_stage2_verified_rate']} | "
            f"{row['end_to_end_verified_rate']} | {row['avg_elapsed_s']} | {row['median_elapsed_s']} | {row['p90_elapsed_s']} | "
            f"{row['avg_verifier_calls']} | {row['p90_verifier_calls']} | {row['avg_edits_success']} | "
            f"{row['median_edits_success']} | {row['max_edits_success']} | {row['avg_stage3_lifted_total_edges_verified']} | "
            f"{row['avg_stage3_expansion_ratio_verified']} | {row['max_stage3_expansion_ratio_verified']} |"
        )
    lines.extend(["", "## Quotient Reduction", ""])
    lines.append("| V source | |V| requested | count | avg original states | avg quotient states | state ratio | avg original transitions | avg quotient transitions | transition ratio | avg quotient ms |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in quotient_rows:
        lines.append(
            f"| {row['V_source']} | {row['V_requested_size']} | {row['quotient_count']} | {row['avg_original_states']} | "
            f"{row['avg_quotient_states']} | {row['avg_state_ratio_q_over_original']} | {row['avg_original_transitions']} | "
            f"{row['avg_quotient_transitions']} | {row['avg_transition_ratio_q_over_original']} | {row['avg_quotient_time_ms']} |"
        )
    lines.extend(["", "## Worst Cases", ""])
    lines.append("| experiment | case | formula | kind | V | timeout | calls | ms | edits | lifted | expansion | verified | stage3 |")
    lines.append("| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in worst_rows[:50]:
        lines.append(
            f"| {row['experiment']} | {row['case_id']} | {row['formula_id']} | {row['formula_kind']} | {row['V_requested_size']} | "
            f"{row['timeout']} | {row['verifier_calls']} | {row['elapsed_ms']} | {row['stage2_edits']} | {row['stage3_lifted_edges']} | "
            f"{row['stage3_expansion_ratio']} | {row['verified']} | {row['stage3_verified']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reviewer-oriented baseline/ablation report")
    parser.add_argument("--results-root", default="results6/linear_ranker_ablation")
    parser.add_argument("--prepared-dir", default="results6/add_delete_prepared")
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument("--output-prefix", default="reviewer_ablation")
    parser.add_argument("--worst-per-metric", type=int, default=25)
    parser.add_argument("--fair-max-case-seconds", default=os.environ.get("FAIR_MAX_CASE_SECONDS", "300"))
    args = parser.parse_args()

    results_root = Path(args.results_root)
    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    summaries = []
    for name in experiments:
        rows = load_runs(results_root, name)
        if not rows:
            continue
        summaries.append(summarize_experiment(name, rows, load_stage3(results_root, name)))

    manifest = read_manifest(Path(args.prepared_dir) / "manifest.json")
    q_rows = quotient_stats(manifest)
    worst_rows = worst_case_rows(results_root, [row["experiment"] for row in summaries], args.worst_per_metric)

    prefix = results_root / args.output_prefix
    write_csv(prefix.with_name(prefix.name + "_overall.csv"), summaries, list(summaries[0].keys()) if summaries else ["experiment"])
    if q_rows:
        write_csv(prefix.with_name(prefix.name + "_quotient_stats.csv"), q_rows, list(q_rows[0].keys()))
    if worst_rows:
        write_csv(prefix.with_name(prefix.name + "_worst_cases.csv"), worst_rows, list(worst_rows[0].keys()))
    write_markdown(prefix.with_name(prefix.name + "_report.md"), summaries, q_rows, worst_rows, args.fair_max_case_seconds)
    print(f"Reviewer ablation report: {prefix.with_name(prefix.name + '_report.md')}")


if __name__ == "__main__":
    main()
