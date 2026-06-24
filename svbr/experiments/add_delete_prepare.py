from __future__ import annotations

import argparse
import gc
import json
import pickle
import re
import time
from pathlib import Path

from svbr.experiments.formula_generation import generate_formula_cases, is_hml_safe_action
from svbr.experiments.progress import print_progress
from svbr.core import parse_aut_header
from svbr.repair.add_delete import RepairLTS, choose_v_actions, parse_v_actions, strong_v_quotient
from svbr.repair.gpu_quotient import strong_v_quotient_torch


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


def parse_int_list(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0:
            raise ValueError("V sizes must be non-negative")
        if value not in values:
            values.append(value)
    return values


def safe_label(text: str) -> str:
    text = text or "empty"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "empty"


def vset_key(actions: set[str]) -> tuple[str, ...]:
    return tuple(sorted(actions))


def case_formula_actions(case: dict, formula_kind: str = "positive") -> set[str]:
    if formula_kind == "positive":
        return set(case.get("positive", {}).get("formula_actions", case.get("formula_actions", [])))
    if formula_kind == "negative_existential":
        return set(case.get("negative_existential_target_meta", {}).get("formula_actions", []))
    if formula_kind == "negative_universal":
        return set(case.get("negative_universal_target_meta", {}).get("formula_actions", []))
    actions = set(case.get("formula_actions", []))
    for key in ["positive", "negative_existential_target_meta", "negative_universal_target_meta"]:
        actions.update(case.get(key, {}).get("formula_actions", []))
    return actions


def choose_v_actions_excluding(model: RepairLTS, v_size: int, policy: str, excluded: set[str]) -> set[str]:
    if v_size <= 0:
        return set()
    counts = model.action_counts()
    eligible = [action for action in counts if action not in excluded]
    if policy == "least-frequent":
        ordered = sorted(eligible, key=lambda action: (counts[action], action))
    elif policy == "most-frequent":
        ordered = sorted(eligible, key=lambda action: (-counts[action], action))
    elif policy == "non-internal-first":
        ordered = sorted(eligible, key=lambda action: (1 if action == "i" else 0, action))
    else:
        ordered = sorted(eligible)
    return set(ordered[:v_size])


def dump_pickle(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_quotient(model: RepairLTS, v_actions: set[str], args):
    if args.quotient_backend == "torch":
        return strong_v_quotient_torch(
            model,
            v_actions,
            device=args.quotient_device,
            strict_device=args.strict_quotient_device,
        )
    return strong_v_quotient(model, v_actions)


def release_quotient_backend_memory(args) -> None:
    if args.quotient_backend != "torch" or not str(args.quotient_device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def resolve_target_state(model: RepairLTS, target_state: int) -> int:
    resolved = model.initial if target_state < 0 else target_state
    if not 0 <= resolved < model.state_count:
        raise ValueError(f"target_state {resolved} is outside 0..{model.state_count - 1}")
    return resolved


def count_initially_unsatisfied(formula_cases: list[dict]) -> int:
    return sum(1 for case in formula_cases if not case.get("initial_satisfied"))


def required_unsatisfied_count(args, formula_cases: list[dict]) -> int:
    return min(max(0, args.min_unsatisfied_formulas), len(formula_cases))


def formula_suite_errors(model: RepairLTS, formula_cases: list[dict], args) -> list[str]:
    errors: list[str] = []
    if args.formulas_per_model > 0 and len(formula_cases) != args.formulas_per_model:
        errors.append(f"formula_count={len(formula_cases)} expected={args.formulas_per_model}")

    expected_difficulties = {
        "easy": args.easy_formula_count,
        "medium": args.medium_formula_count,
        "hard": args.hard_formula_count,
    }
    if args.formulas_per_model == sum(expected_difficulties.values()):
        for difficulty, expected in expected_difficulties.items():
            actual = sum(1 for case in formula_cases if case.get("difficulty") == difficulty)
            if actual != expected:
                errors.append(f"{difficulty}_count={actual} expected={expected}")
        safe_actions = {action for action in model.actions if is_hml_safe_action(action)}
        if safe_actions:
            expected_mixed = min(getattr(args, "mixed_formula_count", 10), len(formula_cases))
            expected_existing = len(formula_cases) - expected_mixed
            actual_mixed = sum(1 for case in formula_cases if case.get("source") == "mixed_existing_missing")
            actual_existing = sum(1 for case in formula_cases if case.get("source") == "existing_only")
            if actual_mixed != expected_mixed:
                errors.append(f"mixed_count={actual_mixed} expected={expected_mixed}")
            if actual_existing != expected_existing:
                errors.append(f"existing_only_count={actual_existing} expected={expected_existing}")
        else:
            generated_only = sum(1 for case in formula_cases if case.get("source") == "generated_missing_only")
            if generated_only != len(formula_cases):
                errors.append(f"generated_missing_only_count={generated_only} expected={len(formula_cases)}")

    actual_unsatisfied = count_initially_unsatisfied(formula_cases)
    required_unsatisfied = required_unsatisfied_count(args, formula_cases)
    if actual_unsatisfied < required_unsatisfied:
        errors.append(f"initially_unsatisfied={actual_unsatisfied} required={required_unsatisfied}")

    model_actions = set(model.actions)
    mixed_cases = [case for case in formula_cases if case.get("source") == "mixed_existing_missing"]
    if model_actions and len(mixed_cases) > 1:
        first_in = sum(1 for case in mixed_cases if case.get("target_action_in_lts"))
        first_missing = len(mixed_cases) - first_in
        if first_in == 0 or first_missing == 0:
            errors.append(f"mixed_first_action_diversity in_lts={first_in} missing={first_missing}")

    for case in formula_cases:
        positive = case.get("positive", {})
        modal_count = int(case.get("modal_action_count", positive.get("modal_action_count", 0)) or 0)
        if modal_count < args.formula_min_actions or modal_count > args.formula_max_actions:
            errors.append(f"{case.get('formula_id')}: modal_count={modal_count}")
        if not positive.get("has_diamond"):
            errors.append(f"{case.get('formula_id')}: no diamond")
        if not positive.get("has_box"):
            errors.append(f"{case.get('formula_id')}: no box")
        if not (positive.get("has_conjunction") or positive.get("has_disjunction")):
            errors.append(f"{case.get('formula_id')}: no conjunction/disjunction")
        if case.get("source") == "existing_only" and case.get("missing_action_count", 0) != 0:
            errors.append(f"{case.get('formula_id')}: existing_only uses missing actions")
        if case.get("source") == "mixed_existing_missing":
            if case.get("missing_action_count", 0) <= 0:
                errors.append(f"{case.get('formula_id')}: mixed has no missing action")
            if case.get("known_action_count", 0) <= 0:
                errors.append(f"{case.get('formula_id')}: mixed has no known action")
        for meta_key in ["positive", "negative_existential_target_meta", "negative_universal_target_meta"]:
            meta = case.get(meta_key, {})
            if meta and not meta.get("formula_satisfiable", True):
                errors.append(f"{case.get('formula_id')}: {meta_key} is logically unsatisfiable")
        for key in ["v_in_actions", "v_out_actions"]:
            invalid = sorted(set(case.get(key, [])) - model_actions)
            if invalid:
                errors.append(f"{case.get('formula_id')}: {key} contains non-LTS actions {invalid}")
    return errors


def generate_until_unsatisfied_quota(model: RepairLTS, model_id: str, model_index: int, args) -> tuple[list[dict], int, int, int, int]:
    attempt = 0
    while True:
        seed = args.formula_seed + model_index * 100000 + attempt
        try:
            formula_cases = generate_formula_cases(
                model,
                model_id,
                formulas_per_model=args.formulas_per_model,
                known_formula_count=args.known_formula_count,
                mixed_formula_count=args.mixed_formula_count,
                easy_formula_count=args.easy_formula_count,
                medium_formula_count=args.medium_formula_count,
                hard_formula_count=args.hard_formula_count,
                min_actions=args.formula_min_actions,
                max_actions=args.formula_max_actions,
                min_unsatisfied_formulas=args.min_unsatisfied_formulas,
                seed=seed,
            )
            required_unsatisfied = required_unsatisfied_count(args, formula_cases)
            actual_unsatisfied = count_initially_unsatisfied(formula_cases)
            errors = formula_suite_errors(model, formula_cases, args)
            if not errors:
                if attempt > 0:
                    print(
                        f"formula generation {model_id}: satisfied quota after {attempt + 1} attempts "
                        f"seed={seed} initially_unsatisfied={actual_unsatisfied}/{len(formula_cases)}"
                    )
                return formula_cases, actual_unsatisfied, required_unsatisfied, attempt + 1, seed
            if attempt == 0 or (attempt + 1) % 10 == 0:
                print(
                    f"formula generation {model_id}: retry {attempt + 1}; "
                    f"initially_unsatisfied={actual_unsatisfied}/{len(formula_cases)} "
                    f"required={required_unsatisfied} seed={seed} reason={errors[0]}"
                )
        except Exception as exc:
            if attempt == 0 or (attempt + 1) % 10 == 0:
                print(f"formula generation {model_id}: retry {attempt + 1}; seed={seed} error={exc!r}")
        attempt += 1


def build_v_sets(model: RepairLTS, args, formula_cases: list[dict] | None = None) -> list[dict]:
    result = []
    labels: set[str] = set()
    model_actions = set(model.actions)

    def add(label: str, actions: set[str], source: str, requested_size: int | None = None) -> dict:
        base_label = label
        suffix = 1
        while label in labels:
            suffix += 1
            label = f"{base_label}_{suffix}"
        labels.add(label)
        item = {
            "v_label": label,
            "v_actions": sorted(actions),
            "source": source,
            "requested_size": requested_size if requested_size is not None else len(actions),
        }
        result.append(item)
        return item

    cases = formula_cases or []
    if cases:
        sizes = parse_int_list(args.v_sizes)
        for case in cases:
            formula_id = case["formula_id"]
            labels_by_kind: dict[str, dict[str, str]] = {}
            explicit_labels_by_kind: dict[str, list[str]] = {}
            for formula_kind in ["positive", "negative_existential", "negative_universal"]:
                excluded = case_formula_actions(case, formula_kind) & model_actions
                labels_by_size: dict[str, str] = {}
                for size in sizes:
                    actions = choose_v_actions_excluding(model, size, args.v_policy, excluded)
                    label = f"case_{safe_label(formula_id)}_{safe_label(formula_kind)}_v{size}_{args.v_policy}_formula_safe"
                    item = add(label, actions, "formula_safe_v_size", size)
                    item["formula_id"] = formula_id
                    item["formula_kind"] = formula_kind
                    item["excluded_formula_actions"] = sorted(excluded)
                    labels_by_size[str(size)] = item["v_label"]
                labels_by_kind[formula_kind] = labels_by_size

                explicit_labels: list[str] = []
                for explicit in args.explicit_v:
                    requested = parse_v_actions(explicit)
                    actions = requested - excluded
                    label = f"case_{safe_label(formula_id)}_{safe_label(formula_kind)}_explicit_{safe_label('_'.join(sorted(actions)))}_formula_safe"
                    item = add(label, actions, "formula_safe_explicit", len(requested))
                    item["formula_id"] = formula_id
                    item["formula_kind"] = formula_kind
                    item["excluded_formula_actions"] = sorted(excluded)
                    item["requested_actions"] = sorted(requested)
                    explicit_labels.append(item["v_label"])
                explicit_labels_by_kind[formula_kind] = explicit_labels
            case["v_size_labels_by_kind"] = labels_by_kind
            case["explicit_v_labels_by_kind"] = explicit_labels_by_kind
            case["v_size_labels"] = labels_by_kind.get("positive", {})
            case["explicit_v_labels"] = explicit_labels_by_kind.get("positive", [])
        # Keep the formula-safe labels above as the default experimental
        # surface, but also materialize ordinary V-size quotients. These are
        # used only by unsafe-forgetting ablations that intentionally allow
        # formula actions to appear in V.

    for size in parse_int_list(args.v_sizes):
        actions = choose_v_actions(model, "", size, args.v_policy)
        add(f"v{size}_{args.v_policy}" if size else "v0_empty", actions, "v_size", size)

    for explicit in args.explicit_v:
        actions = parse_v_actions(explicit)
        add(f"explicit_{safe_label('_'.join(sorted(actions)))}", actions, "explicit", len(actions))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: prepare AUT models and strong-V quotients for add/delete repair")
    parser.add_argument("--aut-dir", default="data/download")
    parser.add_argument("--prepared-dir", default="results/add_delete_prepared")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-states", type=int, default=100000)
    parser.add_argument("--max-transitions", type=int, default=1000000)
    parser.add_argument("--v-sizes", default="0,1,3,5")
    parser.add_argument("--v-policy", choices=["least-frequent", "most-frequent", "deterministic", "non-internal-first"], default="least-frequent")
    parser.add_argument("--explicit-v", action="append", default=[], help="Explicit V set such as a or a:b; may be repeated")
    parser.add_argument("--formulas-per-model", type=int, default=30)
    parser.add_argument("--known-formula-count", type=int, default=20)
    parser.add_argument("--mixed-formula-count", type=int, default=10)
    parser.add_argument("--easy-formula-count", type=int, default=5)
    parser.add_argument("--medium-formula-count", type=int, default=10)
    parser.add_argument("--hard-formula-count", type=int, default=15)
    parser.add_argument("--formula-min-actions", type=int, default=5)
    parser.add_argument("--formula-max-actions", type=int, default=10)
    parser.add_argument("--min-unsatisfied-formulas", type=int, default=30)
    parser.add_argument("--formula-seed", type=int, default=13)
    parser.add_argument("--target-state", type=int, default=-1, help="Original LTS state repaired/checked by generated formulas; -1 means AUT initial state")
    parser.add_argument("--write-model-pickles", action="store_true", help="also store original LTS pickles; Stage 2 does not need them")
    parser.add_argument("--quotient-backend", choices=["cpu", "torch"], default="cpu")
    parser.add_argument("--quotient-device", default="cuda")
    parser.add_argument("--strict-quotient-device", action="store_true")
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    models_dir = prepared_dir / "models"
    quotients_dir = prepared_dir / "quotients"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    all_inputs = []
    selected = []
    for path in discover_aut_files(args.aut_dir, args.recursive):
        try:
            item = read_header(path)
            item["selected"] = True
            item["reason"] = ""
            if item["states"] > args.max_states:
                item["selected"] = False
                item["reason"] = f"states>{args.max_states}"
            elif item["transitions"] > args.max_transitions:
                item["selected"] = False
                item["reason"] = f"transitions>{args.max_transitions}"
        except Exception as exc:
            item = {"path": path, "states": "", "transitions": "", "selected": False, "reason": repr(exc)}
        all_inputs.append(item)
    selected = [item for item in all_inputs if item["selected"]]
    selected.sort(key=lambda item: (item["states"], item["transitions"], str(item["path"])))
    if args.limit > 0:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("No AUT files selected. Adjust --aut-dir, --limit, --max-states, or --max-transitions.")

    manifest = {"version": 4, "settings": vars(args), "models": []}
    selection_rows = []
    for item in all_inputs:
        selection_rows.append(
            {
                "model_path": str(item["path"]),
                "states": item.get("states", ""),
                "transitions": item.get("transitions", ""),
                "selected": "YES" if item in selected else "NO",
                "reason": item.get("reason", ""),
            }
        )

    for index, item in enumerate(selected):
        model_id = f"{safe_label(item['path'].stem)}_{index}"
        print_progress("stage1-model", index, len(selected), current=model_id, status="start")
        model = RepairLTS.from_aut(item["path"])
        target_state = resolve_target_state(model, args.target_state)
        formula_model = RepairLTS(target_state, model.state_count, model.edges)
        model_pickle = ""
        if args.write_model_pickles:
            model_pickle_path = models_dir / f"{model_id}.pkl"
            dump_pickle(model_pickle_path, model)
            model_pickle = str(model_pickle_path.relative_to(prepared_dir))
        formula_cases, actual_unsatisfied, required_unsatisfied, formula_attempts, formula_seed = generate_until_unsatisfied_quota(
            formula_model,
            model_id,
            index,
            args,
        )

        model_meta = {
            "model_id": model_id,
            "model_path": str(item["path"]),
            "model_pickle": model_pickle,
            "initial": model.initial,
            "target_state": target_state,
            "states": model.state_count,
            "transitions": model.transition_count,
            "actions": sorted(model.actions),
            "hml_safe_actions": sorted(action for action in model.actions if is_hml_safe_action(action)),
            "formula_cases": formula_cases,
            "formula_generation_attempts": formula_attempts,
            "formula_seed": formula_seed,
            "required_unsatisfied_formulas": required_unsatisfied,
            "initially_unsatisfied_formulas": actual_unsatisfied,
            "v_sets": [],
        }

        quotient_cache = {}
        v_set_list = build_v_sets(model, args, formula_cases)
        for v_index, v_meta in enumerate(v_set_list, start=1):
            v_actions = set(v_meta["v_actions"])
            key = vset_key(v_actions)
            if key in quotient_cache:
                cached = quotient_cache[key]
                elapsed_ms = 0.0
                q_path = cached["q_path"]
                quotient_states = cached["quotient_states"]
                quotient_transitions = cached["quotient_transitions"]
                quotient_reused = True
            else:
                start = time.perf_counter()
                quotient = compute_quotient(model, v_actions, args)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                q_name = f"{model_id}_{safe_label(v_meta['v_label'])}.pkl"
                q_path = quotients_dir / q_name
                dump_pickle(q_path, quotient)
                quotient_states = quotient.block_count
                quotient_transitions = len(quotient.transitions)
                quotient_cache[key] = {
                    "q_path": q_path,
                    "quotient_states": quotient_states,
                    "quotient_transitions": quotient_transitions,
                }
                del quotient
                release_quotient_backend_memory(args)
                quotient_reused = False
            v_meta.update(
                {
                    "quotient_pickle": str(q_path.relative_to(prepared_dir)),
                    "quotient_states": quotient_states,
                    "quotient_transitions": quotient_transitions,
                    "quotient_time_ms": elapsed_ms,
                    "quotient_reused": quotient_reused,
                }
            )
            model_meta["v_sets"].append(v_meta)
            print(
                f"prepared {model_id} {v_meta['v_label']}: "
                f"source={v_meta.get('source', '')} "
                f"required |V|={v_meta.get('requested_size', len(v_actions))} "
                f"actual |V|={len(v_actions)} quotient_states={quotient_states} "
                f"quotient_transitions={quotient_transitions} "
                f"backend={args.quotient_backend}:{args.quotient_device if args.quotient_backend == 'torch' else 'cpu'}"
            )
            print_progress(
                "stage1-quotient",
                v_index,
                len(v_set_list),
                current=model_id,
                v=v_meta["v_label"],
                q_states=quotient_states,
                q_transitions=quotient_transitions,
            )

        manifest["models"].append(model_meta)
        sat = len(formula_cases) - actual_unsatisfied
        unsat = actual_unsatisfied
        missing = sum(1 for case in formula_cases if case.get("uses_missing_actions"))
        print(
            f"formulas {model_id}: total={len(formula_cases)} "
            f"target_state={target_state} initially_satisfied={sat} initially_unsatisfied={unsat} "
            f"required_unsatisfied>={required_unsatisfied} uses_missing={missing} "
            f"generation_attempts={formula_attempts} seed={formula_seed}"
        )
        quotient_cache.clear()
        del formula_model
        del model
        gc.collect()
        release_quotient_backend_memory(args)
        print_progress("stage1-model", index + 1, len(selected), current=model_id, status="done")

    with (prepared_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    with (prepared_dir / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection_rows, handle, ensure_ascii=False, indent=2)

    print(f"Prepared models: {len(manifest['models'])}")
    print(f"Prepared dir: {prepared_dir}")
    print(f"Manifest: {prepared_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
