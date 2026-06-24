from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
import shutil
import stat
import time
from dataclasses import dataclass, replace
from pathlib import Path

from svbr.core import HMLParser
from svbr.experiments.add_delete_run import RUN_FIELDS, write_csv, yes_no
from svbr.experiments.progress import print_progress
from svbr.io_hints import drop_file_cache
from svbr.repair import CostConfig, RepairConfig
from svbr.repair.add_delete import (
    CANDIDATE_FEATURE_ORDER,
    LEGACY_CANDIDATE_FEATURE_ORDER,
    Edge,
    Candidate,
    GraphCandidateRankerModule,
    RepairLTS,
    allowed_by_mode,
    build_mlp,
    candidate_tensors_for_gnn,
    candidate_feature_matrix,
    candidate_linear_prior_score,
    dump_json,
    first_modal_action,
    formula_is_contradiction,
    formula_actions,
    formula_guided_candidates,
    graph_tensors_for_model,
    generic_candidates,
    make_ranker,
    repair_view,
    run_repair,
    torch_load_checkpoint,
    verify_formula,
    verify_formula_with_edits,
)


ERROR_FIELDS = ["case_id", "model_path", "error"]
SKIP_FIELDS = ["case_id", "model_path", "formula_id", "formula_kind", "out_name", "reason"]


@dataclass(frozen=True)
class ExperimentSpec:
    suite: str
    out_name: str
    task: str
    repair_mode: str
    sf_setting: str
    formula_kind: str = "positive"
    v_kind: str = "size"
    v_size: int = 1
    v_selection: str = "formula_safe"
    ranker: str = "heuristic"
    postprocess: bool = True
    quotient_weight: float = 10.0
    lambda_add_non_v: float = 5.0
    lambda_del_non_v: float = 5.0


@dataclass(frozen=True)
class GnnTrainingBatch:
    model: RepairLTS
    v_actions: frozenset[str]
    candidates: tuple[Candidate, ...]
    targets: tuple[float, ...]


def resolve_prepared_path(prepared_dir: Path, relative: str) -> Path:
    base = prepared_dir.resolve()
    path = Path(relative)
    candidate = path if path.is_absolute() else prepared_dir / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ValueError(f"Prepared manifest path escapes prepared dir: {relative}")
    return resolved


def read_pickle(prepared_dir: Path, relative: str):
    with resolve_prepared_path(prepared_dir, relative).open("rb") as handle:
        try:
            return pickle.load(handle)
        finally:
            drop_file_cache(handle)


def target_state_for_model(model_meta: dict, args=None) -> int:
    if args is not None and getattr(args, "target_state", -1) >= 0:
        target_state = int(args.target_state)
    else:
        target_state = int(model_meta.get("target_state", model_meta.get("initial", 0)))
    states = int(model_meta.get("states", 0) or 0)
    if states > 0 and not 0 <= target_state < states:
        raise ValueError(f"{model_meta.get('model_id', '')}: target_state {target_state} is outside 0..{states - 1}")
    return target_state


def quotient_as_repair_lts(quotient, original_target_state: int) -> RepairLTS:
    """Use the strong-V quotient as the Stage 2 repair surface.

    Quotient states are block ids. Edits produced on this model are therefore
    block-level edits and must be lifted back to the original LTS only after
    Stage 2 has finished.
    """
    if not quotient.state_to_block:
        return RepairLTS(0, 0, frozenset())
    initial_block = int(quotient.state_to_block[int(original_target_state)])
    edges = frozenset(
        Edge(int(src), str(action), int(dst))
        for src, action, dst in quotient.transitions
    )
    return RepairLTS(initial_block, int(quotient.block_count), edges)


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


def formula_texts(spec: ExperimentSpec, formula_case: dict) -> tuple[str, str, dict]:
    if spec.formula_kind == "positive":
        return formula_case["positive_formula"], "", formula_case["positive"]
    if spec.formula_kind == "negative_existential":
        return (
            formula_case["negative_existential_target"],
            formula_case["negative_existential_psi"],
            formula_case["negative_existential_target_meta"],
        )
    if spec.formula_kind == "negative_universal":
        return (
            formula_case["negative_universal_target"],
            formula_case["negative_universal_psi"],
            formula_case["negative_universal_target_meta"],
        )
    raise ValueError(f"Unknown formula kind: {spec.formula_kind}")


def case_target_formula_actions(formula_case: dict) -> set[str]:
    actions = set(formula_case.get("formula_actions", []))
    for key in [
        "positive",
        "negative_existential_target_meta",
        "negative_universal_target_meta",
    ]:
        actions.update(formula_case.get(key, {}).get("formula_actions", []))
    return actions


def case_formula_kind_actions(formula_case: dict, formula_kind: str) -> set[str]:
    if formula_kind == "positive":
        return set(formula_case.get("positive", {}).get("formula_actions", formula_case.get("formula_actions", [])))
    if formula_kind == "negative_existential":
        return set(formula_case.get("negative_existential_target_meta", {}).get("formula_actions", []))
    if formula_kind == "negative_universal":
        return set(formula_case.get("negative_universal_target_meta", {}).get("formula_actions", []))
    return case_target_formula_actions(formula_case)


def with_v_suffix(name: str, v_size: int) -> str:
    return f"{name}_V{v_size}"


def build_specs(
    include_neural: bool = True,
    v_sizes: list[int] | None = None,
    default_ranker: str = "heuristic",
    include_heuristic_comparison: bool = True,
    include_random_comparison: bool = False,
    experiment_profile: str = "full",
    v_selection: str = "formula_safe",
    repair_mode_filter: str = "all",
) -> list[ExperimentSpec]:
    v_sizes = v_sizes or [0, 1, 3, 5]
    specs = []

    def add_ranker_specs(v_size: int, all_formula_kinds: bool = False) -> None:
        formula_specs = [("pos", "positive", "positive")]
        if all_formula_kinds:
            formula_specs.extend(
                [
                    ("neg_exist", "negative", "negative_existential"),
                    ("neg_univ", "negative", "negative_universal"),
                ]
            )
        for label, task, formula_kind in formula_specs:
            heuristic_name = f"ranker_heuristic_{label}" if all_formula_kinds else "ranker_heuristic"
            neural_name = f"ranker_neural_{label}" if all_formula_kinds else "ranker_neural"
            if include_heuristic_comparison:
                specs.append(ExperimentSpec("ranker", with_v_suffix(heuristic_name, v_size), task, "add-delete", "strict_then_escalate", ranker="heuristic", formula_kind=formula_kind, v_size=v_size, v_selection=v_selection))
            if include_random_comparison:
                random_name = f"ranker_random_{label}" if all_formula_kinds else "ranker_random"
                specs.append(ExperimentSpec("ranker", with_v_suffix(random_name, v_size), task, "add-delete", "strict_then_escalate", ranker="random", formula_kind=formula_kind, v_size=v_size, v_selection=v_selection))
            if include_neural:
                specs.append(ExperimentSpec("ranker", with_v_suffix(neural_name, v_size), task, "add-delete", "strict_then_escalate", ranker="neural", formula_kind=formula_kind, v_size=v_size, v_selection=v_selection))

    if experiment_profile == "ranker-add-delete":
        for v_size in v_sizes:
            add_ranker_specs(v_size, all_formula_kinds=True)
        return [spec for spec in specs if repair_mode_filter == "all" or spec.repair_mode == repair_mode_filter]

    if experiment_profile == "repair-mode-ablation":
        for v_size in v_sizes:
            for mode in ["add-only", "delete-only", "add-delete"]:
                specs.append(ExperimentSpec("repair_mode", with_v_suffix(f"pos_{mode}", v_size), "positive", mode, "strict_then_escalate", v_size=v_size, ranker=default_ranker, v_selection=v_selection))
                specs.append(ExperimentSpec("repair_mode", with_v_suffix(f"neg_exist_{mode}", v_size), "negative", mode, "strict_then_escalate", formula_kind="negative_existential", v_size=v_size, ranker=default_ranker, v_selection=v_selection))
                specs.append(ExperimentSpec("repair_mode", with_v_suffix(f"neg_univ_{mode}", v_size), "negative", mode, "strict_then_escalate", formula_kind="negative_universal", v_size=v_size, ranker=default_ranker, v_selection=v_selection))
        return [spec for spec in specs if repair_mode_filter == "all" or spec.repair_mode == repair_mode_filter]

    for v_size in v_sizes:
        specs.extend(
            [
                ExperimentSpec("sf_vs_no_sf", with_v_suffix("no_sf_add_delete", v_size), "positive", "add-delete", "no_sf", v_size=v_size, ranker=default_ranker, quotient_weight=0.0, lambda_add_non_v=0.0, lambda_del_non_v=0.0, v_selection=v_selection),
                ExperimentSpec("sf_vs_no_sf", with_v_suffix("soft_sf_add_delete", v_size), "positive", "add-delete", "soft_sf", v_size=v_size, ranker=default_ranker, v_selection=v_selection),
                ExperimentSpec("sf_vs_no_sf", with_v_suffix("strict_then_escalate_add_delete", v_size), "positive", "add-delete", "strict_then_escalate", v_size=v_size, ranker=default_ranker, v_selection=v_selection),
            ]
        )
    for v_size in v_sizes:
        modes = ["add-delete"] if experiment_profile == "add-delete-only" else ["add-only", "delete-only", "add-delete"]
        for mode in modes:
            specs.append(ExperimentSpec("repair_mode", with_v_suffix(f"pos_{mode}", v_size), "positive", mode, "strict_then_escalate", v_size=v_size, ranker=default_ranker, v_selection=v_selection))
            specs.append(ExperimentSpec("repair_mode", with_v_suffix(f"neg_exist_{mode}", v_size), "negative", mode, "strict_then_escalate", formula_kind="negative_existential", v_size=v_size, ranker=default_ranker, v_selection=v_selection))
            specs.append(ExperimentSpec("repair_mode", with_v_suffix(f"neg_univ_{mode}", v_size), "negative", mode, "strict_then_escalate", formula_kind="negative_universal", v_size=v_size, ranker=default_ranker, v_selection=v_selection))
        specs.extend(
            [
                ExperimentSpec("postprocess", with_v_suffix("post_off", v_size), "negative", "add-delete", "strict_then_escalate", formula_kind="negative_existential", postprocess=False, v_size=v_size, ranker=default_ranker, v_selection=v_selection),
                ExperimentSpec("postprocess", with_v_suffix("post_on", v_size), "negative", "add-delete", "strict_then_escalate", formula_kind="negative_existential", postprocess=True, v_size=v_size, ranker=default_ranker, v_selection=v_selection),
            ]
        )
        add_ranker_specs(v_size, all_formula_kinds=experiment_profile == "add-delete-only")
    return [spec for spec in specs if repair_mode_filter == "all" or spec.repair_mode == repair_mode_filter]


def find_v_meta(model_meta: dict, spec: ExperimentSpec, formula_case: dict) -> dict:
    if spec.v_kind in {"formula_in", "formula_out"}:
        raise KeyError("formula_in/formula_out V sets are disabled because formula actions must not be forgotten.")

    if spec.v_size is not None:
        if spec.v_selection != "unsafe":
            labels_by_kind = formula_case.get("v_size_labels_by_kind", {})
            labels_by_size = labels_by_kind.get(spec.formula_kind, formula_case.get("v_size_labels", {}))
            wanted_label = labels_by_size.get(str(spec.v_size), "")
            if wanted_label:
                for v_meta in model_meta["v_sets"]:
                    if v_meta["v_label"] == wanted_label:
                        return v_meta
                raise KeyError(f"Missing formula-safe |V|={spec.v_size} set {wanted_label} for {model_meta['model_id']}. Re-run stage 1.")

    for v_meta in model_meta["v_sets"]:
        if v_meta.get("source") == "v_size" and int(v_meta.get("requested_size", -1)) == spec.v_size:
            return v_meta
    raise KeyError(f"Missing |V|={spec.v_size} for {model_meta['model_id']}. Re-run stage 1 with --v-sizes including {spec.v_size}.")


def make_config(spec: ExperimentSpec, args) -> RepairConfig:
    costs = CostConfig(
        w_add=1.0,
        w_del=1.0,
        lambda_add_non_v=spec.lambda_add_non_v,
        lambda_del_non_v=spec.lambda_del_non_v,
        quotient_weight=spec.quotient_weight,
    )
    return RepairConfig(
        repair_mode=spec.repair_mode,
        sf_setting=spec.sf_setting,
        ranker=spec.ranker,
        ranker_architecture=args.ranker_architecture if spec.ranker == "neural" else "heuristic",
        gnn_graph_mode=args.gnn_graph_mode,
        model_path=args.ranker_model,
        ranker_device=args.device,
        strict_ranker_device=args.strict_device,
        beam_width=args.beam_width,
        max_iters=args.max_iters,
        candidate_limit=args.candidate_limit,
        candidate_state_limit=args.candidate_state_limit,
        state_scan_limit=args.state_scan_limit,
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
        neural_prefilter_multiplier=args.neural_prefilter_multiplier,
        neural_prefilter_limit=args.neural_prefilter_limit,
        neural_linear_blend=args.neural_linear_blend,
        neural_verify_frontier_only=args.neural_verify_frontier_only,
        neural_verify_top_k=args.neural_verify_top_k,
        max_case_seconds=args.max_case_seconds,
        max_quotient_drift=args.max_quotient_drift,
        drift_mode=args.drift_mode,
        exact_drift_max_transitions=args.exact_drift_max_transitions,
        store_final_model=args.write_repaired_aut,
        postprocess=spec.postprocess,
        include_partition_drift=args.partition_drift,
        seed=args.seed,
        costs=costs,
    )


def make_cegis_oracle_config(config: RepairConfig, args) -> RepairConfig:
    oracle_model = str(getattr(args, "neural_cegis_oracle_model", "") or "").strip()
    use_neural_oracle = bool(oracle_model)
    if use_neural_oracle and not Path(oracle_model).exists():
        print(f"Warning: neural CEGIS oracle model not found: {oracle_model}. Falling back to heuristic oracle.")
        oracle_model = ""
        use_neural_oracle = False
    return replace(
        config,
        ranker="neural" if use_neural_oracle else "heuristic",
        ranker_architecture="linear" if use_neural_oracle else "heuristic",
        model_path=oracle_model,
        search_strategy="beam",
        beam_width=max(config.beam_width, 4),
        max_iters=max(config.max_iters, 16),
        candidate_limit=max(config.candidate_limit, 64),
        candidate_state_limit=max(config.candidate_state_limit, 128),
        state_scan_limit=max(config.state_scan_limit, 5000),
        dynamic_budget=True,
        dynamic_budget_rounds=0,
        dynamic_max_iters=max(config.dynamic_max_iters, 512),
        dynamic_max_beam_width=max(config.dynamic_max_beam_width, 256),
        dynamic_max_candidate_limit=0,
        dynamic_max_candidate_state_limit=0,
        dynamic_max_state_scan_limit=0,
        dynamic_max_minimal_layer_width=max(config.dynamic_max_minimal_layer_width, 32768),
        dynamic_max_minimal_seen_limit=max(config.dynamic_max_minimal_seen_limit, 500000),
        dynamic_final_search_strategy="neural_guided_minimal",
        neural_prefilter_multiplier=config.neural_prefilter_multiplier,
        neural_prefilter_limit=config.neural_prefilter_limit,
        neural_linear_blend=config.neural_linear_blend,
    )


def make_linear_rescue_config(config: RepairConfig, args) -> RepairConfig | None:
    rescue_model = str(getattr(args, "neural_rescue_linear_model", "") or "").strip()
    if not rescue_model:
        return None
    if not Path(rescue_model).exists():
        print(f"Warning: neural rescue linear model not found: {rescue_model}. Rescue fallback disabled.")
        return None
    return replace(
        config,
        ranker="neural",
        ranker_architecture="linear",
        model_path=rescue_model,
        search_strategy="beam",
        max_iters=max(config.max_iters, 16),
        beam_width=max(config.beam_width, 8),
        candidate_limit=max(config.candidate_limit, 64),
        candidate_state_limit=max(config.candidate_state_limit, 128),
        state_scan_limit=max(config.state_scan_limit, 5000),
        dynamic_budget=True,
        dynamic_budget_rounds=0,
        dynamic_max_iters=max(config.dynamic_max_iters, 512),
        dynamic_max_beam_width=max(config.dynamic_max_beam_width, 256),
        dynamic_max_candidate_limit=max(config.dynamic_max_candidate_limit, 4096),
        dynamic_max_candidate_state_limit=max(config.dynamic_max_candidate_state_limit, 2048),
        dynamic_max_state_scan_limit=max(config.dynamic_max_state_scan_limit, 20000),
        dynamic_max_minimal_layer_width=max(config.dynamic_max_minimal_layer_width, 32768),
        dynamic_max_minimal_seen_limit=max(config.dynamic_max_minimal_seen_limit, 500000),
        dynamic_final_search_strategy="neural_guided_minimal",
        neural_linear_blend=0.0,
        max_case_seconds=0.0,
    )


def neural_cegis_retry_config(config: RepairConfig, model: RepairLTS, formula, attempt: int) -> RepairConfig:
    attempt = max(1, attempt)
    formula_action_count = max(1, len(formula_actions(formula)))
    formula_edit_floor = max(1, formula.modal_action_count()) * 2
    finite_edit_cap = max(config.max_iters, formula_edit_floor)
    if config.dynamic_max_iters > 0:
        finite_edit_cap = max(finite_edit_cap, config.dynamic_max_iters)

    max_candidate_limit = (
        config.dynamic_max_candidate_limit
        if config.dynamic_max_candidate_limit > 0
        else 4096
    )
    max_candidate_state_limit = (
        config.dynamic_max_candidate_state_limit
        if config.dynamic_max_candidate_state_limit > 0
        else min(model.state_count, 2048)
    )
    max_state_scan_limit = (
        config.dynamic_max_state_scan_limit
        if config.dynamic_max_state_scan_limit > 0
        else min(model.state_count, 20000)
    )
    max_beam_width = (
        config.dynamic_max_beam_width
        if config.dynamic_max_beam_width > 0
        else max(config.beam_width, min(256, max_candidate_limit))
    )
    max_layer_width = (
        config.dynamic_max_minimal_layer_width
        if config.dynamic_max_minimal_layer_width > 0
        else 32768
    )
    max_seen_limit = (
        config.dynamic_max_minimal_seen_limit
        if config.dynamic_max_minimal_seen_limit > 0
        else 500000
    )
    factor = min(2 ** attempt, 64)

    def grow(current: int, floor: int, cap: int) -> int:
        cap = max(1, cap)
        if current <= 0:
            return min(cap, max(1, floor))
        return min(cap, max(current + 1, current * factor, floor))

    # MLP/GNN keep learning from failed HML checks, but the search remains
    # memory/time-bounded. The ranker orders candidates; HML remains the verifier.
    # Do not switch online CEGIS retries to minimal search by default: hard
    # cases can spend millions of HML checks there and block the whole suite.
    retry_strategy = config.search_strategy
    return replace(
        config,
        search_strategy=retry_strategy,
        max_iters=min(finite_edit_cap, grow(config.max_iters, formula_edit_floor, finite_edit_cap)),
        beam_width=grow(config.beam_width, 8, max_beam_width),
        candidate_limit=grow(config.candidate_limit, 64, max_candidate_limit),
        candidate_state_limit=grow(config.candidate_state_limit, 128, max_candidate_state_limit),
        state_scan_limit=grow(config.state_scan_limit, 1000, max_state_scan_limit),
        minimal_layer_width=grow(config.minimal_layer_width, 2048, max_layer_width),
        minimal_seen_limit=grow(config.minimal_seen_limit, 100000, max_seen_limit),
        neural_prefilter_multiplier=config.neural_prefilter_multiplier,
        neural_prefilter_limit=grow(config.neural_prefilter_limit, 512, max_candidate_limit),
        neural_linear_blend=config.neural_linear_blend,
        dynamic_budget=True,
        dynamic_final_search_strategy="",
    )


def ranker_feature_order(args) -> list[str]:
    if args.ranker_architecture == "linear" and getattr(args, "linear_feature_set", "current") == "legacy_v3":
        return LEGACY_CANDIDATE_FEATURE_ORDER
    return CANDIDATE_FEATURE_ORDER


def ranker_feature_order_from_config(config: RepairConfig) -> list[str]:
    if getattr(config, "ranker_architecture", "") == "linear":
        return CANDIDATE_FEATURE_ORDER
    return CANDIDATE_FEATURE_ORDER


def ranker_training_signature(manifest: dict, args) -> str:
    feature_order = ranker_feature_order(args)
    payload = {
        "feature_order": feature_order,
        "v_sizes": parse_int_list(args.v_sizes),
        "ranker_train_samples": args.ranker_train_samples,
        "ranker_train_model_limit": args.ranker_train_model_limit,
        "ranker_train_formula_limit": args.ranker_train_formula_limit,
        "ranker_train_candidate_limit": args.ranker_train_candidate_limit,
        "ranker_epochs": args.ranker_epochs,
        "ranker_lr": args.ranker_lr,
        "ranker_architecture": args.ranker_architecture,
        "linear_feature_set": getattr(args, "linear_feature_set", "current"),
        "ranker_hidden_dim": args.ranker_hidden_dim,
        "ranker_hidden_layers": args.ranker_hidden_layers,
        "candidate_state_limit": args.candidate_state_limit,
        "state_scan_limit": args.state_scan_limit,
        "seed": args.seed,
        "signature_version": 8,
        "models": [
            {
                "model_id": model.get("model_id"),
                "states": model.get("states"),
                "transitions": model.get("transitions"),
                "target_state": target_state_for_model(model, args),
                "actions": model.get("actions", []),
                "formula_cases": [
                    {
                        "formula_id": case.get("formula_id"),
                        "positive_formula": case.get("positive_formula"),
                        "negative_existential_target": case.get("negative_existential_target"),
                        "negative_universal_target": case.get("negative_universal_target"),
                        "initial_satisfied": case.get("initial_satisfied"),
                        "difficulty": case.get("difficulty"),
                        "source": case.get("source"),
                    }
                    for case in model.get("formula_cases", [])
                ],
                "v_sets": [
                    {
                        "source": v.get("source"),
                        "requested_size": v.get("requested_size"),
                        "v_actions": v.get("v_actions", []),
                    }
                    for v in model.get("v_sets", [])
                    if v.get("source") in {"v_size", "formula_safe_v_size"}
                ],
            }
            for model in manifest.get("models", [])
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_is_trained(path: Path, signature: str) -> bool:
    if not path.exists():
        return False
    try:
        import torch

        payload = torch_load_checkpoint(torch, path, map_location="cpu")
    except Exception:
        return False
    return (
        payload.get("kind") in {"linear_candidate_ranker_v3_trained", "candidate_ranker_v4_trained"}
        and tuple(payload.get("feature_order") or ()) in {tuple(CANDIDATE_FEATURE_ORDER), tuple(LEGACY_CANDIDATE_FEATURE_ORDER)}
        and payload.get("prepared_signature") == signature
    )


def candidate_training_target(candidate, model: RepairLTS, formula, v_actions: set[str]) -> float:
    edit = candidate.edit
    adds = frozenset([edit.edge]) if edit.op == "add" else frozenset()
    dels = frozenset([edit.edge]) if edit.op == "del" else frozenset()
    immediate_success, _checker = verify_formula_with_edits(model, formula, adds, dels)
    if immediate_success:
        return 1.0
    linear_prior = candidate_linear_prior_score(candidate, model, v_actions)
    action_bonus = 0.20 if edit.edge.action in v_actions else 0.0
    symbolic_bonus = max(0.0, 0.25 * (1.0 - min(candidate.symbolic_reason, 10) / 10.0))
    local_bonus = max(0.0, min(candidate.local_score, 1.0)) * 0.30
    path_bonus = 0.15 if candidate.along_counterexample_path else 0.0
    required_action_bonus = 0.10 if candidate.required_modal_action and edit.edge.action == candidate.required_modal_action else 0.0
    next_formula_bonus = 0.10 if candidate.dst_satisfies_next else 0.0
    symbolic_target = min(0.95, action_bonus + symbolic_bonus + local_bonus + path_bonus + required_action_bonus + next_formula_bonus)
    teacher_target = (0.55 * symbolic_target) + (0.45 * linear_prior)
    return min(0.95, max(symbolic_target, teacher_target))


def candidate_training_weight(candidate, target: float) -> float:
    weight = 1.0 + (3.0 * max(0.0, min(target, 1.0)))
    if candidate.along_counterexample_path:
        weight += 0.75
    if candidate.required_modal_action and candidate.edit.edge.action == candidate.required_modal_action:
        weight += 0.50
    if candidate.dst_satisfies_next:
        weight += 0.50
    if candidate.symbolic_reason <= 1:
        weight += 0.50
    return min(6.0, weight)


def weighted_mse(torch_module, pred, y, weights=None):
    loss = (pred - y).pow(2)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def ranker_training_candidates(model: RepairLTS, formula, config: RepairConfig, limit: int, v_actions: set[str] | None = None) -> list[Candidate]:
    v_actions = set() if v_actions is None else v_actions
    raw_candidates = formula_guided_candidates(model, formula, model.initial, True, config)
    raw_candidates.extend(generic_candidates(model, formula, config))
    dedup: dict[tuple[str, Edge], Candidate] = {}
    for candidate in raw_candidates:
        candidate = replace(candidate, formula_modal_depth=formula.modal_depth(), current_edit_count=0)
        edit = candidate.edit
        if not allowed_by_mode(edit, "add-delete"):
            continue
        if edit.op == "add" and edit.edge in model.edges:
            continue
        if edit.op == "del" and edit.edge not in model.edges:
            continue
        key = (edit.op, edit.edge)
        existing = dedup.get(key)
        if existing is None or candidate.local_score > existing.local_score:
            dedup[key] = candidate
    candidates = sorted(
        dedup.values(),
        key=lambda candidate: (
            -candidate_linear_prior_score(candidate, model, v_actions),
            candidate.symbolic_reason,
            -candidate.local_score,
            candidate.edit.op,
            candidate.edit.edge.src,
            candidate.edit.edge.action,
            candidate.edit.edge.dst,
        ),
    )
    return candidates[:limit]


def cegis_counterexample_training_rows(
    model: RepairLTS,
    formula,
    config: RepairConfig,
    v_actions: set[str],
    oracle_result,
    feature_order: list[str],
    limit: int,
) -> tuple[list[list[float]], list[float]]:
    oracle_edits = [("add", edge) for edge in sorted(oracle_result.adds)] + [("del", edge) for edge in sorted(oracle_result.dels)]
    if not oracle_edits:
        return [], []
    oracle_keys = set(oracle_edits)
    rows: list[list[float]] = []
    targets: list[float] = []
    current_adds: set[Edge] = set()
    current_dels: set[Edge] = set()
    step_limit = max(1, limit)

    for step, (oracle_op, oracle_edge) in enumerate(oracle_edits):
        current = repair_view(model, frozenset(current_adds), frozenset(current_dels))
        raw_candidates = formula_guided_candidates(current, formula, current.initial, True, config)
        raw_candidates.extend(generic_candidates(current, formula, config))
        dedup: dict[tuple[str, Edge], Candidate] = {}
        for candidate in raw_candidates:
            candidate = replace(
                candidate,
                formula_modal_depth=formula.modal_depth(),
                current_edit_count=len(current_adds) + len(current_dels),
            )
            edit = candidate.edit
            if not allowed_by_mode(edit, "add-delete"):
                continue
            if edit.op == "add" and edit.edge in current.edges:
                continue
            if edit.op == "del" and edit.edge not in current.edges:
                continue
            key = (edit.op, edit.edge)
            existing = dedup.get(key)
            if existing is None or candidate.local_score > existing.local_score:
                dedup[key] = candidate

        candidates = sorted(
            dedup.values(),
            key=lambda candidate: (
                0 if (candidate.edit.op, candidate.edit.edge) in oracle_keys else 1,
                candidate.symbolic_reason,
                -candidate.local_score,
                candidate.edit.op,
                candidate.edit.edge.src,
                candidate.edit.edge.action,
                candidate.edit.edge.dst,
            ),
        )[:step_limit]
        if candidates:
            candidate_rows = candidate_feature_matrix(candidates, current, v_actions, feature_order=feature_order)
            for candidate, row in zip(candidates, candidate_rows):
                key = (candidate.edit.op, candidate.edit.edge)
                if key in oracle_keys:
                    target = 1.0
                else:
                    target = min(0.20, candidate_training_target(candidate, current, formula, v_actions))
                rows.append(row)
                targets.append(target)

        if oracle_op == "add":
            current_adds.add(oracle_edge)
            current_dels.discard(oracle_edge)
        else:
            current_dels.add(oracle_edge)
            current_adds.discard(oracle_edge)

    return rows, targets


def fine_tune_neural_ranker_from_counterexample(
    ranker,
    model: RepairLTS,
    formula,
    config: RepairConfig,
    v_actions: set[str],
    oracle_result,
    feature_order: list[str],
    epochs: int,
    lr: float,
    candidate_limit: int,
) -> tuple[bool, dict]:
    if ranker is None or not hasattr(ranker, "architecture"):
        return False, {"reason": "ranker_not_neural"}
    if getattr(ranker, "architecture", "") != "mlp":
        return False, {"reason": f"online_training_not_enabled_for_{getattr(ranker, 'architecture', '')}"}
    rows, targets = cegis_counterexample_training_rows(
        model,
        formula,
        config,
        v_actions,
        oracle_result,
        feature_order,
        candidate_limit,
    )
    if not rows:
        return False, {"reason": "no_counterexample_training_rows"}
    torch = ranker.torch
    device = ranker.device
    model_nn = ranker.model
    model_nn.train()
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    y = torch.tensor(targets, dtype=torch.float32, device=device).view(-1, 1)
    weights = torch.clamp(1.0 + (3.0 * y), max=5.0)
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=lr)
    final_loss = 0.0
    for _epoch in range(max(1, epochs)):
        optimizer.zero_grad(set_to_none=True)
        pred = model_nn(x)
        loss = weighted_mse(torch, pred, y, weights)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    model_nn.eval()
    del x, y, weights, optimizer
    if hasattr(torch, "cuda") and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return True, {"samples": len(rows), "epochs": max(1, epochs), "final_loss": final_loss}


def hml_failure_training_candidates(
    model: RepairLTS,
    formula,
    config: RepairConfig,
    v_actions: set[str],
    limit: int,
) -> tuple[list[Candidate], list[float]]:
    candidates = ranker_training_candidates(model, formula, config, max(1, limit), v_actions)
    targets = [candidate_training_target(candidate, model, formula, v_actions) for candidate in candidates]
    return candidates, targets


def fine_tune_neural_ranker_from_hml_failure(
    ranker,
    model: RepairLTS,
    formula,
    config: RepairConfig,
    v_actions: set[str],
    feature_order: list[str],
    epochs: int,
    lr: float,
    candidate_limit: int,
) -> tuple[bool, dict]:
    if ranker is None or not hasattr(ranker, "architecture"):
        return False, {"reason": "ranker_not_neural"}
    candidates, targets = hml_failure_training_candidates(model, formula, config, v_actions, candidate_limit)
    if not candidates:
        return False, {"reason": "no_hml_failure_training_candidates"}
    torch = ranker.torch
    device = ranker.device
    model_nn = ranker.model
    if model_nn is None:
        return False, {"reason": f"online_training_not_enabled_for_{getattr(ranker, 'architecture', '')}"}
    model_nn.train()
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=lr)
    final_loss = 0.0
    architecture = getattr(ranker, "architecture", "")
    y = torch.tensor(targets, dtype=torch.float32, device=device).view(-1, 1)
    weight_values = [candidate_training_weight(candidate, target) for candidate, target in zip(candidates, targets)]
    weights = torch.tensor(weight_values, dtype=torch.float32, device=device).view(-1, 1)
    x = None
    node_features = edge_index = edge_features = None
    candidate_features = candidate_src = candidate_dst = None
    if architecture == "mlp":
        rows = candidate_feature_matrix(candidates, model, v_actions, feature_order=feature_order)
        x = torch.tensor(rows, dtype=torch.float32, device=device)
    elif architecture == "gnn":
        node_features, edge_index, edge_features = graph_tensors_for_model(model, v_actions, torch, device)
        candidate_features, candidate_src, candidate_dst = candidate_tensors_for_gnn(
            candidates,
            model,
            v_actions,
            torch,
            device,
            feature_order=feature_order,
        )
    else:
        del y
        return False, {"reason": f"online_training_not_enabled_for_{architecture}"}
    for _epoch in range(max(1, epochs)):
        optimizer.zero_grad(set_to_none=True)
        if architecture == "mlp":
            pred = model_nn(x)
        elif architecture == "gnn":
            pred = model_nn(node_features, edge_index, edge_features, candidate_features, candidate_src, candidate_dst)
        loss = weighted_mse(torch, pred, y, weights)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())
    model_nn.eval()
    del y, weights, optimizer
    if x is not None:
        del x
    if node_features is not None:
        del node_features, edge_index, edge_features, candidate_features, candidate_src, candidate_dst
    if hasattr(torch, "cuda") and str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return True, {
        "samples": len(candidates),
        "epochs": max(1, epochs),
        "final_loss": final_loss,
        "source": "hml_failure_formula_guided_candidates",
    }


def collect_ranker_training_examples(prepared_dir: Path, manifest: dict, args) -> tuple[list[list[float]], list[float]]:
    features: list[list[float]] = []
    targets: list[float] = []
    feature_order = ranker_feature_order(args)
    config = RepairConfig(
        repair_mode="add-delete",
        sf_setting="strict_then_escalate",
        candidate_limit=args.ranker_train_candidate_limit,
        candidate_state_limit=args.candidate_state_limit,
        state_scan_limit=args.state_scan_limit,
        max_iters=args.max_iters,
        seed=args.seed,
        costs=CostConfig(),
    )

    model_count = len(manifest["models"])
    if args.ranker_train_model_limit > 0:
        model_count = min(model_count, args.ranker_train_model_limit)
    for model_index, model_meta in enumerate(manifest["models"][:model_count]):
        target_state = target_state_for_model(model_meta, args)
        quotient_cache: dict[str, RepairLTS] = {}

        def get_quotient_model(v_meta: dict) -> RepairLTS:
            relative = v_meta["quotient_pickle"]
            if v_meta.get("source") == "v_size":
                cached = quotient_cache.get(relative)
                if cached is None:
                    quotient = read_pickle(prepared_dir, relative)
                    cached = quotient_as_repair_lts(quotient, target_state)
                    quotient_cache[relative] = cached
                    del quotient
                return cached
            quotient = read_pickle(prepared_dir, relative)
            model = quotient_as_repair_lts(quotient, target_state)
            del quotient
            return model

        try:
            global_v_metas = [item for item in model_meta["v_sets"] if item.get("source") == "v_size"]
            v_by_label = {item.get("v_label", ""): item for item in model_meta["v_sets"]}
            formula_cases = model_meta.get("formula_cases", [])[: args.ranker_train_formula_limit]
            for formula_case in formula_cases:
                for formula_kind in ["positive", "negative_existential", "negative_universal"]:
                    labels_by_kind = formula_case.get("v_size_labels_by_kind", {})
                    labels_by_size = labels_by_kind.get(formula_kind, formula_case.get("v_size_labels", {}))
                    if labels_by_size:
                        v_metas = [
                            v_by_label[label]
                            for _size, label in sorted(labels_by_size.items(), key=lambda item: int(item[0]))
                            if label in v_by_label
                        ]
                    else:
                        v_metas = global_v_metas
                    spec = ExperimentSpec("ranker_train", "ranker_train", "positive" if formula_kind == "positive" else "negative", "add-delete", "strict_then_escalate", formula_kind=formula_kind)
                    target_text, _psi_text, _meta = formula_texts(spec, formula_case)
                    formula = HMLParser.parse(target_text)
                    for v_meta in v_metas:
                        model = get_quotient_model(v_meta)
                        try:
                            v_actions = set(v_meta["v_actions"])
                            already_true, _checker = verify_formula(model, formula)
                            if already_true:
                                continue
                            candidates = ranker_training_candidates(model, formula, config, args.ranker_train_candidate_limit, v_actions)
                            candidate_rows = candidate_feature_matrix(candidates, model, v_actions, feature_order=feature_order)
                            for candidate, row in zip(candidates, candidate_rows):
                                features.append(row)
                                targets.append(candidate_training_target(candidate, model, formula, v_actions))
                                if len(features) >= args.ranker_train_samples:
                                    print_progress(
                                        "stage2-ranker-data",
                                        model_index + 1,
                                        model_count,
                                        current=model_meta.get("model_id", ""),
                                        samples=len(features),
                                        status="sample_limit",
                                    )
                                    return features, targets
                        finally:
                            if v_meta.get("source") != "v_size":
                                del model
        finally:
            quotient_cache.clear()
            gc.collect()
        print_progress(
            "stage2-ranker-data",
            model_index + 1,
            model_count,
            current=model_meta.get("model_id", ""),
            samples=len(features),
        )
    return features, targets


def collect_gnn_ranker_training_batches(prepared_dir: Path, manifest: dict, args) -> tuple[list[GnnTrainingBatch], int, float | None, float | None]:
    batches: list[GnnTrainingBatch] = []
    sample_count = 0
    target_min: float | None = None
    target_max: float | None = None
    config = RepairConfig(
        repair_mode="add-delete",
        sf_setting="strict_then_escalate",
        candidate_limit=args.ranker_train_candidate_limit,
        candidate_state_limit=args.candidate_state_limit,
        state_scan_limit=args.state_scan_limit,
        max_iters=args.max_iters,
        seed=args.seed,
        costs=CostConfig(),
    )

    model_count = len(manifest["models"])
    if args.ranker_train_model_limit > 0:
        model_count = min(model_count, args.ranker_train_model_limit)
    for model_index, model_meta in enumerate(manifest["models"][:model_count]):
        target_state = target_state_for_model(model_meta, args)
        quotient_cache: dict[str, RepairLTS] = {}

        def get_quotient_model(v_meta: dict) -> RepairLTS:
            relative = v_meta["quotient_pickle"]
            if v_meta.get("source") == "v_size":
                cached = quotient_cache.get(relative)
                if cached is None:
                    quotient = read_pickle(prepared_dir, relative)
                    cached = quotient_as_repair_lts(quotient, target_state)
                    quotient_cache[relative] = cached
                    del quotient
                return cached
            quotient = read_pickle(prepared_dir, relative)
            model = quotient_as_repair_lts(quotient, target_state)
            del quotient
            return model

        try:
            global_v_metas = [item for item in model_meta["v_sets"] if item.get("source") == "v_size"]
            v_by_label = {item.get("v_label", ""): item for item in model_meta["v_sets"]}
            formula_cases = model_meta.get("formula_cases", [])[: args.ranker_train_formula_limit]
            for formula_case in formula_cases:
                for formula_kind in ["positive", "negative_existential", "negative_universal"]:
                    labels_by_kind = formula_case.get("v_size_labels_by_kind", {})
                    labels_by_size = labels_by_kind.get(formula_kind, formula_case.get("v_size_labels", {}))
                    if labels_by_size:
                        v_metas = [
                            v_by_label[label]
                            for _size, label in sorted(labels_by_size.items(), key=lambda item: int(item[0]))
                            if label in v_by_label
                        ]
                    else:
                        v_metas = global_v_metas
                    spec = ExperimentSpec("ranker_train", "ranker_train", "positive" if formula_kind == "positive" else "negative", "add-delete", "strict_then_escalate", formula_kind=formula_kind)
                    target_text, _psi_text, _meta = formula_texts(spec, formula_case)
                    formula = HMLParser.parse(target_text)
                    for v_meta in v_metas:
                        model = get_quotient_model(v_meta)
                        v_actions = set(v_meta["v_actions"])
                        already_true, _checker = verify_formula(model, formula)
                        if already_true:
                            continue
                        candidates = ranker_training_candidates(model, formula, config, args.ranker_train_candidate_limit, v_actions)
                        if not candidates:
                            continue
                        targets = tuple(candidate_training_target(candidate, model, formula, v_actions) for candidate in candidates)
                        target_min = min(targets) if target_min is None else min(target_min, min(targets))
                        target_max = max(targets) if target_max is None else max(target_max, max(targets))
                        batches.append(GnnTrainingBatch(model, frozenset(v_actions), tuple(candidates), targets))
                        sample_count += len(candidates)
                        if sample_count >= args.ranker_train_samples:
                            print_progress(
                                "stage2-ranker-data",
                                model_index + 1,
                                model_count,
                                current=model_meta.get("model_id", ""),
                                samples=sample_count,
                                status="sample_limit",
                            )
                            return batches, sample_count, target_min, target_max
        finally:
            quotient_cache.clear()
            gc.collect()
        print_progress(
            "stage2-ranker-data",
            model_index + 1,
            model_count,
            current=model_meta.get("model_id", ""),
            samples=sample_count,
        )
    return batches, sample_count, target_min, target_max


def train_ranker_model(args, manifest: dict, prepared_dir: Path) -> None:
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        message = f"Requested CUDA device '{args.device}', but CUDA is not available."
        if args.strict_device:
            raise SystemExit(message)
        print("Warning:", message, "Using CPU for ranker training.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Torch initialized once for neural ranker training: device={device}")

    feature_order = ranker_feature_order(args)
    features = None
    targets = None
    batches = None
    if args.ranker_architecture == "gnn":
        batches, sample_count, target_min, target_max = collect_gnn_ranker_training_batches(prepared_dir, manifest, args)
    else:
        features, targets = collect_ranker_training_examples(prepared_dir, manifest, args)
        sample_count = len(features)
        target_min = min(targets) if targets else None
        target_max = max(targets) if targets else None

    if sample_count == 0:
        print("Warning: no ranker training candidates were collected; falling back to deterministic warm-start weights.")
        model = torch.nn.Linear(len(feature_order), 1).to(device)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
            model.weight[0, feature_order.index("local_score")] = 1.0
            model.weight[0, feature_order.index("symbolic_reason")] = -0.35
        weights = model.weight.detach().view(-1)
        bias = model.bias.detach().view(())
        train_meta = {"samples": 0, "epochs": 0, "final_loss": None, "fallback": True}
    elif args.ranker_architecture == "gnn":
        model = GraphCandidateRankerModule(len(feature_order), args.ranker_hidden_dim, args.ranker_hidden_layers, torch).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ranker_lr)
        final_loss = 0.0
        epoch_report_every = max(1, args.ranker_epochs // 10)
        for epoch in range(args.ranker_epochs):
            total_loss = 0.0
            total_items = 0
            for batch in batches:
                optimizer.zero_grad(set_to_none=True)
                node_features, edge_index, edge_features = graph_tensors_for_model(batch.model, set(batch.v_actions), torch, device)
                candidate_features, candidate_src, candidate_dst = candidate_tensors_for_gnn(
                    list(batch.candidates),
                    batch.model,
                    set(batch.v_actions),
                    torch,
                    device,
                    feature_order=feature_order,
                )
                y = torch.tensor(batch.targets, dtype=torch.float32, device=device).view(-1, 1)
                weights = torch.clamp(1.0 + (3.0 * y), max=5.0)
                pred = model(node_features, edge_index, edge_features, candidate_features, candidate_src, candidate_dst)
                loss = weighted_mse(torch, pred, y, weights)
                loss.backward()
                optimizer.step()
                items = len(batch.targets)
                total_loss += float(loss.detach().cpu().item()) * items
                total_items += items
            final_loss = total_loss / max(1, total_items)
            if (epoch + 1) == args.ranker_epochs or (epoch + 1) % epoch_report_every == 0:
                print_progress(
                    "stage2-ranker-epoch",
                    epoch + 1,
                    args.ranker_epochs,
                    samples=sample_count,
                    loss=f"{final_loss:.6f}",
                )
        weights = None
        bias = None
        train_meta = {
            "samples": sample_count,
            "epochs": args.ranker_epochs,
            "final_loss": final_loss,
            "fallback": False,
            "target_min": target_min,
            "target_max": target_max,
        }
    else:
        x = torch.tensor(features, dtype=torch.float32, device=device)
        y = torch.tensor(targets, dtype=torch.float32, device=device).view(-1, 1)
        if args.ranker_architecture == "mlp":
            model = build_mlp(len(feature_order), args.ranker_hidden_dim, args.ranker_hidden_layers, torch).to(device)
        else:
            model = torch.nn.Linear(len(feature_order), 1).to(device)
            with torch.no_grad():
                model.weight.zero_()
                model.bias.zero_()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ranker_lr)
        final_loss = 0.0
        epoch_report_every = max(1, args.ranker_epochs // 10)
        for epoch in range(args.ranker_epochs):
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            if args.ranker_architecture == "linear":
                loss = torch.nn.functional.mse_loss(pred, y)
            else:
                weights = torch.clamp(1.0 + (3.0 * y), max=5.0)
                loss = weighted_mse(torch, pred, y, weights)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu().item())
            if (epoch + 1) == args.ranker_epochs or (epoch + 1) % epoch_report_every == 0:
                print_progress(
                    "stage2-ranker-epoch",
                    epoch + 1,
                    args.ranker_epochs,
                    samples=len(features),
                    loss=f"{final_loss:.6f}",
                )
        if args.ranker_architecture == "linear":
            weights = model.weight.detach().view(-1)
            bias = model.bias.detach().view(())
        else:
            weights = None
            bias = None
        train_meta = {
            "samples": sample_count,
            "epochs": args.ranker_epochs,
            "final_loss": final_loss,
            "fallback": False,
            "target_min": target_min,
            "target_max": target_max,
        }

    path = Path(args.ranker_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_kind = "linear_candidate_ranker_v3_trained" if args.ranker_architecture == "linear" and getattr(args, "linear_feature_set", "current") == "legacy_v3" else "candidate_ranker_v4_trained"
    payload = {
        "kind": payload_kind,
        "architecture": args.ranker_architecture,
        "feature_order": feature_order,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "training": train_meta,
        "prepared_signature": ranker_training_signature(manifest, args),
        "created_by": "svbr.experiments.add_delete_prepared_run",
    }
    if args.ranker_architecture == "linear" or sample_count == 0:
        payload["architecture"] = "linear"
        payload["weights"] = weights.detach().cpu()
        payload["bias"] = bias.detach().cpu()
    else:
        payload["hidden_dim"] = args.ranker_hidden_dim
        payload["hidden_layers"] = args.ranker_hidden_layers
        payload["model_state"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(payload, path)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    with (results_root / "ranker_training.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(path),
                "kind": payload["kind"],
                "architecture": payload["architecture"],
                "linear_feature_set": getattr(args, "linear_feature_set", "current"),
                "hidden_dim": payload.get("hidden_dim", ""),
                "hidden_layers": payload.get("hidden_layers", ""),
                "feature_order": payload["feature_order"],
                "prepared_signature": payload["prepared_signature"],
                "training": train_meta,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Ranker checkpoint: {path}")
    print(f"Ranker training: samples={train_meta['samples']} epochs={train_meta['epochs']} final_loss={train_meta['final_loss']}")
    features = None
    targets = None
    batches = None
    try:
        del x
    except UnboundLocalError:
        pass
    try:
        del y
    except UnboundLocalError:
        pass
    try:
        del model
    except UnboundLocalError:
        pass
    try:
        del optimizer
    except UnboundLocalError:
        pass
    gc.collect()
    release_accelerator_cache(str(device))


def ensure_ranker_model(args, manifest: dict, prepared_dir: Path) -> None:
    needs_neural_ranker = args.include_neural or getattr(args, "default_ranker", "heuristic") == "neural"
    if not needs_neural_ranker:
        return
    path = Path(args.ranker_model)
    signature = ranker_training_signature(manifest, args)
    if path.exists() and not args.force_ranker_train and checkpoint_is_trained(path, signature):
        return
    train_ranker_model(args, manifest, prepared_dir)


def validate_prepared_manifest(manifest: dict) -> None:
    missing_formula_models = [model.get("model_id", "") for model in manifest.get("models", []) if not model.get("formula_cases")]
    if missing_formula_models:
        raise SystemExit("Prepared manifest has no formula_cases. Re-run stage 1 with the updated add_delete_prepare.py.")

    settings = manifest.get("settings", {})
    try:
        expected_v_sizes = set(parse_int_list(str(settings.get("v_sizes", "0,1,3,5"))))
    except Exception:
        expected_v_sizes = {0, 1, 3, 5}

    for model in manifest.get("models", []):
        model_id = model.get("model_id", "")
        model_actions = set(model.get("actions", []))
        v_by_label = {v.get("v_label", ""): v for v in model.get("v_sets", [])}
        formula_cases = model.get("formula_cases", [])
        for case in formula_cases:
            formula_id = case.get("formula_id", "")
            target_texts = [
                ("positive", case.get("positive_formula", "")),
                ("negative_existential", case.get("negative_existential_target", "")),
                ("negative_universal", case.get("negative_universal_target", "")),
            ]
            for formula_kind, target_text in target_texts:
                if target_text and formula_is_contradiction(HMLParser.parse(target_text)):
                    raise SystemExit(
                        f"{model_id}/{formula_id}/{formula_kind}: target formula is logically unsatisfiable. "
                        "Re-run stage 1 with the updated formula generator."
                    )
        uses_formula_safe_v = any(case.get("v_size_labels") for case in formula_cases)
        if uses_formula_safe_v:
            for case in formula_cases:
                labels_by_kind = case.get("v_size_labels_by_kind", {})
                if not labels_by_kind:
                    raise SystemExit(
                        f"{model_id}/{case.get('formula_id', '')}: prepared V labels are outdated. "
                        "Re-run Stage 1 so positive/negative formulas each get their own formula-safe V sets."
                    )
                for formula_kind, labels_by_size in labels_by_kind.items():
                    missing_v_sizes = {str(size) for size in expected_v_sizes} - set(labels_by_size)
                    if missing_v_sizes:
                        raise SystemExit(
                            f"{model_id}/{case.get('formula_id', '')}/{formula_kind}: missing formula-safe V sizes {sorted(missing_v_sizes)}. "
                            "Re-run stage 1."
                        )
                    formula_action_set = case_formula_kind_actions(case, formula_kind) & model_actions
                    for size_text, label in labels_by_size.items():
                        v_meta = v_by_label.get(label)
                        if v_meta is None:
                            raise SystemExit(f"{model_id}/{case.get('formula_id', '')}/{formula_kind}: missing V set {label}. Re-run stage 1.")
                        overlap = sorted(formula_action_set & set(v_meta.get("v_actions", [])))
                        if overlap:
                            raise SystemExit(
                                f"{model_id}/{case.get('formula_id', '')}/{formula_kind}: formula actions appear in V set {label}: {overlap}."
                            )
                        requested = int(v_meta.get("requested_size", int(size_text)))
                        actual = len(v_meta.get("v_actions", []))
                        expected_actual = min(requested, len(model_actions - formula_action_set))
                        if actual != expected_actual:
                            raise SystemExit(
                                f"{model_id}/{case.get('formula_id', '')}/{formula_kind}: requested |V|={requested} has actual |V|={actual}, "
                                f"expected {expected_actual} after excluding formula actions."
                            )
        else:
            v_size_sets = [v for v in model.get("v_sets", []) if v.get("source") == "v_size"]
            present_v_sizes = {int(v.get("requested_size", -1)) for v in v_size_sets}
            missing_v_sizes = expected_v_sizes - present_v_sizes
            if missing_v_sizes:
                raise SystemExit(f"{model_id}: prepared manifest is missing V sizes {sorted(missing_v_sizes)}. Re-run stage 1.")
        for v_meta in model.get("v_sets", []):
            invalid = sorted(set(v_meta.get("v_actions", [])) - model_actions)
            if invalid:
                raise SystemExit(f"{model_id}: V set {v_meta.get('v_label', '')} contains non-LTS actions {invalid}.")
            if v_meta.get("source") == "v_size" and not uses_formula_safe_v:
                requested = int(v_meta.get("requested_size", len(v_meta.get("v_actions", []))))
                actual = len(v_meta.get("v_actions", []))
                expected_actual = min(requested, len(model_actions))
                if actual != expected_actual:
                    raise SystemExit(
                        f"{model_id}: requested |V|={requested} has actual |V|={actual}, expected {expected_actual}."
                    )

        mixed_cases = [case for case in formula_cases if case.get("source") == "mixed_existing_missing"]
        expected_formula_count = int(settings.get("formulas_per_model", len(formula_cases)) or len(formula_cases))
        if expected_formula_count and len(formula_cases) != expected_formula_count:
            raise SystemExit(f"{model_id}: formula_count={len(formula_cases)} expected={expected_formula_count}. Re-run stage 1.")
        hml_safe_actions = set(model.get("hml_safe_actions", model.get("actions", [])))
        if hml_safe_actions:
            expected_mixed = min(int(settings.get("mixed_formula_count", 0) or 0), len(formula_cases))
            actual_existing = sum(1 for case in formula_cases if case.get("source") == "existing_only")
            expected_existing = len(formula_cases) - expected_mixed
            if len(mixed_cases) != expected_mixed or actual_existing != expected_existing:
                raise SystemExit(
                    f"{model_id}: formula source counts are wrong "
                    f"(mixed={len(mixed_cases)}/{expected_mixed}, existing={actual_existing}/{expected_existing})."
                )
        else:
            generated_only = sum(1 for case in formula_cases if case.get("source") == "generated_missing_only")
            if generated_only != len(formula_cases):
                raise SystemExit(f"{model_id}: expected all formulas to be generated_missing_only.")

        if model_actions and len(mixed_cases) > 1:
            target_in_lts = sum(1 for case in mixed_cases if case.get("target_action_in_lts"))
            target_missing = len(mixed_cases) - target_in_lts
            if target_in_lts == 0 or target_missing == 0:
                raise SystemExit(
                    f"{model_id}: mixed formulas do not diversify first target action "
                    f"(in_lts={target_in_lts}, missing={target_missing}). Re-run stage 1."
                )



def row_for_result(
    model_meta: dict,
    formula_case: dict,
    formula_meta: dict,
    spec: ExperimentSpec,
    v_meta: dict,
    quotient_relative: str,
    target_text: str,
    psi_text: str,
    result,
    out_dir: Path,
    case_id: str,
    model: RepairLTS,
    formula,
    initial_satisfied_on_surface: bool,
    ranker_architecture: str = "",
    gnn_graph_mode: str = "",
    search_strategy: str = "",
    write_repaired_aut: bool = False,
) -> dict:
    target_state = int(model_meta.get("effective_target_state", model_meta.get("target_state", model_meta.get("initial", 0))))
    script_path = out_dir / "edit_scripts" / f"{case_id}.json"
    repaired_path = ""
    result_payload = result.edit_script_json()
    dump_json(
        script_path,
        {
            "case_id": case_id,
            "repair_surface": "quotient_lts",
            "model_path": model_meta["model_path"],
            "model_pickle": model_meta.get("model_pickle", ""),
            "original_initial": model_meta.get("initial", ""),
            "original_target_state": target_state,
            "original_states": model_meta.get("states", ""),
            "original_transitions": model_meta.get("transitions", ""),
            "quotient_pickle": quotient_relative,
            "quotient_initial": model.initial,
            "quotient_states": model.state_count,
            "quotient_transitions": model.transition_count,
            "quotient_state_semantics": "state ids are strong-V-bisimulation block ids",
            "task_type": spec.task,
            "repair_mode": spec.repair_mode,
            "sf_setting": spec.sf_setting,
            "ranker": spec.ranker,
            "V_selection": spec.v_selection,
            "V_requested_size": v_meta.get("requested_size", len(v_meta["v_actions"])),
            "V_size": len(v_meta["v_actions"]),
            "V_size_note": f"required |V|={v_meta.get('requested_size', len(v_meta['v_actions']))}, actual |V|={len(v_meta['v_actions'])} after excluding formula actions",
            "formula_id": formula_case["formula_id"],
            "formula_kind": spec.formula_kind,
            "V_actions": v_meta["v_actions"],
            "target_formula": target_text,
            "psi": psi_text,
            "lts_prime_to_lts_double_prime_operations": {
                "surface": "quotient_lts_blocks",
                "adds": result_payload["adds"],
                "dels": result_payload["dels"],
            },
            "result": result_payload,
        },
    )
    if result.success and write_repaired_aut:
        repaired_path = str(out_dir / "repaired_quotient_aut" / f"{case_id}.aut")
        result.final_model.write_aut(repaired_path)
    log_path = out_dir / "logs" / f"{case_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.message + "\n", encoding="utf-8")

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
        "ranker_architecture": ranker_architecture,
        "gnn_graph_mode": gnn_graph_mode,
        "search_strategy": search_strategy,
        "V_requested_size": v_meta.get("requested_size", len(v_meta["v_actions"])),
        "V_size": len(v_meta["v_actions"]),
        "V_size_note": f"required |V|={v_meta.get('requested_size', len(v_meta['v_actions']))}, actual |V|={len(v_meta['v_actions'])} after excluding formula actions",
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
        "formula_initial_satisfied": yes_no(initial_satisfied_on_surface),
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
        "repaired_path": repaired_path,
        "edit_script_path": str(script_path),
    }


def summarize_rows(rows: list[dict], errors: list[dict]) -> dict:
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


def f(row: dict, field: str) -> float:
    try:
        return float(row.get(field, 0) or 0)
    except ValueError:
        return 0.0


class CsvSink:
    def __init__(self, path: Path, fieldnames: list[str], flush_every: int = 100):
        self.path = path
        self.fieldnames = fieldnames
        self.flush_every = max(1, flush_every)
        self.rows_written = 0
        self.handle = None
        self.writer = None

    def __enter__(self) -> "CsvSink":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames, extrasaction="ignore")
        self.writer.writeheader()
        return self

    def writerow(self, row: dict) -> None:
        if self.writer is None:
            raise RuntimeError("CsvSink is not open")
        self.writer.writerow(row)
        self.rows_written += 1
        if self.handle is not None and self.rows_written % self.flush_every == 0:
            self.handle.flush()

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()


class RunningSummary:
    def __init__(self) -> None:
        self.runs = 0
        self.errors = 0
        self.successes = 0
        self.verified = 0
        self.actual_cost_sum = 0.0
        self.non_v_add_sum = 0.0
        self.non_v_del_sum = 0.0
        self.quotient_drift_sum = 0.0
        self.verifier_calls_sum = 0.0
        self.elapsed_ms_sum = 0.0

    def add_row(self, row: dict) -> None:
        self.runs += 1
        if row.get("success") == "YES":
            self.successes += 1
        if row.get("verified") == "YES":
            self.verified += 1
        self.actual_cost_sum += f(row, "actual_cost")
        self.non_v_add_sum += f(row, "nonV_add_edges")
        self.non_v_del_sum += f(row, "nonV_del_edges")
        self.quotient_drift_sum += f(row, "quotient_drift")
        self.verifier_calls_sum += f(row, "verifier_calls")
        self.elapsed_ms_sum += f(row, "elapsed_ms")

    def add_error(self) -> None:
        self.errors += 1

    def to_dict(self) -> dict:
        runs = self.runs
        return {
            "runs": runs,
            "errors": self.errors,
            "successes": self.successes,
            "verified": self.verified,
            "success_rate": self.successes / runs if runs else 0.0,
            "verified_rate": self.verified / runs if runs else 0.0,
            "avg_actual_cost": self.actual_cost_sum / runs if runs else 0.0,
            "avg_nonV_edits": (self.non_v_add_sum + self.non_v_del_sum) / runs if runs else 0.0,
            "avg_quotient_drift": self.quotient_drift_sum / runs if runs else 0.0,
            "avg_verifier_calls": self.verifier_calls_sum / runs if runs else 0.0,
            "avg_elapsed_ms": self.elapsed_ms_sum / runs if runs else 0.0,
        }


def group_key(row: dict, stratified: bool = False) -> tuple[str, ...]:
    key = [
        row["task_type"],
        row["repair_mode"],
        row["sf_setting"],
        row["ranker"],
        row.get("ranker_architecture", ""),
        str(row.get("V_requested_size", "")),
        str(row.get("V_size", "")),
        row["target_action_in_V"],
    ]
    if stratified:
        key.extend([row.get("formula_difficulty", ""), row.get("formula_source", ""), row.get("formula_kind", "")])
    return tuple(key)


class GroupSummary:
    def __init__(self, stratified: bool = False) -> None:
        self.stratified = stratified
        self.groups: dict[tuple[str, ...], dict[str, float | int]] = {}

    def add_row(self, row: dict) -> None:
        key = group_key(row, self.stratified)
        group = self.groups.setdefault(
            key,
            {
                "N": 0,
                "success": 0,
                "verified": 0,
                "actual_cost": 0.0,
                "non_v_edits": 0.0,
                "quotient_drift": 0.0,
                "verifier_calls": 0.0,
                "elapsed_ms": 0.0,
            },
        )
        group["N"] += 1
        if row.get("success") == "YES":
            group["success"] += 1
        if row.get("verified") == "YES":
            group["verified"] += 1
        group["actual_cost"] += f(row, "actual_cost")
        group["non_v_edits"] += f(row, "nonV_add_edges") + f(row, "nonV_del_edges")
        group["quotient_drift"] += f(row, "quotient_drift")
        group["verifier_calls"] += f(row, "verifier_calls")
        group["elapsed_ms"] += f(row, "elapsed_ms")

    def rows(self) -> list[dict]:
        rows = []
        for key, group in sorted(self.groups.items()):
            n = int(group["N"])
            row = {
                "group": " / ".join(key),
                "task_type": key[0],
                "repair_mode": key[1],
                "sf_setting": key[2],
                "ranker": key[3],
                "ranker_architecture": key[4],
                "V_requested_size": key[5],
                "V_size": key[6],
                "target_action_in_V": key[7],
                "N": n,
                "success": int(group["success"]),
                "verified": int(group["verified"]),
                "success_rate": f"{int(group['success']) / n:.6f}",
                "verified_rate": f"{int(group['verified']) / n:.6f}",
                "avg_cost": f"{float(group['actual_cost']) / n:.6f}",
                "avg_nonV_edits": f"{float(group['non_v_edits']) / n:.6f}",
                "avg_quotient_drift": f"{float(group['quotient_drift']) / n:.6f}",
                "avg_verifier_calls": f"{float(group['verifier_calls']) / n:.6f}",
                "avg_elapsed_ms": f"{float(group['elapsed_ms']) / n:.6f}",
            }
            if self.stratified:
                row.update({"formula_difficulty": key[8], "formula_source": key[9], "formula_kind": key[10]})
            rows.append(row)
        return rows


def write_group_accumulator(accumulator: GroupSummary | None, out: Path) -> None:
    rows = accumulator.rows() if accumulator is not None else []
    if rows:
        write_csv(out, rows, list(rows[0].keys()))
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")


def release_accelerator_cache(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def trim_process_memory() -> None:
    if os.name != "posix":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def release_runtime_memory(device: str, trim_process: bool = True) -> None:
    gc.collect()
    release_accelerator_cache(device)
    if trim_process:
        trim_process_memory()


def current_rss_mb() -> str:
    try:
        status = Path("/proc/self/status")
        if status.exists():
            for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return f"{int(parts[1]) / 1024:.1f}"
    except Exception:
        pass
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > 10_000_000:
            return f"{rss / (1024 * 1024):.1f}"
        return f"{rss / 1024:.1f}"
    except Exception:
        return ""


def remove_tree(path: Path) -> None:
    def make_writable_and_retry(func, name, _exc_info):
        os.chmod(name, stat.S_IWRITE | stat.S_IREAD)
        func(name)

    shutil.rmtree(path, onerror=make_writable_and_retry)


def write_group_summary(rows: list[dict], out: Path, stratified: bool = False) -> None:
    groups = {}
    for row in rows:
        key = [
            row["task_type"],
            row["repair_mode"],
            row["sf_setting"],
            row["ranker"],
            str(row.get("V_requested_size", "")),
            str(row.get("V_size", "")),
            row["target_action_in_V"],
        ]
        if stratified:
            key.extend([row.get("formula_difficulty", ""), row.get("formula_source", ""), row.get("formula_kind", "")])
        groups.setdefault(tuple(key), []).append(row)
    summary_rows = []
    for key, group in sorted(groups.items()):
        success = sum(1 for row in group if row["success"] == "YES")
        verified = sum(1 for row in group if row["verified"] == "YES")
        non_v = [f(row, "nonV_add_edges") + f(row, "nonV_del_edges") for row in group]
        row = {
            "group": " / ".join(key),
            "task_type": key[0],
            "repair_mode": key[1],
            "sf_setting": key[2],
            "ranker": key[3],
            "V_requested_size": key[4],
            "V_size": key[5],
            "target_action_in_V": key[6],
            "N": len(group),
            "success": success,
            "verified": verified,
            "success_rate": f"{success / len(group):.6f}",
            "verified_rate": f"{verified / len(group):.6f}",
            "avg_cost": f"{sum(f(item, 'actual_cost') for item in group) / len(group):.6f}",
            "avg_nonV_edits": f"{sum(non_v) / len(non_v):.6f}",
            "avg_quotient_drift": f"{sum(f(item, 'quotient_drift') for item in group) / len(group):.6f}",
            "avg_verifier_calls": f"{sum(f(item, 'verifier_calls') for item in group) / len(group):.6f}",
            "avg_elapsed_ms": f"{sum(f(item, 'elapsed_ms') for item in group) / len(group):.6f}",
        }
        if stratified:
            row.update({"formula_difficulty": key[7], "formula_source": key[8], "formula_kind": key[9]})
        summary_rows.append(row)
    if summary_rows:
        write_csv(out, summary_rows, list(summary_rows[0].keys()))
    else:
        out.write_text("", encoding="utf-8")


def selected_formula_cases(model_meta: dict, args) -> list[dict]:
    cases = list(model_meta.get("formula_cases") or [])
    if args.formula_id:
        wanted = set(args.formula_id)
        cases = [case for case in cases if case.get("formula_id") in wanted]
    if args.formula_limit > 0:
        cases = cases[: args.formula_limit]
    return cases


def run_prepared_case(
    model_meta: dict,
    model: RepairLTS,
    formula_case: dict,
    formula_meta: dict,
    spec: ExperimentSpec,
    v_meta: dict,
    quotient_relative: str,
    target_text: str,
    psi_text: str,
    formula,
    initial_satisfied_on_surface: bool,
    config: RepairConfig,
    out_dir: Path,
    case_id: str,
    ranker,
    write_repaired_aut: bool = False,
    neural_cegis_enabled: bool = False,
    neural_cegis_attempts: int = 0,
    neural_cegis_epochs: int = 4,
    neural_cegis_lr: float = 0.001,
    neural_cegis_candidate_limit: int = 256,
    neural_cegis_adopt_oracle: bool = False,
    oracle_config: RepairConfig | None = None,
    oracle_ranker=None,
    neural_rescue_enabled: bool = False,
    rescue_config: RepairConfig | None = None,
    rescue_ranker=None,
) -> dict:
    case_started = time.perf_counter()
    result = run_repair(
        model,
        formula,
        set(v_meta["v_actions"]),
        config,
        original_quotient=None,
        ranker=ranker,
        case_id=case_id,
    )
    if (
        neural_cegis_enabled
        and spec.ranker == "neural"
        and getattr(config, "ranker_architecture", "") in {"mlp", "gnn"}
        and not result.verified
    ):
        v_actions = set(v_meta["v_actions"])
        training_notes: list[str] = []
        accumulated_calls = result.verifier_calls
        accumulated_iters = result.cex_iters
        attempts_done = 0
        max_attempts = max(0, neural_cegis_attempts)
        last_retry_budget: tuple | None = None
        repeated_budget_failures = 0
        while not result.verified and (max_attempts == 0 or attempts_done < max_attempts):
            attempts_done += 1
            retry_config = neural_cegis_retry_config(config, model, formula, attempts_done)
            retry_budget_key = (
                retry_config.search_strategy,
                retry_config.max_iters,
                retry_config.beam_width,
                retry_config.candidate_limit,
                retry_config.candidate_state_limit,
                retry_config.state_scan_limit,
                retry_config.minimal_layer_width,
                retry_config.minimal_seen_limit,
            )
            if retry_budget_key == last_retry_budget:
                repeated_budget_failures += 1
            else:
                repeated_budget_failures = 0
                last_retry_budget = retry_budget_key
            if max_attempts == 0 and repeated_budget_failures > 2:
                training_notes.append(
                    "stopped_at_memory_safe_ceiling_after_repeated_failed_retries"
                )
                break
            trained, train_meta = fine_tune_neural_ranker_from_hml_failure(
                ranker,
                model,
                formula,
                retry_config,
                v_actions,
                ranker_feature_order_from_config(config),
                neural_cegis_epochs,
                neural_cegis_lr,
                neural_cegis_candidate_limit,
            )
            training_notes.append(f"attempt={attempts_done}, trained={trained}, meta={train_meta}")
            if not trained:
                break
            retried_result = run_repair(
                model,
                formula,
                v_actions,
                retry_config,
                original_quotient=None,
                ranker=ranker,
                case_id=f"{case_id}:cegis_retry{attempts_done}",
            )
            accumulated_calls += retried_result.verifier_calls
            accumulated_iters += retried_result.cex_iters
            retried_result.verifier_calls = accumulated_calls
            retried_result.cex_iters = accumulated_iters
            retried_result.elapsed_ms = (time.perf_counter() - case_started) * 1000.0
            retried_result.message = (
                f"{retried_result.message} Neural CEGIS retry {attempts_done}; "
                f"retry_budget=({retry_config.search_strategy}, max_iters={retry_config.max_iters}, "
                f"beam={retry_config.beam_width}, candidate_limit={retry_config.candidate_limit}, "
                f"candidate_state_limit={retry_config.candidate_state_limit}, state_scan_limit={retry_config.state_scan_limit}); "
                f"HML-failure online training: {'; '.join(training_notes)}"
            )
            result = retried_result
        if not result.verified and neural_cegis_adopt_oracle and oracle_config is not None and oracle_ranker is not None:
            oracle_result = run_repair(
                model,
                formula,
                v_actions,
                oracle_config,
                original_quotient=None,
                ranker=oracle_ranker,
                case_id=f"{case_id}:cegis_oracle_optional",
            )
            if oracle_result.verified:
                oracle_result.verifier_calls += result.verifier_calls
                oracle_result.cex_iters += result.cex_iters
                oracle_result.elapsed_ms = (time.perf_counter() - case_started) * 1000.0
                oracle_result.stage = f"optional_cegis_oracle_{oracle_result.stage}"
                oracle_result.message = (
                    f"{oracle_result.message} Optional oracle adoption after pure neural CEGIS failed. "
                    f"This row should be treated as oracle-assisted, not pure MLP/GNN. "
                    f"Online training: {'; '.join(training_notes) if training_notes else 'not_performed'}"
                )
                result = oracle_result
        if not result.verified and neural_rescue_enabled and rescue_config is not None and rescue_ranker is not None:
            rescue_result = run_repair(
                model,
                formula,
                v_actions,
                rescue_config,
                original_quotient=None,
                ranker=rescue_ranker,
                case_id=f"{case_id}:linear_rescue",
            )
            rescue_result.verifier_calls += result.verifier_calls
            rescue_result.cex_iters += result.cex_iters
            rescue_result.elapsed_ms = (time.perf_counter() - case_started) * 1000.0
            if rescue_result.verified:
                rescue_result.stage = f"linear_rescue_{rescue_result.stage}"
                rescue_result.message = (
                    f"{rescue_result.message} Linear rescue fallback after MLP/GNN primary failed. "
                    f"Primary message: {result.message}"
                )
                result = rescue_result
            else:
                result.message = (
                    f"{result.message} Linear rescue fallback also failed: {rescue_result.message}"
                )
    return row_for_result(
        model_meta,
        formula_case,
        formula_meta,
        spec,
        v_meta,
        quotient_relative,
        target_text,
        psi_text,
        result,
        out_dir,
        case_id,
        model,
        formula,
        initial_satisfied_on_surface,
        ranker_architecture=getattr(config, "ranker_architecture", ""),
        gnn_graph_mode=getattr(config, "gnn_graph_mode", ""),
        search_strategy=config.search_strategy,
        write_repaired_aut=write_repaired_aut,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: run prepared add/delete repair experiments with one neural-ranker GPU context")
    parser.add_argument("--prepared-dir", default="results/add_delete_prepared")
    parser.add_argument("--results-root", default="results/add_delete_run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strict-device", action="store_true")
    parser.add_argument("--ranker-model", default="models/add_delete_ranker.pt")
    parser.add_argument("--v-sizes", default="0,1,3,5")
    parser.add_argument("--force-ranker-train", action="store_true")
    parser.add_argument("--ranker-train-samples", type=int, default=8000)
    parser.add_argument("--ranker-train-model-limit", type=int, default=0, help="0 means use all prepared models")
    parser.add_argument("--ranker-train-formula-limit", type=int, default=30)
    parser.add_argument("--ranker-train-candidate-limit", type=int, default=64)
    parser.add_argument("--ranker-epochs", type=int, default=20)
    parser.add_argument("--ranker-lr", type=float, default=0.03)
    parser.add_argument("--ranker-architecture", choices=["linear", "mlp", "gnn"], default="mlp")
    parser.add_argument("--gnn-graph-mode", choices=["dynamic", "static"], default="dynamic", help="dynamic scores the edited overlay graph; static caches only the base quotient graph for speed ablations")
    parser.add_argument("--linear-feature-set", choices=["current", "legacy_v3"], default="current", help="Use legacy_v3 to reproduce the older 8-feature GPU linear ranker")
    parser.add_argument("--ranker-hidden-dim", type=int, default=64)
    parser.add_argument("--ranker-hidden-layers", type=int, default=2)
    parser.add_argument("--neural-prefilter-multiplier", type=int, default=4, help="For MLP/GNN, score only the cheap linear-prior top candidate_limit*N candidates")
    parser.add_argument("--neural-prefilter-limit", type=int, default=512, help="For MLP/GNN, minimum cheap linear-prior top-K kept before neural reranking")
    parser.add_argument("--neural-linear-blend", type=float, default=0.35, help="For MLP/GNN, blend this fraction of the lightweight linear prior into the neural score")
    parser.add_argument("--neural-verify-frontier-only", action="store_true", help="For MLP/GNN beam search, verify only the retained neural frontier instead of every generated candidate")
    parser.add_argument("--no-neural-verify-frontier-only", dest="neural_verify_frontier_only", action="store_false")
    parser.set_defaults(neural_verify_frontier_only=True)
    parser.add_argument("--neural-verify-top-k", type=int, default=0, help="For MLP/GNN delayed verification, verify top K retained frontier nodes per depth; 0 means beam width")
    parser.add_argument("--neural-cegis-retrain", action="store_true", help="For MLP/GNN runs, use verifier/oracle counterexamples to fine-tune after failed repairs")
    parser.add_argument("--no-neural-cegis-retrain", dest="neural_cegis_retrain", action="store_false")
    parser.set_defaults(neural_cegis_retrain=False)
    parser.add_argument("--neural-cegis-attempts", type=int, default=0, help="Online fine-tune/retry attempts after one neural repair failure; 0 expands until success or the memory-safe retry ceiling is repeatedly hit")
    parser.add_argument("--neural-cegis-epochs", type=int, default=4)
    parser.add_argument("--neural-cegis-lr", type=float, default=0.001)
    parser.add_argument("--neural-cegis-candidate-limit", type=int, default=256)
    parser.add_argument("--neural-cegis-oracle-model", default="", help="Optional stable neural-ranker checkpoint used as CEGIS oracle; empty uses heuristic oracle")
    parser.add_argument("--neural-cegis-adopt-oracle", action="store_true", help="Optional debug fallback: adopt the verifier-confirmed oracle repair if online neural retry still fails")
    parser.add_argument("--no-neural-cegis-adopt-oracle", dest="neural_cegis_adopt_oracle", action="store_false")
    parser.set_defaults(neural_cegis_adopt_oracle=False)
    parser.add_argument("--neural-rescue-linear", action="store_true", help="For MLP/GNN runs, use the lightweight contextual linear ranker as a verified rescue fallback")
    parser.add_argument("--no-neural-rescue-linear", dest="neural_rescue_linear", action="store_false")
    parser.set_defaults(neural_rescue_linear=False)
    parser.add_argument("--neural-rescue-linear-model", default="", help="Checkpoint for the lightweight contextual linear rescue ranker")
    parser.add_argument("--default-ranker", choices=["heuristic", "neural", "random"], default="heuristic", help="Ranker used by the main non-ranker-comparison experiment groups")
    parser.add_argument("--include-heuristic-comparison", action="store_true")
    parser.add_argument("--no-include-heuristic-comparison", dest="include_heuristic_comparison", action="store_false")
    parser.set_defaults(include_heuristic_comparison=True)
    parser.add_argument("--include-random-comparison", action="store_true", help="Add random-ordering ranker specs for ranker-add-delete comparisons")
    parser.add_argument("--no-include-random-comparison", dest="include_random_comparison", action="store_false")
    parser.set_defaults(include_random_comparison=False)
    parser.add_argument("--experiment-profile", choices=["full", "add-delete-only", "ranker-add-delete", "repair-mode-ablation"], default="full", help="Use add-delete-only, ranker-add-delete, or repair-mode-ablation to reduce the suite")
    parser.add_argument("--v-selection", choices=["formula_safe", "unsafe"], default="formula_safe", help="formula_safe excludes target-formula actions from V; unsafe uses ordinary V-size sets for ablation")
    parser.add_argument("--repair-mode-filter", choices=["all", "add-only", "delete-only", "add-delete"], default="all", help="Keep only one repair mode from profiles that include repair-mode ablations")
    parser.add_argument("--include-neural", action="store_true")
    parser.add_argument("--no-include-neural", dest="include_neural", action="store_false")
    parser.set_defaults(include_neural=True)
    parser.add_argument("--formula-limit", type=int, default=0, help="Debug cap per model; 0 means use all prepared formula cases")
    parser.add_argument("--formula-id", action="append", default=[], help="Run only one prepared formula id; may be repeated")
    parser.add_argument("--target-state", type=int, default=-1, help="Override manifest target_state for Stage 2 repair; -1 uses prepared target_state or AUT initial")
    parser.add_argument("--skip-initially-satisfied", action="store_true")
    parser.add_argument("--no-skip-initially-satisfied", dest="skip_initially_satisfied", action="store_false")
    parser.set_defaults(skip_initially_satisfied=True)
    parser.add_argument("--skip-unsatisfiable-targets", action="store_true", help="Skip logically unsatisfiable HML targets before dynamic search")
    parser.add_argument("--no-skip-unsatisfiable-targets", dest="skip_unsatisfiable_targets", action="store_false")
    parser.set_defaults(skip_unsatisfiable_targets=True)
    parser.add_argument("--clean-outputs", action="store_true")
    parser.add_argument("--no-clean-outputs", dest="clean_outputs", action="store_false")
    parser.set_defaults(clean_outputs=True)
    parser.add_argument("--write-repaired-aut", action="store_true", help="Write full repaired AUT files during Stage 2; off by default to keep memory and I/O low")
    parser.add_argument("--progress-every", type=int, default=1, help="Print Stage 2 progress every N prepared models per experiment")
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, default=16, help="Max candidates kept per expansion; 0 uses an automatic bounded cap")
    parser.add_argument("--candidate-state-limit", type=int, default=64, help="Max candidate states considered; 0 uses the current bounded candidate-state policy")
    parser.add_argument("--state-scan-limit", type=int, default=1000, help="Max states scanned for formula-satisfying destinations; 0 scans all states")
    parser.add_argument("--search-strategy", choices=["beam", "neural_guided_minimal"], default="beam")
    parser.add_argument("--minimal-layer-width", type=int, default=2048, help="Max nodes kept per edit-depth layer for neural_guided_minimal; keep positive to bound memory")
    parser.add_argument("--minimal-seen-limit", type=int, default=500000, help="Max script keys remembered by neural_guided_minimal; keep positive to bound memory")
    parser.add_argument("--max-case-seconds", type=float, default=0.0, help="Per Stage-2 case wall-clock cap in seconds; 0 disables")
    parser.add_argument("--dynamic-repair-budget", action="store_true", help="Retry failed Stage 2 repairs with expanded search/candidate budgets")
    parser.add_argument("--no-dynamic-repair-budget", dest="dynamic_repair_budget", action="store_false")
    parser.set_defaults(dynamic_repair_budget=False)
    parser.add_argument("--dynamic-budget-rounds", type=int, default=0, help="0 keeps doubling failed cases until the configured safety ceilings are saturated")
    parser.add_argument("--dynamic-max-iters", type=int, default=512, help="Max iters for dynamic Stage 2 retries; 0 chooses a model/formula-based cap")
    parser.add_argument("--dynamic-max-beam-width", type=int, default=256)
    parser.add_argument("--dynamic-max-candidate-limit", type=int, default=0, help="0 chooses an automatic bounded widest-retry candidate cap")
    parser.add_argument("--dynamic-max-candidate-state-limit", type=int, default=0, help="0 chooses an automatic bounded widest-retry state cap")
    parser.add_argument("--dynamic-max-state-scan-limit", type=int, default=0, help="0 means scan all quotient states on the widest retry")
    parser.add_argument("--dynamic-max-minimal-layer-width", type=int, default=32768)
    parser.add_argument("--dynamic-max-minimal-seen-limit", type=int, default=500000)
    parser.add_argument("--dynamic-final-search-strategy", choices=["", "beam", "neural_guided_minimal"], default="")
    parser.add_argument("--max-quotient-drift", type=int, default=1000000000)
    parser.add_argument("--drift-mode", choices=["estimate", "auto", "exact"], default="estimate")
    parser.add_argument("--exact-drift-max-transitions", type=int, default=200000)
    parser.add_argument("--stage2-max-states", type=int, default=100000)
    parser.add_argument("--stage2-max-transitions", type=int, default=200000)
    parser.add_argument("--cache-quotient-models", action="store_true", help="Reuse quotient RepairLTS objects within one prepared model; off by default to lower RSS")
    parser.add_argument("--trim-memory-every-case", action="store_true", help="Run gc, CUDA empty_cache, and malloc_trim after every Stage 2 formula case")
    parser.add_argument("--no-trim-memory-every-case", dest="trim_memory_every_case", action="store_false")
    parser.set_defaults(trim_memory_every_case=True)
    parser.add_argument("--case-progress-every", type=int, default=0, help="Print formula-case progress every N cases within each model; 0 disables")
    parser.add_argument("--partition-drift", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prepared_dir = Path(args.prepared_dir)
    results_root = Path(args.results_root)
    with (prepared_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    validate_prepared_manifest(manifest)

    ensure_ranker_model(args, manifest, prepared_dir)
    v_sizes = parse_int_list(args.v_sizes)
    specs = build_specs(
        args.include_neural,
        v_sizes,
        default_ranker=args.default_ranker,
        include_heuristic_comparison=args.include_heuristic_comparison,
        experiment_profile=args.experiment_profile,
        include_random_comparison=args.include_random_comparison,
        v_selection=args.v_selection,
        repair_mode_filter=args.repair_mode_filter,
    )
    neural_ranker = None
    heuristic_ranker = None
    cegis_oracle_ranker = None
    cegis_oracle_ranker_key = None
    rescue_linear_ranker = None
    rescue_linear_ranker_key = None
    suite_summaries: dict[str, GroupSummary] = {}
    suite_summaries_by_formula: dict[str, GroupSummary] = {}
    total_repair_units = len(specs) * len(manifest["models"])
    completed_repair_units = 0

    for spec_index, spec in enumerate(specs, start=1):
        out_dir = results_root / spec.out_name
        if args.clean_outputs and out_dir.exists():
            resolved = out_dir.resolve()
            root_resolved = results_root.resolve()
            try:
                resolved.relative_to(root_resolved)
            except ValueError:
                raise SystemExit(f"Refusing to remove outside results root: {resolved}")
            try:
                remove_tree(out_dir)
            except OSError as exc:
                print(f"Warning: could not fully clean {out_dir}: {exc}. Current CSV/log/script files will be overwritten in place.")
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = RunningSummary()
        skipped_count = 0
        config = make_config(spec, args)
        oracle_config = None
        oracle_ranker = None
        rescue_config = None
        rescue_ranker = None
        if spec.ranker == "neural":
            if neural_ranker is None:
                neural_ranker = make_ranker(config)
                print(f"Neural ranker loaded once: model={args.ranker_model} device={args.device}")
            ranker = neural_ranker
            if args.neural_rescue_linear and args.ranker_architecture in {"mlp", "gnn"}:
                rescue_config = make_linear_rescue_config(config, args)
                if rescue_config is not None:
                    rescue_key = (rescue_config.model_path, rescue_config.ranker_device, rescue_config.strict_ranker_device)
                    if rescue_linear_ranker is None or rescue_linear_ranker_key != rescue_key:
                        rescue_linear_ranker = make_ranker(rescue_config)
                        rescue_linear_ranker_key = rescue_key
                        print(f"Neural rescue linear ranker loaded: model={rescue_config.model_path}")
                    rescue_ranker = rescue_linear_ranker
            if args.neural_cegis_retrain and args.neural_cegis_adopt_oracle and args.ranker_architecture in {"mlp", "gnn"}:
                oracle_config = make_cegis_oracle_config(config, args)
                oracle_key = (oracle_config.ranker, oracle_config.model_path, oracle_config.ranker_device, oracle_config.gnn_graph_mode)
                if cegis_oracle_ranker is None or cegis_oracle_ranker_key != oracle_key:
                    cegis_oracle_ranker = make_ranker(oracle_config)
                    cegis_oracle_ranker_key = oracle_key
                    print(
                        "Neural CEGIS oracle loaded: "
                        f"ranker={oracle_config.ranker} model={oracle_config.model_path or 'heuristic'}"
                    )
                oracle_ranker = cegis_oracle_ranker
        else:
            if heuristic_ranker is None:
                heuristic_ranker = make_ranker(config)
            ranker = heuristic_ranker

        with (
            CsvSink(out_dir / "runs.csv", RUN_FIELDS) as runs_sink,
            CsvSink(out_dir / "errors.csv", ERROR_FIELDS) as errors_sink,
            CsvSink(out_dir / "skipped_initially_satisfied.csv", SKIP_FIELDS) as skipped_sink,
        ):
            for model_index, model_meta in enumerate(manifest["models"]):
                processed_models = model_index + 1
                target_state = target_state_for_model(model_meta, args)
                model_meta["effective_target_state"] = target_state
                quotient_cache: dict[str, RepairLTS] = {}

                def get_quotient_model(v_meta: dict) -> tuple[RepairLTS, str]:
                    relative = v_meta["quotient_pickle"]
                    if args.cache_quotient_models and v_meta.get("source") in {"v_size", "formula_safe_v_size"}:
                        if relative not in quotient_cache:
                            quotient = read_pickle(prepared_dir, relative)
                            quotient_cache[relative] = quotient_as_repair_lts(quotient, target_state)
                            del quotient
                        return quotient_cache[relative], relative
                    quotient = read_pickle(prepared_dir, relative)
                    model = quotient_as_repair_lts(quotient, target_state)
                    del quotient
                    return model, relative

                try:
                    formula_cases_for_model = selected_formula_cases(model_meta, args)
                    for case_index, formula_case in enumerate(formula_cases_for_model, start=1):
                        case_id = f"{model_meta['model_id']}_{formula_case['formula_id']}_{spec.out_name}"
                        v_meta = None
                        target_text = None
                        psi_text = None
                        formula_meta = None
                        repair_model = None
                        quotient_relative = None
                        formula = None
                        row = None
                        _checker = None
                        case_status = "start"
                        case_start = time.perf_counter()
                        try:
                            v_meta = find_v_meta(model_meta, spec, formula_case)
                            target_text, psi_text, formula_meta = formula_texts(spec, formula_case)
                            q_states = int(v_meta.get("quotient_states", 0) or 0)
                            q_transitions = int(v_meta.get("quotient_transitions", 0) or 0)
                            if args.stage2_max_states > 0 and q_states > args.stage2_max_states:
                                case_status = "skipped_quotient_states"
                                skipped_sink.writerow(
                                    {
                                        "case_id": case_id,
                                        "model_path": model_meta.get("model_path", ""),
                                        "formula_id": formula_case.get("formula_id", ""),
                                        "formula_kind": spec.formula_kind,
                                        "out_name": spec.out_name,
                                        "reason": f"quotient_states>{args.stage2_max_states}",
                                    }
                                )
                                skipped_count += 1
                                continue
                            if args.stage2_max_transitions > 0 and q_transitions > args.stage2_max_transitions:
                                case_status = "skipped_quotient_transitions"
                                skipped_sink.writerow(
                                    {
                                        "case_id": case_id,
                                        "model_path": model_meta.get("model_path", ""),
                                        "formula_id": formula_case.get("formula_id", ""),
                                        "formula_kind": spec.formula_kind,
                                        "out_name": spec.out_name,
                                        "reason": f"quotient_transitions>{args.stage2_max_transitions}",
                                    }
                                )
                                skipped_count += 1
                                continue
                            repair_model, quotient_relative = get_quotient_model(v_meta)
                            formula = HMLParser.parse(target_text)
                            if args.skip_unsatisfiable_targets and formula_is_contradiction(formula):
                                case_status = "skipped_unsatisfiable_target"
                                skipped_sink.writerow(
                                    {
                                        "case_id": case_id,
                                        "model_path": model_meta.get("model_path", ""),
                                        "formula_id": formula_case.get("formula_id", ""),
                                        "formula_kind": spec.formula_kind,
                                        "out_name": spec.out_name,
                                        "reason": "logically_unsatisfiable_target",
                                    }
                                )
                                skipped_count += 1
                                continue
                            initial_satisfied_on_surface, _checker = verify_formula(repair_model, formula)
                            if args.skip_initially_satisfied and initial_satisfied_on_surface:
                                case_status = "skipped_initially_satisfied"
                                skipped_sink.writerow(
                                    {
                                        "case_id": case_id,
                                        "model_path": model_meta.get("model_path", ""),
                                        "formula_id": formula_case.get("formula_id", ""),
                                        "formula_kind": spec.formula_kind,
                                        "out_name": spec.out_name,
                                        "reason": "initially_satisfied_on_quotient",
                                    }
                                )
                                skipped_count += 1
                                continue
                            row = run_prepared_case(
                                model_meta,
                                repair_model,
                                formula_case,
                                formula_meta,
                                spec,
                                v_meta,
                                quotient_relative,
                                target_text,
                                psi_text,
                                formula,
                                initial_satisfied_on_surface,
                                config,
                                out_dir,
                                case_id,
                                ranker,
                                write_repaired_aut=args.write_repaired_aut,
                                neural_cegis_enabled=args.neural_cegis_retrain,
                                neural_cegis_attempts=args.neural_cegis_attempts,
                                neural_cegis_epochs=args.neural_cegis_epochs,
                                neural_cegis_lr=args.neural_cegis_lr,
                                neural_cegis_candidate_limit=args.neural_cegis_candidate_limit,
                                neural_cegis_adopt_oracle=args.neural_cegis_adopt_oracle,
                                oracle_config=oracle_config,
                                oracle_ranker=oracle_ranker,
                                neural_rescue_enabled=args.neural_rescue_linear,
                                rescue_config=rescue_config,
                                rescue_ranker=rescue_ranker,
                            )
                            runs_sink.writerow(row)
                            summary.add_row(row)
                            suite_summaries.setdefault(spec.suite, GroupSummary()).add_row(row)
                            suite_summaries_by_formula.setdefault(spec.suite, GroupSummary(stratified=True)).add_row(row)
                            case_status = "success" if row.get("success") == "YES" else "no_repair"
                        except Exception as exc:
                            case_status = "error"
                            errors_sink.writerow({"case_id": case_id, "model_path": model_meta.get("model_path", ""), "error": repr(exc)})
                            summary.add_error()
                            print(f"[ERROR] {case_id}: {exc}")
                        finally:
                            v_meta = None
                            target_text = None
                            psi_text = None
                            formula_meta = None
                            repair_model = None
                            quotient_relative = None
                            formula = None
                            row = None
                            _checker = None
                            if args.trim_memory_every_case:
                                release_runtime_memory(args.device, trim_process=True)
                            if args.case_progress_every > 0 and (
                                case_index == len(formula_cases_for_model) or case_index % args.case_progress_every == 0
                            ):
                                print_progress(
                                    "stage2-case",
                                    case_index,
                                    len(formula_cases_for_model),
                                    spec=spec.out_name,
                                    model=model_meta.get("model_id", ""),
                                    current=case_id,
                                    status=case_status,
                                    elapsed_ms=f"{(time.perf_counter() - case_start) * 1000.0:.1f}",
                                    rss_mb=current_rss_mb(),
                                )
                finally:
                    quotient_cache.clear()
                    release_runtime_memory(args.device, trim_process=args.trim_memory_every_case)
                completed_repair_units += 1
                if args.progress_every > 0 and (
                    processed_models == len(manifest["models"]) or processed_models % args.progress_every == 0
                ):
                    print_progress(
                        "stage2-repair",
                        completed_repair_units,
                        total_repair_units,
                        spec=f"{spec_index}/{len(specs)}:{spec.out_name}",
                        model=f"{processed_models}/{len(manifest['models'])}",
                        current=model_meta.get("model_id", ""),
                        runs=summary.runs,
                        skipped=skipped_count,
                        errors=summary.errors,
                    )

        with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, ensure_ascii=False, indent=2)
        print(f"{spec.out_name}: runs={summary.runs} skipped_initially_satisfied={skipped_count} errors={summary.errors}")

    summary_targets = {
        "sf_vs_no_sf": "sf_vs_no_sf_summary.csv",
        "repair_mode": "repair_mode_summary.csv",
        "postprocess": "postprocess_summary.csv",
        "ranker": "ranker_summary.csv",
    }
    for suite, name in summary_targets.items():
        write_group_accumulator(suite_summaries.get(suite), results_root / name)
        write_group_accumulator(suite_summaries_by_formula.get(suite), results_root / name.replace(".csv", "_by_formula.csv"))
    print(f"Stage 2 complete: {results_root}")


if __name__ == "__main__":
    main()
