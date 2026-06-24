from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

from svbr.core import Formula, HMLParser
from svbr.experiments.progress import print_progress
from svbr.io_hints import drop_file_cache
from svbr.repair.add_delete import (
    Edge,
    OverlayHMLChecker,
    RepairLTS,
    escape_aut_label,
    verify_formula_with_edits,
)


def iter_edit_scripts(results_root: Path, script_list: Path | None = None):
    if script_list is not None:
        base_dir = script_list.parent
        with script_list.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                text = raw_line.strip()
                if not text or text.startswith("#"):
                    continue
                path = Path(text)
                if not path.is_absolute() and not path.exists():
                    candidate = base_dir / path
                    if candidate.exists():
                        path = candidate
                yield path
        return
    yield from sorted(results_root.glob("*/edit_scripts/*.json"))


def materialize_output_dir(script_path: Path, source_results_root: Path, output_results_root: Path) -> Path:
    source_spec_dir = script_path.parents[1]
    try:
        relative = source_spec_dir.resolve().relative_to(source_results_root.resolve())
    except ValueError:
        relative = Path(source_spec_dir.name)
    return output_results_root / relative


def edge_from_json(payload: dict) -> Edge:
    return Edge(int(payload["src"]), str(payload["action"]), int(payload["dst"]))


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def edges_to_json(edges: frozenset[Edge]) -> list[dict]:
    return [edge.to_json() for edge in sorted(edges)]


def write_writeback_operations(
    path: Path,
    case_id: str,
    target_formula: str,
    stage2_adds: frozenset[Edge],
    stage2_dels: frozenset[Edge],
    stage3_adds: frozenset[Edge],
    stage3_dels: frozenset[Edge],
    materialized_verified: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "case_id": case_id,
                "target_formula": target_formula,
                "lts_prime_to_lts_double_prime": {
                    "surface": "quotient_lts_blocks",
                    "adds": edges_to_json(stage2_adds),
                    "dels": edges_to_json(stage2_dels),
                },
                "lts_double_prime_template_to_original_lts": {
                    "surface": "original_lts_states",
                    "selection": "minimal_required_template_instances",
                    "adds": edges_to_json(stage3_adds),
                    "dels": edges_to_json(stage3_dels),
                },
                "materialized_verified": materialized_verified,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")


def trim_process_memory() -> None:
    if os.name != "posix":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def release_runtime_memory(trim_process: bool = True) -> None:
    gc.collect()
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
    return ""


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


def first_state_by_block(state_to_block: tuple[int, ...]) -> dict[int, int]:
    reps: dict[int, int] = {}
    for state, block in enumerate(state_to_block):
        reps.setdefault(int(block), state)
    return reps


def states_by_block(state_to_block: tuple[int, ...], wanted_blocks: set[int] | None = None) -> dict[int, list[int]]:
    blocks: dict[int, list[int]] = {}
    for state, block in enumerate(state_to_block):
        block_id = int(block)
        if wanted_blocks is not None and block_id not in wanted_blocks:
            continue
        blocks.setdefault(block_id, []).append(state)
    return blocks


def quotient_add_blocks(adds: frozenset[Edge]) -> set[int]:
    blocks: set[int] = set()
    for edge in adds:
        blocks.add(edge.src)
        blocks.add(edge.dst)
    return blocks


def quotient_delete_keys(dels: frozenset[Edge]) -> set[tuple[int, str, int]]:
    return {(edge.src, edge.action, edge.dst) for edge in dels}


def is_deleted_by_quotient(edge: Edge, state_to_block: tuple[int, ...], delete_keys: set[tuple[int, str, int]]) -> bool:
    return (int(state_to_block[edge.src]), edge.action, int(state_to_block[edge.dst])) in delete_keys


def concrete_edge_for_add(
    model: RepairLTS,
    block_states: dict[int, list[int]],
    q_edge: Edge,
    concrete_dels: frozenset[Edge],
    target_state: int | None = None,
) -> Edge | None:
    src_states = block_states.get(q_edge.src)
    dst_states = block_states.get(q_edge.dst)
    if not src_states:
        raise ValueError(f"Quotient add edge uses missing source block {q_edge.src}")
    if not dst_states:
        raise ValueError(f"Quotient add edge uses missing destination block {q_edge.dst}")
    if target_state in src_states:
        src_states = [target_state] + [state for state in src_states if state != target_state]
    if target_state in dst_states:
        dst_states = [target_state] + [state for state in dst_states if state != target_state]
    for src in src_states:
        for dst in dst_states:
            edge = Edge(src, q_edge.action, dst)
            if edge not in model.edges or edge in concrete_dels:
                return edge
    return None


def concrete_edge_for_delete(
    model: RepairLTS,
    state_to_block: tuple[int, ...],
    q_edge: Edge,
    target_state: int | None = None,
) -> Edge | None:
    candidates = [
        edge
        for edge in model.edges
        if int(state_to_block[edge.src]) == q_edge.src
        and edge.action == q_edge.action
        and int(state_to_block[edge.dst]) == q_edge.dst
    ]
    preferred = [edge for edge in candidates if edge.src == target_state]
    if preferred:
        return min(preferred)
    return min(candidates) if candidates else None


def lift_quotient_edits_to_concrete(
    model: RepairLTS,
    state_to_block: tuple[int, ...],
    adds: frozenset[Edge],
    dels: frozenset[Edge],
    target_state: int | None = None,
) -> tuple[frozenset[Edge], frozenset[Edge]]:
    """Map each quotient edit to at most one concrete edge in the original LTS."""
    block_states = states_by_block(state_to_block, quotient_add_blocks(adds))
    concrete_dels = {
        edge
        for q_edge in dels
        for edge in [concrete_edge_for_delete(model, state_to_block, q_edge, target_state)]
        if edge is not None
    }
    concrete_dels_frozen = frozenset(concrete_dels)
    concrete_adds = {
        edge
        for q_edge in adds
        for edge in [concrete_edge_for_add(model, block_states, q_edge, concrete_dels_frozen, target_state)]
        if edge is not None
    }
    return frozenset(concrete_adds), concrete_dels_frozen


def concrete_edge_key(edge: Edge, state_to_block: tuple[int, ...]) -> tuple[int, str, int]:
    return (int(state_to_block[edge.src]), edge.action, int(state_to_block[edge.dst]))


def concrete_edit_allowed_by_quotient(
    edge: Edge,
    op: str,
    state_to_block: tuple[int, ...],
    q_add_keys: set[tuple[int, str, int]],
    q_del_keys: set[tuple[int, str, int]],
) -> bool:
    if not 0 <= edge.src < len(state_to_block) or not 0 <= edge.dst < len(state_to_block):
        return False
    key = concrete_edge_key(edge, state_to_block)
    if op == "add":
        return key in q_add_keys
    if op == "del":
        return key in q_del_keys
    return False


def original_formula_holds(
    model: RepairLTS,
    formula: Formula,
    adds: frozenset[Edge],
    dels: frozenset[Edge],
    target_state: int,
) -> bool:
    checker = OverlayHMLChecker(model, adds, dels)
    return checker.eval(target_state, formula)


@dataclass(frozen=True)
class ConcreteLiftCandidate:
    op: str
    edge: Edge
    reason: str
    priority: int = 0
    batch_key: tuple | None = None
    template_key: tuple[int, str, int] | None = None


def required_batch_key(parent_key: tuple | None, *parts) -> tuple:
    if parent_key is None:
        return ("required", *parts)
    return (*parent_key, *parts)


def append_limited(target: list[ConcreteLiftCandidate], items: list[ConcreteLiftCandidate], limit: int | None) -> bool:
    if not items:
        return False
    if limit is None:
        target.extend(items)
        return False
    remaining = limit - len(target)
    if remaining <= 0:
        return True
    target.extend(items[:remaining])
    return len(target) >= limit


def ordered_block_states(block_states: dict[int, list[int]], block: int, preferred: int | None = None) -> list[int]:
    states = list(block_states.get(block, []))
    if preferred is not None and preferred in states:
        states.remove(preferred)
        states.insert(0, preferred)
    return states


def fallback_add_priority(direct_priority: int) -> int:
    # direct_priority is -100 + depth; for fallback edges, prefer extending the
    # current counterexample path before trying more sibling representatives.
    return -direct_priority


def add_candidates_for_quotient_edge(
    model: RepairLTS,
    checker: OverlayHMLChecker,
    state_to_block: tuple[int, ...],
    block_states: dict[int, list[int]],
    q_add_keys: set[tuple[int, str, int]],
    adds: set[Edge],
    dels: set[Edge],
    src_state: int,
    action: str,
    child: Formula,
    want_child: bool,
    priority: int = 0,
    batch_key: tuple | None = None,
    limit: int | None = None,
) -> list[ConcreteLiftCandidate]:
    if not 0 <= src_state < len(state_to_block):
        return []
    src_block = int(state_to_block[src_state])
    fallback: ConcreteLiftCandidate | None = None
    for q_src, q_action, q_dst in sorted(q_add_keys):
        if q_src != src_block or q_action != action:
            continue
        template_key = (q_src, q_action, q_dst)
        for dst_state in ordered_block_states(block_states, q_dst):
            edge = Edge(src_state, action, dst_state)
            if edge in adds:
                continue
            if edge in model.edges and edge not in dels:
                continue
            # The counterexample selects a concrete source path. The Stage 2
            # LTS'' template remains authoritative for both endpoint blocks.
            candidate = ConcreteLiftCandidate("add", edge, "path_guided_add", priority, batch_key, template_key)
            if checker.eval(dst_state, child) == want_child:
                return [candidate]
            if fallback is None:
                fallback = ConcreteLiftCandidate("add", edge, "path_guided_add_fallback", fallback_add_priority(priority), batch_key, template_key)
    return [fallback] if fallback is not None else []


def fallback_if_empty(
    adds: set[Edge],
    dels: set[Edge],
    concrete_adds: frozenset[Edge],
    concrete_dels: frozenset[Edge],
) -> tuple[frozenset[Edge], frozenset[Edge]]:
    if adds or dels:
        return frozenset(adds), frozenset(dels)
    return concrete_adds, concrete_dels


def delete_candidate_for_quotient_edge(
    edge: Edge,
    state_to_block: tuple[int, ...],
    q_del_keys: set[tuple[int, str, int]],
    dels: set[Edge],
    priority: int = 0,
    batch_key: tuple | None = None,
) -> list[ConcreteLiftCandidate]:
    if edge in dels:
        return []
    if not 0 <= edge.src < len(state_to_block) or not 0 <= edge.dst < len(state_to_block):
        return []
    key = (int(state_to_block[edge.src]), edge.action, int(state_to_block[edge.dst]))
    if key not in q_del_keys:
        return []
    return [ConcreteLiftCandidate("del", edge, "path_guided_del", priority, batch_key, key)]


def overlay_successors(
    model: RepairLTS,
    adds: set[Edge],
    dels: set[Edge],
    state: int,
    action: str | None = None,
    adds_by_src: dict[int, list[Edge]] | None = None,
) -> list[Edge]:
    if not 0 <= state < model.state_count:
        return []
    values = [
        edge
        for edge in model.edge_adjacency()[state]
        if (action is None or edge.action == action) and edge not in dels
    ]
    base_edges = model.edges
    added_edges = adds_by_src.get(state, []) if adds_by_src is not None else [edge for edge in adds if edge.src == state]
    for edge in added_edges:
        if edge.src != state:
            continue
        if action is not None and edge.action != action:
            continue
        if edge in base_edges and edge not in dels:
            continue
        values.append(edge)
    return sorted(values)


def path_guided_lift_candidates(
    model: RepairLTS,
    checker: OverlayHMLChecker,
    formula: Formula,
    state: int,
    want_true: bool,
    state_to_block: tuple[int, ...],
    block_states: dict[int, list[int]],
    q_add_keys: set[tuple[int, str, int]],
    q_del_keys: set[tuple[int, str, int]],
    adds: set[Edge],
    dels: set[Edge],
    depth: int = 0,
    batch_key: tuple | None = None,
    candidate_limit: int | None = None,
) -> list[ConcreteLiftCandidate]:
    if depth > 8 or checker.eval(state, formula) == want_true:
        return []
    direct_priority = -100 + depth

    if formula.kind in {"true", "false"}:
        return []
    if formula.kind == "not":
        # Keep the modal operator intact under negation and only flip polarity:
        # !<> is handled as diamond/want_false, ![] as box/want_false.
        return path_guided_lift_candidates(
            model,
            checker,
            formula.left,
            state,
            not want_true,
            state_to_block,
            block_states,
            q_add_keys,
            q_del_keys,
            adds,
            dels,
            depth + 1,
            batch_key,
            candidate_limit,
        )
    if formula.kind == "and":
        result: list[ConcreteLiftCandidate] = []
        if want_true:
            for index, child in enumerate([formula.left, formula.right]):
                if not checker.eval(state, child):
                    if append_limited(result, path_guided_lift_candidates(model, checker, child, state, True, state_to_block, block_states, q_add_keys, q_del_keys, adds, dels, depth + 1, required_batch_key(batch_key, "and_true", state, index), candidate_limit), candidate_limit):
                        return result
        else:
            for child in [formula.left, formula.right]:
                if checker.eval(state, child):
                    candidates = path_guided_lift_candidates(model, checker, child, state, False, state_to_block, block_states, q_add_keys, q_del_keys, adds, dels, depth + 1, batch_key, candidate_limit)
                    if candidates:
                        return candidates
        return result
    if formula.kind == "or":
        result: list[ConcreteLiftCandidate] = []
        if want_true:
            for child in [formula.left, formula.right]:
                if not checker.eval(state, child):
                    candidates = path_guided_lift_candidates(model, checker, child, state, True, state_to_block, block_states, q_add_keys, q_del_keys, adds, dels, depth + 1, batch_key, candidate_limit)
                    if candidates:
                        return candidates
        else:
            for index, child in enumerate([formula.left, formula.right]):
                if checker.eval(state, child):
                    if append_limited(result, path_guided_lift_candidates(model, checker, child, state, False, state_to_block, block_states, q_add_keys, q_del_keys, adds, dels, depth + 1, required_batch_key(batch_key, "or_false", state, index), candidate_limit), candidate_limit):
                        return result
        return result
    if formula.kind == "diamond":
        result: list[ConcreteLiftCandidate] = []
        action = formula.action
        successors = overlay_successors(model, adds, dels, state, action, checker.adds_by_src)
        if want_true:
            # One repaired successor is enough for an existential obligation.
            for edge in successors:
                if not checker.eval(edge.dst, formula.left):
                    candidates = path_guided_lift_candidates(model, checker, formula.left, edge.dst, True, state_to_block, block_states, q_add_keys, q_del_keys, adds, dels, depth + 1, batch_key, candidate_limit)
                    if candidates:
                        return candidates
            return add_candidates_for_quotient_edge(model, checker, state_to_block, block_states, q_add_keys, adds, dels, state, action, formula.left, True, direct_priority, batch_key, candidate_limit)
        else:
            for edge in successors:
                if checker.eval(edge.dst, formula.left):
                    edge_batch_key = required_batch_key(batch_key, "diamond_false", state, action, edge.dst)
                    if append_limited(result, delete_candidate_for_quotient_edge(edge, state_to_block, q_del_keys, dels, direct_priority, edge_batch_key), candidate_limit):
                        return result
                    if append_limited(result, path_guided_lift_candidates(model, checker, formula.left, edge.dst, False, state_to_block, block_states, q_add_keys, q_del_keys, adds, dels, depth + 1, edge_batch_key, candidate_limit), candidate_limit):
                        return result
        return result
    if formula.kind == "box":
        result: list[ConcreteLiftCandidate] = []
        action = formula.action
        successors = overlay_successors(model, adds, dels, state, action, checker.adds_by_src)
        if want_true:
            # Every violating successor is required for a universal obligation.
            for edge in successors:
                if not checker.eval(edge.dst, formula.left):
                    edge_batch_key = required_batch_key(batch_key, "box_true", state, action, edge.dst)
                    if append_limited(result, delete_candidate_for_quotient_edge(edge, state_to_block, q_del_keys, dels, direct_priority, edge_batch_key), candidate_limit):
                        return result
                    if append_limited(result, path_guided_lift_candidates(model, checker, formula.left, edge.dst, True, state_to_block, block_states, q_add_keys, q_del_keys, adds, dels, depth + 1, edge_batch_key, candidate_limit), candidate_limit):
                        return result
        else:
            candidates = add_candidates_for_quotient_edge(model, checker, state_to_block, block_states, q_add_keys, adds, dels, state, action, formula.left, False, direct_priority, batch_key, candidate_limit)
            if candidates:
                return candidates
            for edge in successors:
                if checker.eval(edge.dst, formula.left):
                    candidates = path_guided_lift_candidates(model, checker, formula.left, edge.dst, False, state_to_block, block_states, q_add_keys, q_del_keys, adds, dels, depth + 1, batch_key, candidate_limit)
                    if candidates:
                        return candidates
        return result
    return []


def counterexample_guided_lift(
    model: RepairLTS,
    state_to_block: tuple[int, ...],
    q_adds: frozenset[Edge],
    q_dels: frozenset[Edge],
    target_text: str,
    target_state: int,
    concrete_adds: frozenset[Edge],
    concrete_dels: frozenset[Edge],
    max_iters: int,
    single_per_quotient_edge: bool = False,
    initial_budget: int | None = None,
    batch_size: int = 512,
    progress_every: int = 0,
    case_id: str = "",
    max_seconds: float = 0.0,
    quotient_fill_after: int = 0,
) -> tuple[frozenset[Edge], frozenset[Edge], int]:
    if max_iters <= 0 or not target_text:
        return concrete_adds, concrete_dels, 0

    if initial_budget is None:
        initial_budget = max_iters
    active_budget = min(max_iters, max(1, initial_budget))
    formula = HMLParser.parse(target_text)
    q_add_keys = quotient_delete_keys(q_adds)
    q_del_keys = quotient_delete_keys(q_dels)
    block_states = states_by_block(state_to_block, quotient_add_blocks(q_adds))
    adds: set[Edge] = set()
    dels: set[Edge] = set()
    seen: set[tuple[str, Edge]] = set()
    iteration = 0
    last_progress_iter = 0
    started = time.perf_counter()
    batch_size = max(1, batch_size)
    candidate_pool_limit = max(32, batch_size)
    last_candidates_count = 0
    last_reason = ""

    def candidate_is_valid(
        candidate: ConcreteLiftCandidate,
        covered_add_keys: set[tuple[int, str, int]],
        covered_del_keys: set[tuple[int, str, int]],
    ) -> bool:
        edge = candidate.edge
        if (candidate.op, edge) in seen:
            return False
        quotient_key = concrete_edge_key(edge, state_to_block)
        # Counterexamples only prioritize concrete representatives. Every
        # applied edit must still instantiate its exact Stage 2 LTS'' template.
        if candidate.template_key is None or quotient_key != candidate.template_key:
            return False
        if not concrete_edit_allowed_by_quotient(edge, candidate.op, state_to_block, q_add_keys, q_del_keys):
            return False
        if single_per_quotient_edge and candidate.op == "add" and quotient_key in covered_add_keys:
            return False
        if single_per_quotient_edge and candidate.op == "del" and quotient_key in covered_del_keys:
            return False
        if candidate.op == "add" and edge in model.edges and edge not in dels:
            return False
        if candidate.op == "del" and edge not in model.edges:
            return False
        return True

    while iteration < max_iters:
        elapsed = time.perf_counter() - started
        if max_seconds > 0 and elapsed >= max_seconds:
            raise TimeoutError(
                f"Stage 3 lift timed out after {elapsed:.1f}s: "
                f"case={case_id}, edits={iteration}, adds={len(adds)}, dels={len(dels)}, max_iters={max_iters}"
            )
        frozen_adds = frozenset(adds)
        frozen_dels = frozenset(dels)
        checker = OverlayHMLChecker(model, frozen_adds, frozen_dels)
        if checker.eval(target_state, formula):
            return frozen_adds, frozen_dels, iteration
        covered_add_keys = {concrete_edge_key(edge, state_to_block) for edge in adds} if single_per_quotient_edge else set()
        covered_del_keys = {concrete_edge_key(edge, state_to_block) for edge in dels} if single_per_quotient_edge else set()
        candidates = path_guided_lift_candidates(
            model,
            checker,
            formula,
            target_state,
            True,
            state_to_block,
            block_states,
            q_add_keys,
            q_del_keys,
            adds,
            dels,
            candidate_limit=candidate_pool_limit,
        )
        last_candidates_count = len(candidates)
        candidates.sort(key=lambda item: (item.priority, item.op, item.edge.src, item.edge.action, item.edge.dst))

        chosen_batch: list[ConcreteLiftCandidate] = []
        chosen_batch_keys: set[tuple] = set()
        chosen_edit_keys: set[tuple[str, Edge]] = set()
        batch_covered_add_keys = set(covered_add_keys)
        batch_covered_del_keys = set(covered_del_keys)
        for candidate in candidates:
            if not candidate_is_valid(candidate, batch_covered_add_keys, batch_covered_del_keys):
                continue
            chosen_batch = [candidate]
            chosen_edit_keys.add((candidate.op, candidate.edge))
            quotient_key = concrete_edge_key(candidate.edge, state_to_block)
            if candidate.op == "add":
                batch_covered_add_keys.add(quotient_key)
            else:
                batch_covered_del_keys.add(quotient_key)
            break

        if chosen_batch and chosen_batch[0].batch_key is not None and batch_size > 1:
            first = chosen_batch[0]
            chosen_batch_keys.add(first.batch_key)
            for candidate in candidates:
                if len(chosen_batch) >= batch_size:
                    break
                if candidate is first:
                    continue
                if candidate.priority != first.priority:
                    break
                if candidate.batch_key is None or candidate.batch_key in chosen_batch_keys:
                    continue
                if (candidate.op, candidate.edge) in chosen_edit_keys:
                    continue
                if not candidate_is_valid(candidate, batch_covered_add_keys, batch_covered_del_keys):
                    continue
                chosen_batch.append(candidate)
                chosen_batch_keys.add(candidate.batch_key)
                chosen_edit_keys.add((candidate.op, candidate.edge))
                quotient_key = concrete_edge_key(candidate.edge, state_to_block)
                if candidate.op == "add":
                    batch_covered_add_keys.add(quotient_key)
                else:
                    batch_covered_del_keys.add(quotient_key)

        if not chosen_batch:
            next_adds, next_dels = fallback_if_empty(adds, dels, concrete_adds, concrete_dels)
            return next_adds, next_dels, iteration
        last_reason = chosen_batch[0].reason
        for chosen in chosen_batch:
            if iteration >= max_iters:
                break
            seen.add((chosen.op, chosen.edge))
            if chosen.op == "add":
                adds.add(chosen.edge)
            elif chosen.op == "del":
                dels.add(chosen.edge)
            iteration += 1

        if progress_every > 0 and (iteration == max_iters or iteration - last_progress_iter >= progress_every):
            last_progress_iter = iteration
            print_progress(
                "stage3-lift",
                iteration,
                max_iters,
                case=case_id,
                active_budget=active_budget,
                batch=len(chosen_batch),
                candidates=last_candidates_count,
                reason=last_reason,
                adds=len(adds),
                dels=len(dels),
                rss_mb=current_rss_mb(),
                elapsed_s=f"{time.perf_counter() - started:.1f}",
            )
            release_runtime_memory(trim_process=False)

        if iteration >= active_budget:
            frozen_adds = frozenset(adds)
            frozen_dels = frozenset(dels)
            if original_formula_holds(model, formula, frozen_adds, frozen_dels, target_state):
                return frozen_adds, frozen_dels, iteration
            if active_budget >= max_iters:
                break
            while active_budget < max_iters and active_budget <= iteration:
                active_budget = min(max_iters, max(active_budget + 1, active_budget * 2))

    next_adds, next_dels = fallback_if_empty(adds, dels, concrete_adds, concrete_dels)
    return next_adds, next_dels, max_iters


def lifted_add_count(
    model: RepairLTS,
    state_to_block: tuple[int, ...],
    reps: dict[int, int],
    adds: frozenset[Edge],
    delete_keys: set[tuple[int, str, int]],
) -> int:
    count = 0
    for q_edge in adds:
        if q_edge.dst not in reps:
            raise ValueError(f"Quotient add edge uses missing destination block {q_edge.dst}")
        dst_rep = reps[q_edge.dst]
        source_found = False
        for src, block in enumerate(state_to_block):
            if int(block) != q_edge.src:
                continue
            source_found = True
            edge = Edge(src, q_edge.action, dst_rep)
            if edge not in model.edges or is_deleted_by_quotient(edge, state_to_block, delete_keys):
                count += 1
        if not source_found:
            raise ValueError(f"Quotient add edge uses missing source block {q_edge.src}")
    return count


def lifted_del_count(model: RepairLTS, state_to_block: tuple[int, ...], delete_keys: set[tuple[int, str, int]]) -> int:
    return sum(1 for edge in model.edges if is_deleted_by_quotient(edge, state_to_block, delete_keys))


def write_aut_stream(
    model: RepairLTS,
    path: Path,
    transition_count: int,
    state_to_block: tuple[int, ...] | None = None,
    reps: dict[int, int] | None = None,
    q_adds: frozenset[Edge] = frozenset(),
    q_delete_keys: set[tuple[int, str, int]] | None = None,
    adds: frozenset[Edge] = frozenset(),
    dels: frozenset[Edge] = frozenset(),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q_delete_keys = q_delete_keys or set()
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"des ({model.initial},{transition_count},{model.state_count})\n")
        for edge in model.edges:
            if state_to_block is not None:
                if is_deleted_by_quotient(edge, state_to_block, q_delete_keys):
                    continue
            elif edge in dels:
                continue
            handle.write(f'({edge.src},"{escape_aut_label(edge.action)}",{edge.dst})\n')

        if state_to_block is not None:
            if reps is None:
                raise ValueError("Missing quotient block representatives")
            for q_edge in sorted(q_adds):
                dst_rep = reps[q_edge.dst]
                for src, block in enumerate(state_to_block):
                    if int(block) != q_edge.src:
                        continue
                    edge = Edge(src, q_edge.action, dst_rep)
                    if edge in model.edges and not is_deleted_by_quotient(edge, state_to_block, q_delete_keys):
                        continue
                    handle.write(f'({edge.src},"{escape_aut_label(edge.action)}",{edge.dst})\n')
        else:
            for edge in sorted(adds):
                if edge in model.edges and edge not in dels:
                    continue
                handle.write(f'({edge.src},"{escape_aut_label(edge.action)}",{edge.dst})\n')


class QuotientLiftedHMLChecker:
    def __init__(self, base: RepairLTS, quotient, adds: frozenset[Edge], dels: frozenset[Edge]):
        self.base = base
        self.base_adjacency = base.adjacency()
        self.state_to_block = tuple(int(item) for item in quotient.state_to_block)
        self.reps = first_state_by_block(self.state_to_block)
        self.delete_keys = quotient_delete_keys(dels)
        self.adds_by_src_block: dict[int, list[tuple[str, int]]] = {}
        for edge in adds:
            if edge.dst not in self.reps:
                raise ValueError(f"Quotient add edge uses missing destination block {edge.dst}")
            self.adds_by_src_block.setdefault(edge.src, []).append((edge.action, self.reps[edge.dst]))
        self.memo: dict[tuple[int, int], bool] = {}
        self.visited_entries = 0
        self.scanned_transitions = 0

    def edge_deleted(self, src: int, action: str, dst: int) -> bool:
        return (self.state_to_block[src], action, self.state_to_block[dst]) in self.delete_keys

    def added_edges_for_state(self, state: int) -> list[tuple[str, int]]:
        return self.adds_by_src_block.get(self.state_to_block[state], [])

    def eval(self, state: int, formula: Formula) -> bool:
        key = (state, id(formula))
        if key in self.memo:
            return self.memo[key]
        self.visited_entries += 1

        if formula.kind == "true":
            value = True
        elif formula.kind == "false":
            value = False
        elif formula.kind == "not":
            value = not self.eval(state, formula.left)
        elif formula.kind == "and":
            value = self.eval(state, formula.left) and self.eval(state, formula.right)
        elif formula.kind == "or":
            value = self.eval(state, formula.left) or self.eval(state, formula.right)
        elif formula.kind == "diamond":
            value = False
            for action, dst in self.base_adjacency[state]:
                if action != formula.action or self.edge_deleted(state, action, dst):
                    continue
                self.scanned_transitions += 1
                if self.eval(dst, formula.left):
                    value = True
                    break
            if not value:
                for action, dst in self.added_edges_for_state(state):
                    if action != formula.action:
                        continue
                    self.scanned_transitions += 1
                    if self.eval(dst, formula.left):
                        value = True
                        break
        elif formula.kind == "box":
            value = True
            for action, dst in self.base_adjacency[state]:
                if action != formula.action or self.edge_deleted(state, action, dst):
                    continue
                self.scanned_transitions += 1
                if not self.eval(dst, formula.left):
                    value = False
                    break
            if value:
                for action, dst in self.added_edges_for_state(state):
                    if action != formula.action:
                        continue
                    self.scanned_transitions += 1
                    if not self.eval(dst, formula.left):
                        value = False
                        break
        else:
            raise ValueError(f"Unknown formula kind: {formula.kind}")

        self.memo[key] = value
        return value


def verify_quotient_lifted(model: RepairLTS, quotient, adds: frozenset[Edge], dels: frozenset[Edge], target_text: str, target_state: int | None = None) -> tuple[str, str]:
    if not target_text:
        return "", ""
    try:
        formula = HMLParser.parse(target_text)
        checker = QuotientLiftedHMLChecker(model, quotient, adds, dels)
        ok = checker.eval(model.initial if target_state is None else target_state, formula)
        return yes_no(ok), ""
    except Exception as exc:
        return "ERROR", repr(exc)


def verify_original_lifted(model: RepairLTS, adds: frozenset[Edge], dels: frozenset[Edge], target_text: str, target_state: int | None = None) -> tuple[str, str]:
    if not target_text:
        return "", ""
    try:
        formula = HMLParser.parse(target_text)
        if target_state is None:
            ok, _checker = verify_formula_with_edits(model, formula, adds, dels)
        else:
            checker = OverlayHMLChecker(model, adds, dels)
            ok = checker.eval(target_state, formula)
        return yes_no(ok), ""
    except Exception as exc:
        return "ERROR", repr(exc)


def script_target_state(payload: dict, model: RepairLTS | None = None, override: int = -1) -> int:
    if override >= 0:
        target_state = override
    else:
        target_state = int(payload.get("original_target_state", payload.get("target_state", payload.get("original_initial", -1))))
        if target_state < 0 and model is not None:
            target_state = model.initial
    if model is not None and not 0 <= target_state < model.state_count:
        raise ValueError(f"target_state {target_state} is outside 0..{model.state_count - 1}")
    return target_state


def materialize_one(
    script_path: Path,
    prepared_dir: Path,
    force: bool = False,
    target_state_override: int = -1,
    cex_lift_iters: int = 0,
    cex_lift_mode: str = "closure",
    cex_batch_size: int = 512,
    case_progress_every: int = 0,
    max_case_seconds: float = 0.0,
    quotient_fill_after: int = 0,
    trim_memory_every_case: bool = True,
    source_results_root: Path | None = None,
    output_results_root: Path | None = None,
) -> dict:
    case_started = time.perf_counter()
    with script_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    result = payload.get("result", {})
    case_id = payload.get("case_id", script_path.stem)
    source_results_root = source_results_root or script_path.parents[2]
    output_results_root = output_results_root or source_results_root
    out_dir = materialize_output_dir(script_path, source_results_root, output_results_root)
    repaired_path = out_dir / "repaired_aut" / f"{case_id}.aut"
    operations_path = out_dir / "writeback_operations" / f"{case_id}.json"

    row = {
        "case_id": case_id,
        "script_path": str(script_path),
        "model_path": payload.get("model_path", ""),
        "target_state": script_target_state(payload, override=target_state_override),
        "repair_surface": payload.get("repair_surface", "original_lts"),
        "quotient_pickle": payload.get("quotient_pickle", ""),
        "repaired_path": str(repaired_path),
        "stage2_block_operations_path": str(script_path),
        "stage3_writeback_operations_path": "",
        "status": "",
        "add_edges": 0,
        "del_edges": 0,
        "lifted_add_edges": 0,
        "lifted_del_edges": 0,
        "lifting_strategy": "single",
        "lifting_mode": cex_lift_mode,
        "lifting_iters": 0,
        "lifting_initial_budget": cex_lift_iters,
        "lifting_budget_cap": cex_lift_iters,
        "lifting_batch_size": cex_batch_size,
        "lifting_timeout_seconds": max_case_seconds,
        "strict_minimal_lifting": "YES",
        "elapsed_ms": "0.000",
        "rss_mb": "",
        "materialized_verified": "",
        "verification_error": "",
        "error": "",
    }

    if not result.get("success", False):
        row["status"] = "skipped_unsuccessful"
        row["elapsed_ms"] = f"{(time.perf_counter() - case_started) * 1000.0:.3f}"
        row["rss_mb"] = current_rss_mb()
        return row
    if repaired_path.exists() and not force:
        existing_model = None
        try:
            existing_model = RepairLTS.from_aut(str(repaired_path))
            existing_target_state = script_target_state(payload, existing_model, target_state_override)
            verified, verify_error = verify_original_lifted(
                existing_model,
                frozenset(),
                frozenset(),
                payload.get("target_formula", ""),
                existing_target_state,
            )
            row["target_state"] = existing_target_state
            row["materialized_verified"] = verified
            row["verification_error"] = verify_error
            if operations_path.exists():
                row["stage3_writeback_operations_path"] = str(operations_path)
            row["status"] = "skipped_exists"
        except Exception as exc:
            row["status"] = "error"
            row["error"] = repr(exc)
        finally:
            if existing_model is not None:
                existing_model._adjacency_cache = None
                existing_model._edge_adjacency_cache = None
                existing_model._actions_cache = None
                existing_model._degree_cache = None
            existing_model = None
            release_runtime_memory(trim_process=trim_memory_every_case)
            row["elapsed_ms"] = f"{(time.perf_counter() - case_started) * 1000.0:.3f}"
            row["rss_mb"] = current_rss_mb()
        return row

    adds = frozenset(edge_from_json(edge) for edge in result.get("adds", []))
    dels = frozenset(edge_from_json(edge) for edge in result.get("dels", []))
    row["add_edges"] = len(adds)
    row["del_edges"] = len(dels)

    model = None
    try:
        model = RepairLTS.from_aut(payload["model_path"])
        target_state = script_target_state(payload, model, target_state_override)
        row["target_state"] = target_state
        if payload.get("repair_surface") == "quotient_lts":
            quotient = read_pickle(prepared_dir, payload["quotient_pickle"])
            state_to_block = tuple(int(item) for item in quotient.state_to_block)
            del quotient
            concrete_adds, concrete_dels = lift_quotient_edits_to_concrete(model, state_to_block, adds, dels, target_state)
            lift_budget_cap = cex_lift_iters
            if cex_lift_mode == "closure" and cex_lift_iters > 0:
                lift_budget_cap = model.transition_count
            row["lifting_budget_cap"] = lift_budget_cap
            concrete_adds, concrete_dels, lifting_iters = counterexample_guided_lift(
                model,
                state_to_block,
                adds,
                dels,
                payload.get("target_formula", ""),
                target_state,
                concrete_adds,
                concrete_dels,
                lift_budget_cap,
                single_per_quotient_edge=(cex_lift_mode != "closure"),
                initial_budget=cex_lift_iters,
                batch_size=cex_batch_size,
                progress_every=case_progress_every,
                case_id=case_id,
                max_seconds=max_case_seconds,
                quotient_fill_after=quotient_fill_after,
            )
            lifted_del_edges = len(concrete_dels)
            lifted_add_edges = sum(1 for edge in concrete_adds if edge not in model.edges or edge in concrete_dels)
            transition_count = model.transition_count - lifted_del_edges + lifted_add_edges
            write_aut_stream(
                model,
                repaired_path,
                transition_count,
                adds=concrete_adds,
                dels=concrete_dels,
            )
            verified, verify_error = verify_original_lifted(model, concrete_adds, concrete_dels, payload.get("target_formula", ""), target_state)
            writeback_adds = concrete_adds
            writeback_dels = concrete_dels
            row["lifting_iters"] = lifting_iters
            if lifting_iters > 0:
                row["lifting_strategy"] = "cex_guided_closure" if cex_lift_mode == "closure" else "cex_guided_single"
        else:
            lifted_del_edges = len(dels & model.edges)
            lifted_add_edges = sum(1 for edge in adds if edge not in model.edges or edge in dels)
            transition_count = model.transition_count - lifted_del_edges + lifted_add_edges
            write_aut_stream(model, repaired_path, transition_count, adds=adds, dels=dels)
            verified, verify_error = verify_original_lifted(model, adds, dels, payload.get("target_formula", ""), target_state)
            writeback_adds = adds
            writeback_dels = dels
        write_writeback_operations(
            operations_path,
            case_id,
            payload.get("target_formula", ""),
            adds,
            dels,
            writeback_adds,
            writeback_dels,
            verified,
        )
        row["lifted_add_edges"] = lifted_add_edges
        row["lifted_del_edges"] = lifted_del_edges
        row["stage3_writeback_operations_path"] = str(operations_path)
        row["materialized_verified"] = verified
        row["verification_error"] = verify_error
        row["status"] = "written"
    except Exception as exc:
        row["status"] = "error"
        row["error"] = repr(exc)
    finally:
        if model is not None:
            model._adjacency_cache = None
            model._edge_adjacency_cache = None
            model._actions_cache = None
            model._degree_cache = None
        model = None
        release_runtime_memory(trim_process=trim_memory_every_case)
        row["elapsed_ms"] = f"{(time.perf_counter() - case_started) * 1000.0:.3f}"
        row["rss_mb"] = current_rss_mb()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize repaired AUT files from Stage 2 edit_scripts after the main run")
    parser.add_argument("--results-root", default="results/add_delete_run")
    parser.add_argument("--output-results-root", default="", help="Write materialized AUT files/report here while reading Stage 2 edit_scripts from --results-root")
    parser.add_argument("--prepared-dir", default="results/add_delete_prepared")
    parser.add_argument("--script-list", default="", help="Optional newline-delimited edit_script paths to materialize instead of scanning --results-root")
    parser.add_argument("--force", action="store_true", help="overwrite existing repaired_aut files")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--case-start-every", type=int, default=0, help="Print the edit_script path before every N Stage 3 cases; 0 disables")
    parser.add_argument("--target-state", type=int, default=-1, help="Override edit script target state; -1 uses script target or AUT initial")
    parser.add_argument("--cex-lift-iters", type=int, default=16, help="Initial counterexample-guided concrete lifting budget; closure mode doubles this budget until verified or the original LTS transition count is reached")
    parser.add_argument("--cex-lift-mode", choices=["single", "closure"], default="closure", help="single keeps one concrete edit per quotient edge; closure keeps adding counterexample-guided concrete edits until verification succeeds or no candidate remains")
    parser.add_argument("--cex-batch-size", type=int, default=512, help="Maximum number of independent counterexample-guided concrete edits to apply in one Stage 3 lift batch")
    parser.add_argument("--case-progress-every", type=int, default=0, help="Print in-case Stage 3 lifting progress every N concrete edits; 0 disables")
    parser.add_argument("--max-case-seconds", type=float, default=0.0, help="Abort one Stage 3 lift case after this many seconds and save it as unresolved; 0 disables")
    parser.add_argument("--quotient-fill-after", type=int, default=0, help="Deprecated compatibility option; Stage 3 always keeps strict minimal template instantiation")
    parser.add_argument("--trim-memory-every-case", choices=["0", "1"], default="1", help="Run gc and malloc_trim after every materialized case")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_results_root = Path(args.output_results_root) if args.output_results_root else results_root
    prepared_dir = Path(args.prepared_dir)
    script_list = Path(args.script_list) if args.script_list else None
    script_count = sum(1 for _path in iter_edit_scripts(results_root, script_list))
    total = min(script_count, args.limit) if args.limit > 0 else script_count
    report_path = output_results_root / "materialize_repaired_aut.csv"
    unresolved_report_path = output_results_root / "materialize_repaired_aut_unresolved.csv"
    unresolved_scripts_path = output_results_root / "stage3_unresolved_scripts.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "script_path",
        "model_path",
        "target_state",
        "repair_surface",
        "quotient_pickle",
        "repaired_path",
        "stage2_block_operations_path",
        "stage3_writeback_operations_path",
        "status",
        "add_edges",
        "del_edges",
        "lifted_add_edges",
        "lifted_del_edges",
        "lifting_strategy",
        "lifting_mode",
        "lifting_iters",
        "lifting_initial_budget",
        "lifting_budget_cap",
        "lifting_batch_size",
        "lifting_timeout_seconds",
        "strict_minimal_lifting",
        "elapsed_ms",
        "rss_mb",
        "materialized_verified",
        "verification_error",
        "error",
    ]
    written_count = 0
    verified_count = 0
    error_count = 0
    unresolved_rows = []
    if args.limit > 0:
        limit = args.limit
    else:
        limit = 0
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, script_path in enumerate(iter_edit_scripts(results_root, script_list), start=1):
            if limit > 0 and index > limit:
                break
            if args.case_start_every > 0 and (index == 1 or index == total or index % args.case_start_every == 0):
                print_progress(
                    "stage3-case",
                    index,
                    total,
                    current=script_path,
                )
            row = materialize_one(
                script_path,
                prepared_dir,
                force=args.force,
                target_state_override=args.target_state,
                cex_lift_iters=args.cex_lift_iters,
                cex_lift_mode=args.cex_lift_mode,
                cex_batch_size=args.cex_batch_size,
                case_progress_every=args.case_progress_every,
                max_case_seconds=args.max_case_seconds,
                quotient_fill_after=args.quotient_fill_after,
                trim_memory_every_case=args.trim_memory_every_case == "1",
                source_results_root=results_root,
                output_results_root=output_results_root,
            )
            writer.writerow(row)
            handle.flush()
            if row["status"] == "written":
                written_count += 1
            if row["materialized_verified"] == "YES":
                verified_count += 1
            if row["status"] == "error":
                error_count += 1
            if row["status"] == "error" or (row["status"] == "written" and row["materialized_verified"] != "YES"):
                unresolved_rows.append(row)
            if args.progress_every > 0 and (index == total or index % args.progress_every == 0):
                print_progress(
                    "stage3-materialize",
                    index,
                    total,
                    written=written_count,
                    verified=verified_count,
                    errors=error_count,
                    current=script_path,
                )
    with unresolved_report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unresolved_rows)
    with unresolved_scripts_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in unresolved_rows:
            handle.write(str(row["script_path"]) + "\n")
    print(f"Materialized report: {report_path}")
    print(f"Unresolved Stage3 cases: {len(unresolved_rows)}")
    print(f"Unresolved report: {unresolved_report_path}")
    print(f"Unresolved scripts: {unresolved_scripts_path}")


if __name__ == "__main__":
    main()
