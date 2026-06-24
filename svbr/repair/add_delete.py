from __future__ import annotations

import json
import gc
import hashlib
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from svbr.core import (
    Formula,
    HMLParser,
    hml_formula_is_contradiction as hml_semantic_contradiction,
    hml_formula_is_tautology as hml_semantic_tautology,
    parse_aut_header,
    parse_aut_transition,
)
from svbr.io_hints import drop_file_cache


@dataclass(frozen=True, order=True)
class Edge:
    src: int
    action: str
    dst: int

    def to_json(self) -> dict:
        return {"src": self.src, "action": self.action, "dst": self.dst}


@dataclass(frozen=True, order=True)
class Edit:
    op: str
    edge: Edge
    reason: str = ""

    def to_json(self) -> dict:
        payload = {"op": self.op, **self.edge.to_json()}
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass
class RepairLTS:
    initial: int
    state_count: int
    edges: frozenset[Edge]
    _actions_cache: set[str] | None = field(default=None, init=False, repr=False, compare=False)
    _action_counts_cache: Counter | None = field(default=None, init=False, repr=False, compare=False)
    _degree_cache: tuple[Counter, Counter] | None = field(default=None, init=False, repr=False, compare=False)
    _adjacency_cache: list[list[tuple[str, int]]] | None = field(default=None, init=False, repr=False, compare=False)
    _edge_adjacency_cache: list[list[Edge]] | None = field(default=None, init=False, repr=False, compare=False)
    _edge_action_adjacency_cache: list[dict[str, list[Edge]]] | None = field(default=None, init=False, repr=False, compare=False)

    @staticmethod
    def from_aut(path: str | Path) -> "RepairLTS":
        path = Path(path)
        with path.open("r", encoding="utf-8-sig", errors="strict") as handle:
            try:
                initial, _declared_transitions, states = parse_aut_header(handle.readline())
                if states <= 0:
                    raise ValueError(f"{path}: AUT state count must be positive, got {states}")
                if not 0 <= initial < states:
                    raise ValueError(f"{path}: initial state {initial} is outside 0..{states - 1}")
                edges = []
                for raw_line in handle:
                    parsed = parse_aut_transition(raw_line)
                    if parsed is None:
                        continue
                    src, action, dst = parsed
                    if not 0 <= src < states or not 0 <= dst < states:
                        raise ValueError(f"{path}: transition ({src}, {action}, {dst}) uses a state outside 0..{states - 1}")
                    edges.append(Edge(src, action.strip(), dst))
            finally:
                drop_file_cache(handle)
        return RepairLTS(initial=initial, state_count=states, edges=frozenset(edges))

    @property
    def actions(self) -> set[str]:
        cached = getattr(self, "_actions_cache", None)
        if cached is None:
            cached = {edge.action for edge in self.edges}
            self._actions_cache = cached
        return cached

    @property
    def transition_count(self) -> int:
        return len(self.edges)

    def action_counts(self) -> Counter:
        cached = getattr(self, "_action_counts_cache", None)
        if cached is None:
            cached = Counter(edge.action for edge in self.edges)
            self._action_counts_cache = cached
        return cached

    def degree_counts(self) -> tuple[Counter, Counter]:
        cached = getattr(self, "_degree_cache", None)
        if cached is None:
            out_degree = Counter(edge.src for edge in self.edges)
            in_degree = Counter(edge.dst for edge in self.edges)
            cached = (out_degree, in_degree)
            self._degree_cache = cached
        return cached

    def degree_for(self, src: int, dst: int) -> tuple[int, int]:
        out_degree, in_degree = self.degree_counts()
        return out_degree[src], in_degree[dst]

    def adjacency(self) -> list[list[tuple[str, int]]]:
        cached = getattr(self, "_adjacency_cache", None)
        if cached is not None:
            return cached
        adjacency = [[] for _ in range(self.state_count)]
        for edge in self.edges:
            if 0 <= edge.src < self.state_count and 0 <= edge.dst < self.state_count:
                adjacency[edge.src].append((edge.action, edge.dst))
        self._adjacency_cache = adjacency
        return adjacency

    def edge_adjacency(self) -> list[list[Edge]]:
        cached = getattr(self, "_edge_adjacency_cache", None)
        if cached is not None:
            return cached
        adjacency = [[] for _ in range(self.state_count)]
        for edge in self.edges:
            if 0 <= edge.src < self.state_count and 0 <= edge.dst < self.state_count:
                adjacency[edge.src].append(edge)
        self._edge_adjacency_cache = adjacency
        return adjacency

    def edge_action_adjacency(self) -> list[dict[str, list[Edge]]]:
        cached = getattr(self, "_edge_action_adjacency_cache", None)
        if cached is not None:
            return cached
        adjacency: list[dict[str, list[Edge]]] = [dict() for _ in range(self.state_count)]
        for edge in self.edges:
            if 0 <= edge.src < self.state_count and 0 <= edge.dst < self.state_count:
                adjacency[edge.src].setdefault(edge.action, []).append(edge)
        self._edge_action_adjacency_cache = adjacency
        return adjacency

    def successors(self, state: int, action: str | None = None) -> list[Edge]:
        if not 0 <= state < self.state_count:
            return []
        if action is None:
            return list(self.edge_adjacency()[state])
        return list(self.edge_action_adjacency()[state].get(action, []))

    def apply_script(self, adds: Iterable[Edge], dels: Iterable[Edge]) -> "RepairLTS":
        adds = frozenset(adds)
        dels = frozenset(dels)
        if not adds and not dels:
            return self
        next_edges = set(self.edges)
        next_edges.difference_update(dels)
        next_edges.update(adds)
        return RepairLTS(self.initial, self.state_count, frozenset(next_edges))

    def write_aut(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii", newline="\n") as handle:
            handle.write(f"des ({self.initial},{len(self.edges)},{self.state_count})\n")
            for edge in sorted(self.edges):
                handle.write(f'({edge.src},"{escape_aut_label(edge.action)}",{edge.dst})\n')


def escape_aut_label(label: str) -> str:
    return label.replace("\\", "\\\\").replace('"', '\\"')


class EdgeOverlay:
    def __init__(self, base: RepairLTS, adds: frozenset[Edge], dels: frozenset[Edge]):
        self.base = base
        self.adds = adds
        self.dels = dels

    def __contains__(self, edge: Edge) -> bool:
        if edge in self.dels:
            return False
        if edge in self.adds:
            return True
        return edge in self.base.edges

    def __iter__(self):
        for edge in self.base.edges:
            if edge not in self.dels:
                yield edge
        for edge in self.adds:
            if edge not in self.base.edges or edge in self.dels:
                yield edge

    def __len__(self) -> int:
        removed = sum(1 for edge in self.dels if edge in self.base.edges)
        added = sum(1 for edge in self.adds if edge not in self.base.edges or edge in self.dels)
        return self.base.transition_count - removed + added


class OverlayAdjacency:
    def __init__(self, base: RepairLTS, adds: frozenset[Edge], dels: frozenset[Edge]):
        self.base = base
        self.adds_by_src: dict[int, list[Edge]] = {}
        for edge in adds:
            self.adds_by_src.setdefault(edge.src, []).append(edge)
        self.dels = dels
        self.memo: dict[int, list[tuple[str, int]]] = {}

    def __getitem__(self, state: int) -> list[tuple[str, int]]:
        cached = self.memo.get(state)
        if cached is not None:
            return cached
        values = [
            (edge.action, edge.dst)
            for edge in self.base.successors(state)
            if edge not in self.dels
        ]
        base_edges = self.base.edges
        for edge in self.adds_by_src.get(state, []):
            if edge in base_edges and edge not in self.dels:
                continue
            values.append((edge.action, edge.dst))
        self.memo[state] = values
        return values


class OverlayRepairLTS:
    def __init__(self, base: RepairLTS, adds: frozenset[Edge], dels: frozenset[Edge]):
        self.base = base
        self.adds = adds
        self.dels = dels
        self.initial = base.initial
        self.state_count = base.state_count
        self.edges = EdgeOverlay(base, adds, dels)
        self._actions_cache: set[str] | None = None
        self._action_counts_cache: Counter | None = None
        self._degree_cache: tuple[Counter, Counter] | None = None
        self._adjacency_cache: OverlayAdjacency | None = None
        self._adds_by_src: dict[int, list[Edge]] = {}
        for edge in adds:
            self._adds_by_src.setdefault(edge.src, []).append(edge)

    @property
    def transition_count(self) -> int:
        return len(self.edges)

    @property
    def actions(self) -> set[str]:
        if self._actions_cache is None:
            self._actions_cache = set(self.base.actions) | {edge.action for edge in self.adds}
        return self._actions_cache

    def action_counts(self) -> Counter:
        if self._action_counts_cache is None:
            counts = Counter(self.base.action_counts())
            for edge in self.dels:
                if edge in self.base.edges:
                    counts[edge.action] -= 1
            for edge in self.adds:
                if edge not in self.base.edges or edge in self.dels:
                    counts[edge.action] += 1
            self._action_counts_cache = counts
        return self._action_counts_cache

    def degree_counts(self) -> tuple[Counter, Counter]:
        if self._degree_cache is None:
            base_out, base_in = self.base.degree_counts()
            out_degree = Counter(base_out)
            in_degree = Counter(base_in)
            for edge in self.dels:
                if edge in self.base.edges:
                    out_degree[edge.src] -= 1
                    in_degree[edge.dst] -= 1
            for edge in self.adds:
                if edge not in self.base.edges or edge in self.dels:
                    out_degree[edge.src] += 1
                    in_degree[edge.dst] += 1
            self._degree_cache = (out_degree, in_degree)
        return self._degree_cache

    def degree_for(self, src: int, dst: int) -> tuple[int, int]:
        out_degree, in_degree = self.degree_counts()
        return out_degree[src], in_degree[dst]

    def adjacency(self) -> OverlayAdjacency:
        if self._adjacency_cache is None:
            self._adjacency_cache = OverlayAdjacency(self.base, self.adds, self.dels)
        return self._adjacency_cache

    def successors(self, state: int, action: str | None = None) -> list[Edge]:
        if not 0 <= state < self.state_count:
            return []
        values = [edge for edge in self.base.successors(state, action) if edge not in self.dels]
        base_edges = self.base.edges
        for edge in self._adds_by_src.get(state, []):
            if action is not None and edge.action != action:
                continue
            if edge in base_edges and edge not in self.dels:
                continue
            values.append(edge)
        return values


def repair_view(base: RepairLTS, adds: frozenset[Edge], dels: frozenset[Edge]):
    if not adds and not dels:
        return base
    return OverlayRepairLTS(base, adds, dels)


class ConcreteHMLChecker:
    def __init__(self, model: RepairLTS):
        self.model = model
        self.adjacency = model.adjacency()
        self.memo: dict[tuple[int, int], bool] = {}
        self.visited_entries = 0
        self.scanned_transitions = 0

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
            for action, dst in self.adjacency[state]:
                self.scanned_transitions += 1
                if action == formula.action and self.eval(dst, formula.left):
                    value = True
                    break
        elif formula.kind == "box":
            value = True
            for action, dst in self.adjacency[state]:
                self.scanned_transitions += 1
                if action == formula.action and not self.eval(dst, formula.left):
                    value = False
                    break
        else:
            raise ValueError(f"Unknown formula kind: {formula.kind}")

        self.memo[key] = value
        return value


class OverlayHMLChecker:
    def __init__(self, base: RepairLTS, adds: frozenset[Edge], dels: frozenset[Edge]):
        self.base = base
        self.base_adjacency = base.adjacency()
        self.adds_by_src: dict[int, list[Edge]] = {}
        for edge in adds:
            self.adds_by_src.setdefault(edge.src, []).append(edge)
        self.dels = dels
        self.memo: dict[tuple[int, int], bool] = {}
        self.visited_entries = 0
        self.scanned_transitions = 0

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
                if action != formula.action or Edge(state, action, dst) in self.dels:
                    continue
                self.scanned_transitions += 1
                if self.eval(dst, formula.left):
                    value = True
                    break
            if not value:
                for edge in self.adds_by_src.get(state, []):
                    if edge.action != formula.action:
                        continue
                    self.scanned_transitions += 1
                    if self.eval(edge.dst, formula.left):
                        value = True
                        break
        elif formula.kind == "box":
            value = True
            for action, dst in self.base_adjacency[state]:
                if action != formula.action or Edge(state, action, dst) in self.dels:
                    continue
                self.scanned_transitions += 1
                if not self.eval(dst, formula.left):
                    value = False
                    break
            if value:
                for edge in self.adds_by_src.get(state, []):
                    if edge.action != formula.action:
                        continue
                    self.scanned_transitions += 1
                    if not self.eval(edge.dst, formula.left):
                        value = False
                        break
        else:
            raise ValueError(f"Unknown formula kind: {formula.kind}")

        self.memo[key] = value
        return value


@dataclass
class QuotientSignature:
    state_to_block: tuple[int, ...]
    transitions: frozenset[tuple[int, str, int]]

    @property
    def block_count(self) -> int:
        return 0 if not self.state_to_block else max(self.state_to_block) + 1


def strong_v_quotient(model: RepairLTS, v_actions: set[str]) -> QuotientSignature:
    visible = [[] for _ in range(model.state_count)]
    for edge in model.edges:
        if edge.action not in v_actions:
            visible[edge.src].append((edge.action, edge.dst))
    for edges in visible:
        edges.sort()

    block = [0] * model.state_count
    while True:
        signature_to_block: OrderedDict[tuple, int] = OrderedDict()
        next_block = [0] * model.state_count
        for state in range(model.state_count):
            signature = tuple(sorted({(action, block[dst]) for action, dst in visible[state]}))
            if signature not in signature_to_block:
                signature_to_block[signature] = len(signature_to_block)
            next_block[state] = signature_to_block[signature]
        if next_block == block:
            quotient_edges = set()
            for edge in model.edges:
                if edge.action in v_actions:
                    continue
                quotient_edges.add((block[edge.src], edge.action, block[edge.dst]))
            return QuotientSignature(tuple(block), frozenset(quotient_edges))
        block = next_block


def quotient_drift(before: QuotientSignature, after: QuotientSignature, include_partition: bool) -> int:
    transition_drift = len(before.transitions.symmetric_difference(after.transitions))
    if not include_partition:
        return transition_drift
    return transition_drift + abs(before.block_count - after.block_count)


@dataclass
class CostConfig:
    w_add: float = 1.0
    w_del: float = 1.0
    lambda_add_non_v: float = 5.0
    lambda_del_non_v: float = 5.0
    quotient_weight: float = 10.0

    def for_setting(self, sf_setting: str) -> "CostConfig":
        if sf_setting == "no_sf":
            return CostConfig(
                w_add=self.w_add,
                w_del=self.w_del,
                lambda_add_non_v=0.0,
                lambda_del_non_v=0.0,
                quotient_weight=0.0,
            )
        return self


@dataclass
class RepairConfig:
    repair_mode: str = "add-delete"
    sf_setting: str = "strict_then_escalate"
    ranker: str = "heuristic"
    ranker_architecture: str = ""
    gnn_graph_mode: str = "dynamic"
    model_path: str = ""
    ranker_device: str = "cpu"
    strict_ranker_device: bool = False
    beam_width: int = 8
    max_iters: int = 16
    candidate_limit: int = 64
    candidate_state_limit: int = 256
    state_scan_limit: int = 5000
    search_strategy: str = "beam"
    minimal_layer_width: int = 2048
    minimal_seen_limit: int = 500_000
    dynamic_budget: bool = False
    dynamic_budget_rounds: int = 0
    dynamic_max_iters: int = 512
    dynamic_max_beam_width: int = 256
    dynamic_max_candidate_limit: int = 0
    dynamic_max_candidate_state_limit: int = 0
    dynamic_max_state_scan_limit: int = 0
    dynamic_max_minimal_layer_width: int = 32_768
    dynamic_max_minimal_seen_limit: int = 500_000
    dynamic_final_search_strategy: str = ""
    neural_prefilter_multiplier: int = 4
    neural_prefilter_limit: int = 512
    neural_linear_blend: float = 0.35
    neural_verify_frontier_only: bool = True
    neural_verify_top_k: int = 0
    max_case_seconds: float = 0.0
    max_quotient_drift: int = 1_000_000_000
    drift_mode: str = "estimate"
    exact_drift_max_transitions: int = 200_000
    store_final_model: bool = True
    postprocess: bool = True
    include_partition_drift: bool = False
    seed: int = 0
    search_progress_label: str = ""
    search_progress_every: int = 0
    costs: CostConfig = field(default_factory=CostConfig)


@dataclass
class ScriptMetrics:
    add_edges: int
    del_edges: int
    non_v_add_edges: int
    non_v_del_edges: int
    quotient_drift: int
    cost: float

    def objective_tuple(self) -> tuple[float, float, float]:
        non_v = self.non_v_add_edges + self.non_v_del_edges
        return (non_v, self.quotient_drift, self.cost)


@dataclass
class RepairResult:
    success: bool
    verified: bool
    final_model: RepairLTS
    adds: frozenset[Edge]
    dels: frozenset[Edge]
    raw_metrics: ScriptMetrics
    actual_metrics: ScriptMetrics
    verifier_calls: int
    cex_iters: int
    post_removed_add: int = 0
    post_restored_del: int = 0
    elapsed_ms: float = 0.0
    stage: str = ""
    message: str = ""

    def edit_script_json(self) -> dict:
        return {
            "success": self.success,
            "verified": self.verified,
            "stage": self.stage,
            "adds": [edge.to_json() for edge in sorted(self.adds)],
            "dels": [edge.to_json() for edge in sorted(self.dels)],
            "raw_metrics": self.raw_metrics.__dict__,
            "actual_metrics": self.actual_metrics.__dict__,
            "verifier_calls": self.verifier_calls,
            "cex_iters": self.cex_iters,
            "post_removed_add": self.post_removed_add,
            "post_restored_del": self.post_restored_del,
            "message": self.message,
        }


@dataclass(frozen=True)
class Candidate:
    edit: Edit
    symbolic_reason: int
    local_score: float
    formula_modal_depth: int = 0
    subformula_kind: str = ""
    required_modal_action: str = ""
    along_counterexample_path: bool = False
    dst_satisfies_next: bool = False
    current_edit_count: int = 0
    remaining_formula_depth: int = 0
    context_formula_kinds: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SearchNode:
    adds: frozenset[Edge] = frozenset()
    dels: frozenset[Edge] = frozenset()

    def key(self) -> tuple:
        return (tuple(sorted(self.adds)), tuple(sorted(self.dels)))

    def with_edit(self, edit: Edit) -> "SearchNode":
        adds = set(self.adds)
        dels = set(self.dels)
        if edit.op == "add":
            adds.add(edit.edge)
            dels.discard(edit.edge)
        elif edit.op == "del":
            dels.add(edit.edge)
            adds.discard(edit.edge)
        else:
            raise ValueError(f"Unknown edit op: {edit.op}")
        return SearchNode(frozenset(adds), frozenset(dels))


def parse_v_actions(text: str | None) -> set[str]:
    if not text:
        return set()
    normalized = text.replace(",", ":")
    return {piece.strip() for piece in normalized.split(":") if piece.strip()}


def choose_v_actions(model: RepairLTS, explicit: str | None, v_size: int, policy: str, excluded: set[str] | None = None) -> set[str]:
    excluded = excluded or set()
    explicit_actions = parse_v_actions(explicit) - excluded
    if explicit_actions:
        return explicit_actions
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


def formula_actions(formula: Formula) -> set[str]:
    return set(formula.actions())


def first_modal_action(formula: Formula) -> str:
    if formula.kind in {"diamond", "box"}:
        return formula.action
    if formula.left is not None:
        left = first_modal_action(formula.left)
        if left:
            return left
    if formula.right is not None:
        return first_modal_action(formula.right)
    return ""


def formula_is_diamond_true(formula: Formula) -> tuple[bool, str]:
    return (formula.kind == "diamond" and formula_is_tautology(formula.left), formula.action if formula.kind == "diamond" else "")


def formula_is_box_modality(formula: Formula) -> tuple[bool, str]:
    return (formula.kind == "box", formula.action if formula.kind == "box" else "")


def formula_is_tautology(formula: Formula) -> bool:
    return hml_semantic_tautology(formula)


def formula_is_contradiction(formula: Formula) -> bool:
    return hml_semantic_contradiction(formula)


def verify_formula(model: RepairLTS, formula: Formula) -> tuple[bool, ConcreteHMLChecker]:
    checker = ConcreteHMLChecker(model)
    return checker.eval(model.initial, formula), checker


def verify_formula_with_edits(
    base: RepairLTS,
    formula: Formula,
    adds: frozenset[Edge],
    dels: frozenset[Edge],
) -> tuple[bool, OverlayHMLChecker]:
    checker = OverlayHMLChecker(base, adds, dels)
    return checker.eval(base.initial, formula), checker


def edits_are_quotient_invisible(adds: frozenset[Edge], dels: frozenset[Edge], v_actions: set[str]) -> bool:
    return all(edge.action in v_actions for edge in adds) and all(edge.action in v_actions for edge in dels)


def should_compute_exact_drift(base: RepairLTS, config: RepairConfig, compute_drift: bool) -> bool:
    if not compute_drift:
        return False
    if config.drift_mode == "exact":
        return True
    if config.drift_mode == "auto":
        return base.transition_count <= config.exact_drift_max_transitions
    return False


def estimated_quotient_drift(adds: frozenset[Edge], dels: frozenset[Edge], v_actions: set[str]) -> int:
    return sum(1 for edge in adds if edge.action not in v_actions) + sum(1 for edge in dels if edge.action not in v_actions)


def script_metrics(
    base: RepairLTS,
    adds: frozenset[Edge],
    dels: frozenset[Edge],
    v_actions: set[str],
    original_quotient: QuotientSignature,
    config: RepairConfig,
    compute_drift: bool = True,
) -> ScriptMetrics:
    if not compute_drift or edits_are_quotient_invisible(adds, dels, v_actions):
        drift = 0
    elif should_compute_exact_drift(base, config, compute_drift):
        repaired = base.apply_script(adds, dels)
        repaired_quotient = strong_v_quotient(repaired, v_actions)
        drift = quotient_drift(original_quotient, repaired_quotient, config.include_partition_drift)
    else:
        drift = estimated_quotient_drift(adds, dels, v_actions)
    costs = config.costs.for_setting(config.sf_setting)
    non_v_add = sum(1 for edge in adds if edge.action not in v_actions)
    non_v_del = sum(1 for edge in dels if edge.action not in v_actions)
    cost = (
        costs.w_add * len(adds)
        + costs.w_del * len(dels)
        + costs.lambda_add_non_v * non_v_add
        + costs.lambda_del_non_v * non_v_del
        + costs.quotient_weight * drift
    )
    return ScriptMetrics(
        add_edges=len(adds),
        del_edges=len(dels),
        non_v_add_edges=non_v_add,
        non_v_del_edges=non_v_del,
        quotient_drift=drift,
        cost=cost,
    )


def compute_drift_during_search(config: RepairConfig) -> bool:
    return config.sf_setting != "no_sf" or config.max_quotient_drift < 1_000_000_000


def search_objective(metrics: ScriptMetrics, config: RepairConfig) -> tuple[float, float, float]:
    if config.sf_setting == "no_sf":
        return (0.0, 0.0, metrics.cost)
    return metrics.objective_tuple()


class Ranker:
    def score(self, candidate: Candidate, model: RepairLTS, v_actions: set[str]) -> float:
        return candidate.local_score

    def score_many(self, candidates: list[Candidate], model: RepairLTS, v_actions: set[str]) -> list[float]:
        return [self.score(candidate, model, v_actions) for candidate in candidates]


class RandomRanker(Ranker):
    def __init__(self, seed: int = 0):
        self.seed = int(seed)

    def score(self, candidate: Candidate, model: RepairLTS, v_actions: set[str]) -> float:
        edge = candidate.edit.edge
        key = f"{self.seed}|{candidate.edit.op}|{edge.src}|{edge.action}|{edge.dst}".encode("utf-8")
        raw = hashlib.blake2b(key, digest_size=8).digest()
        return int.from_bytes(raw, "big") / float(1 << 64)


def candidate_linear_prior_score(candidate: Candidate, model: RepairLTS, v_actions: set[str]) -> float:
    edge = candidate.edit.edge
    score = 0.05
    score += 0.35 * max(0.0, min(candidate.local_score, 1.0))
    score += 0.15 if candidate.symbolic_reason <= 1 else 0.08 if candidate.symbolic_reason <= 5 else 0.0
    score += 0.15 if candidate.dst_satisfies_next else 0.0
    score += 0.12 if candidate.required_modal_action and edge.action == candidate.required_modal_action else 0.0
    score += 0.10 if candidate.along_counterexample_path else 0.0
    score += 0.05 if edge.action in v_actions else 0.0
    score -= 0.05 if candidate.edit.op == "del" and candidate.symbolic_reason >= 8 else 0.0
    return max(0.0, min(1.0, score))


CANDIDATE_FEATURE_ORDER = [
    "is_add",
    "is_delete",
    "action_in_V",
    "edge_present",
    "action_frequency",
    "source_out_degree",
    "dest_in_degree",
    "src_is_initial",
    "dst_is_initial",
    "self_loop",
    "reason_complete_diamond",
    "reason_break_diamond",
    "reason_box",
    "reason_generic",
    "symbolic_reason",
    "local_score",
    "formula_modal_depth",
    "subformula_is_diamond",
    "subformula_is_box",
    "subformula_is_not",
    "subformula_is_and",
    "subformula_is_or",
    "action_matches_required_modal",
    "along_counterexample_path",
    "dst_satisfies_next",
    "current_edit_count",
    "remaining_formula_depth",
]

LEGACY_CANDIDATE_FEATURE_ORDER = [
    "is_add",
    "is_delete",
    "action_in_V",
    "edge_present",
    "source_out_degree",
    "dest_in_degree",
    "symbolic_reason",
    "local_score",
]

GNN_NODE_FEATURE_DIM = 5
GNN_EDGE_FEATURE_DIM = 4


@dataclass
class CachedGnnBaseGraph:
    base: RepairLTS
    edge_to_index: dict[Edge, int]
    action_to_id: dict[str, int]
    base_action_count_by_name: dict[str, int]
    initial_column: object
    state_column: object
    base_out_degree: object
    base_in_degree: object
    base_self_loop_count: object
    base_action_counts: object
    base_edge_index: object
    base_edge_action_ids: object
    base_edge_in_v: object
    base_edge_self_loop: object
    base_edge_src_initial: object


def candidate_feature_values(
    candidate: Candidate,
    model: RepairLTS,
    v_actions: set[str],
    degree_counts: tuple[Counter, Counter] | None = None,
    action_counts: Counter | None = None,
) -> dict[str, float]:
    edge = candidate.edit.edge
    if degree_counts is not None:
        out_degrees, in_degrees = degree_counts
        out_degree = out_degrees[edge.src]
        in_degree = in_degrees[edge.dst]
    elif hasattr(model, "degree_for"):
        out_degree, in_degree = model.degree_for(edge.src, edge.dst)
    else:
        out_degrees, in_degrees = model.degree_counts()
        out_degree = out_degrees[edge.src]
        in_degree = in_degrees[edge.dst]
    counts = action_counts if action_counts is not None else model.action_counts()
    action_count = counts[edge.action]
    reason = candidate.edit.reason
    return {
        "is_add": 1.0 if candidate.edit.op == "add" else 0.0,
        "is_delete": 1.0 if candidate.edit.op == "del" else 0.0,
        "action_in_V": 1.0 if edge.action in v_actions else 0.0,
        "edge_present": 1.0 if edge in model.edges else 0.0,
        "action_frequency": min(action_count, 50) / 50.0,
        "source_out_degree": min(out_degree, 20) / 20.0,
        "dest_in_degree": min(in_degree, 20) / 20.0,
        "src_is_initial": 1.0 if edge.src == model.initial else 0.0,
        "dst_is_initial": 1.0 if edge.dst == model.initial else 0.0,
        "self_loop": 1.0 if edge.src == edge.dst else 0.0,
        "reason_complete_diamond": 1.0 if reason == "complete_diamond" else 0.0,
        "reason_break_diamond": 1.0 if reason == "break_diamond_witness" else 0.0,
        "reason_box": 1.0 if reason in {"remove_bad_box_successor", "break_box_with_counterexample"} else 0.0,
        "reason_generic": 1.0 if reason.startswith("generic_") else 0.0,
        "symbolic_reason": min(candidate.symbolic_reason, 10) / 10.0,
        "local_score": candidate.local_score,
        "formula_modal_depth": min(candidate.formula_modal_depth, 20) / 20.0,
        "subformula_is_diamond": 1.0 if "diamond" in candidate.context_formula_kinds or candidate.subformula_kind == "diamond" else 0.0,
        "subformula_is_box": 1.0 if "box" in candidate.context_formula_kinds or candidate.subformula_kind == "box" else 0.0,
        "subformula_is_not": 1.0 if "not" in candidate.context_formula_kinds or candidate.subformula_kind == "not" else 0.0,
        "subformula_is_and": 1.0 if "and" in candidate.context_formula_kinds or candidate.subformula_kind == "and" else 0.0,
        "subformula_is_or": 1.0 if "or" in candidate.context_formula_kinds or candidate.subformula_kind == "or" else 0.0,
        "action_matches_required_modal": 1.0 if candidate.required_modal_action and edge.action == candidate.required_modal_action else 0.0,
        "along_counterexample_path": 1.0 if candidate.along_counterexample_path else 0.0,
        "dst_satisfies_next": 1.0 if candidate.dst_satisfies_next else 0.0,
        "current_edit_count": min(candidate.current_edit_count, 64) / 64.0,
        "remaining_formula_depth": min(candidate.remaining_formula_depth, 20) / 20.0,
    }


def candidate_feature_vector(
    candidate: Candidate,
    model: RepairLTS,
    v_actions: set[str],
    feature_order: list[str] | tuple[str, ...] = CANDIDATE_FEATURE_ORDER,
    degree_counts: tuple[Counter, Counter] | None = None,
    action_counts: Counter | None = None,
) -> list[float]:
    values = candidate_feature_values(candidate, model, v_actions, degree_counts=degree_counts, action_counts=action_counts)
    return [values[name] for name in feature_order]


def candidate_feature_matrix(
    candidates: list[Candidate],
    model: RepairLTS,
    v_actions: set[str],
    feature_order: list[str] | tuple[str, ...] = CANDIDATE_FEATURE_ORDER,
) -> list[list[float]]:
    if not candidates:
        return []
    degree_counts = model.degree_counts()
    action_counts = model.action_counts()
    return [
        candidate_feature_vector(
            candidate,
            model,
            v_actions,
            feature_order=feature_order,
            degree_counts=degree_counts,
            action_counts=action_counts,
        )
        for candidate in candidates
    ]


def build_mlp(input_dim: int, hidden_dim: int, hidden_layers: int, torch_module):
    layers = []
    current_dim = input_dim
    for _index in range(max(1, hidden_layers)):
        layers.append(torch_module.nn.Linear(current_dim, hidden_dim))
        layers.append(torch_module.nn.ReLU())
        current_dim = hidden_dim
    layers.append(torch_module.nn.Linear(current_dim, 1))
    return torch_module.nn.Sequential(*layers)


def build_gnn_mlp(input_dim: int, hidden_dim: int, torch_module):
    return torch_module.nn.Sequential(
        torch_module.nn.Linear(input_dim, hidden_dim),
        torch_module.nn.ReLU(),
        torch_module.nn.Linear(hidden_dim, hidden_dim),
    )


class GraphCandidateRankerModule:
    def __new__(cls, candidate_dim: int, hidden_dim: int, hidden_layers: int, torch_module):
        class _GraphCandidateRanker(torch_module.nn.Module):
            def __init__(self):
                super().__init__()
                self.node_proj = torch_module.nn.Linear(GNN_NODE_FEATURE_DIM, hidden_dim)
                self.edge_mlps = torch_module.nn.ModuleList(
                    build_gnn_mlp((hidden_dim * 2) + GNN_EDGE_FEATURE_DIM, hidden_dim, torch_module)
                    for _layer in range(max(1, hidden_layers))
                )
                self.update_mlps = torch_module.nn.ModuleList(
                    build_gnn_mlp(hidden_dim * 2, hidden_dim, torch_module)
                    for _layer in range(max(1, hidden_layers))
                )
                self.score_head = torch_module.nn.Sequential(
                    torch_module.nn.Linear((hidden_dim * 2) + candidate_dim, hidden_dim),
                    torch_module.nn.ReLU(),
                    torch_module.nn.Linear(hidden_dim, 1),
                )

            def encode(self, node_features, edge_index, edge_features):
                h = torch_module.relu(self.node_proj(node_features))
                for edge_mlp, update_mlp in zip(self.edge_mlps, self.update_mlps):
                    agg = torch_module.zeros_like(h)
                    if edge_index.numel() > 0:
                        src = edge_index[0]
                        dst = edge_index[1]
                        msg_input = torch_module.cat([h[src], h[dst], edge_features], dim=1)
                        msg = edge_mlp(msg_input)
                        agg.index_add_(0, dst, msg)
                    h = torch_module.relu(update_mlp(torch_module.cat([h, agg], dim=1)))
                return h

            def forward(self, node_features, edge_index, edge_features, candidate_features, candidate_src, candidate_dst):
                h = self.encode(node_features, edge_index, edge_features)
                if h.numel() == 0:
                    h = torch_module.zeros((1, hidden_dim), dtype=node_features.dtype, device=node_features.device)
                max_state = h.shape[0] - 1
                src = candidate_src.clamp(0, max_state)
                dst = candidate_dst.clamp(0, max_state)
                candidate_input = torch_module.cat([h[src], h[dst], candidate_features], dim=1)
                return self.score_head(candidate_input)

        return _GraphCandidateRanker()


def graph_tensors_for_model(model: RepairLTS, v_actions: set[str], torch_module, device):
    out_degrees, in_degrees = model.degree_counts()
    action_counts = model.action_counts()
    node_rows = []
    denom = max(1, model.state_count - 1)
    self_loop_states = {edge.src for edge in model.edges if edge.src == edge.dst}
    for state in range(model.state_count):
        node_rows.append(
            [
                1.0 if state == model.initial else 0.0,
                min(out_degrees[state], 50) / 50.0,
                min(in_degrees[state], 50) / 50.0,
                state / denom,
                1.0 if state in self_loop_states else 0.0,
            ]
        )
    if not node_rows:
        node_rows = [[0.0] * GNN_NODE_FEATURE_DIM]
    edges = sorted(model.edges)
    edge_index_rows = [[edge.src, edge.dst] for edge in edges]
    edge_feature_rows = [
        [
            min(action_counts[edge.action], 50) / 50.0,
            1.0 if edge.action in v_actions else 0.0,
            1.0 if edge.src == edge.dst else 0.0,
            1.0 if edge.src == model.initial else 0.0,
        ]
        for edge in edges
    ]
    node_features = torch_module.tensor(node_rows, dtype=torch_module.float32, device=device)
    if edge_index_rows:
        edge_index = torch_module.tensor(edge_index_rows, dtype=torch_module.long, device=device).t().contiguous()
        edge_features = torch_module.tensor(edge_feature_rows, dtype=torch_module.float32, device=device)
    else:
        edge_index = torch_module.empty((2, 0), dtype=torch_module.long, device=device)
        edge_features = torch_module.empty((0, GNN_EDGE_FEATURE_DIM), dtype=torch_module.float32, device=device)
    return node_features, edge_index, edge_features


def candidate_tensors_for_gnn(
    candidates: list[Candidate],
    model: RepairLTS,
    v_actions: set[str],
    torch_module,
    device,
    feature_order: list[str] | tuple[str, ...] = CANDIDATE_FEATURE_ORDER,
):
    candidate_features = candidate_feature_matrix(candidates, model, v_actions, feature_order=feature_order)
    src = [candidate.edit.edge.src for candidate in candidates]
    dst = [candidate.edit.edge.dst for candidate in candidates]
    return (
        torch_module.tensor(candidate_features, dtype=torch_module.float32, device=device),
        torch_module.tensor(src, dtype=torch_module.long, device=device),
        torch_module.tensor(dst, dtype=torch_module.long, device=device),
    )


def torch_load_checkpoint(torch_module, path: str | Path, map_location: str = "cpu"):
    try:
        return torch_module.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch_module.load(path, map_location=map_location)


class NeuralRanker(Ranker):
    def __init__(self, model_path: str, device: str = "cpu", strict_device: bool = False, gnn_graph_mode: str = "dynamic"):
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            message = f"Requested neural ranker device '{device}', but CUDA is not available."
            if strict_device:
                raise SystemExit(message)
            print("Warning:", message, "Using CPU.")
            device = "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.gnn_graph_mode = gnn_graph_mode
        if self.gnn_graph_mode not in {"static", "dynamic"}:
            raise ValueError(f"Unknown GNN graph mode: {self.gnn_graph_mode}")
        self._graph_tensor_cache: OrderedDict[tuple[int, tuple[str, ...], str], tuple] = OrderedDict()
        self._graph_tensor_cache_limit = 4 if self.device.type == "cuda" else 16
        self._base_graph_cache: OrderedDict[tuple[int, int, tuple[str, ...], str], CachedGnnBaseGraph] = OrderedDict()
        self._base_graph_cache_limit = 4 if self.device.type == "cuda" else 16
        payload = torch_load_checkpoint(torch, model_path, map_location="cpu")
        self.feature_order = list(payload.get("feature_order") or CANDIDATE_FEATURE_ORDER)
        if self.feature_order not in [CANDIDATE_FEATURE_ORDER, LEGACY_CANDIDATE_FEATURE_ORDER]:
            raise ValueError("Neural ranker checkpoint feature order does not match current code.")
        self.architecture = payload.get("architecture", "linear")
        self.model = None
        self.weights = ()
        self.bias = 0.0
        if self.architecture == "mlp":
            hidden_dim = int(payload.get("hidden_dim", 64))
            hidden_layers = int(payload.get("hidden_layers", 2))
            self.model = build_mlp(len(self.feature_order), hidden_dim, hidden_layers, torch).to(self.device)
            self.model.load_state_dict(payload["model_state"])
            self.model.eval()
            self.weights_tensor = None
            self.bias_tensor = None
        elif self.architecture == "gnn":
            hidden_dim = int(payload.get("hidden_dim", 64))
            hidden_layers = int(payload.get("hidden_layers", 2))
            self.model = GraphCandidateRankerModule(len(self.feature_order), hidden_dim, hidden_layers, torch).to(self.device)
            self.model.load_state_dict(payload["model_state"])
            self.model.eval()
            self.weights_tensor = None
            self.bias_tensor = None
        else:
            weights_tensor = payload["weights"].float().view(-1)
            if len(weights_tensor) != len(self.feature_order):
                raise ValueError(f"Neural ranker weight count {len(weights_tensor)} does not match {len(self.feature_order)} features.")
            self.weights = tuple(float(value) for value in weights_tensor.cpu().tolist())
            self.bias = float(torch.as_tensor(payload.get("bias", 0.0), dtype=torch.float32).view(()).item())
            self.weights_tensor = weights_tensor.to(self.device)
            self.bias_tensor = torch.tensor(self.bias, dtype=torch.float32, device=self.device)

    def _graph_model_for_scoring(self, model: RepairLTS) -> RepairLTS:
        if self.gnn_graph_mode == "static" and hasattr(model, "base"):
            return model.base
        return model

    def _cached_base_graph(self, base: RepairLTS, v_actions: set[str]) -> CachedGnnBaseGraph:
        key = (id(base), id(base.edges), tuple(sorted(v_actions)), str(self.device))
        cached = self._base_graph_cache.get(key)
        if cached is not None:
            self._base_graph_cache.move_to_end(key)
            return cached

        action_counts = base.action_counts()
        out_degrees, in_degrees = base.degree_counts()
        edges = tuple(sorted(base.edges))
        action_names = sorted(action_counts)
        action_to_id = {action: index for index, action in enumerate(action_names)}
        edge_to_index = {edge: index for index, edge in enumerate(edges)}
        action_count_values = [float(action_counts[action]) for action in action_names]
        self_loop_counts = [0.0] * base.state_count
        for edge in edges:
            if edge.src == edge.dst:
                self_loop_counts[edge.src] += 1.0

        torch_module = self.torch
        denom = max(1, base.state_count - 1)
        initial_column = torch_module.tensor([1.0 if state == base.initial else 0.0 for state in range(base.state_count)], dtype=torch_module.float32, device=self.device)
        state_column = torch_module.tensor([state / denom for state in range(base.state_count)], dtype=torch_module.float32, device=self.device)
        base_out_degree = torch_module.tensor([float(out_degrees[state]) for state in range(base.state_count)], dtype=torch_module.float32, device=self.device)
        base_in_degree = torch_module.tensor([float(in_degrees[state]) for state in range(base.state_count)], dtype=torch_module.float32, device=self.device)
        base_self_loop_count = torch_module.tensor(self_loop_counts, dtype=torch_module.float32, device=self.device)
        base_action_counts = torch_module.tensor(action_count_values, dtype=torch_module.float32, device=self.device)

        if edges:
            base_edge_index = torch_module.tensor([[edge.src, edge.dst] for edge in edges], dtype=torch_module.long, device=self.device).t().contiguous()
            base_edge_action_ids = torch_module.tensor([action_to_id[edge.action] for edge in edges], dtype=torch_module.long, device=self.device)
            base_edge_in_v = torch_module.tensor([1.0 if edge.action in v_actions else 0.0 for edge in edges], dtype=torch_module.float32, device=self.device)
            base_edge_self_loop = torch_module.tensor([1.0 if edge.src == edge.dst else 0.0 for edge in edges], dtype=torch_module.float32, device=self.device)
            base_edge_src_initial = torch_module.tensor([1.0 if edge.src == base.initial else 0.0 for edge in edges], dtype=torch_module.float32, device=self.device)
        else:
            base_edge_index = torch_module.empty((2, 0), dtype=torch_module.long, device=self.device)
            base_edge_action_ids = torch_module.empty((0,), dtype=torch_module.long, device=self.device)
            base_edge_in_v = torch_module.empty((0,), dtype=torch_module.float32, device=self.device)
            base_edge_self_loop = torch_module.empty((0,), dtype=torch_module.float32, device=self.device)
            base_edge_src_initial = torch_module.empty((0,), dtype=torch_module.float32, device=self.device)

        cached = CachedGnnBaseGraph(
            base=base,
            edge_to_index=edge_to_index,
            action_to_id=action_to_id,
            base_action_count_by_name={action: int(action_counts[action]) for action in action_names},
            initial_column=initial_column,
            state_column=state_column,
            base_out_degree=base_out_degree,
            base_in_degree=base_in_degree,
            base_self_loop_count=base_self_loop_count,
            base_action_counts=base_action_counts,
            base_edge_index=base_edge_index,
            base_edge_action_ids=base_edge_action_ids,
            base_edge_in_v=base_edge_in_v,
            base_edge_self_loop=base_edge_self_loop,
            base_edge_src_initial=base_edge_src_initial,
        )
        self._base_graph_cache[key] = cached
        while len(self._base_graph_cache) > self._base_graph_cache_limit:
            self._base_graph_cache.popitem(last=False)
        return cached

    def _dynamic_overlay_graph_tensors(self, model: RepairLTS, v_actions: set[str]):
        cache = self._cached_base_graph(model.base, v_actions)
        torch_module = self.torch
        del_edges = [edge for edge in model.dels if edge in cache.edge_to_index]
        add_edges = sorted(edge for edge in model.adds if edge not in model.base.edges or edge in model.dels)

        out_degree = cache.base_out_degree.clone()
        in_degree = cache.base_in_degree.clone()
        self_loop_count = cache.base_self_loop_count.clone()
        action_counts = cache.base_action_counts.clone()

        out_indices: list[int] = []
        out_values: list[float] = []
        in_indices: list[int] = []
        in_values: list[float] = []
        loop_indices: list[int] = []
        loop_values: list[float] = []
        action_delta = Counter()

        for edge in del_edges:
            out_indices.append(edge.src)
            out_values.append(-1.0)
            in_indices.append(edge.dst)
            in_values.append(-1.0)
            if edge.src == edge.dst:
                loop_indices.append(edge.src)
                loop_values.append(-1.0)
            action_delta[edge.action] -= 1

        for edge in add_edges:
            out_indices.append(edge.src)
            out_values.append(1.0)
            in_indices.append(edge.dst)
            in_values.append(1.0)
            if edge.src == edge.dst:
                loop_indices.append(edge.src)
                loop_values.append(1.0)
            action_delta[edge.action] += 1

        if out_indices:
            out_degree.index_add_(
                0,
                torch_module.tensor(out_indices, dtype=torch_module.long, device=self.device),
                torch_module.tensor(out_values, dtype=torch_module.float32, device=self.device),
            )
            in_degree.index_add_(
                0,
                torch_module.tensor(in_indices, dtype=torch_module.long, device=self.device),
                torch_module.tensor(in_values, dtype=torch_module.float32, device=self.device),
            )
        if loop_indices:
            self_loop_count.index_add_(
                0,
                torch_module.tensor(loop_indices, dtype=torch_module.long, device=self.device),
                torch_module.tensor(loop_values, dtype=torch_module.float32, device=self.device),
            )

        action_indices: list[int] = []
        action_values: list[float] = []
        for action, delta in action_delta.items():
            action_id = cache.action_to_id.get(action)
            if action_id is not None and delta:
                action_indices.append(action_id)
                action_values.append(float(delta))
        if action_indices:
            action_counts.index_add_(
                0,
                torch_module.tensor(action_indices, dtype=torch_module.long, device=self.device),
                torch_module.tensor(action_values, dtype=torch_module.float32, device=self.device),
            )

        node_features = torch_module.stack(
            [
                cache.initial_column,
                torch_module.clamp(out_degree, min=0.0, max=50.0) / 50.0,
                torch_module.clamp(in_degree, min=0.0, max=50.0) / 50.0,
                cache.state_column,
                (self_loop_count > 0).to(dtype=torch_module.float32),
            ],
            dim=1,
        )

        edge_count = cache.base_edge_index.shape[1]
        if edge_count:
            base_action_frequency = torch_module.clamp(action_counts[cache.base_edge_action_ids], min=0.0, max=50.0) / 50.0
            base_edge_features = torch_module.stack(
                [
                    base_action_frequency,
                    cache.base_edge_in_v,
                    cache.base_edge_self_loop,
                    cache.base_edge_src_initial,
                ],
                dim=1,
            )
            if del_edges:
                keep = torch_module.ones(edge_count, dtype=torch_module.bool, device=self.device)
                del_indices = torch_module.tensor([cache.edge_to_index[edge] for edge in del_edges], dtype=torch_module.long, device=self.device)
                keep[del_indices] = False
                edge_index = cache.base_edge_index[:, keep]
                edge_features = base_edge_features[keep]
            else:
                edge_index = cache.base_edge_index
                edge_features = base_edge_features
        else:
            edge_index = cache.base_edge_index
            edge_features = torch_module.empty((0, GNN_EDGE_FEATURE_DIM), dtype=torch_module.float32, device=self.device)

        if add_edges:
            add_edge_index = torch_module.tensor([[edge.src, edge.dst] for edge in add_edges], dtype=torch_module.long, device=self.device).t().contiguous()
            add_feature_rows = []
            for edge in add_edges:
                action_count = cache.base_action_count_by_name.get(edge.action, 0) + action_delta[edge.action]
                add_feature_rows.append(
                    [
                        min(max(action_count, 0), 50) / 50.0,
                        1.0 if edge.action in v_actions else 0.0,
                        1.0 if edge.src == edge.dst else 0.0,
                        1.0 if edge.src == model.initial else 0.0,
                    ]
                )
            add_edge_features = torch_module.tensor(add_feature_rows, dtype=torch_module.float32, device=self.device)
            edge_index = torch_module.cat([edge_index, add_edge_index], dim=1)
            edge_features = torch_module.cat([edge_features, add_edge_features], dim=0)

        return node_features, edge_index, edge_features

    def _graph_tensors(self, model: RepairLTS, v_actions: set[str]):
        key = (id(model), tuple(sorted(v_actions)), str(self.device))
        cached = self._graph_tensor_cache.get(key)
        if cached is not None:
            self._graph_tensor_cache.move_to_end(key)
            return cached
        tensors = graph_tensors_for_model(model, v_actions, self.torch, self.device)
        self._graph_tensor_cache[key] = tensors
        while len(self._graph_tensor_cache) > self._graph_tensor_cache_limit:
            self._graph_tensor_cache.popitem(last=False)
        return tensors

    def score(self, candidate: Candidate, model: RepairLTS, v_actions: set[str]) -> float:
        return self.score_many([candidate], model, v_actions)[0]

    def score_many(self, candidates: list[Candidate], model: RepairLTS, v_actions: set[str]) -> list[float]:
        if not candidates:
            return []
        if self.architecture == "mlp":
            rows = candidate_feature_matrix(candidates, model, v_actions, feature_order=self.feature_order)
            with self.torch.inference_mode():
                x = self.torch.tensor(rows, dtype=self.torch.float32, device=self.device)
                scores = self.model(x).view(-1)
                return [float(value) for value in scores.detach().cpu().tolist()]
        if self.architecture == "gnn":
            with self.torch.inference_mode():
                if self.gnn_graph_mode == "dynamic" and hasattr(model, "base"):
                    node_features, edge_index, edge_features = self._dynamic_overlay_graph_tensors(model, v_actions)
                else:
                    graph_model = self._graph_model_for_scoring(model)
                    node_features, edge_index, edge_features = self._graph_tensors(graph_model, v_actions)
                candidate_features, candidate_src, candidate_dst = candidate_tensors_for_gnn(
                    candidates,
                    model,
                    v_actions,
                    self.torch,
                    self.device,
                    feature_order=self.feature_order,
                )
                scores = self.model(node_features, edge_index, edge_features, candidate_features, candidate_src, candidate_dst).view(-1)
                return [float(value) for value in scores.detach().cpu().tolist()]
        if self.device.type == "cpu":
            rows = candidate_feature_matrix(candidates, model, v_actions, feature_order=self.feature_order)
            return [sum(feature * weight for feature, weight in zip(row, self.weights)) + self.bias for row in rows]
        rows = candidate_feature_matrix(candidates, model, v_actions, feature_order=self.feature_order)
        with self.torch.inference_mode():
            x = self.torch.tensor(rows, dtype=self.torch.float32, device=self.device)
            scores = x.mv(self.weights_tensor) + self.bias_tensor
            return [float(value) for value in scores.detach().cpu().tolist()]


def make_ranker(config: RepairConfig) -> Ranker:
    if config.ranker == "neural":
        if not config.model_path:
            raise ValueError("--ranker neural requires --model-path")
        if not Path(config.model_path).exists():
            raise FileNotFoundError(config.model_path)
        return NeuralRanker(config.model_path, config.ranker_device, config.strict_ranker_device, config.gnn_graph_mode)
    if config.ranker == "random":
        return RandomRanker(config.seed)
    return Ranker()


def candidate_states(model: RepairLTS, config: RepairConfig) -> list[int]:
    if config.candidate_state_limit > 0:
        limit = min(model.state_count, config.candidate_state_limit)
    else:
        # Treat 0 as "auto" rather than "all states" for the hot repair path.
        # Explicitly set a large positive limit when a full scan is intended.
        scan_cap = config.state_scan_limit if config.state_scan_limit > 0 else model.state_count
        limit = min(model.state_count, scan_cap, 4096)
    states = list(range(limit))
    if model.initial not in states:
        states.insert(0, model.initial)
    return states if config.candidate_state_limit <= 0 else states[: config.candidate_state_limit]


def states_satisfying(
    model: RepairLTS,
    checker: ConcreteHMLChecker,
    formula: Formula,
    want_true: bool,
    config: RepairConfig,
) -> list[int]:
    limit = model.state_count if config.state_scan_limit <= 0 else min(model.state_count, config.state_scan_limit)
    found = []
    result_cap = config.candidate_state_limit if config.candidate_state_limit > 0 else min(4096, limit)
    preferred = candidate_states(model, config)
    preferred_set = set(preferred)
    for state in preferred:
        if checker.eval(state, formula) == want_true:
            found.append(state)
            if len(found) >= result_cap:
                return found
    for state in range(limit):
        if state in preferred_set:
            continue
        if checker.eval(state, formula) == want_true:
            found.append(state)
            if len(found) >= result_cap:
                return found
    return found


def candidate_generation_cap(config: RepairConfig) -> int:
    if config.candidate_limit > 0:
        return max(32, min(4096, config.candidate_limit * 4))
    if config.candidate_state_limit > 0:
        return max(64, min(4096, config.candidate_state_limit * 2))
    # A literal "keep every candidate" mode is theoretically complete but can
    # materialize |S| * |Act| candidates per node. Keep 0-valued CLI limits as
    # an automatic bounded mode instead of an unbounded one.
    return 4096


def fallback_candidate_states(model: RepairLTS, config: RepairConfig) -> list[int]:
    states = candidate_states(model, config)
    if not states:
        return states
    if config.candidate_limit > 0:
        limit = max(16, min(128, config.candidate_limit // 4 or 1))
    else:
        limit = 64
    return states[: min(len(states), limit)]


def candidates_full(candidates: list[Candidate], cap: int) -> bool:
    return cap > 0 and len(candidates) >= cap


def extend_bounded(candidates: list[Candidate], new_candidates: list[Candidate], cap: int) -> None:
    if not new_candidates or candidates_full(candidates, cap):
        return
    remaining = cap - len(candidates) if cap > 0 else len(new_candidates)
    candidates.extend(new_candidates[:remaining])


def add_candidate(
    candidates: list[Candidate],
    op: str,
    edge: Edge,
    reason: str,
    symbolic_reason: int,
    score: float,
    *,
    subformula: Formula | None = None,
    required_modal_action: str = "",
    along_counterexample_path: bool = False,
    dst_satisfies_next: bool = False,
    context_formula_kinds: frozenset[str] = frozenset(),
) -> None:
    candidates.append(
        Candidate(
            Edit(op, edge, reason),
            symbolic_reason,
            score,
            subformula_kind=subformula.kind if subformula is not None else "",
            required_modal_action=required_modal_action,
            along_counterexample_path=along_counterexample_path,
            dst_satisfies_next=dst_satisfies_next,
            remaining_formula_depth=subformula.modal_depth() if subformula is not None else 0,
            context_formula_kinds=context_formula_kinds,
        )
    )


def formula_guided_candidates(
    model: RepairLTS,
    formula: Formula,
    state: int,
    want_true: bool,
    config: RepairConfig,
    depth: int = 0,
    checker: ConcreteHMLChecker | None = None,
    context_formula_kinds: frozenset[str] = frozenset(),
) -> list[Candidate]:
    max_depth = max(4, formula.modal_depth() + 2)
    if depth > max_depth:
        return []
    if checker is None:
        checker = ConcreteHMLChecker(model)
    current = checker.eval(state, formula)
    if current == want_true:
        return []

    candidates: list[Candidate] = []
    cap = candidate_generation_cap(config)
    kind = formula.kind
    context_formula_kinds = context_formula_kinds | {kind}
    if kind in {"true", "false"}:
        return candidates
    if kind == "not":
        return formula_guided_candidates(model, formula.left, state, not want_true, config, depth + 1, checker, context_formula_kinds)
    if kind == "and":
        if want_true:
            for child in [formula.left, formula.right]:
                if not checker.eval(state, child):
                    extend_bounded(candidates, formula_guided_candidates(model, child, state, True, config, depth + 1, checker, context_formula_kinds), cap)
                    if candidates_full(candidates, cap):
                        return candidates
        else:
            for child in [formula.left, formula.right]:
                if checker.eval(state, child):
                    extend_bounded(candidates, formula_guided_candidates(model, child, state, False, config, depth + 1, checker, context_formula_kinds), cap)
                    if candidates_full(candidates, cap):
                        return candidates
        return candidates
    if kind == "or":
        if want_true:
            for child in [formula.left, formula.right]:
                if not checker.eval(state, child):
                    extend_bounded(candidates, formula_guided_candidates(model, child, state, True, config, depth + 1, checker, context_formula_kinds), cap)
                    if candidates_full(candidates, cap):
                        return candidates
        else:
            for child in [formula.left, formula.right]:
                if checker.eval(state, child):
                    extend_bounded(candidates, formula_guided_candidates(model, child, state, False, config, depth + 1, checker, context_formula_kinds), cap)
                    if candidates_full(candidates, cap):
                        return candidates
        return candidates
    if kind == "diamond":
        action = formula.action
        if want_true:
            good_states = states_satisfying(model, checker, formula.left, True, config)
            fallback_states = fallback_candidate_states(model, config)
            dsts = unique_ints(good_states + fallback_states)
            for dst in dsts:
                edge = Edge(state, action, dst)
                if edge not in model.edges:
                    dst_satisfies_next = checker.eval(dst, formula.left)
                    score = 1.0 if dst_satisfies_next else 0.35
                    add_candidate(
                        candidates,
                        "add",
                        edge,
                        "complete_diamond",
                        0,
                        score,
                        subformula=formula,
                        required_modal_action=action,
                        along_counterexample_path=True,
                        dst_satisfies_next=dst_satisfies_next,
                        context_formula_kinds=context_formula_kinds,
                    )
                    if candidates_full(candidates, cap):
                        return candidates
            for edge in model.successors(state, action):
                if not checker.eval(edge.dst, formula.left):
                    extend_bounded(candidates, formula_guided_candidates(model, formula.left, edge.dst, True, config, depth + 1, checker, context_formula_kinds), cap)
                    if candidates_full(candidates, cap):
                        return candidates
        else:
            for edge in model.successors(state, action):
                if checker.eval(edge.dst, formula.left):
                    add_candidate(
                        candidates,
                        "del",
                        edge,
                        "break_diamond_witness",
                        0,
                        1.0,
                        subformula=formula,
                        required_modal_action=action,
                        along_counterexample_path=True,
                        dst_satisfies_next=True,
                        context_formula_kinds=context_formula_kinds,
                    )
                    if candidates_full(candidates, cap):
                        return candidates
                    extend_bounded(candidates, formula_guided_candidates(model, formula.left, edge.dst, False, config, depth + 1, checker, context_formula_kinds), cap)
                    if candidates_full(candidates, cap):
                        return candidates
        return candidates
    if kind == "box":
        action = formula.action
        if want_true:
            for edge in model.successors(state, action):
                if not checker.eval(edge.dst, formula.left):
                    add_candidate(
                        candidates,
                        "del",
                        edge,
                        "remove_bad_box_successor",
                        0,
                        1.0,
                        subformula=formula,
                        required_modal_action=action,
                        along_counterexample_path=True,
                        dst_satisfies_next=False,
                        context_formula_kinds=context_formula_kinds,
                    )
                    if candidates_full(candidates, cap):
                        return candidates
                    extend_bounded(candidates, formula_guided_candidates(model, formula.left, edge.dst, True, config, depth + 1, checker, context_formula_kinds), cap)
                    if candidates_full(candidates, cap):
                        return candidates
        else:
            bad_states = states_satisfying(model, checker, formula.left, False, config)
            fallback_states = fallback_candidate_states(model, config)
            for dst in unique_ints(bad_states + fallback_states):
                edge = Edge(state, action, dst)
                if edge not in model.edges:
                    child_ok = checker.eval(dst, formula.left)
                    dst_satisfies_next = child_ok == want_true
                    score = 1.0 if not child_ok else 0.25
                    add_candidate(
                        candidates,
                        "add",
                        edge,
                        "break_box_with_counterexample",
                        0,
                        score,
                        subformula=formula,
                        required_modal_action=action,
                        along_counterexample_path=True,
                        dst_satisfies_next=dst_satisfies_next,
                        context_formula_kinds=context_formula_kinds,
                    )
                    if candidates_full(candidates, cap):
                        return candidates
            for edge in model.successors(state, action):
                if checker.eval(edge.dst, formula.left):
                    extend_bounded(candidates, formula_guided_candidates(model, formula.left, edge.dst, False, config, depth + 1, checker, context_formula_kinds), cap)
                    if candidates_full(candidates, cap):
                        return candidates
        return candidates
    return candidates


def unique_ints(values: Iterable[int]) -> list[int]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def generic_candidates(model: RepairLTS, formula: Formula, config: RepairConfig) -> list[Candidate]:
    actions = sorted(formula_actions(formula) | model.actions)
    required_action = first_modal_action(formula)
    if not actions:
        actions = ["tau"]
    states = candidate_states(model, config)
    exhaustive = False
    candidates: list[Candidate] = []
    for action in sorted(formula_actions(formula)):
        dst_pool = states if exhaustive else states[: min(8, len(states))]
        for dst in dst_pool:
            edge = Edge(model.initial, action, dst)
            if edge not in model.edges:
                add_candidate(candidates, "add", edge, "generic_formula_add", 5, 0.1, subformula=formula, required_modal_action=required_action)
        successor_pool = sorted(model.successors(model.initial, action))
        if not exhaustive:
            successor_pool = successor_pool[:16]
        for edge in successor_pool:
            add_candidate(candidates, "del", edge, "generic_formula_del", 5, 0.1, subformula=formula, required_modal_action=required_action)
    if not candidates:
        successor_pool = sorted(model.successors(model.initial))
        if not exhaustive:
            successor_pool = successor_pool[:16]
        for edge in successor_pool:
            add_candidate(candidates, "del", edge, "generic_del", 8, 0.01, subformula=formula, required_modal_action=required_action)
        action_pool = actions if exhaustive else actions[:4]
        dst_pool = states if exhaustive else states[:4]
        for action in action_pool:
            for dst in dst_pool:
                edge = Edge(model.initial, action, dst)
                if edge not in model.edges:
                    add_candidate(candidates, "add", edge, "generic_add", 8, 0.01, subformula=formula, required_modal_action=required_action)
    return candidates


def allowed_by_mode(edit: Edit, repair_mode: str) -> bool:
    if repair_mode == "add-only":
        return edit.op == "add"
    if repair_mode == "delete-only":
        return edit.op == "del"
    return edit.op in {"add", "del"}


def is_heavy_neural_ranker(ranker: Ranker) -> bool:
    return getattr(ranker, "architecture", "") in {"mlp", "gnn"}


def neural_prefilter_cap(config: RepairConfig, candidate_count: int) -> int:
    if candidate_count <= 0:
        return 0
    limit = max(1, config.neural_prefilter_limit)
    if config.candidate_limit > 0:
        limit = max(limit, config.candidate_limit * max(1, config.neural_prefilter_multiplier))
    return min(candidate_count, limit)


def filter_and_rank_candidates(
    base: RepairLTS,
    current: RepairLTS,
    node: SearchNode,
    formula: Formula,
    v_actions: set[str],
    stage: str,
    config: RepairConfig,
    ranker: Ranker,
) -> list[Candidate]:
    candidates = formula_guided_candidates(current, formula, current.initial, True, config)
    candidates.extend(generic_candidates(current, formula, config))

    dedup: dict[tuple[str, Edge], Candidate] = {}
    for candidate in candidates:
        candidate = replace(
            candidate,
            formula_modal_depth=formula.modal_depth(),
            current_edit_count=len(node.adds) + len(node.dels),
        )
        edit = candidate.edit
        if not allowed_by_mode(edit, config.repair_mode):
            continue
        if stage == "strict" and edit.edge.action not in v_actions:
            continue
        if edit.op == "add" and edit.edge in current.edges:
            continue
        if edit.op == "del" and edit.edge not in current.edges:
            continue
        if edit.op == "add" and edit.edge in node.adds:
            continue
        if edit.op == "del" and edit.edge in node.dels:
            continue
        key = (edit.op, edit.edge)
        existing = dedup.get(key)
        if existing is None or candidate.local_score > existing.local_score:
            dedup[key] = candidate

    ranked_candidates = list(dedup.values())
    linear_prior_scores = {
        candidate: candidate_linear_prior_score(candidate, current, v_actions)
        for candidate in ranked_candidates
    }

    def cheap_sort_key(candidate: Candidate) -> tuple:
        edge = candidate.edit.edge
        non_v_tier = 0 if config.sf_setting == "no_sf" or edge.action in v_actions else 1
        quotient_tier = 0 if config.sf_setting == "no_sf" or edge.action in v_actions else 1
        op_cost = config.costs.w_add if candidate.edit.op == "add" else config.costs.w_del
        return (
            non_v_tier,
            quotient_tier,
            op_cost,
            -linear_prior_scores[candidate],
            candidate.symbolic_reason,
            candidate.edit.op,
            edge.src,
            edge.action,
            edge.dst,
        )

    if is_heavy_neural_ranker(ranker):
        cap = neural_prefilter_cap(config, len(ranked_candidates))
        if cap > 0 and len(ranked_candidates) > cap:
            ranked_candidates = sorted(ranked_candidates, key=cheap_sort_key)[:cap]

    neural_scores = dict(zip(ranked_candidates, ranker.score_many(ranked_candidates, current, v_actions)))
    blend = max(0.0, min(1.0, config.neural_linear_blend if is_heavy_neural_ranker(ranker) else 0.0))
    ranker_scores = {
        candidate: (1.0 - blend) * neural_scores[candidate] + blend * linear_prior_scores[candidate]
        for candidate in ranked_candidates
    }

    def sort_key(candidate: Candidate) -> tuple:
        edge = candidate.edit.edge
        non_v_tier = 0 if config.sf_setting == "no_sf" or edge.action in v_actions else 1
        quotient_tier = 0 if config.sf_setting == "no_sf" or edge.action in v_actions else 1
        neural_score = ranker_scores[candidate]
        op_cost = config.costs.w_add if candidate.edit.op == "add" else config.costs.w_del
        return (non_v_tier, quotient_tier, op_cost, -neural_score, candidate.symbolic_reason, candidate.edit.op, edge.src, edge.action, edge.dst)

    ranked_candidates = sorted(ranked_candidates, key=sort_key)
    if config.candidate_limit <= 0:
        return ranked_candidates
    return ranked_candidates[: config.candidate_limit]


def _grow_limit(current: int, hard_limit: int, factor: int = 2) -> int:
    if hard_limit <= 0:
        return 0
    if current <= 0:
        return 0
    return min(hard_limit, max(current + 1, current * factor))


def dynamic_budget_configs(config: RepairConfig, base: RepairLTS, formula: Formula) -> list[RepairConfig]:
    configs = [config]
    if not config.dynamic_budget:
        return configs

    modal_depth_budget = max(config.max_iters, formula.modal_action_count() * 2)
    transition_budget = max(config.max_iters, min(512, max(64, base.transition_count + base.state_count)))
    max_iters = config.dynamic_max_iters if config.dynamic_max_iters > 0 else max(modal_depth_budget, transition_budget)
    max_beam_width = config.dynamic_max_beam_width if config.dynamic_max_beam_width > 0 else 0
    estimated_candidates = max(256, base.state_count * max(1, len(formula_actions(formula))))
    max_candidate_limit = config.dynamic_max_candidate_limit if config.dynamic_max_candidate_limit > 0 else min(4096, estimated_candidates)
    max_candidate_state_limit = config.dynamic_max_candidate_state_limit if config.dynamic_max_candidate_state_limit > 0 else base.state_count
    max_state_scan_limit = config.dynamic_max_state_scan_limit if config.dynamic_max_state_scan_limit > 0 else base.state_count
    max_layer_width = config.dynamic_max_minimal_layer_width if config.dynamic_max_minimal_layer_width > 0 else 0
    max_seen_limit = config.dynamic_max_minimal_seen_limit if config.dynamic_max_minimal_seen_limit > 0 else 0

    current = config
    round_index = 0
    while config.dynamic_budget_rounds <= 0 or round_index < config.dynamic_budget_rounds:
        next_config = replace(
            current,
            max_iters=_grow_limit(current.max_iters, max_iters),
            beam_width=_grow_limit(current.beam_width, max_beam_width),
            candidate_limit=_grow_limit(current.candidate_limit, max_candidate_limit),
            candidate_state_limit=_grow_limit(current.candidate_state_limit, max_candidate_state_limit),
            state_scan_limit=_grow_limit(current.state_scan_limit, max_state_scan_limit),
            minimal_layer_width=_grow_limit(current.minimal_layer_width, max_layer_width, factor=4),
            minimal_seen_limit=_grow_limit(current.minimal_seen_limit, max_seen_limit, factor=2),
        )
        if next_config == current:
            break
        configs.append(next_config)
        current = next_config
        round_index += 1

    final_strategy = config.dynamic_final_search_strategy.strip()
    if final_strategy and final_strategy != current.search_strategy:
        configs.append(
            replace(
                current,
                search_strategy=final_strategy,
                max_iters=max(current.max_iters, max_iters),
                beam_width=max(current.beam_width, max_beam_width) if max_beam_width > 0 else current.beam_width,
                candidate_limit=max_candidate_limit,
                candidate_state_limit=max_candidate_state_limit,
                state_scan_limit=max_state_scan_limit,
                minimal_layer_width=max_layer_width,
                minimal_seen_limit=max_seen_limit,
            )
        )
    return configs


def budget_description(config: RepairConfig) -> str:
    return (
        f"strategy={config.search_strategy}, max_iters={config.max_iters}, beam={config.beam_width}, "
        f"candidate_limit={config.candidate_limit}, candidate_state_limit={config.candidate_state_limit}, "
        f"state_scan_limit={config.state_scan_limit}, layer_width={config.minimal_layer_width}, "
        f"seen_limit={config.minimal_seen_limit}"
    )


def search_budget_message(verifier_calls: int, stage_start: float, config: RepairConfig) -> str:
    if config.max_case_seconds > 0 and time.perf_counter() - stage_start >= config.max_case_seconds:
        return f"Case time cap exhausted: {time.perf_counter() - stage_start:.1f}s/{config.max_case_seconds:.1f}s."
    return ""


def run_repair(
    base: RepairLTS,
    target_formula: Formula | str,
    v_actions: set[str],
    config: RepairConfig,
    original_quotient: QuotientSignature | None = None,
    ranker: Ranker | None = None,
    case_id: str = "",
) -> RepairResult:
    start = time.perf_counter()
    formula = HMLParser.parse(target_formula) if isinstance(target_formula, str) else target_formula
    if formula_is_contradiction(formula):
        metrics = script_metrics(
            base,
            frozenset(),
            frozenset(),
            v_actions,
            QuotientSignature(tuple(), frozenset()),
            config,
            compute_drift=False,
        )
        return RepairResult(
            success=False,
            verified=False,
            final_model=base,
            adds=frozenset(),
            dels=frozenset(),
            raw_metrics=metrics,
            actual_metrics=metrics,
            verifier_calls=0,
            cex_iters=0,
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
            stage=config.search_strategy,
            message="Target formula is syntactically unsatisfiable under HML tautology rules.",
        )
    if original_quotient is None and should_compute_exact_drift(base, config, compute_drift=config.sf_setting != "no_sf"):
        original_quotient = strong_v_quotient(base, v_actions)
    if original_quotient is None:
        original_quotient = QuotientSignature(tuple(), frozenset())
    if ranker is None:
        ranker = make_ranker(config)

    stages = stages_for_setting(config.sf_setting)
    total_calls_all = 0
    total_iters_all = 0
    last_message = ""
    failed_attempts: list[str] = []

    for attempt_index, attempt_config in enumerate(dynamic_budget_configs(config, base, formula), start=1):
        if config.max_case_seconds > 0 and time.perf_counter() - start >= config.max_case_seconds:
            last_message = f"Case time cap exhausted: {time.perf_counter() - start:.1f}s/{config.max_case_seconds:.1f}s."
            break
        run_config = attempt_config
        if attempt_index > 1:
            label = f" case={case_id}" if case_id else ""
            print(f"[stage2-dynamic]{label} retry={attempt_index} {budget_description(attempt_config)}", flush=True)
            progress_every = max(1, min(16, attempt_config.max_iters // 8 if attempt_config.max_iters > 0 else 1))
            run_config = replace(
                attempt_config,
                search_progress_label=(f"case={case_id} " if case_id else "") + f"retry={attempt_index}",
                search_progress_every=progress_every,
            )
        total_calls = 0
        total_iters = 0
        attempt_message = ""

        for stage in stages:
            stage_config = run_config
            if config.max_case_seconds > 0:
                remaining = config.max_case_seconds - (time.perf_counter() - start)
                if remaining <= 0:
                    attempt_message = f"Case time cap exhausted: {time.perf_counter() - start:.1f}s/{config.max_case_seconds:.1f}s."
                    break
                stage_config = replace(run_config, max_case_seconds=remaining)
            if stage_config.search_strategy == "neural_guided_minimal":
                result, calls, iters, message = neural_guided_minimal_search_stage(base, formula, v_actions, original_quotient, stage_config, ranker, stage)
            elif stage_config.search_strategy == "beam":
                result, calls, iters, message = beam_search_stage(base, formula, v_actions, original_quotient, stage_config, ranker, stage)
            else:
                raise ValueError(f"Unknown search_strategy: {stage_config.search_strategy}")
            total_calls += calls
            total_iters += iters
            if result is not None:
                total_calls_all += total_calls
                total_iters_all += total_iters
                result.verifier_calls = total_calls_all
                result.cex_iters = total_iters_all
                result.elapsed_ms = (time.perf_counter() - start) * 1000.0
                result.stage = stage
                if attempt_index > 1:
                    result.message = (
                        f"{result.message} Dynamic budget succeeded on attempt {attempt_index} "
                        f"after {attempt_index - 1} failed budget(s): {budget_description(attempt_config)}."
                    )
                return result
            attempt_message = message

        total_calls_all += total_calls
        total_iters_all += total_iters
        last_message = attempt_message
        failed_attempts.append(f"attempt {attempt_index}: {attempt_message}; {budget_description(run_config)}")
        if len(failed_attempts) > 8:
            failed_attempts.pop(0)
        gc.collect()
        if config.max_case_seconds > 0 and time.perf_counter() - start >= config.max_case_seconds:
            break

    empty_metrics = script_metrics(
        base,
        frozenset(),
        frozenset(),
        v_actions,
        original_quotient,
        config,
        compute_drift=config.sf_setting != "no_sf",
    )
    return RepairResult(
        success=False,
        verified=False,
        final_model=base,
        adds=frozenset(),
        dels=frozenset(),
        raw_metrics=empty_metrics,
        actual_metrics=empty_metrics,
        verifier_calls=total_calls_all,
        cex_iters=total_iters_all,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        stage="/".join(stages),
        message=(last_message or "No repair found within the configured budget.")
        + ((" Dynamic budget attempts: " + " | ".join(failed_attempts)) if failed_attempts else ""),
    )


def successful_result_from_node(
    base: RepairLTS,
    formula: Formula,
    node: SearchNode,
    metrics: ScriptMetrics,
    v_actions: set[str],
    original_quotient: QuotientSignature,
    config: RepairConfig,
    verifier_calls: int,
    cex_iters: int,
    message: str,
) -> tuple[RepairResult, int]:
    raw_metrics = (
        metrics
        if compute_drift_during_search(config)
        else script_metrics(base, node.adds, node.dels, v_actions, original_quotient, config)
    )
    final_adds, final_dels, actual_metrics, post_removed, post_restored, post_calls = postprocess_script(
        base,
        formula,
        node.adds,
        node.dels,
        v_actions,
        original_quotient,
        config,
    )
    final_model = base.apply_script(final_adds, final_dels) if config.store_final_model else base
    return (
        RepairResult(
            success=True,
            verified=True,
            final_model=final_model,
            adds=final_adds,
            dels=final_dels,
            raw_metrics=raw_metrics,
            actual_metrics=actual_metrics,
            verifier_calls=verifier_calls + post_calls,
            cex_iters=cex_iters,
            post_removed_add=post_removed,
            post_restored_del=post_restored,
            message=message,
        ),
        post_calls,
    )


def stages_for_setting(sf_setting: str) -> list[str]:
    if sf_setting == "strict_then_escalate":
        return ["strict", "escalate"]
    if sf_setting in {"no_sf", "soft_sf"}:
        return ["soft"]
    raise ValueError(f"Unknown sf_setting: {sf_setting}")


def neural_guided_minimal_search_stage(
    base: RepairLTS,
    formula: Formula,
    v_actions: set[str],
    original_quotient: QuotientSignature,
    config: RepairConfig,
    ranker: Ranker,
    stage: str,
) -> tuple[RepairResult | None, int, int, str]:
    stage_start = time.perf_counter()
    verifier_calls = 0
    cex_iters = 0
    initial_truth, _checker = verify_formula(base, formula)
    verifier_calls += 1
    if initial_truth:
        metrics = script_metrics(
            base,
            frozenset(),
            frozenset(),
            v_actions,
            original_quotient,
            config,
            compute_drift=config.sf_setting != "no_sf",
        )
        return (
            RepairResult(True, True, base, frozenset(), frozenset(), metrics, metrics, verifier_calls, cex_iters, message="Already satisfied."),
            verifier_calls,
            cex_iters,
            "Already satisfied.",
        )

    current_layer = [SearchNode()]
    seen = {current_layer[0].key()}
    best_objective: tuple[float, float, float] | None = None
    best_node: SearchNode | None = None
    measure_drift = compute_drift_during_search(config)
    layer_width = max(0, config.minimal_layer_width)
    seen_limit = max(0, config.minimal_seen_limit)

    for depth in range(1, config.max_iters + 1):
        cex_iters = depth
        next_items: list[tuple[tuple[float, float, float], int, tuple, SearchNode]] = []
        success_items: list[tuple[tuple[float, float, float], int, tuple, SearchNode, ScriptMetrics]] = []
        next_seen: set[tuple] = set()
        candidate_order = 0
        for node in current_layer:
            current = repair_view(base, node.adds, node.dels)
            candidates = filter_and_rank_candidates(base, current, node, formula, v_actions, stage, config, ranker)
            for candidate in candidates:
                budget_message = search_budget_message(verifier_calls, stage_start, config)
                if budget_message:
                    return None, verifier_calls, cex_iters, budget_message
                candidate_order += 1
                next_node = node.with_edit(candidate.edit)
                key = next_node.key()
                if key in seen or key in next_seen:
                    continue
                if seen_limit <= 0 or len(next_seen) < seen_limit:
                    next_seen.add(key)
                metrics = script_metrics(
                    base,
                    next_node.adds,
                    next_node.dels,
                    v_actions,
                    original_quotient,
                    config,
                    compute_drift=measure_drift,
                )
                if metrics.quotient_drift > config.max_quotient_drift:
                    continue
                objective = search_objective(metrics, config)
                truth, _checker = verify_formula_with_edits(base, formula, next_node.adds, next_node.dels)
                verifier_calls += 1
                if truth:
                    success_items.append((objective, candidate_order, key, next_node, metrics))
                    if layer_width > 0 and len(success_items) > layer_width * 2:
                        success_items.sort(key=lambda item: (item[0], item[1], item[2]))
                        del success_items[layer_width:]
                    continue
                next_items.append((objective, candidate_order, key, next_node))
                if layer_width > 0 and len(next_items) > layer_width * 2:
                    next_items.sort(key=lambda item: (item[0], item[1], item[2]))
                    del next_items[layer_width:]
                if best_objective is None or objective < best_objective:
                    best_objective = objective
                    best_node = next_node
        if success_items:
            success_items.sort(key=lambda item: (item[0], item[1], item[2]))
            _objective, _order, _key, success_node, success_metrics = success_items[0]
            result, post_calls = successful_result_from_node(
                base,
                formula,
                success_node,
                success_metrics,
                v_actions,
                original_quotient,
                config,
                verifier_calls,
                cex_iters,
                f"Verified minimal repair found at edit depth {depth}.",
            )
            verifier_calls += post_calls
            return result, verifier_calls, cex_iters, "Verified minimal repair found."
        if not next_items:
            return None, verifier_calls, cex_iters, "No admissible candidates remain."
        next_items.sort(key=lambda item: (item[0], item[1], item[2]))
        if layer_width > 0 and len(next_items) > layer_width:
            next_items = next_items[:layer_width]
        kept_keys = [key for _objective, _order, key, _node in next_items]
        if seen_limit > 0 and len(seen) + len(kept_keys) > seen_limit:
            seen = set(kept_keys)
        else:
            seen.update(kept_keys)
        current_layer = [node for _objective, _order, _key, node in next_items]
        if config.search_progress_every > 0 and (
            depth == 1 or depth % config.search_progress_every == 0 or depth == config.max_iters
        ):
            label = f" {config.search_progress_label}" if config.search_progress_label else ""
            print(
                f"[stage2-search]{label} strategy=neural_guided_minimal stage={stage} depth={depth}/{config.max_iters} "
                f"layer={len(current_layer)} seen={len(seen)} verifier_calls={verifier_calls}",
                flush=True,
            )

    if best_objective is not None and best_node is not None:
        edits = len(best_node.adds) + len(best_node.dels)
        return None, verifier_calls, cex_iters, f"Minimal search budget exhausted. Best objective tried: {best_objective}, edits={edits}."
    return None, verifier_calls, cex_iters, "Minimal search budget exhausted before producing any candidate."


def beam_search_stage(
    base: RepairLTS,
    formula: Formula,
    v_actions: set[str],
    original_quotient: QuotientSignature,
    config: RepairConfig,
    ranker: Ranker,
    stage: str,
) -> tuple[RepairResult | None, int, int, str]:
    stage_start = time.perf_counter()
    verifier_calls = 0
    cex_iters = 0
    initial_truth, _checker = verify_formula(base, formula)
    verifier_calls += 1
    if initial_truth:
        metrics = script_metrics(
            base,
            frozenset(),
            frozenset(),
            v_actions,
            original_quotient,
            config,
            compute_drift=config.sf_setting != "no_sf",
        )
        return (
            RepairResult(True, True, base, frozenset(), frozenset(), metrics, metrics, verifier_calls, cex_iters, message="Already satisfied."),
            verifier_calls,
            cex_iters,
            "Already satisfied.",
        )

    frontier = [SearchNode()]
    seen = {frontier[0].key()}
    best_objective: tuple[float, float, float] | None = None
    best_node: SearchNode | None = None
    measure_drift = compute_drift_during_search(config)
    seen_limit = max(0, config.minimal_seen_limit)
    delayed_verify = is_heavy_neural_ranker(ranker) and config.neural_verify_frontier_only

    for iteration in range(1, config.max_iters + 1):
        cex_iters = iteration
        next_items: list[tuple[tuple[float, float, float], SearchNode, ScriptMetrics]] = []
        next_seen: set[tuple] = set()
        for node in frontier:
            current = repair_view(base, node.adds, node.dels)
            candidates = filter_and_rank_candidates(base, current, node, formula, v_actions, stage, config, ranker)
            for candidate in candidates:
                budget_message = search_budget_message(verifier_calls, stage_start, config)
                if budget_message:
                    return None, verifier_calls, cex_iters, budget_message
                next_node = node.with_edit(candidate.edit)
                key = next_node.key()
                if key in seen or key in next_seen:
                    continue
                next_seen.add(key)
                if seen_limit <= 0 or len(seen) < seen_limit:
                    seen.add(key)
                metrics = script_metrics(
                    base,
                    next_node.adds,
                    next_node.dels,
                    v_actions,
                    original_quotient,
                    config,
                    compute_drift=measure_drift,
                )
                if metrics.quotient_drift > config.max_quotient_drift:
                    continue
                objective = search_objective(metrics, config)
                if not delayed_verify:
                    truth, _checker = verify_formula_with_edits(base, formula, next_node.adds, next_node.dels)
                    verifier_calls += 1
                    if truth:
                        result, post_calls = successful_result_from_node(
                            base,
                            formula,
                            next_node,
                            metrics,
                            v_actions,
                            original_quotient,
                            config,
                            verifier_calls,
                            cex_iters,
                            "Verified repair found.",
                        )
                        verifier_calls += post_calls
                        return result, verifier_calls, cex_iters, "Verified repair found."
                next_items.append((objective, next_node, metrics))
                frontier_width = max(1, config.beam_width)
                if len(next_items) > frontier_width * 4:
                    next_items.sort(key=lambda item: item[0])
                    del next_items[frontier_width * 2 :]
                if best_objective is None or objective < best_objective:
                    best_objective = objective
                    best_node = next_node
        if not next_items:
            return None, verifier_calls, cex_iters, "No admissible candidates remain."
        next_items.sort(key=lambda item: item[0])
        frontier_width = max(1, config.beam_width)
        kept_items = next_items[:frontier_width]
        if delayed_verify:
            verify_limit = config.neural_verify_top_k if config.neural_verify_top_k > 0 else frontier_width
            for _objective, candidate_node, candidate_metrics in kept_items[: max(1, verify_limit)]:
                budget_message = search_budget_message(verifier_calls, stage_start, config)
                if budget_message:
                    return None, verifier_calls, cex_iters, budget_message
                truth, _checker = verify_formula_with_edits(base, formula, candidate_node.adds, candidate_node.dels)
                verifier_calls += 1
                if truth:
                    result, post_calls = successful_result_from_node(
                        base,
                        formula,
                        candidate_node,
                        candidate_metrics,
                        v_actions,
                        original_quotient,
                        config,
                        verifier_calls,
                        cex_iters,
                        "Verified neural-frontier repair found.",
                    )
                    verifier_calls += post_calls
                    return result, verifier_calls, cex_iters, "Verified neural-frontier repair found."
        frontier = [node for _score, node, _metrics in kept_items]
        if seen_limit > 0 and len(seen) >= seen_limit:
            seen = {node.key() for node in frontier}
        if config.search_progress_every > 0 and (
            iteration == 1 or iteration % config.search_progress_every == 0 or iteration == config.max_iters
        ):
            label = f" {config.search_progress_label}" if config.search_progress_label else ""
            print(
                f"[stage2-search]{label} strategy=beam stage={stage} iter={iteration}/{config.max_iters} "
                f"frontier={len(frontier)} generated={len(next_items)} seen={len(seen)} verifier_calls={verifier_calls}"
                f"{' delayed_verify=1' if delayed_verify else ''}",
                flush=True,
            )

    if best_objective is not None and best_node is not None:
        edits = len(best_node.adds) + len(best_node.dels)
        return None, verifier_calls, cex_iters, f"Budget exhausted. Best objective tried: {best_objective}, edits={edits}."
    return None, verifier_calls, cex_iters, "Budget exhausted before producing any candidate."


def postprocess_script(
    base: RepairLTS,
    formula: Formula,
    adds: frozenset[Edge],
    dels: frozenset[Edge],
    v_actions: set[str],
    original_quotient: QuotientSignature,
    config: RepairConfig,
) -> tuple[frozenset[Edge], frozenset[Edge], ScriptMetrics, int, int, int]:
    current_adds = set(adds)
    current_dels = set(dels)
    calls = 0
    removed_add = 0
    restored_del = 0
    measure_drift = compute_drift_during_search(config)

    if not config.postprocess:
        metrics = script_metrics(base, frozenset(current_adds), frozenset(current_dels), v_actions, original_quotient, config)
        return frozenset(current_adds), frozenset(current_dels), metrics, removed_add, restored_del, calls

    improved = True
    while improved:
        improved = False
        for edge in sorted(list(current_adds)):
            trial_adds = frozenset(item for item in current_adds if item != edge)
            trial_dels = frozenset(current_dels)
            truth, _checker = verify_formula_with_edits(base, formula, trial_adds, trial_dels)
            calls += 1
            if not truth:
                continue
            trial_metrics = script_metrics(
                base,
                trial_adds,
                trial_dels,
                v_actions,
                original_quotient,
                config,
                compute_drift=measure_drift,
            )
            current_metrics = script_metrics(
                base,
                frozenset(current_adds),
                frozenset(current_dels),
                v_actions,
                original_quotient,
                config,
                compute_drift=measure_drift,
            )
            if trial_metrics.cost <= current_metrics.cost:
                current_adds.remove(edge)
                removed_add += 1
                improved = True
                break
        if improved:
            continue
        for edge in sorted(list(current_dels)):
            trial_adds = frozenset(current_adds)
            trial_dels = frozenset(item for item in current_dels if item != edge)
            truth, _checker = verify_formula_with_edits(base, formula, trial_adds, trial_dels)
            calls += 1
            if not truth:
                continue
            trial_metrics = script_metrics(
                base,
                trial_adds,
                trial_dels,
                v_actions,
                original_quotient,
                config,
                compute_drift=measure_drift,
            )
            current_metrics = script_metrics(
                base,
                frozenset(current_adds),
                frozenset(current_dels),
                v_actions,
                original_quotient,
                config,
                compute_drift=measure_drift,
            )
            if trial_metrics.cost <= current_metrics.cost:
                current_dels.remove(edge)
                restored_del += 1
                improved = True
                break

    metrics = script_metrics(base, frozenset(current_adds), frozenset(current_dels), v_actions, original_quotient, config)
    return frozenset(current_adds), frozenset(current_dels), metrics, removed_add, restored_del, calls


def dump_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
