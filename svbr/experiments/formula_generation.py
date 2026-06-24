from __future__ import annotations

import random
from collections import Counter

from svbr.core import Formula, HMLParser, hml_formula_is_satisfiable
from svbr.repair.add_delete import RepairLTS, first_modal_action, verify_formula


FORBIDDEN_ACTION_CHARS = set("<>[]\r\n")
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def is_hml_safe_action(action: str) -> bool:
    return bool(action.strip()) and not any(char in action for char in FORBIDDEN_ACTION_CHARS)


def modal_actions_in_order(formula: Formula) -> list[str]:
    actions: list[str] = []

    def visit(node: Formula) -> None:
        if node.kind in {"diamond", "box"}:
            actions.append(node.action or "")
        if node.left is not None:
            visit(node.left)
        if node.right is not None:
            visit(node.right)

    visit(formula)
    return actions


def formula_has_kind(formula: Formula, kind: str) -> bool:
    if formula.kind == kind:
        return True
    return bool(formula.left is not None and formula_has_kind(formula.left, kind)) or bool(
        formula.right is not None and formula_has_kind(formula.right, kind)
    )


def generated_missing_actions(existing_actions: set[str], model_id: str, count: int) -> list[str]:
    actions = []
    suffix = 0
    while len(actions) < count:
        candidate = f"missing_{model_id}_{suffix}"
        suffix += 1
        if candidate not in existing_actions and is_hml_safe_action(candidate):
            actions.append(candidate)
    return actions


def target_modal_count(difficulty: str, index: int, min_actions: int, max_actions: int) -> int:
    ranges = {
        "easy": (5, 6),
        "medium": (7, 8),
        "hard": (9, 10),
    }
    low, high = ranges[difficulty]
    low = max(low, min_actions)
    high = min(high, max_actions)
    if low > high:
        low, high = min_actions, max_actions
    return low + (index % (high - low + 1))


def allocate_mixed_by_difficulty(difficulty_counts: dict[str, int], mixed_count: int) -> dict[str, int]:
    total = sum(difficulty_counts.values())
    if total <= 0 or mixed_count <= 0:
        return {difficulty: 0 for difficulty in DIFFICULTY_ORDER}
    raw = {difficulty: difficulty_counts[difficulty] * mixed_count / total for difficulty in DIFFICULTY_ORDER}
    allocation = {difficulty: min(difficulty_counts[difficulty], int(raw[difficulty])) for difficulty in DIFFICULTY_ORDER}
    remaining = mixed_count - sum(allocation.values())
    remainders = sorted(
        DIFFICULTY_ORDER,
        key=lambda difficulty: (raw[difficulty] - int(raw[difficulty]), difficulty == "hard", difficulty == "medium"),
        reverse=True,
    )
    while remaining > 0:
        changed = False
        for difficulty in remainders:
            if allocation[difficulty] < difficulty_counts[difficulty]:
                allocation[difficulty] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    return allocation


def build_difficulty_schedule(
    easy_count: int,
    medium_count: int,
    hard_count: int,
    formulas_per_model: int,
    mixed_formula_count: int,
) -> list[tuple[str, bool]]:
    counts = {"easy": max(0, easy_count), "medium": max(0, medium_count), "hard": max(0, hard_count)}
    if sum(counts.values()) == 0:
        base = [("easy", False)] * 5 + [("medium", False)] * 10 + [("hard", False)] * 15
        counts = {"easy": 5, "medium": 10, "hard": 15}
    else:
        base = []
        for difficulty in DIFFICULTY_ORDER:
            base.extend([(difficulty, False)] * counts[difficulty])
    if formulas_per_model > 0:
        if len(base) < formulas_per_model:
            refill = list(base)
            while len(base) < formulas_per_model and refill:
                base.extend(refill)
        base = base[:formulas_per_model]
        counts = {difficulty: sum(1 for item, _missing in base if item == difficulty) for difficulty in DIFFICULTY_ORDER}

    mixed_targets = allocate_mixed_by_difficulty(counts, min(mixed_formula_count, len(base)))
    schedule: list[tuple[str, bool]] = []
    for difficulty in DIFFICULTY_ORDER:
        count = counts[difficulty]
        mixed = mixed_targets[difficulty]
        schedule.extend((difficulty, True) for _index in range(mixed))
        schedule.extend((difficulty, False) for _index in range(count - mixed))
    return schedule


def sample_actions(actions: list[str], count: int, rng: random.Random) -> list[str]:
    if not actions:
        raise ValueError("Cannot sample actions from an empty list")
    result = []
    bag = list(actions)
    while len(result) < count:
        rng.shuffle(bag)
        for action in bag:
            result.append(action)
            if len(result) == count:
                break
    return result


def action_sequence(
    existing_actions: list[str],
    missing_actions: list[str],
    count: int,
    use_missing: bool,
    rng: random.Random,
    prefer_missing_first: bool = False,
) -> list[str]:
    if not use_missing and existing_actions:
        return sample_actions(existing_actions, count, rng)
    if use_missing and existing_actions:
        missing_count = max(1, min(count - 1, count // 3))
        known_count = count - missing_count
        known = sample_actions(existing_actions, known_count, rng)
        missing = sample_actions(missing_actions, missing_count, rng)
        sequence = known + missing
        first = missing[0] if prefer_missing_first else sequence[0]
        rest = [action for index, action in enumerate(sequence) if not (action == first and index == sequence.index(first))]
        rng.shuffle(rest)
        if first not in existing_actions and not prefer_missing_first:
            first = sample_actions(existing_actions, 1, rng)[0]
        return [first] + rest
    return sample_actions(missing_actions, count, rng)


def build_chain(actions: list[str], variant: int, force_first: str | None = None) -> Formula:
    modalities: list[str] = []
    for index, _action in enumerate(actions):
        if index == 0 and force_first:
            modalities.append(force_first)
        elif (index + variant) % 3 == 0:
            modalities.append("box")
        else:
            modalities.append("diamond")
    if len(modalities) > 1 and len(set(modalities)) == 1:
        modalities[-1] = "box" if modalities[-1] == "diamond" else "diamond"

    formula = Formula("true")
    for action, modality in reversed(list(zip(actions, modalities))):
        formula = Formula(modality, action=action, left=formula)
    return formula


def build_box_chain(actions: list[str]) -> Formula:
    formula = Formula("true")
    for action in reversed(actions):
        formula = Formula("box", action=action, left=formula)
    return formula


def build_diamond_chain(actions: list[str]) -> Formula:
    formula = Formula("true")
    for action in reversed(actions):
        formula = Formula("diamond", action=action, left=formula)
    return formula


def action_path_exists(model: RepairLTS, actions: list[str]) -> bool:
    states = {model.initial}
    for action in actions:
        next_states = set()
        for state in states:
            for edge in model.successors(state, action):
                next_states.add(edge.dst)
        if not next_states:
            return False
        states = next_states
    return bool(states)


def find_action_path(
    model: RepairLTS,
    existing_actions: list[str],
    length: int,
    rng: random.Random,
    want_exists: bool,
) -> list[str] | None:
    for _attempt in range(512):
        candidate = sample_actions(existing_actions, length, rng)
        if action_path_exists(model, candidate) == want_exists:
            return candidate

    for action in existing_actions:
        candidate = [action] * length
        if action_path_exists(model, candidate) == want_exists:
            return candidate

    frontier = [[]]
    max_checked = 4096
    checked = 0
    while frontier and checked < max_checked:
        prefix = frontier.pop(0)
        if len(prefix) == length:
            checked += 1
            if action_path_exists(model, prefix) == want_exists:
                return prefix
            continue
        for action in existing_actions:
            frontier.append(prefix + [action])
    return None


def build_complex_formula(actions: list[str], difficulty: str, variant: int, force_first: str | None = None) -> Formula:
    count = len(actions)
    if count < 5:
        raise ValueError("Generated formulas require at least five modal action occurrences")

    if difficulty == "easy":
        left_count = max(2, count // 2)
        left = build_chain(actions[:left_count], variant, force_first)
        right = build_chain(actions[left_count:], variant + 1)
        op = "and" if variant % 2 == 0 else "or"
        return Formula(op, left=left, right=right)

    if difficulty == "medium":
        first_count = max(2, count // 3)
        second_count = max(2, (count - first_count) // 2)
        first = build_chain(actions[:first_count], variant, force_first)
        second = build_chain(actions[first_count : first_count + second_count], variant + 1)
        third = build_chain(actions[first_count + second_count :], variant + 2)
        inner = Formula("and" if variant % 2 == 0 else "or", left=second, right=third)
        outer = "or" if variant % 2 == 0 else "and"
        return Formula(outer, left=first, right=inner)

    first_count = 3
    second_count = 3
    if count - first_count - second_count < 2:
        second_count = 2
    first = build_chain(actions[:first_count], variant, force_first)
    second = build_chain(actions[first_count : first_count + second_count], variant + 1)
    third = build_chain(actions[first_count + second_count :], variant + 2)
    if variant % 2 == 1:
        third = Formula("not", left=third)
    return Formula("and", left=Formula("or", left=first, right=second), right=third)


def build_unsatisfied_mixed_positive(
    existing_actions: list[str],
    missing_actions: list[str],
    count: int,
    difficulty: str,
    variant: int,
    rng: random.Random,
    first_action_in_lts: bool | None = None,
) -> Formula:
    max_impossible_len = max(1, min(3, count - 2))
    missing_action = sample_actions(missing_actions, 1, rng)[0]

    if first_action_in_lts is False or not existing_actions:
        impossible_len = 1
    elif first_action_in_lts is True:
        impossible_len = rng.randint(2, max_impossible_len)
    else:
        impossible_len = rng.randint(1, max_impossible_len)

    if existing_actions and impossible_len > 1:
        prefix = sample_actions(existing_actions, impossible_len - 1, rng)
    else:
        prefix = []
    impossible_actions = prefix + [missing_action]
    tail_count = max(1, count - len(impossible_actions))
    if existing_actions:
        tail_actions = action_sequence(existing_actions, missing_actions, tail_count, True, rng)
    else:
        tail_actions = sample_actions(missing_actions, tail_count, rng)
    impossible = build_diamond_chain(impossible_actions)
    split = max(2, min(tail_count - 1, tail_count // 2))
    left = build_chain(tail_actions[:split], variant + 2)
    right = build_chain(tail_actions[split:], variant + 3)
    context = Formula("or", left=left, right=right)
    if first_action_in_lts is False:
        return Formula("and", left=impossible, right=context)
    if first_action_in_lts is True:
        if rng.random() < 0.5:
            return Formula("and", left=context, right=impossible)
        return Formula("and", left=impossible, right=context)
    if rng.random() < 0.5:
        return Formula("and", left=impossible, right=context)
    return Formula("and", left=context, right=impossible)


def build_unsatisfied_existing_positive(
    model: RepairLTS,
    existing_actions: list[str],
    count: int,
    difficulty: str,
    variant: int,
    rng: random.Random,
) -> Formula:
    left_count = max(2, min(count - 2, count // 2))
    right_count = count - left_count
    context = Formula("or", left=build_box_chain(sample_actions(existing_actions, right_count, rng)), right=Formula("false"))

    absent_path = find_action_path(model, existing_actions, left_count, rng, want_exists=False)
    if absent_path is not None:
        return Formula("and", left=build_diamond_chain(absent_path), right=context)

    present_path = find_action_path(model, existing_actions, left_count, rng, want_exists=True)
    if present_path is None:
        present_path = sample_actions(existing_actions, left_count, rng)
    inner = Formula("and", left=build_diamond_chain(present_path), right=context)
    return Formula("not", left=inner)


def describe_formula(text: str, model: RepairLTS, existing_actions: set[str]) -> dict:
    formula = HMLParser.parse(text)
    sequence = modal_actions_in_order(formula)
    counter = Counter(sequence)
    known_occurrences = sum(count for action, count in counter.items() if action in existing_actions)
    missing_occurrences = len(sequence) - known_occurrences
    initial_satisfied, _checker = verify_formula(model, formula)
    return {
        "text": str(formula),
        "modal_action_count": len(sequence),
        "modal_depth": formula.modal_depth(),
        "formula_actions": sorted(counter),
        "first_action": first_modal_action(formula),
        "known_action_count": known_occurrences,
        "missing_action_count": missing_occurrences,
        "uses_missing_actions": missing_occurrences > 0,
        "has_conjunction": formula_has_kind(formula, "and"),
        "has_disjunction": formula_has_kind(formula, "or"),
        "has_diamond": formula_has_kind(formula, "diamond"),
        "has_box": formula_has_kind(formula, "box"),
        "initial_satisfied": initial_satisfied,
        "formula_satisfiable": hml_formula_is_satisfiable(formula),
    }


def make_target_not(psi_text: str) -> str:
    return str(Formula("not", left=HMLParser.parse(psi_text)))


def choose_v_out_action(existing_actions: list[str], target_action: str) -> list[str]:
    for action in existing_actions:
        if action != target_action:
            return [action]
    return []


def generate_checked_positive_formula(
    model: RepairLTS,
    existing_set: set[str],
    existing_actions: list[str],
    missing_actions: list[str],
    difficulty: str,
    source: str,
    count: int,
    index: int,
    require_unsatisfied: bool,
    seen_texts: set[str],
    rng: random.Random,
    first_action_in_lts: bool | None = None,
) -> tuple[Formula, dict, int, bool]:
    attempts = 0
    fallback_used = False
    use_missing = source != "existing_only"
    max_attempts = 160

    while attempts < max_attempts:
        variant = index + attempts * 37
        if require_unsatisfied and source == "existing_only" and existing_actions:
            formula = build_unsatisfied_existing_positive(model, existing_actions, count, difficulty, variant, rng)
            fallback_used = True
        elif require_unsatisfied and source != "existing_only":
            formula = build_unsatisfied_mixed_positive(
                existing_actions,
                missing_actions,
                count,
                difficulty,
                variant,
                rng,
                first_action_in_lts=first_action_in_lts,
            )
            fallback_used = True
        else:
            actions = action_sequence(
                existing_actions,
                missing_actions,
                count,
                use_missing,
                rng,
                prefer_missing_first=source != "existing_only",
            )
            formula = build_complex_formula(actions, difficulty, variant, force_first="diamond")

        meta = describe_formula(str(formula), model, existing_set)
        duplicate = meta["text"] in seen_texts
        satisfied_ok = not require_unsatisfied or not meta["initial_satisfied"]
        satisfiable_ok = bool(meta["formula_satisfiable"])
        if satisfiable_ok and satisfied_ok and (not duplicate or attempts >= max_attempts // 2):
            return formula, meta, attempts + 1, fallback_used
        attempts += 1

    if require_unsatisfied and source == "existing_only" and existing_actions:
        formula = build_unsatisfied_existing_positive(model, existing_actions, count, difficulty, index + max_attempts, rng)
    else:
        formula = build_unsatisfied_mixed_positive(
            existing_actions,
            missing_actions,
            count,
            difficulty,
            index + max_attempts,
            rng,
            first_action_in_lts=first_action_in_lts,
        )
    meta = describe_formula(str(formula), model, existing_set)
    if not meta["formula_satisfiable"]:
        raise RuntimeError(f"Could not generate a satisfiable formula for {difficulty}/{source}")
    if require_unsatisfied and meta["initial_satisfied"]:
        raise RuntimeError(f"Could not generate an initially unsatisfied formula for {difficulty}/{source}")
    return formula, meta, attempts + 1, True


def generate_checked_negative_pair(
    model: RepairLTS,
    existing_set: set[str],
    existing_actions: list[str],
    missing_actions: list[str],
    difficulty: str,
    source: str,
    count: int,
    index: int,
    use_missing: bool,
    rng: random.Random,
    force_first: str,
) -> tuple[dict, dict]:
    max_attempts = 160
    for attempt in range(max_attempts):
        actions = action_sequence(
            existing_actions,
            missing_actions,
            count,
            use_missing,
            rng,
            prefer_missing_first=source == "mixed_existing_missing",
        )
        psi = build_complex_formula(actions, difficulty, index + attempt * 41, force_first=force_first)
        psi_meta = describe_formula(str(psi), model, existing_set)
        target_meta = describe_formula(make_target_not(psi_meta["text"]), model, existing_set)
        if target_meta["formula_satisfiable"]:
            return psi_meta, target_meta
    raise RuntimeError(f"Could not generate satisfiable negative target for {difficulty}/{source}/{force_first}")


def generate_formula_cases(
    model: RepairLTS,
    model_id: str,
    formulas_per_model: int = 30,
    known_formula_count: int = 20,
    mixed_formula_count: int = 10,
    easy_formula_count: int = 5,
    medium_formula_count: int = 10,
    hard_formula_count: int = 15,
    min_actions: int = 5,
    max_actions: int = 10,
    min_unsatisfied_formulas: int = 30,
    seed: int = 13,
) -> list[dict]:
    schedule = build_difficulty_schedule(easy_formula_count, medium_formula_count, hard_formula_count, formulas_per_model, mixed_formula_count)
    formulas_per_model = len(schedule)
    if not schedule:
        return []
    min_unsatisfied_formulas = min(max(0, min_unsatisfied_formulas), formulas_per_model)
    if min_actions < 1 or max_actions < min_actions:
        raise ValueError("--formula-min-actions/--formula-max-actions are invalid")
    if known_formula_count + mixed_formula_count != formulas_per_model:
        known_formula_count = formulas_per_model - mixed_formula_count

    existing_set = {action for action in model.actions if is_hml_safe_action(action)}
    existing_actions = sorted(existing_set)
    missing_actions = generated_missing_actions(existing_set, model_id, max(max_actions, 10) + 4)
    rng = random.Random(seed)
    cases: list[dict] = []
    seen_positive_texts: set[str] = set()
    mixed_first_action_flags: list[bool] = []
    if existing_actions:
        mixed_slots = sum(1 for _difficulty, scheduled_missing in schedule if scheduled_missing)
        if mixed_slots == 1:
            mixed_first_action_flags = [bool(rng.getrandbits(1))]
        elif mixed_slots > 1:
            existing_first_count = max(1, mixed_slots // 2)
            missing_first_count = max(1, mixed_slots - existing_first_count)
            if existing_first_count + missing_first_count > mixed_slots:
                existing_first_count -= 1
            mixed_first_action_flags = [True] * existing_first_count + [False] * missing_first_count
            rng.shuffle(mixed_first_action_flags)
    mixed_flag_index = 0

    for index, (difficulty, scheduled_missing) in enumerate(schedule):
        count = target_modal_count(difficulty, index, min_actions, max_actions)
        use_missing = scheduled_missing or not existing_actions
        source = "mixed_existing_missing" if use_missing and existing_actions else "existing_only"
        if not existing_actions:
            source = "generated_missing_only"
        first_action_in_lts: bool | None = None
        if source == "mixed_existing_missing" and mixed_flag_index < len(mixed_first_action_flags):
            first_action_in_lts = mixed_first_action_flags[mixed_flag_index]
            mixed_flag_index += 1

        require_unsatisfied = index < min_unsatisfied_formulas
        _positive_formula, positive, attempts, fallback_used = generate_checked_positive_formula(
            model,
            existing_set,
            existing_actions,
            missing_actions,
            difficulty,
            source,
            count,
            index,
            require_unsatisfied,
            seen_positive_texts,
            rng,
            first_action_in_lts=first_action_in_lts,
        )
        seen_positive_texts.add(positive["text"])

        neg_exist_source, neg_exist_target = generate_checked_negative_pair(
            model,
            existing_set,
            existing_actions,
            missing_actions,
            difficulty,
            source,
            count,
            index + 31,
            use_missing,
            rng,
            force_first="diamond",
        )
        neg_univ_source, neg_univ_target = generate_checked_negative_pair(
            model,
            existing_set,
            existing_actions,
            missing_actions,
            difficulty,
            source,
            count,
            index + 67,
            use_missing,
            rng,
            force_first="box",
        )

        target_action = positive["first_action"]

        cases.append(
            {
                "formula_id": f"f{index:02d}_{difficulty}_{source}",
                "difficulty": difficulty,
                "source": source,
                "positive_formula": positive["text"],
                "negative_existential_psi": neg_exist_source["text"],
                "negative_existential_target": neg_exist_target["text"],
                "negative_universal_psi": neg_univ_source["text"],
                "negative_universal_target": neg_univ_target["text"],
                "target_action": target_action,
                "target_action_in_lts": target_action in existing_set,
                "v_in_actions": [],
                "v_out_actions": [],
                "positive": positive,
                "negative_existential_source": neg_exist_source,
                "negative_existential_target_meta": neg_exist_target,
                "negative_universal_source": neg_univ_source,
                "negative_universal_target_meta": neg_univ_target,
                "modal_action_count": positive["modal_action_count"],
                "formula_actions": positive["formula_actions"],
                "known_action_count": positive["known_action_count"],
                "missing_action_count": positive["missing_action_count"],
                "uses_missing_actions": positive["uses_missing_actions"],
                "initial_satisfied": positive["initial_satisfied"],
                "repair_eligible": not positive["initial_satisfied"],
                "required_unsatisfied": require_unsatisfied,
                "forced_unsatisfied": require_unsatisfied,
                "generation_attempts": attempts,
                "fallback_unsatisfied_formula": fallback_used,
                "mixed_first_action_required_in_lts": first_action_in_lts,
            }
        )
    return cases
