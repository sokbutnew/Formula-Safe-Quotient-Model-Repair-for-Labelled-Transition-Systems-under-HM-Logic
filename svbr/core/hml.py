from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Formula:
    kind: str
    action: str | None = None
    left: object | None = None
    right: object | None = None

    def subformulas(self):
        result = OrderedDict()

        def visit(formula):
            result.setdefault(str(formula), formula)
            if formula.left is not None:
                visit(formula.left)
            if formula.right is not None:
                visit(formula.right)

        visit(self)
        return list(result.values())

    def actions(self) -> set[str]:
        result = set()

        def visit(formula):
            if formula.kind in {"diamond", "box"}:
                result.add(formula.action)
            if formula.left is not None:
                visit(formula.left)
            if formula.right is not None:
                visit(formula.right)

        visit(self)
        return result

    def modal_action_count(self) -> int:
        if self.kind in {"true", "false"}:
            return 0
        if self.kind == "not":
            return self.left.modal_action_count()
        if self.kind in {"and", "or"}:
            return self.left.modal_action_count() + self.right.modal_action_count()
        if self.kind in {"diamond", "box"}:
            return 1 + self.left.modal_action_count()
        raise ValueError(f"Unknown formula kind: {self.kind}")

    def modal_depth(self) -> int:
        if self.kind in {"true", "false"}:
            return 0
        if self.kind == "not":
            return self.left.modal_depth()
        if self.kind in {"and", "or"}:
            return max(self.left.modal_depth(), self.right.modal_depth())
        if self.kind in {"diamond", "box"}:
            return 1 + self.left.modal_depth()
        raise ValueError(f"Unknown formula kind: {self.kind}")

    def __str__(self) -> str:
        if self.kind == "true":
            return "true"
        if self.kind == "false":
            return "false"
        if self.kind == "not":
            return "!" + parenthesize(self.left)
        if self.kind == "and":
            return f"({self.left} & {self.right})"
        if self.kind == "or":
            return f"({self.left} | {self.right})"
        if self.kind == "diamond":
            return f"<{self.action}>{parenthesize(self.left)}"
        if self.kind == "box":
            return f"[{self.action}]{parenthesize(self.left)}"
        raise ValueError(f"Unknown formula kind: {self.kind}")


def parenthesize(formula: Formula) -> str:
    if formula.kind in {"true", "false", "not", "diamond", "box"}:
        return str(formula)
    return f"({formula})"


class HMLParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    @staticmethod
    def parse(text: str) -> Formula:
        parser = HMLParser(text)
        formula = parser.parse_or()
        parser.skip_ws()
        if parser.pos != len(parser.text):
            raise ValueError(f"Unexpected token at position {parser.pos}")
        return formula

    def parse_or(self):
        formula = self.parse_and()
        while True:
            self.skip_ws()
            if not self.consume("|"):
                return formula
            formula = Formula("or", left=formula, right=self.parse_and())

    def parse_and(self):
        formula = self.parse_unary()
        while True:
            self.skip_ws()
            if not self.consume("&"):
                return formula
            formula = Formula("and", left=formula, right=self.parse_unary())

    def parse_unary(self):
        self.skip_ws()
        if self.consume("!"):
            return Formula("not", left=self.parse_unary())
        if self.consume("<"):
            action = self.read_until(">")
            return Formula("diamond", action=action, left=self.parse_unary())
        if self.consume("["):
            action = self.read_until("]")
            return Formula("box", action=action, left=self.parse_unary())
        if self.consume("("):
            formula = self.parse_or()
            self.expect(")")
            return formula
        if self.consume_keyword("true") or self.consume_keyword("tt"):
            return Formula("true")
        if self.consume_keyword("false") or self.consume_keyword("ff"):
            return Formula("false")
        raise ValueError(f"Expected HML formula at position {self.pos}")

    def read_until(self, marker: str) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] != marker:
            self.pos += 1
        if self.pos >= len(self.text):
            raise ValueError(f"Missing {marker} after action")
        action = self.text[start : self.pos].strip()
        self.pos += 1
        if not action:
            raise ValueError("Action label cannot be empty")
        return action

    def expect(self, char: str) -> None:
        self.skip_ws()
        if not self.consume(char):
            raise ValueError(f"Expected {char} at position {self.pos}")

    def consume(self, char: str) -> bool:
        if self.text.startswith(char, self.pos):
            self.pos += len(char)
            return True
        return False

    def consume_keyword(self, keyword: str) -> bool:
        self.skip_ws()
        end = self.pos + len(keyword)
        if self.text[self.pos : end].lower() != keyword:
            return False
        if end < len(self.text) and (self.text[end].isalnum() or self.text[end] == "_"):
            return False
        self.pos = end
        return True

    def skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1


def hml_to_nnf(formula: Formula, negate: bool = False) -> Formula:
    if formula.kind == "true":
        return Formula("false") if negate else formula
    if formula.kind == "false":
        return Formula("true") if negate else formula
    if formula.kind == "not":
        return hml_to_nnf(formula.left, not negate)
    if formula.kind == "and":
        kind = "or" if negate else "and"
        return Formula(kind, left=hml_to_nnf(formula.left, negate), right=hml_to_nnf(formula.right, negate))
    if formula.kind == "or":
        kind = "and" if negate else "or"
        return Formula(kind, left=hml_to_nnf(formula.left, negate), right=hml_to_nnf(formula.right, negate))
    if formula.kind == "diamond":
        kind = "box" if negate else "diamond"
        return Formula(kind, action=formula.action, left=hml_to_nnf(formula.left, negate))
    if formula.kind == "box":
        kind = "diamond" if negate else "box"
        return Formula(kind, action=formula.action, left=hml_to_nnf(formula.left, negate))
    raise ValueError(f"Unknown formula kind: {formula.kind}")


def _flatten_formula(kind: str, formulas: tuple[Formula, ...]) -> list[Formula]:
    result: list[Formula] = []
    for formula in formulas:
        if formula.kind == kind:
            result.extend(_flatten_formula(kind, (formula.left, formula.right)))
        else:
            result.append(formula)
    return result


def hml_formula_is_satisfiable(formula: Formula) -> bool:
    """Return whether an HML formula has some pointed LTS model.

    HML has no atomic propositions here, so satisfiability is governed by
    boolean structure plus modal obligations.  In NNF, boxes constrain every
    successor for one action, while each diamond may choose its own successor.
    A conjunction is unsatisfiable exactly when a required diamond successor
    cannot satisfy its child together with all boxes for the same action.
    """

    nnf = hml_to_nnf(formula)

    @lru_cache(maxsize=None)
    def sat_one(text: str) -> bool:
        return sat_all((HMLParser.parse(text),))

    def sat_all(formulas: tuple[Formula, ...]) -> bool:
        items = _flatten_formula("and", formulas)
        if any(item.kind == "false" for item in items):
            return False
        items = [item for item in items if item.kind != "true"]
        if not items:
            return True

        for index, item in enumerate(items):
            if item.kind == "or":
                rest = tuple(items[:index] + items[index + 1 :])
                return sat_all((item.left,) + rest) or sat_all((item.right,) + rest)

        boxes_by_action: dict[str, list[Formula]] = {}
        diamonds: list[Formula] = []
        for item in items:
            if item.kind == "box":
                boxes_by_action.setdefault(item.action or "", []).append(item.left)
            elif item.kind == "diamond":
                diamonds.append(item)
            elif item.kind == "and":
                return sat_all(tuple(_flatten_formula("and", (item,)) + [other for other in items if other is not item]))
            elif item.kind == "not":
                return sat_one(str(hml_to_nnf(item)))
            else:
                raise ValueError(f"Expected NNF HML formula, got {item.kind}")

        for diamond in diamonds:
            obligations = tuple([diamond.left] + boxes_by_action.get(diamond.action or "", []))
            key = " & ".join(sorted(str(item) for item in obligations))
            if not sat_one(f"({key})" if len(obligations) > 1 else key):
                return False
        return True

    return sat_all((nnf,))


def hml_formula_is_contradiction(formula: Formula) -> bool:
    return not hml_formula_is_satisfiable(formula)


def hml_formula_is_tautology(formula: Formula) -> bool:
    return not hml_formula_is_satisfiable(hml_to_nnf(formula, negate=True))
