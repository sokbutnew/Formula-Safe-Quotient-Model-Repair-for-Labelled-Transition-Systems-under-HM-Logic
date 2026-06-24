from __future__ import annotations

import unittest

from svbr.core import HMLParser
from svbr.repair.add_delete import (
    CostConfig,
    Edge,
    RepairConfig,
    RepairLTS,
    candidate_feature_values,
    dynamic_budget_configs,
    formula_guided_candidates,
)


class ContextualRankerTests(unittest.TestCase):
    def test_formula_guided_candidate_records_counterexample_context(self):
        model = RepairLTS(initial=0, state_count=2, edges=frozenset())
        formula = HMLParser.parse("!(<a>true | [b]false)")
        config = RepairConfig(candidate_state_limit=2, state_scan_limit=2, costs=CostConfig())

        candidates = formula_guided_candidates(model, formula, model.initial, True, config)

        self.assertTrue(candidates)
        candidate = next(item for item in candidates if item.edit.edge.action == "b")
        values = candidate_feature_values(candidate, model, set())
        self.assertEqual(values["subformula_is_not"], 1.0)
        self.assertEqual(values["subformula_is_or"], 1.0)
        self.assertEqual(values["subformula_is_box"], 1.0)
        self.assertEqual(values["action_matches_required_modal"], 1.0)
        self.assertEqual(values["along_counterexample_path"], 1.0)
        self.assertEqual(values["dst_satisfies_next"], 1.0)

    def test_zero_dynamic_rounds_stops_after_safety_ceiling_saturates(self):
        model = RepairLTS(initial=0, state_count=3, edges=frozenset({Edge(0, "a", 1)}))
        formula = HMLParser.parse("<a><b>true")
        config = RepairConfig(
            dynamic_budget=True,
            dynamic_budget_rounds=0,
            max_iters=1,
            beam_width=1,
            candidate_limit=1,
            candidate_state_limit=1,
            state_scan_limit=1,
            dynamic_max_iters=8,
            dynamic_max_beam_width=8,
            dynamic_max_candidate_limit=8,
            dynamic_max_candidate_state_limit=3,
            dynamic_max_state_scan_limit=3,
            dynamic_max_minimal_layer_width=32,
            dynamic_max_minimal_seen_limit=32,
            minimal_layer_width=1,
            minimal_seen_limit=1,
            costs=CostConfig(),
        )

        configs = dynamic_budget_configs(config, model, formula)

        self.assertGreater(len(configs), 1)
        self.assertLess(len(configs), 16)
        self.assertEqual(configs[-1].max_iters, 8)
        self.assertEqual(configs[-1].candidate_state_limit, 3)
        self.assertEqual(configs[-1].state_scan_limit, 3)


if __name__ == "__main__":
    unittest.main()
