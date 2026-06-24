import unittest

from svbr.core import HMLParser
from svbr.repair import CostConfig, RepairConfig, RepairLTS, run_repair
from svbr.repair.add_delete import ConcreteHMLChecker


class AddDeleteRepairTests(unittest.TestCase):
    def test_positive_repair_adds_missing_witness_in_strict_first_stage(self):
        model = RepairLTS.from_aut("data/hml_deadlock.aut")
        formula = HMLParser.parse("<z>true")
        config = RepairConfig(
            repair_mode="add-only",
            sf_setting="strict_then_escalate",
            max_iters=4,
            candidate_state_limit=8,
            costs=CostConfig(),
        )

        result = run_repair(model, formula, {"z"}, config)

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(result.stage, "strict")
        self.assertEqual(result.actual_metrics.add_edges, 1)
        self.assertTrue(any(edge.action == "z" for edge in result.adds))
        self.assertTrue(ConcreteHMLChecker(result.final_model).eval(result.final_model.initial, formula))

    def test_negative_existential_repair_deletes_witness_in_strict_first_stage(self):
        model = RepairLTS.from_aut("data/hml_deadlock.aut")
        formula = HMLParser.parse("!<a>true")
        config = RepairConfig(
            repair_mode="delete-only",
            sf_setting="strict_then_escalate",
            max_iters=4,
            candidate_state_limit=8,
            costs=CostConfig(),
        )

        result = run_repair(model, formula, {"a"}, config)

        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(result.stage, "strict")
        self.assertEqual(result.actual_metrics.del_edges, 1)
        self.assertTrue(all(edge.action == "a" for edge in result.dels))
        self.assertTrue(ConcreteHMLChecker(result.final_model).eval(result.final_model.initial, formula))

    def test_strict_then_escalate_reports_non_v_edit(self):
        model = RepairLTS.from_aut("data/hml_deadlock.aut")
        formula = HMLParser.parse("<z>true")
        config = RepairConfig(
            repair_mode="add-only",
            sf_setting="strict_then_escalate",
            max_iters=4,
            candidate_state_limit=8,
            costs=CostConfig(lambda_add_non_v=5.0, quotient_weight=10.0),
        )

        result = run_repair(model, formula, {"a"}, config)

        self.assertTrue(result.success)
        self.assertEqual(result.stage, "escalate")
        self.assertEqual(result.actual_metrics.non_v_add_edges, 1)

    def test_add_only_cannot_break_existing_diamond(self):
        model = RepairLTS.from_aut("data/hml_deadlock.aut")
        formula = HMLParser.parse("!<a>true")
        config = RepairConfig(
            repair_mode="add-only",
            sf_setting="strict_then_escalate",
            max_iters=2,
            candidate_state_limit=8,
            costs=CostConfig(),
        )

        result = run_repair(model, formula, {"a"}, config)

        self.assertFalse(result.success)
        self.assertFalse(result.verified)

    def test_no_sf_candidate_choice_is_not_biased_by_v(self):
        model = RepairLTS.from_aut("data/hml_deadlock.aut")
        formula = HMLParser.parse("(<y>true | <z>true)")
        config = RepairConfig(
            repair_mode="add-only",
            sf_setting="no_sf",
            max_iters=2,
            candidate_state_limit=4,
            costs=CostConfig(),
        )

        result_y = run_repair(model, formula, {"y"}, config)
        result_z = run_repair(model, formula, {"z"}, config)

        self.assertTrue(result_y.success)
        self.assertTrue(result_z.success)
        self.assertEqual(result_y.adds, result_z.adds)


if __name__ == "__main__":
    unittest.main()
