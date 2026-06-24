import unittest
from argparse import Namespace

from svbr.core import HMLParser, hml_formula_is_contradiction, hml_formula_is_satisfiable
from svbr.experiments.add_delete_prepare import build_v_sets, formula_suite_errors
from svbr.experiments.add_delete_prepared_run import validate_prepared_manifest
from svbr.experiments.formula_generation import generate_formula_cases
from svbr.repair.add_delete import Edge, RepairLTS


class FormulaGenerationTests(unittest.TestCase):
    def formula_args(self):
        return Namespace(
            formulas_per_model=30,
            easy_formula_count=5,
            medium_formula_count=10,
            hard_formula_count=15,
            known_formula_count=20,
            mixed_formula_count=10,
            formula_min_actions=5,
            formula_max_actions=10,
            min_unsatisfied_formulas=30,
        )

    def test_formula_suite_tracks_model_actions_and_missing_actions(self):
        model = RepairLTS.from_aut("data/hml_branching.aut")
        cases = generate_formula_cases(model, "branching", seed=5)

        self.assertEqual(30, len(cases))
        self.assertEqual(5, sum(1 for case in cases if case["difficulty"] == "easy"))
        self.assertEqual(10, sum(1 for case in cases if case["difficulty"] == "medium"))
        self.assertEqual(15, sum(1 for case in cases if case["difficulty"] == "hard"))
        self.assertEqual(20, sum(1 for case in cases if case["source"] == "existing_only"))
        self.assertEqual(10, sum(1 for case in cases if case["source"] == "mixed_existing_missing"))
        self.assertEqual(30, sum(1 for case in cases if not case["initial_satisfied"]))
        self.assertEqual(30, sum(1 for case in cases if case["repair_eligible"]))
        self.assertEqual(30, sum(1 for case in cases if case["required_unsatisfied"]))

        for case in cases:
            formula = HMLParser.parse(case["positive_formula"])
            self.assertGreaterEqual(formula.modal_action_count(), 5)
            self.assertLessEqual(formula.modal_action_count(), 10)
            self.assertTrue(hml_formula_is_satisfiable(formula))
            self.assertTrue(hml_formula_is_satisfiable(HMLParser.parse(case["negative_existential_target"])))
            self.assertTrue(hml_formula_is_satisfiable(HMLParser.parse(case["negative_universal_target"])))
            self.assertTrue(case["positive"]["has_diamond"])
            self.assertTrue(case["positive"]["has_box"])
            self.assertTrue(case["positive"]["has_conjunction"] or case["positive"]["has_disjunction"])

        for case in [case for case in cases if case["source"] == "existing_only"]:
            self.assertEqual(0, case["missing_action_count"])
            self.assertFalse(case["uses_missing_actions"])

        mixed_cases = [case for case in cases if case["source"] == "mixed_existing_missing"]
        self.assertGreater(sum(1 for case in mixed_cases if case["target_action_in_lts"]), 0)
        self.assertGreater(sum(1 for case in mixed_cases if not case["target_action_in_lts"]), 0)

        for case in mixed_cases:
            self.assertGreater(case["missing_action_count"], 0)
            self.assertGreater(case["known_action_count"], 0)
            self.assertTrue(case["uses_missing_actions"])
            self.assertEqual([], case["v_in_actions"])
            self.assertEqual([], case["v_out_actions"])

        self.assertEqual([], formula_suite_errors(model, cases, self.formula_args()))

    def test_hml_satisfiability_detects_modal_contradictions(self):
        self.assertFalse(hml_formula_is_satisfiable(HMLParser.parse("<a>true & [a]false")))
        self.assertFalse(hml_formula_is_satisfiable(HMLParser.parse("!(<a>true | [a]<b>true)")))
        self.assertFalse(hml_formula_is_satisfiable(HMLParser.parse("<a>false")))
        self.assertTrue(hml_formula_is_satisfiable(HMLParser.parse("[a]false")))
        self.assertTrue(hml_formula_is_satisfiable(HMLParser.parse("![a]false")))
        self.assertTrue(hml_formula_is_contradiction(HMLParser.parse("<a>true & [a]false")))

    def test_formula_suite_validation_rejects_mixed_target_collapse(self):
        model = RepairLTS.from_aut("data/hml_branching.aut")
        cases = generate_formula_cases(model, "branching", seed=5)
        for case in cases:
            if case["source"] == "mixed_existing_missing":
                case["target_action_in_lts"] = False

        errors = formula_suite_errors(model, cases, self.formula_args())
        self.assertTrue(any("mixed_first_action_diversity" in error for error in errors))

    def test_prepared_manifest_validation_rejects_wrong_v_actual_size(self):
        manifest = {
            "settings": {"v_sizes": "0,1,3,5"},
            "models": [
                {
                    "model_id": "bad",
                    "actions": ["a", "b"],
                    "v_sets": [
                        {"source": "v_size", "requested_size": 0, "v_actions": []},
                        {"source": "v_size", "requested_size": 1, "v_actions": ["a"]},
                        {"source": "v_size", "requested_size": 3, "v_actions": ["a"]},
                        {"source": "v_size", "requested_size": 5, "v_actions": ["a", "b"]},
                    ],
                    "formula_cases": [
                        {"formula_id": "f00", "source": "existing_only", "initial_satisfied": False}
                    ],
                }
            ],
        }

        with self.assertRaises(SystemExit):
            validate_prepared_manifest(manifest)

    def test_prepared_manifest_validation_rejects_old_formula_safe_v_labels(self):
        manifest = {
            "version": 3,
            "settings": {"v_sizes": "0", "formulas_per_model": 1, "mixed_formula_count": 0},
            "models": [
                {
                    "model_id": "old",
                    "actions": ["a", "b"],
                    "hml_safe_actions": ["a", "b"],
                    "v_sets": [
                        {"source": "formula_safe_v_size", "v_label": "old_v0", "requested_size": 0, "v_actions": []}
                    ],
                    "formula_cases": [
                        {
                            "formula_id": "f00",
                            "source": "existing_only",
                            "initial_satisfied": False,
                            "v_size_labels": {"0": "old_v0"},
                            "positive": {"formula_actions": ["a"]},
                        }
                    ],
                }
            ],
        }

        with self.assertRaises(SystemExit):
            validate_prepared_manifest(manifest)

    def test_reuses_actions_when_lts_has_small_action_alphabet(self):
        model = RepairLTS(
            initial=0,
            state_count=2,
            edges=frozenset(
                {
                    Edge(0, "a", 1),
                    Edge(1, "b", 0),
                }
            ),
        )
        cases = generate_formula_cases(
            model,
            "two_actions",
            formulas_per_model=3,
            known_formula_count=3,
            mixed_formula_count=0,
            easy_formula_count=1,
            medium_formula_count=1,
            hard_formula_count=1,
            min_unsatisfied_formulas=0,
            seed=9,
        )

        self.assertEqual(3, len(cases))
        for case in cases:
            formula = HMLParser.parse(case["positive_formula"])
            self.assertGreaterEqual(formula.modal_action_count(), 5)
            self.assertLessEqual(len(case["positive"]["formula_actions"]), 2)
            self.assertGreater(formula.modal_action_count(), len(case["positive"]["formula_actions"]))
            self.assertEqual(0, case["missing_action_count"])

    def test_min_unsatisfied_can_be_disabled_for_existing_only_suite(self):
        model = RepairLTS(
            initial=0,
            state_count=2,
            edges=frozenset(
                {
                    Edge(0, "a", 1),
                    Edge(1, "b", 0),
                }
            ),
        )
        cases = generate_formula_cases(
            model,
            "two_actions",
            formulas_per_model=3,
            known_formula_count=3,
            mixed_formula_count=0,
            easy_formula_count=1,
            medium_formula_count=1,
            hard_formula_count=1,
            min_unsatisfied_formulas=0,
            seed=9,
        )

        self.assertEqual(3, len(cases))
        self.assertTrue(all(case["source"] == "existing_only" for case in cases))

    def test_v_sizes_keep_requested_size_when_actions_are_fewer_than_five(self):
        model = RepairLTS(
            initial=0,
            state_count=2,
            edges=frozenset(
                {
                    Edge(0, "a", 1),
                    Edge(1, "b", 0),
                }
            ),
        )
        args = Namespace(v_sizes="0,1,3,5", v_policy="least-frequent", explicit_v=[])
        v_sets = build_v_sets(model, args, [])
        by_requested = {item["requested_size"]: item for item in v_sets if item["source"] == "v_size"}

        self.assertEqual({0, 1, 3, 5}, set(by_requested))
        self.assertEqual([], by_requested[0]["v_actions"])
        self.assertEqual(1, len(by_requested[1]["v_actions"]))
        self.assertEqual(2, len(by_requested[3]["v_actions"]))
        self.assertEqual(2, len(by_requested[5]["v_actions"]))

    def test_missing_only_formulas_do_not_put_missing_actions_in_v(self):
        model = RepairLTS(initial=0, state_count=1, edges=frozenset())
        cases = generate_formula_cases(
            model,
            "empty",
            formulas_per_model=3,
            known_formula_count=0,
            mixed_formula_count=3,
            easy_formula_count=1,
            medium_formula_count=1,
            hard_formula_count=1,
            seed=11,
        )

        self.assertEqual(3, len(cases))
        self.assertTrue(all(case["source"] == "generated_missing_only" for case in cases))
        self.assertTrue(all(case["v_in_actions"] == [] for case in cases))
        self.assertTrue(all(case["v_out_actions"] == [] for case in cases))

        args = Namespace(v_sizes="0,1,3,5", v_policy="least-frequent", explicit_v=[])
        v_sets = build_v_sets(model, args, cases)
        by_requested = {item["requested_size"]: item for item in v_sets if item["source"] == "formula_safe_v_size"}
        self.assertEqual({0, 1, 3, 5}, set(by_requested))
        self.assertTrue(all(item["v_actions"] == [] for item in by_requested.values()))

    def test_formula_safe_v_sets_exclude_all_formula_actions(self):
        model = RepairLTS(
            initial=0,
            state_count=4,
            edges=frozenset(
                {
                    Edge(0, "a", 1),
                    Edge(0, "b", 2),
                    Edge(0, "c", 3),
                    Edge(0, "d", 3),
                }
            ),
        )
        cases = [
            {
                "formula_id": "f00",
                "formula_actions": ["a", "c"],
                "positive": {"formula_actions": ["a", "c"]},
                "negative_existential_target_meta": {"formula_actions": ["d"]},
                "negative_universal_target_meta": {"formula_actions": ["c"]},
            }
        ]
        args = Namespace(v_sizes="0,1,3,5", v_policy="deterministic", explicit_v=[])

        v_sets = build_v_sets(model, args, cases)
        self.assertEqual({"positive", "negative_existential", "negative_universal"}, set(cases[0]["v_size_labels_by_kind"]))
        self.assertEqual({"0", "1", "3", "5"}, set(cases[0]["v_size_labels_by_kind"]["positive"]))
        by_label = {item["v_label"]: item for item in v_sets}

        positive_v3 = by_label[cases[0]["v_size_labels_by_kind"]["positive"]["3"]]
        neg_exist_v3 = by_label[cases[0]["v_size_labels_by_kind"]["negative_existential"]["3"]]
        neg_univ_v3 = by_label[cases[0]["v_size_labels_by_kind"]["negative_universal"]["3"]]

        self.assertTrue(set(positive_v3["v_actions"]).isdisjoint({"a", "c"}))
        self.assertEqual(["b", "d"], positive_v3["v_actions"])
        self.assertTrue(set(neg_exist_v3["v_actions"]).isdisjoint({"d"}))
        self.assertEqual(["a", "b", "c"], neg_exist_v3["v_actions"])
        self.assertTrue(set(neg_univ_v3["v_actions"]).isdisjoint({"c"}))
        self.assertEqual(["a", "b", "d"], neg_univ_v3["v_actions"])


if __name__ == "__main__":
    unittest.main()
