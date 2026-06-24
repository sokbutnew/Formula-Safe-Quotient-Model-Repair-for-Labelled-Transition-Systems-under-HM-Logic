from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from svbr.experiments.materialize_repaired_aut import (
    add_candidates_for_quotient_edge,
    counterexample_guided_lift,
    lift_quotient_edits_to_concrete,
    path_guided_lift_candidates,
    verify_original_lifted,
    write_writeback_operations,
)
from svbr.core import HMLParser
from svbr.repair.add_delete import CostConfig, Edge, OverlayHMLChecker, RepairConfig, RepairLTS, run_repair


class MaterializeRepairedAutTests(unittest.TestCase):
    def test_diamond_lift_keeps_one_satisfying_successor_path(self):
        model = RepairLTS(
            initial=0,
            state_count=4,
            edges=frozenset({Edge(0, "x", 1), Edge(0, "x", 2)}),
        )
        state_to_block = (0, 1, 1, 2)
        checker = OverlayHMLChecker(model, frozenset(), frozenset())

        candidates = path_guided_lift_candidates(
            model,
            checker,
            HMLParser.parse("<x><b>true"),
            0,
            True,
            state_to_block,
            {0: [0], 1: [1, 2], 2: [3]},
            {(1, "b", 2)},
            set(),
            set(),
            set(),
        )

        self.assertEqual([candidate.edge for candidate in candidates], [Edge(1, "b", 3)])

    def test_box_lift_keeps_all_required_successor_paths(self):
        model = RepairLTS(
            initial=0,
            state_count=4,
            edges=frozenset({Edge(0, "x", 1), Edge(0, "x", 2)}),
        )
        state_to_block = (0, 1, 1, 2)
        checker = OverlayHMLChecker(model, frozenset(), frozenset())

        candidates = path_guided_lift_candidates(
            model,
            checker,
            HMLParser.parse("[x]<b>true"),
            0,
            True,
            state_to_block,
            {0: [0], 1: [1, 2], 2: [3]},
            {(1, "b", 2)},
            set(),
            set(),
            set(),
        )

        self.assertEqual(
            [candidate.edge for candidate in candidates],
            [Edge(1, "b", 3), Edge(2, "b", 3)],
        )
        self.assertEqual(
            [candidate.template_key for candidate in candidates],
            [(1, "b", 2), (1, "b", 2)],
        )

    def test_path_guided_add_keeps_one_destination_representative(self):
        model = RepairLTS(initial=0, state_count=4, edges=frozenset())
        state_to_block = (0, 1, 1, 1)
        checker = OverlayHMLChecker(model, frozenset(), frozenset())

        candidates = add_candidates_for_quotient_edge(
            model,
            checker,
            state_to_block,
            {1: [1, 2, 3]},
            {(0, "x", 1)},
            set(),
            set(),
            0,
            "x",
            HMLParser.parse("true"),
            True,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].edge, Edge(0, "x", 1))
        self.assertEqual(candidates[0].template_key, (0, "x", 1))

    def test_path_guided_add_uses_template_destination_block_not_unrelated_state(self):
        model = RepairLTS(
            initial=0,
            state_count=4,
            edges=frozenset({Edge(3, "z", 3)}),
        )
        state_to_block = (0, 1, 2, 3)
        checker = OverlayHMLChecker(model, frozenset(), frozenset())

        candidates = add_candidates_for_quotient_edge(
            model,
            checker,
            state_to_block,
            {1: [1], 2: [2], 3: [3]},
            {(0, "x", 2)},
            set(),
            set(),
            0,
            "x",
            HMLParser.parse("<z>true"),
            True,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].edge, Edge(0, "x", 2))
        self.assertEqual(candidates[0].template_key, (0, "x", 2))
        self.assertEqual(candidates[0].reason, "path_guided_add_fallback")

    def test_quotient_edit_maps_to_single_concrete_edge(self):
        model = RepairLTS(
            initial=0,
            state_count=4,
            edges=frozenset({Edge(1, "x", 3)}),
        )
        state_to_block = (0, 0, 1, 1)

        adds, dels = lift_quotient_edits_to_concrete(
            model,
            state_to_block,
            adds=frozenset({Edge(0, "y", 1)}),
            dels=frozenset({Edge(0, "x", 1)}),
        )

        self.assertEqual(adds, frozenset({Edge(0, "y", 2)}))
        self.assertEqual(dels, frozenset({Edge(1, "x", 3)}))

    def test_quotient_add_uses_next_missing_concrete_pair(self):
        model = RepairLTS(
            initial=0,
            state_count=4,
            edges=frozenset({Edge(0, "y", 2)}),
        )
        state_to_block = (0, 0, 1, 1)

        adds, dels = lift_quotient_edits_to_concrete(
            model,
            state_to_block,
            adds=frozenset({Edge(0, "y", 1)}),
            dels=frozenset(),
        )

        self.assertEqual(adds, frozenset({Edge(0, "y", 3)}))
        self.assertEqual(dels, frozenset())

    def test_quotient_add_prefers_requested_target_state_as_source(self):
        model = RepairLTS(
            initial=0,
            state_count=4,
            edges=frozenset(),
        )
        state_to_block = (0, 0, 1, 1)

        adds, dels = lift_quotient_edits_to_concrete(
            model,
            state_to_block,
            adds=frozenset({Edge(0, "y", 1)}),
            dels=frozenset(),
            target_state=1,
        )

        self.assertEqual(adds, frozenset({Edge(1, "y", 2)}))
        self.assertEqual(dels, frozenset())

    def test_counterexample_guided_lift_keeps_one_delete_per_quotient_edge(self):
        model = RepairLTS(
            initial=0,
            state_count=3,
            edges=frozenset({Edge(0, "x", 1), Edge(0, "x", 2)}),
        )
        state_to_block = (0, 1, 1)
        q_dels = frozenset({Edge(0, "x", 1)})

        adds, dels = lift_quotient_edits_to_concrete(
            model,
            state_to_block,
            adds=frozenset(),
            dels=q_dels,
            target_state=0,
        )
        self.assertEqual(dels, frozenset({Edge(0, "x", 1)}))

        guided_adds, guided_dels, iters = counterexample_guided_lift(
            model,
            state_to_block,
            q_adds=frozenset(),
            q_dels=q_dels,
            target_text="[x]false",
            target_state=0,
            concrete_adds=adds,
            concrete_dels=dels,
            max_iters=4,
            single_per_quotient_edge=True,
        )

        self.assertEqual(guided_adds, frozenset())
        self.assertEqual(guided_dels, frozenset({Edge(0, "x", 1)}))
        self.assertGreaterEqual(iters, 1)

        closure_adds, closure_dels, _iters = counterexample_guided_lift(
            model,
            state_to_block,
            q_adds=frozenset(),
            q_dels=q_dels,
            target_text="[x]false",
            target_state=0,
            concrete_adds=adds,
            concrete_dels=dels,
            max_iters=4,
            single_per_quotient_edge=False,
        )
        self.assertEqual(closure_adds, frozenset())
        self.assertEqual(closure_dels, frozenset({Edge(0, "x", 1), Edge(0, "x", 2)}))

    def test_counterexample_guided_lift_keeps_one_add_per_quotient_edge(self):
        model = RepairLTS(
            initial=0,
            state_count=5,
            edges=frozenset({Edge(0, "i", 1), Edge(0, "i", 2), Edge(1, "i", 4), Edge(2, "i", 4)}),
        )
        state_to_block = (0, 1, 1, 2, 3)
        q_adds = frozenset({Edge(1, "leader", 2)})

        adds, dels = lift_quotient_edits_to_concrete(
            model,
            state_to_block,
            adds=q_adds,
            dels=frozenset(),
            target_state=0,
        )
        self.assertEqual(adds, frozenset({Edge(1, "leader", 3)}))

        guided_adds, guided_dels, _iters = counterexample_guided_lift(
            model,
            state_to_block,
            q_adds=q_adds,
            q_dels=frozenset(),
            target_text="!(<i>[leader]<i>true)",
            target_state=0,
            concrete_adds=adds,
            concrete_dels=dels,
            max_iters=4,
            single_per_quotient_edge=True,
        )

        self.assertEqual(guided_adds, frozenset({Edge(1, "leader", 3)}))
        self.assertEqual(guided_dels, frozenset())
        self.assertEqual(verify_original_lifted(model, guided_adds, guided_dels, "!(<i>[leader]<i>true)", 0)[0], "NO")

        closure_adds, closure_dels, _iters = counterexample_guided_lift(
            model,
            state_to_block,
            q_adds=q_adds,
            q_dels=frozenset(),
            target_text="!(<i>[leader]<i>true)",
            target_state=0,
            concrete_adds=adds,
            concrete_dels=dels,
            max_iters=4,
            single_per_quotient_edge=False,
        )
        self.assertEqual(closure_adds, frozenset({Edge(1, "leader", 3), Edge(2, "leader", 3)}))
        self.assertEqual(closure_dels, frozenset())
        self.assertEqual(verify_original_lifted(model, closure_adds, closure_dels, "!(<i>[leader]<i>true)", 0)[0], "YES")

    def test_closure_lift_grows_beyond_initial_budget(self):
        model = RepairLTS(
            initial=0,
            state_count=7,
            edges=frozenset({
                Edge(0, "i", 1),
                Edge(0, "i", 2),
                Edge(0, "i", 3),
                Edge(0, "i", 4),
                Edge(1, "i", 6),
                Edge(2, "i", 6),
                Edge(3, "i", 6),
                Edge(4, "i", 6),
            }),
        )
        state_to_block = (0, 1, 1, 1, 1, 2, 3)
        q_adds = frozenset({Edge(1, "leader", 2)})
        concrete_adds, concrete_dels = lift_quotient_edits_to_concrete(
            model,
            state_to_block,
            adds=q_adds,
            dels=frozenset(),
            target_state=0,
        )

        guided_adds, guided_dels, iters = counterexample_guided_lift(
            model,
            state_to_block,
            q_adds=q_adds,
            q_dels=frozenset(),
            target_text="!(<i>[leader]<i>true)",
            target_state=0,
            concrete_adds=concrete_adds,
            concrete_dels=concrete_dels,
            max_iters=10,
            initial_budget=2,
        )

        self.assertEqual(guided_adds, frozenset({
            Edge(1, "leader", 5),
            Edge(2, "leader", 5),
            Edge(3, "leader", 5),
            Edge(4, "leader", 5),
        }))
        self.assertEqual(guided_dels, frozenset())
        self.assertGreater(iters, 2)
        self.assertEqual(verify_original_lifted(model, guided_adds, guided_dels, "!(<i>[leader]<i>true)", 0)[0], "YES")

    def test_counterexample_guided_lift_follows_overlay_added_edges(self):
        model = RepairLTS(
            initial=0,
            state_count=6,
            edges=frozenset({Edge(1, "b", 5), Edge(3, "d", 4)}),
        )
        state_to_block = (0, 1, 2, 3, 4, 2)
        q_adds = frozenset({Edge(0, "a", 1), Edge(2, "c", 3)})

        adds, dels = lift_quotient_edits_to_concrete(
            model,
            state_to_block,
            adds=q_adds,
            dels=frozenset(),
            target_state=0,
        )
        self.assertEqual(adds, frozenset({Edge(0, "a", 1), Edge(2, "c", 3)}))
        self.assertEqual(verify_original_lifted(model, adds, dels, "<a><b><c><d>true", 0)[0], "NO")

        guided_adds, guided_dels, _iters = counterexample_guided_lift(
            model,
            state_to_block,
            q_adds=q_adds,
            q_dels=frozenset(),
            target_text="<a><b><c><d>true",
            target_state=0,
            concrete_adds=adds,
            concrete_dels=dels,
            max_iters=4,
        )

        self.assertEqual(guided_adds, frozenset({Edge(0, "a", 1), Edge(5, "c", 3)}))
        self.assertEqual(guided_dels, frozenset())
        self.assertEqual(verify_original_lifted(model, guided_adds, guided_dels, "<a><b><c><d>true", 0)[0], "YES")

    def test_existential_fallback_extends_current_path_before_trying_siblings(self):
        model = RepairLTS(
            initial=0,
            state_count=6,
            edges=frozenset({Edge(4, "c", 5)}),
        )
        state_to_block = (0, 1, 1, 1, 2, 3)
        q_adds = frozenset({Edge(0, "a", 1), Edge(1, "b", 2)})
        concrete_adds, concrete_dels = lift_quotient_edits_to_concrete(
            model,
            state_to_block,
            adds=q_adds,
            dels=frozenset(),
            target_state=0,
        )

        guided_adds, guided_dels, iters = counterexample_guided_lift(
            model,
            state_to_block,
            q_adds=q_adds,
            q_dels=frozenset(),
            target_text="<a><b><c>true",
            target_state=0,
            concrete_adds=concrete_adds,
            concrete_dels=concrete_dels,
            max_iters=2,
        )

        self.assertEqual(guided_adds, frozenset({Edge(0, "a", 1), Edge(1, "b", 4)}))
        self.assertEqual(guided_dels, frozenset())
        self.assertEqual(iters, 2)
        self.assertEqual(verify_original_lifted(model, guided_adds, guided_dels, "<a><b><c>true", 0)[0], "YES")

    def test_negated_diamond_is_universal_delete_obligation(self):
        model = RepairLTS(
            initial=0,
            state_count=3,
            edges=frozenset({Edge(0, "x", 1), Edge(0, "x", 2)}),
        )
        state_to_block = (0, 1, 1)
        q_dels = frozenset({Edge(0, "x", 1)})
        concrete_adds, concrete_dels = lift_quotient_edits_to_concrete(
            model,
            state_to_block,
            adds=frozenset(),
            dels=q_dels,
            target_state=0,
        )

        guided_adds, guided_dels, _iters = counterexample_guided_lift(
            model,
            state_to_block,
            q_adds=frozenset(),
            q_dels=q_dels,
            target_text="!<x>true",
            target_state=0,
            concrete_adds=concrete_adds,
            concrete_dels=concrete_dels,
            max_iters=4,
        )

        self.assertEqual(guided_adds, frozenset())
        self.assertEqual(guided_dels, frozenset({Edge(0, "x", 1), Edge(0, "x", 2)}))
        self.assertEqual(verify_original_lifted(model, guided_adds, guided_dels, "!<x>true", 0)[0], "YES")

    def test_negated_box_is_existential_add_obligation(self):
        model = RepairLTS(initial=0, state_count=3, edges=frozenset())
        state_to_block = (0, 1, 1)
        q_adds = frozenset({Edge(0, "x", 1)})
        concrete_adds, concrete_dels = lift_quotient_edits_to_concrete(
            model,
            state_to_block,
            adds=q_adds,
            dels=frozenset(),
            target_state=0,
        )

        guided_adds, guided_dels, _iters = counterexample_guided_lift(
            model,
            state_to_block,
            q_adds=q_adds,
            q_dels=frozenset(),
            target_text="![x]false",
            target_state=0,
            concrete_adds=concrete_adds,
            concrete_dels=concrete_dels,
            max_iters=4,
        )

        self.assertEqual(len(guided_adds), 1)
        self.assertEqual(guided_dels, frozenset())
        self.assertEqual(verify_original_lifted(model, guided_adds, guided_dels, "![x]false", 0)[0], "YES")

    def test_existential_repair_does_not_write_unneeded_quotient_edges(self):
        model = RepairLTS(
            initial=0,
            state_count=4,
            edges=frozenset({Edge(0, "x", 1), Edge(2, "y", 3)}),
        )
        state_to_block = (0, 1, 2, 3)
        q_adds = frozenset({Edge(0, "x", 2), Edge(1, "y", 3)})
        concrete_adds, concrete_dels = lift_quotient_edits_to_concrete(
            model,
            state_to_block,
            adds=q_adds,
            dels=frozenset(),
            target_state=0,
        )

        guided_adds, guided_dels, _iters = counterexample_guided_lift(
            model,
            state_to_block,
            q_adds=q_adds,
            q_dels=frozenset(),
            target_text="<x><y>true",
            target_state=0,
            concrete_adds=concrete_adds,
            concrete_dels=concrete_dels,
            max_iters=4,
        )

        self.assertEqual(guided_adds, frozenset({Edge(1, "y", 3)}))
        self.assertEqual(guided_dels, frozenset())
        self.assertEqual(verify_original_lifted(model, guided_adds, guided_dels, "<x><y>true", 0)[0], "YES")

    def test_writeback_operations_lists_stage2_and_stage3_edits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "writeback.json"
            write_writeback_operations(
                path,
                "case-1",
                "<x>true",
                frozenset({Edge(0, "x", 1)}),
                frozenset({Edge(1, "y", 2)}),
                frozenset({Edge(10, "x", 20)}),
                frozenset({Edge(20, "y", 30)}),
                "YES",
            )

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["lts_prime_to_lts_double_prime"]["adds"], [{"src": 0, "action": "x", "dst": 1}])
        self.assertEqual(payload["lts_double_prime_template_to_original_lts"]["adds"], [{"src": 10, "action": "x", "dst": 20}])
        self.assertEqual(payload["materialized_verified"], "YES")

    def test_neural_guided_minimal_search_finds_shortest_edit_depth(self):
        model = RepairLTS(
            initial=0,
            state_count=4,
            edges=frozenset({Edge(1, "b", 3)}),
        )
        config = RepairConfig(
            repair_mode="add-delete",
            sf_setting="no_sf",
            ranker="heuristic",
            search_strategy="neural_guided_minimal",
            max_iters=3,
            candidate_limit=16,
            candidate_state_limit=8,
            state_scan_limit=8,
            minimal_layer_width=64,
            costs=CostConfig(),
        )

        result = run_repair(model, "<a><b>true", set(), config)

        self.assertTrue(result.success)
        self.assertEqual(len(result.adds) + len(result.dels), 1)
        self.assertEqual(result.message, "Verified minimal repair found at edit depth 1.")

    def test_unsatisfiable_modal_tautology_is_rejected_before_search(self):
        model = RepairLTS(
            initial=0,
            state_count=2,
            edges=frozenset({Edge(0, "leader", 1)}),
        )
        config = RepairConfig(
            repair_mode="add-delete",
            sf_setting="no_sf",
            ranker="heuristic",
            search_strategy="neural_guided_minimal",
            max_iters=8,
            costs=CostConfig(),
        )

        result = run_repair(model, "!(<leader>[i]true | [leader]<i><leader>true)", set(), config)

        self.assertFalse(result.success)
        self.assertEqual(result.verifier_calls, 0)
        self.assertIn("syntactically unsatisfiable", result.message)

    def test_deprecated_quotient_fill_after_does_not_fill_whole_block(self):
        model = RepairLTS(
            initial=0,
            state_count=8,
            edges=frozenset({
                Edge(0, "i", 1),
                Edge(0, "i", 2),
                Edge(0, "i", 3),
                Edge(1, "i", 7),
                Edge(2, "i", 7),
                Edge(3, "i", 7),
            }),
        )
        state_to_block = (0, 1, 1, 1, 3, 2, 2, 3)
        q_adds = frozenset({Edge(1, "leader", 2)})
        concrete_adds, concrete_dels = lift_quotient_edits_to_concrete(
            model,
            state_to_block,
            adds=q_adds,
            dels=frozenset(),
            target_state=0,
        )

        guided_adds, guided_dels, _iters = counterexample_guided_lift(
            model,
            state_to_block,
            q_adds=q_adds,
            q_dels=frozenset(),
            target_text="!(<i>[leader]<i>true)",
            target_state=0,
            concrete_adds=concrete_adds,
            concrete_dels=concrete_dels,
            max_iters=10,
            quotient_fill_after=1,
        )

        self.assertEqual(guided_dels, frozenset())
        self.assertEqual(guided_adds, frozenset({
            Edge(1, "leader", 5),
            Edge(2, "leader", 5),
            Edge(3, "leader", 5),
        }))
        self.assertNotIn(Edge(1, "leader", 6), guided_adds)
        self.assertEqual(verify_original_lifted(model, guided_adds, guided_dels, "!(<i>[leader]<i>true)", 0)[0], "YES")


if __name__ == "__main__":
    unittest.main()
