import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional neural dependency
    torch = None

from svbr.repair.add_delete import (
    CANDIDATE_FEATURE_ORDER,
    Edge,
    GraphCandidateRankerModule,
    NeuralRanker,
    RepairLTS,
    graph_tensors_for_model,
    repair_view,
)


@unittest.skipIf(torch is None, "torch is not installed")
class GnnRankerTests(unittest.TestCase):
    def test_dynamic_overlay_cache_matches_full_graph_tensors(self):
        base = RepairLTS(
            0,
            4,
            frozenset(
                {
                    Edge(0, "a", 1),
                    Edge(1, "b", 2),
                    Edge(2, "a", 2),
                    Edge(3, "c", 0),
                }
            ),
        )
        overlay = repair_view(
            base,
            frozenset({Edge(0, "d", 3), Edge(1, "a", 1)}),
            frozenset({Edge(3, "c", 0)}),
        )
        v_actions = {"a", "d"}
        model = GraphCandidateRankerModule(len(CANDIDATE_FEATURE_ORDER), 8, 1, torch)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "gnn.pt"
            torch.save(
                {
                    "architecture": "gnn",
                    "feature_order": CANDIDATE_FEATURE_ORDER,
                    "hidden_dim": 8,
                    "hidden_layers": 1,
                    "model_state": model.state_dict(),
                },
                checkpoint,
            )
            ranker = NeuralRanker(str(checkpoint), device="cpu", gnn_graph_mode="dynamic")

            expected_nodes, expected_edge_index, expected_edge_features = graph_tensors_for_model(
                overlay,
                v_actions,
                torch,
                torch.device("cpu"),
            )
            actual_nodes, actual_edge_index, actual_edge_features = ranker._dynamic_overlay_graph_tensors(overlay, v_actions)

        self.assertTrue(torch.equal(expected_nodes, actual_nodes))
        self.assertEqual(
            self._edge_rows(expected_edge_index, expected_edge_features),
            self._edge_rows(actual_edge_index, actual_edge_features),
        )

    @staticmethod
    def _edge_rows(edge_index, edge_features):
        return sorted(
            (
                int(edge_index[0, index]),
                int(edge_index[1, index]),
                tuple(float(value) for value in edge_features[index]),
            )
            for index in range(edge_index.shape[1])
        )


if __name__ == "__main__":
    unittest.main()
