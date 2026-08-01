from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import torch

from project.run_scripts.ode_edit_motivation.contracts import LowRankFactor
from project.run_scripts.ode_edit_motivation.microseq_analysis import (
    BRANCH_NATIVE,
    BRANCH_ODE,
    CHECKPOINT_EVENT,
    STREAM_SCHEMA,
    load_stream,
)
from project.run_scripts.ode_edit_motivation.microseq_evaluator import (
    EVALUATOR_RUN_IDS,
    _execution_envelope,
    _gini,
    cumulative_geometry,
    load_counterfact_evaluation_fields,
    merge_checkpoint_streams,
    run_microseq_evaluator,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import SanitizedJsonlWriter


class MicroseqEvaluatorTests(unittest.TestCase):
    def test_evaluation_decode_is_source_ordered_after_controller_barrier(self) -> None:
        source = inspect.getsource(run_microseq_evaluator)
        self.assertLess(
            source.index("verify_controller_pair("),
            source.index("evaluation_loader("),
        )

    def test_evaluation_loader_keeps_exact_first_four_in_memory(self) -> None:
        rows = []
        for index in range(4):
            rows.append(
                {
                    "case_id": index,
                    "requested_rewrite": {
                        "target_true": {"str": f"old-{index}"},
                    },
                    "neighborhood_prompts": [f"n{index}-{item}" for item in range(6)],
                    "generation_prompts": [f"g{index}-{item}" for item in range(5)],
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data/counterfact/counterfact.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(rows), encoding="utf-8")
            fields = load_counterfact_evaluation_fields(root, ("3", "1", "0", "2"))
        self.assertEqual([field.case_id for field in fields], ["3", "1", "0", "2"])
        self.assertEqual(fields[0].neighborhood_prompts, ("n3-0", "n3-1", "n3-2", "n3-3"))
        self.assertEqual(fields[1].generation_prompts, ("g1-0", "g1-1", "g1-2", "g1-3"))
        self.assertEqual(fields[2].target_true, "old-0")

    def test_cumulative_geometry_matches_dense_quadratic_forms(self) -> None:
        covariance = {
            4: torch.tensor([[2.0, 0.5], [0.5, 1.0]]),
            5: torch.tensor([[1.0, 0.0], [0.0, 3.0]]),
            6: torch.eye(2),
        }
        factors = (
            LowRankFactor("w4", torch.tensor([[1.0], [2.0]]), torch.tensor([[0.5], [1.0]]), "a" * 64),
            LowRankFactor("w4", torch.tensor([[0.5], [-1.0]]), torch.tensor([[1.0], [0.25]]), "b" * 64),
            LowRankFactor("w5", torch.tensor([[2.0], [0.5]]), torch.tensor([[0.25], [1.5]]), "c" * 64),
        )
        result = cumulative_geometry(
            factors,
            covariance_by_layer=covariance,
            layer_by_weight={"w4": 4, "w5": 5, "w6": 6},
            w0_denominators={4: 10.0, 5: 20.0, 6: 30.0},
            device=torch.device("cpu"),
        )
        delta4 = factors[0].left @ factors[0].right.T + factors[1].left @ factors[1].right.T
        delta5 = factors[2].left @ factors[2].right.T
        energy4 = float(torch.trace(delta4 @ covariance[4] @ delta4.T))
        energy5 = float(torch.trace(delta5 @ covariance[5] @ delta5.T))
        capacities = (energy4 / 10.0, energy5 / 20.0, 0.0)
        expected_frobenius = float(torch.sqrt(torch.sum(delta4.square()) + torch.sum(delta5.square())))
        self.assertAlmostEqual(result["capacity_sum"], sum(capacities), places=5)
        self.assertAlmostEqual(result["max_layer_share"], max(capacities) / sum(capacities), places=5)
        self.assertAlmostEqual(result["layer_gini"], _gini(capacities), places=5)
        self.assertAlmostEqual(result["cumulative_frobenius"], expected_frobenius, places=5)

    def test_merge_resequences_two_isolated_branch_streams(self) -> None:
        contract = {"same": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for branch in (BRANCH_NATIVE, BRANCH_ODE):
                path = root / f"{branch}.jsonl"
                with SanitizedJsonlWriter(path, branch, schema_version=STREAM_SCHEMA) as writer:
                    for index in range(1, 5):
                        writer.write(
                            CHECKPOINT_EVENT,
                            {
                                "model_alias": "llama3-8b-inst",
                                "branch": branch,
                                "edit_index": index,
                                "fixed_contract": contract,
                            },
                        )
                paths[branch] = path
            merged = merge_checkpoint_streams(
                native_path=paths[BRANCH_NATIVE],
                ode_path=paths[BRANCH_ODE],
                output_path=root / "merged.jsonl",
                model_alias="llama3-8b-inst",
                run_id="merged-test",
            )
            records = load_stream(merged)
        self.assertEqual(len(records), 8)
        self.assertEqual([item["branch"] for item in records[:4]], [BRANCH_NATIVE] * 4)
        self.assertEqual([item["branch"] for item in records[4:]], [BRANCH_ODE] * 4)

    def test_evaluator_identity_is_model_and_branch_locked(self) -> None:
        for branch, by_model in EVALUATOR_RUN_IDS.items():
            for model, run_id in by_model.items():
                _execution_envelope(branch, model, run_id)
        with self.assertRaisesRegex(Exception, "identity differs"):
            _execution_envelope(
                BRANCH_ODE,
                "llama3-8b-inst",
                EVALUATOR_RUN_IDS[BRANCH_NATIVE]["llama3-8b-inst"],
            )


if __name__ == "__main__":
    unittest.main()
