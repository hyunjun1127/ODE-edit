from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from project.run_scripts.fzcb_edit.comparators import true_frozen_schedule
from project.run_scripts.fzcb_edit.contracts import Arm, ScientificBoundary
from project.run_scripts.fzcb_edit.journal import ArmJournal
from project.run_scripts.fzcb_edit.linear import full_projected_sensitivity, scalar_rectification
from project.run_scripts.fzcb_edit.rollout import observe_candidate_rollout
from project.run_scripts.fzcb_edit.sensitivity import finite_difference_sweep, sketch_ladder_receipt
from project.run_scripts.fzcb_edit.tech_r1_joint_launcher import joint_matrix
from project.run_scripts.fzcb_edit.tolerances import UnitTolerancePolicy, synthetic_calibration_receipt
from project.run_scripts.fzcb_edit.transaction import WeightSnapshot
from project.run_scripts.fzcb_edit.verifier import verify_barrier_state, verify_terminal


class DenseOperator:
    def __init__(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix.float()
        self.output_dimension, self.coefficient_dimension = self.matrix.shape

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        return self.matrix @ value

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return self.matrix.T @ value


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2, dtype=torch.float32))


class TechR1FocusedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = UnitTolerancePolicy.load()

    def test_tolerance_unit_separation_and_scale_invariance(self) -> None:
        calibration = synthetic_calibration_receipt(self.policy)
        self.assertEqual(
            calibration["unit_separation"]["action_rate"], ["tau_cbf"],
        )
        first = self.policy.resolve(0.5)
        second = self.policy.resolve(50.0)
        self.assertAlmostEqual(first.tau_cbf / 0.5, second.tau_cbf / 50.0)
        self.assertAlmostEqual(first.tau_budget / 0.5, second.tau_budget / 50.0)
        self.assertAlmostEqual(
            first.tau_grad / (0.5**0.5), second.tau_grad / (50.0**0.5),
        )
        self.assertEqual(first.tau_z, second.tau_z)
        self.assertEqual(first.tau_range, second.tau_range)

    def test_positive_psi_uses_action_rate_tolerance_only(self) -> None:
        correction, receipt = scalar_rectification(
            0.01,
            torch.tensor([0.2, 0.0], dtype=torch.float32),
            tau_cbf=1e-5,
        )
        self.assertTrue(receipt["barrier_active"])
        self.assertGreater(float(torch.linalg.vector_norm(correction)), 0.0)
        self.assertNotIn("tau_z", receipt)

    def test_full_projection_matches_explicit_small_null_projection(self) -> None:
        matrix = torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )
        operator = DenseOperator(matrix)
        gradient = torch.tensor([0.3, -0.2, 0.8, 0.4], dtype=torch.float32)
        projected, receipt = full_projected_sensitivity(
            operator,
            gradient,
            tau_range=1e-5,
            tau_null=1e-5,
            maximum_iterations=64,
            refinement_count=2,
        )
        explicit = gradient - matrix.T @ torch.linalg.solve(matrix @ matrix.T, matrix @ gradient)
        self.assertTrue(torch.allclose(projected, explicit, atol=2e-5, rtol=2e-5))
        self.assertLess(receipt["null_residual"], 1e-5)

    def test_fd_sweep_repeat_and_sketch_ladder(self) -> None:
        sweep = finite_difference_sweep(
            lambda scale: (3.0 + 2.0 * scale + 0.25 * scale * scale, {"scale": scale}),
            base_epsilon=self.policy.fd_base_epsilon,
            multipliers=self.policy.fd_multipliers,
            repeats=self.policy.fd_repeats,
            tau_grad=1e-3,
        )
        self.assertAlmostEqual(sweep.selected_derivative, 2.0, places=4)
        receipt = sketch_ladder_receipt(
            [float(index + 1) / 1000.0 for index in range(128)],
            coefficient_dimension=256,
            output_dimension=64,
            ladder=(2, 8, 32, 128),
            seeds=tuple(range(128)),
        )
        self.assertEqual([row["k"] for row in receipt["ladder"]], [2, 8, 32, 128])
        self.assertFalse(receipt["full_gradient_available"])

    def test_frozen_c_identity(self) -> None:
        rows = true_frozen_schedule(
            initial_budget=2.0,
            progress_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
            tau_budget=1e-9,
        )
        self.assertTrue(all(row["pass"] for row in rows))
        self.assertTrue(all(abs(float(row["h_cc"])) <= 1e-9 for row in rows))

    def test_corrector_action_is_not_free_and_intermediate_violation_fails(self) -> None:
        tolerances = self.policy.resolve(2.0)
        with self.assertRaises(ScientificBoundary):
            verify_terminal(
                closure=0.0,
                spent_action=2.0 + 2.0 * tolerances.tau_budget,
                initial_budget=2.0,
                current_viability=True,
                all_waypoints_strict=True,
                corrector_action_included=True,
                rollback_exact=True,
                tolerances=tolerances,
            )
        with self.assertRaises(ScientificBoundary):
            verify_barrier_state(
                initial_budget=2.0,
                spent_action=1.2,
                predicted_suffix=0.9,
                tolerances=tolerances,
                stage="intermediate",
            )

    def test_failure_journal_preserves_prior_arm_and_is_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = ArmJournal.create(Path(directory) / "journal", "llama3-8b-inst", (0,))
            first = journal.append(0, Arm.OFFICIAL_MEMIT, {"status": "TERMINAL_VALID"})
            second = journal.append(1, Arm.FZCB, {"status": "SUBSPACE_INCONCLUSIVE"})
            self.assertTrue(Path(first["path"]).is_file())
            self.assertTrue(Path(second["path"]).is_file())
            self.assertNotEqual(first["sha256"], second["sha256"])
            sealed = journal.seal()
            self.assertEqual(sealed["member_count"], 2)

    def test_realized_rollout_failure_is_observation_only(self) -> None:
        def fail() -> tuple[float, tuple[float, ...], dict[str, object]]:
            raise ScientificBoundary("diagnostic rollout closure")

        receipt = observe_candidate_rollout(
            candidate_id="accepted",
            predicted_suffix_after=0.5,
            execute=fail,
        )
        self.assertEqual(receipt["failure_count"], 1)
        self.assertEqual(receipt["controller_selection_influence_count"], 0)
        self.assertEqual(
            receipt["status"], "PREDICTIVE_ROLLOUT_INVALID_OBSERVATION_ONLY",
        )

    def test_rollback_exact_identity(self) -> None:
        model = ToyModel()
        snapshot = WeightSnapshot.capture(model, ("weight",))
        with torch.no_grad():
            model.weight.add_(0.25)
        receipt = snapshot.restore(model)
        self.assertTrue(receipt["bytes_exact"] and receipt["pointer_exact"])

    def test_joint_launcher_matrix_is_exact_and_failure_independent(self) -> None:
        rows = joint_matrix()
        self.assertEqual([row["array_index"] for row in rows], [0, 1])
        self.assertEqual(
            [row["model"] for row in rows],
            ["llama3-8b-inst", "qwen2.5-7b-inst"],
        )
        self.assertEqual(rows[0]["arm_contract"], rows[1]["arm_contract"])
        self.assertTrue(all(row["same_request_order_seed_contract"] for row in rows))


if __name__ == "__main__":
    unittest.main()
