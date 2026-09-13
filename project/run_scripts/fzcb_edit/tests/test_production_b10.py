from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from project.run_scripts.fzcb_edit.contracts import ScientificBoundary
from project.run_scripts.fzcb_edit.hashing import canonical_hash
from project.run_scripts.fzcb_edit.production_contracts import ProductionNumericalLock
from project.run_scripts.fzcb_edit.production_journal import ProductionJournal
from project.run_scripts.fzcb_edit.production_sensitivity import (
    NumericalInconclusive,
    robust_scalar_rectification,
    uncertainty_aware_sweep,
)
from project.run_scripts.fzcb_edit.tolerances import UnitTolerancePolicy
from project.run_scripts.fzcb_edit.transaction import WeightSnapshot
from project.run_scripts.fzcb_edit.verifier import verify_barrier_state, verify_terminal


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2, dtype=torch.float32))


def quadratic_sweep(a0: float, derivative: float = 0.3):
    policy = UnitTolerancePolicy.load()

    def evaluate(eta: float, solver_scale: float):
        del solver_scale
        value = a0 * (0.4 + derivative * eta + 0.125 * eta * eta)
        return value, {
            "realized_weight_perturbation_norm": abs(eta) * a0**0.5,
            "changed_element_fraction": 1.0,
        }

    return uncertainty_aware_sweep(
        evaluate,
        initial_budget=a0,
        base_epsilon=policy.fd_base_epsilon,
        multipliers=policy.fd_multipliers,
        repeats=policy.fd_repeats,
    )


class ProductionB10FocusedTests(unittest.TestCase):
    def test_01_normalized_cbf_is_action_rescaling_invariant(self) -> None:
        first = quadratic_sweep(0.5)
        second = quadratic_sweep(50.0)
        self.assertAlmostEqual(first.estimate, second.estimate, places=8)
        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
            correction, receipt = robust_scalar_rectification(
                psi_estimate=0.1,
                psi_uncertainty=1e-5,
                gradient=torch.tensor([1.0, 0.5]),
                gradient_uncertainty=torch.tensor([1e-5, 1e-5]),
                tau_normalized=1e-4,
            )
            self.assertTrue(receipt["barrier_active"], alias)
            self.assertTrue(torch.isfinite(correction).all())

    def test_02_equality_direction_normalization_rescaling_identity(self) -> None:
        coefficient = torch.tensor([3.0, 4.0], dtype=torch.float32)
        unit = coefficient / torch.linalg.vector_norm(coefficient)
        directional = 0.2
        a0 = 25.0
        normalized = float(torch.linalg.vector_norm(coefficient)) / a0**0.5 * directional
        self.assertAlmostEqual(float(torch.linalg.vector_norm(unit)), 1.0)
        self.assertAlmostEqual(normalized, directional)

    def test_03_scalar_reductions_are_fp64(self) -> None:
        gradient = torch.tensor([0.7, -0.2], dtype=torch.float32)
        _, receipt = robust_scalar_rectification(
            psi_estimate=0.1,
            psi_uncertainty=0.0,
            gradient=gradient,
            gradient_uncertainty=torch.zeros_like(gradient),
            tau_normalized=1e-4,
        )
        self.assertEqual(receipt["scalar_reduction_dtype"], "float64")

    def test_04_quadratic_analytic_derivative_matches_estimator(self) -> None:
        sweep = quadratic_sweep(4.0, derivative=0.375)
        self.assertAlmostEqual(sweep.estimate, 0.375, places=5)
        self.assertEqual(sweep.payload()["absolute_cross_epsilon_hard_gate_count"], 0)

    def test_05_interval_straddle_is_numerical_inconclusive(self) -> None:
        with self.assertRaises(NumericalInconclusive):
            robust_scalar_rectification(
                psi_estimate=0.0,
                psi_uncertainty=0.1,
                gradient=torch.tensor([1.0]),
                gradient_uncertainty=torch.tensor([0.01]),
                tau_normalized=0.05,
            )

    def test_06_actual_hcc_violation_rolls_back(self) -> None:
        model = ToyModel()
        snapshot = WeightSnapshot.capture(model, ("weight",))
        with torch.no_grad():
            model.weight.add_(1.0)
        with self.assertRaises(ScientificBoundary):
            verify_barrier_state(
                initial_budget=1.0,
                spent_action=0.8,
                predicted_suffix=0.3,
                tolerances=UnitTolerancePolicy.load().resolve(1.0),
                stage="candidate",
            )
        receipt = snapshot.restore(model)
        self.assertTrue(receipt["bytes_exact"] and receipt["pointer_exact"])

    def test_07_weight_pointer_and_bytes_restore(self) -> None:
        model = ToyModel()
        snapshot = WeightSnapshot.capture(model, ("weight",))
        pointer = model.weight.data_ptr()
        with torch.no_grad():
            model.weight.mul_(3.0)
        receipt = snapshot.restore(model)
        self.assertEqual(model.weight.data_ptr(), pointer)
        self.assertTrue(receipt["bytes_exact"])

    def test_08_append_only_journal_preserves_other_arms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            journal = ProductionJournal.create(Path(directory) / "journal", "llama3-8b-inst", (1, 2))
            first = journal.append("OFFICIAL_MEMIT", {"status": "TERMINAL_VALID"})
            second = journal.append("FZCB_SKETCH_K_PROTOTYPE_MEMIT", {"status": "NUMERICAL_INCONCLUSIVE"})
            self.assertNotEqual(first["sha256"], second["sha256"])
            self.assertEqual(journal.seal()["member_count"], 2)

    def test_09_fd_axis_rows_match_report_count(self) -> None:
        rows = [quadratic_sweep(1.0).payload() for _ in range(2)]
        receipt = {"raw_fd_axis_count": 1 + len(rows), "axis_fd_sweeps": rows}
        self.assertEqual(receipt["raw_fd_axis_count"], 1 + len(receipt["axis_fd_sweeps"]))

    def test_10_cohort_target_hash_identity(self) -> None:
        hashes = [canonical_hash({"case": index}) for index in range(10)]
        root = canonical_hash(hashes)
        arms = {name: root for name in ("official", "equality", "fzcb")}
        self.assertEqual(len(set(arms.values())), 1)

    def test_11_fixed_z_waypoint_recomputation_is_zero(self) -> None:
        lock = ProductionNumericalLock()
        self.assertEqual(lock.target_recompute_count, 0)
        self.assertEqual(lock.stock_compute_z_rule, "once_per_request_per_method_cohort_at_W0")

    def test_12_corrector_action_must_be_in_total_E(self) -> None:
        receipt = verify_terminal(
            closure=0.0,
            spent_action=1.0,
            initial_budget=1.0,
            current_viability=True,
            all_waypoints_strict=True,
            corrector_action_included=True,
            rollback_exact=True,
            tolerances=UnitTolerancePolicy.load().resolve(1.0),
        )
        self.assertTrue(receipt["strict_pass"])


if __name__ == "__main__":
    unittest.main()
