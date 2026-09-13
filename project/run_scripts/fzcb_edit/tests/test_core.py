from __future__ import annotations

import math
import unittest

import torch

from project.run_scripts.fzcb_edit.contracts import ScientificBoundary
from project.run_scripts.fzcb_edit.forbidden_imports import scan
from project.run_scripts.fzcb_edit.geometry import FullModelControlOperator, gauge_free_whitened_right
from project.run_scripts.fzcb_edit.linear import (
    equality_null_projection,
    scalar_rectification,
    solve_minimum_action,
    suffix_value,
)
from project.run_scripts.fzcb_edit.transaction import AtomicEditTransaction, WeightSnapshot


class DenseOperator:
    def __init__(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix.float()
        self.output_dimension, self.coefficient_dimension = self.matrix.shape

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        return self.matrix @ value

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return self.matrix.T @ value


class ToyConfig:
    hidden_size = 2


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
        self.config = ToyConfig()

    def forward(self, x: torch.Tensor, **_: object):
        hidden = (x @ self.weight.T).unsqueeze(1)
        return type("Output", (), {"hidden_states": (hidden, hidden)})()


class CoreTests(unittest.TestCase):
    def test_forbidden_superseded_controller_imports_absent(self) -> None:
        from pathlib import Path

        self.assertEqual(scan(Path("project/run_scripts/fzcb_edit")), [])

    def test_factorized_writer_matches_dense_toy_and_removes_gauge(self) -> None:
        covariance = torch.tensor([[2.0, 0.2], [0.2, 1.5]], dtype=torch.float32)
        native = torch.tensor([[1.0, 0.5], [2.0, 1.0]], dtype=torch.float32)  # rank-one request gauge
        right, receipt = gauge_free_whitened_right(native, covariance, 1e-4)
        self.assertEqual(receipt["reduced_rank"], 1)
        self.assertEqual(receipt["removed_gauge_dimension"], 1)
        coefficient = torch.tensor([[0.7], [-0.2]], dtype=torch.float32)
        factorized = coefficient @ right
        dense = torch.einsum("or,ri->oi", coefficient, right)
        self.assertTrue(torch.allclose(factorized, dense, atol=1e-7, rtol=1e-7))
        action = torch.trace(factorized @ (covariance + 1e-4 * torch.eye(2)) @ factorized.T)
        self.assertAlmostEqual(float(action), float(torch.dot(coefficient.reshape(-1), coefficient.reshape(-1))), places=5)

    def test_matrix_free_adjoint_and_minimum_action_kkt(self) -> None:
        matrix = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]], dtype=torch.float32)
        operator = DenseOperator(matrix)
        x = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float32)
        y = torch.tensor([-0.5, 0.8], dtype=torch.float32)
        self.assertTrue(torch.allclose(torch.dot(operator.apply(x), y), torch.dot(x, operator.adjoint(y)), atol=1e-6))
        rhs = torch.tensor([0.3, -0.2], dtype=torch.float32)
        solution = solve_minimum_action(
            operator, rhs, tolerance=1e-5, maximum_iterations=50, refinement_count=2,
        )
        self.assertLess(float(torch.linalg.vector_norm(operator.apply(solution.coefficient) - rhs)), 1e-5)
        expected = matrix.T @ torch.linalg.solve(matrix @ matrix.T, rhs)
        self.assertTrue(torch.allclose(solution.coefficient, expected, atol=2e-5, rtol=2e-5))

    def test_suffix_kkt_and_frozen_identity(self) -> None:
        operator = DenseOperator(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
        demand = torch.tensor([0.4, -0.3], dtype=torch.float32)
        rate = solve_minimum_action(operator, demand, tolerance=1e-6, maximum_iterations=10, refinement_count=1)
        a0 = 0.5 * rate.action_squared
        for progress in (0.0, 0.25, 0.5, 0.75):
            residual = (1.0 - progress) * demand
            suffix = solve_minimum_action(operator, residual, tolerance=1e-6, maximum_iterations=10, refinement_count=1)
            value = suffix_value(suffix, 1.0 - progress)
            spent = progress * a0
            self.assertAlmostEqual(spent + value, a0, places=6)

    def test_equality_null_and_scalar_rectification(self) -> None:
        operator = DenseOperator(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32))
        seed = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float32)
        null, receipt = equality_null_projection(
            operator, seed, tolerance=1e-6, maximum_iterations=20, refinement_count=1,
        )
        self.assertLess(float(torch.linalg.vector_norm(operator.apply(null))), 1e-6)
        inactive, inactive_receipt = scalar_rectification(
            -0.1, torch.tensor([1.0, 0.0]), tau_cbf=1e-6,
        )
        self.assertTrue(torch.equal(inactive, torch.zeros(2)))
        self.assertFalse(inactive_receipt["barrier_active"])
        correction, receipt = scalar_rectification(
            0.25, torch.tensor([1.0, 0.0]), tau_cbf=1e-6,
        )
        residual = 0.5 * float(torch.dot(correction, correction)) + float(correction[0]) + 0.25
        self.assertLessEqual(residual, 1e-6)
        self.assertTrue(receipt["barrier_active"])
        with self.assertRaises(ScientificBoundary):
            scalar_rectification(
                1.0, torch.tensor([0.1, 0.0]), tau_cbf=1e-6,
            )

    def test_actual_torch_func_operator_and_orientation(self) -> None:
        model = ToyModel()
        batch = {
            "x": torch.tensor([[1.5, -0.5]], dtype=torch.float32),
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
            "fzcb_lookup_indices": torch.tensor([0], dtype=torch.long),
        }
        operator = FullModelControlOperator(
            model, batch, 0, ("weight",), (torch.eye(2),), direct_gram_floor=1.0,
        )
        coefficient = torch.tensor([0.2, 0.3, -0.4, 0.1], dtype=torch.float32)
        cotangent = torch.tensor([0.7, -0.2], dtype=torch.float32)
        forward = torch.dot(operator.apply(coefficient), cotangent)
        adjoint = torch.dot(coefficient, operator.adjoint(cotangent))
        self.assertTrue(torch.allclose(forward, adjoint, atol=1e-6, rtol=1e-6))
        tangent = operator.coefficient_to_tangents(coefficient)[0]
        explicit = coefficient.reshape(2, 2) @ torch.eye(2)
        self.assertTrue(torch.equal(tangent, explicit))

    def test_snapshot_rollback_and_terminal_atomic_commit(self) -> None:
        model = ToyModel()
        entry = WeightSnapshot.capture(model, ("weight",))
        with torch.no_grad():
            model.weight.add_(1.0)
        receipt = entry.restore(model)
        self.assertTrue(receipt["bytes_exact"] and receipt["pointer_exact"])
        with torch.no_grad():
            model.weight.add_(torch.tensor([[0.1, 0.0], [0.0, -0.1]]))
        terminal = WeightSnapshot.capture(model, ("weight",))
        entry.restore(model)
        transaction = AtomicEditTransaction(model, entry)
        committed = transaction.commit_terminal(terminal)
        self.assertEqual(committed["commit_count"], 1)
        self.assertEqual(committed["intermediate_commit_count"], 0)
        self.assertEqual(terminal.current_root(model), terminal.root)


if __name__ == "__main__":
    unittest.main()
