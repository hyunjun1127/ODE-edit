from __future__ import annotations

import unittest

import torch

from project.run_scripts.fzcb_completion_value.algebra import (
    dimension_relative_floor,
    frozen_geometry_identity,
    macro_action_receipt,
    project_equality_null,
    scale_null_direction,
    solve_minimum_action,
    suffix_value,
)
from project.run_scripts.fzcb_completion_value.contracts import ScientificBoundary


class DenseOperator:
    def __init__(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix.float()
        self.output_dimension, self.coefficient_dimension = self.matrix.shape

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        return self.matrix @ value

    def adjoint(self, value: torch.Tensor) -> torch.Tensor:
        return self.matrix.T @ value


class AlgebraTests(unittest.TestCase):
    def test_adjoint_null_action_and_suffix_identity(self) -> None:
        generator = torch.Generator().manual_seed(7)
        matrix = torch.randn(5, 12, generator=generator)
        operator = DenseOperator(matrix)
        left = torch.randn(12, generator=generator)
        right = torch.randn(5, generator=generator)
        self.assertAlmostEqual(
            float(torch.dot(operator.apply(left), right)),
            float(torch.dot(left, operator.adjoint(right))), places=5,
        )
        tolerance = dimension_relative_floor(5) * 16
        h = torch.ones(12)
        rhs = torch.randn(5, generator=generator)
        solved = solve_minimum_action(
            operator, rhs, h, relative_tolerance=tolerance, max_iterations=24
        )
        self.assertLess(solved.range_residual, tolerance)
        projected, receipt = project_equality_null(
            operator, torch.randn(12, generator=generator),
            relative_tolerance=tolerance, max_iterations=24,
        )
        self.assertLess(receipt["null_residual"], tolerance)
        scaled = scale_null_direction(projected, solved.action_norm_squared ** 0.5, 0.1)
        self.assertGreater(float(torch.linalg.vector_norm(scaled)), 0.0)
        action0 = suffix_value(solved, 1.0)
        identity = frozen_geometry_identity(action0, (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertEqual(identity["maximum_absolute_error"], 0.0)

    def test_range_infeasible_is_typed(self) -> None:
        matrix = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        with self.assertRaisesRegex(ScientificBoundary, "typed range failure"):
            solve_minimum_action(
                DenseOperator(matrix), torch.tensor([0.0, 1.0]), torch.ones(2),
                relative_tolerance=1e-5, max_iterations=8,
            )

    def test_final_corrected_coefficient_action_accounting(self) -> None:
        predictor = torch.tensor([1.0, -2.0, 0.5])
        correction = torch.tensor([0.2, 0.3, -0.1])
        final = predictor + correction
        receipt = macro_action_receipt(float(torch.dot(final, final)), 0.25)
        self.assertEqual(receipt["absolute_error"], 0.0)
        self.assertAlmostEqual(
            receipt["step_action"],
            float(0.25 * 0.5 * torch.dot(final, final)), places=7,
        )


if __name__ == "__main__":
    unittest.main()
