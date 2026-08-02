from __future__ import annotations

import unittest

import torch

from project.run_scripts.ode_edit_motivation.capacity_geometry import (
    CapacityGeometryError,
    compute_w0_denominators,
    weight_c_energy,
)


class _Toy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        self.second = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.first.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
            self.second.weight.copy_(torch.tensor([[0.5, -1.0]]))


class CapacityGeometryTests(unittest.TestCase):
    def test_blocked_energy_matches_dense_trace(self) -> None:
        weight = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [-2.0, 0.5]], dtype=torch.float64
        )
        covariance = torch.tensor([[2.0, 0.25], [0.25, 1.0]])
        expected = float(torch.trace(weight @ covariance.double() @ weight.T))
        self.assertAlmostEqual(
            weight_c_energy(weight, covariance, row_block=1),
            expected,
            places=10,
        )

    def test_denominators_follow_locked_layer_order(self) -> None:
        model = _Toy()
        covariance = {4: torch.eye(2), 5: torch.tensor([[2.0, 0.0], [0.0, 1.0]])}
        result = compute_w0_denominators(
            model,
            covariance,
            {"first.weight": 4, "second.weight": 5},
            expected_layers=(4, 5),
        )
        self.assertEqual(tuple(result), (4, 5))
        self.assertAlmostEqual(result[4], 30.0)
        self.assertAlmostEqual(result[5], 1.5)

    def test_missing_layer_fails_closed(self) -> None:
        with self.assertRaises(CapacityGeometryError):
            compute_w0_denominators(
                _Toy(),
                {4: torch.eye(2)},
                {"first.weight": 4},
                expected_layers=(4, 5),
            )


if __name__ == "__main__":
    unittest.main()
