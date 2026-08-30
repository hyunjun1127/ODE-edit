from __future__ import annotations

import unittest

import torch

from project.run_scripts.barrier_usefulness.transport import factor_dense, reconstruct


class TransportTests(unittest.TestCase):
    def test_compact_transport_is_bounded(self) -> None:
        torch.manual_seed(11)
        matrix = torch.randn(13, 3) @ torch.randn(3, 17)
        factors = factor_dense(matrix, tolerance=3.0517578125e-5)
        observed = reconstruct(factors, device=torch.device("cpu"))
        self.assertLess(float((observed - matrix).norm() / matrix.norm()), 3.0517578125e-5)
        self.assertLessEqual(factors["rank"], 3)


if __name__ == "__main__":
    unittest.main()
