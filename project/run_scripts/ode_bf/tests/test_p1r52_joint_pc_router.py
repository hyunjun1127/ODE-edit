from __future__ import annotations

import unittest

import numpy as np
import torch

from project.run_scripts.ode_bf.p1r52_joint_pc_router import (
    JointPCReferenceDegenerate,
    _kkt_residual,
    proxies_from_entry_problem,
    solve_joint_pc_router,
)
from project.run_scripts.ode_bf.p1r52_residual_reserve_pc_router import QuadraticProxy
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem


def _barrier(label: str, diagonal: tuple[float, ...]) -> QuadraticBarrier:
    zero = np.zeros(5, dtype=np.float64)
    gram = np.diag(np.asarray(diagonal, dtype=np.float64))
    return QuadraticBarrier(label, 0.0, zero, gram, 100.0, "layer-local-diagonal")


class JointPCRouterTests(unittest.TestCase):
    def test_kkt_certificate_excludes_inactive_inequalities(self) -> None:
        pi = np.asarray((0.0, 0.25, 0.25, 0.25, 0.25), dtype=np.float64)
        stationarity, complementarity = _kkt_residual(
            pi,
            1.0,
            p_slack=0.0,
            c_slack=0.1,
            p_gradient=np.asarray((2.0, 1.0, 1.0, 1.0, 1.0)),
            c_gradient=np.asarray((-10.0, 1.0, 1.0, 1.0, 1.0)),
        )
        self.assertLessEqual(stationarity, 1.0e-12)
        self.assertLessEqual(complementarity, 1.0e-12)

    def test_exact_one_stage_minimax_is_finite_and_simplex(self) -> None:
        p = QuadraticProxy(
            "P",
            0.0,
            torch.zeros(5, dtype=torch.float32),
            torch.diag(torch.tensor((9.0, 4.0, 1.0, 4.0, 9.0), dtype=torch.float32)),
        )
        c = QuadraticProxy(
            "C",
            0.0,
            torch.zeros(5, dtype=torch.float32),
            torch.diag(torch.tensor((1.0, 4.0, 9.0, 4.0, 1.0), dtype=torch.float32)),
        )
        result = solve_joint_pc_router(p, c)
        receipt = result.receipt
        self.assertAlmostEqual(float(result.pi.sum()), 1.0, places=6)
        self.assertTrue(bool(torch.all(result.pi >= 0.0)))
        self.assertLessEqual(receipt.p_constraint_slack, 1.0e-5)
        self.assertLessEqual(receipt.c_constraint_slack, 1.0e-5)
        self.assertEqual(receipt.weighted_sum_count, 0)
        self.assertEqual(receipt.lexicographic_stage_count, 0)
        self.assertEqual(receipt.fallback_count, 0)
        self.assertEqual(receipt.model_forward_count, 0)

    def test_reference_degeneracy_is_typed_without_fallback(self) -> None:
        zero = torch.zeros(5, dtype=torch.float32)
        q = torch.zeros((5, 5), dtype=torch.float32)
        with self.assertRaisesRegex(JointPCReferenceDegenerate, "JOINT_PC_REFERENCE_DEGENERATE"):
            solve_joint_pc_router(
                QuadraticProxy("P", 1.0, zero, q),
                QuadraticProxy("C", 2.0, zero, q),
            )

    def test_existing_problem_maps_p_and_capacity_without_h_or_budget(self) -> None:
        problem = RoutingProblem(
            np.ones(5),
            np.diag(np.asarray((2.0, 4.0, 6.0, 8.0, 10.0))),
            np.eye(5),
            1.0,
            np.ones(5),
            1.0,
            1.0,
            _barrier("historical", (7.0,) * 5),
            _barrier("pretrained", (1.0, 2.0, 3.0, 4.0, 5.0)),
        )
        p, c = proxies_from_entry_problem(problem)
        self.assertTrue(torch.equal(p.quadratic, torch.diag(torch.arange(1.0, 6.0))))
        self.assertTrue(torch.equal(c.quadratic, torch.diag(torch.arange(1.0, 6.0))))
        self.assertEqual(p.constant, 0.0)
        self.assertEqual(c.constant, 0.0)


if __name__ == "__main__":
    unittest.main()
