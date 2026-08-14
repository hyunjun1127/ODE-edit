from __future__ import annotations

import inspect
from pathlib import Path
import unittest

import numpy as np

from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.fixed_e8_soft_routing import FixedE8Arm
from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    p1r24_disable_historical,
    solve_p1r24_matched_routing,
)
from project.run_scripts.ode_bf.p1r39_independent_b10x10_panel import (
    LOCK_FILE,
    METHODS,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r39_normalized_gradient_target import (
    prepare_p1r39_target_proposal,
    select_p1r39_target_proposal,
)
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem


ROOT = Path(__file__).resolve().parents[4]
TARGET_SHA256 = "98356016cd87d258f9c9e889ccb59f6f14e3f869e0f9284d2c90721e38c6537a"


def _barrier(label: str, linear: np.ndarray, gram: np.ndarray) -> QuadraticBarrier:
    return QuadraticBarrier(label, 0.0, linear, gram, 1.0e6, "layer-local-diagonal")


def _problem() -> RoutingProblem:
    slopes = np.asarray((2.0, 1.5, 1.0, 0.5, -0.25), dtype=np.float64)
    return RoutingProblem(
        slopes,
        np.diag(np.asarray((1.5, 1.7, 1.9, 2.1, 2.3))),
        np.diag(np.asarray((1.0, 1.2, 1.4, 1.6, 1.8))),
        100.0,
        np.ones(5),
        1.0e-12,
        1.0e-12,
        _barrier("historical", np.zeros(5), np.zeros((5, 5))),
        _barrier(
            "pretrained",
            np.asarray((-0.8, 0.4, 0.3, 0.2, 0.1)),
            np.diag(np.asarray((2.0, 1.0, 3.0, 4.0, 5.0))),
        ),
    )


class P1R39A1SoftExtensionTests(unittest.TestCase):
    def test_target_policy_bytes_and_tensor_only_path_are_unchanged(self) -> None:
        target = ROOT / "project/run_scripts/ode_bf/p1r39_normalized_gradient_target.py"
        self.assertEqual(sha256_file(target), TARGET_SHA256)
        for function in (prepare_p1r39_target_proposal, select_p1r39_target_proposal):
            source = inspect.getsource(function)
            for forbidden in ("model(", ".backward(", "P1R38AdamState"):
                self.assertNotIn(forbidden, source)

    def test_soft_lock_and_namespace_are_distinct(self) -> None:
        lock, _ = load_and_validate_lock(
            ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(METHODS, ["PR-P1R39-NORMALIZED-GRADIENT-SOFT"])
        self.assertEqual(lock["routing_arm"], "SOFT")
        self.assertEqual(lock["target_or_demand_attenuation_count"], 0)
        self.assertEqual(lock["accepted_p1r39_checkpoint"], "763457560f2efb177a56310dfd87526772cf8158")

    def test_soft_preserves_identical_requested_and_predicted_strength(self) -> None:
        problem = p1r24_disable_historical(_problem())
        rho = 0.75
        neutral = solve_p1r24_matched_routing(
            problem, arm=FixedE8Arm.NEUTRAL, rho_write=rho
        )
        soft = solve_p1r24_matched_routing(
            problem, arm=FixedE8Arm.SOFT, rho_write=rho
        )
        self.assertAlmostEqual(neutral.rho_write, soft.rho_write, places=12)
        self.assertAlmostEqual(neutral.predicted_progress, rho, places=8)
        self.assertAlmostEqual(soft.predicted_progress, rho, places=8)
        self.assertLessEqual(soft.selected_p, neutral.selected_p + 1.0e-7)


if __name__ == "__main__":
    unittest.main()
