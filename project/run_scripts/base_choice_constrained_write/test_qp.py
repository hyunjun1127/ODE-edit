"""CPU linear-QP tests only: no model, GPU, Slurm, or Llama PASS claim."""
from __future__ import annotations

from dataclasses import replace
import itertools
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .qp import (DEFAULT_QP_POLICY, QPInfeasible, QPInputError,
                 QPSolverFailure, QPZeroRowInfeasible, audit_same_problem, solve)


def from_rows(h, b, order="gss", **kwargs):
    h = np.asarray(h, dtype=np.float64)
    result = solve(h @ h.T, b, [f"pair-{i:04}" for i in range(len(h))], order, **kwargs)
    return np.asarray(result["alpha"]) @ h, result


def enumerated_projection(h, b):
    """Independent small exact-active-set oracle, only in tests."""
    h, b = np.asarray(h, dtype=float), np.asarray(b, dtype=float)
    candidates = []
    for size in range(min(len(h), h.shape[1]) + 1):
        for active in itertools.combinations(range(len(h)), size):
            rows = h[list(active)]
            if size:
                try:
                    multiplier = np.linalg.solve(rows @ rows.T, b[list(active)])
                except np.linalg.LinAlgError:
                    continue
                if np.any(multiplier < -1e-9):
                    continue
                direction = multiplier @ rows
            else:
                direction = np.zeros(h.shape[1])
            if np.all(h @ direction >= b - 1e-9):
                candidates.append(direction)
    return min(candidates, key=lambda d: d @ d) if candidates else None


class QPTests(unittest.TestCase):
    def assertCertified(self, result):
        self.assertEqual(result["status"], "CERTIFIED")
        self.assertTrue(result["diagnostics"]["KKT_pass"])
        self.assertEqual(result["diagnostics"]["all_rows_checked"], len(result["alpha"]))
        self.assertEqual(result["spectrum"]["ridge"], 0)
        self.assertTrue(np.all(np.asarray(result["alpha"]) >= 0))
        json.dumps(result, allow_nan=False)

    def test_positive_sign_and_safe_negative_rhs(self):
        direction, result = from_rows([[2, 0]], [.134])
        np.testing.assert_allclose(direction, [.067, 0], atol=1e-14)
        np.testing.assert_allclose(result["alpha"], [.0335], atol=1e-14)
        self.assertCertified(result)
        direction, result = from_rows([[2, 0]], [-.3])
        np.testing.assert_array_equal(direction, [0, 0])
        self.assertEqual(result["working_set_history"], [])
        self.assertCertified(result)

    def test_safe_row_can_become_binding(self):
        direction, result = from_rows([[1, 0], [-1, 1]], [1, -.25])
        np.testing.assert_allclose(direction, [1, .75], atol=1e-13)
        self.assertGreater(result["alpha"][1], 0)
        self.assertCertified(result)

    def test_exposed_competitor_rows_must_both_remain(self):
        direction, result = from_rows([[1, 0], [-20, 1]], [.1, -1])
        np.testing.assert_allclose(direction, [.1, 1], atol=1e-12)
        self.assertCertified(result)

    def test_same_direction_stricter_offset_and_nonunique_dual(self):
        h = np.array([[1, 0], [1, 0], [2, 0], [0, 1], [0, 1]], dtype=float)
        b = [1, 2, 4, 1, 1]
        for order in ("full", "gss", "most_violation"):
            direction, result = from_rows(h, b, order)
            np.testing.assert_allclose(direction, [2, 1], atol=1e-12)
            self.assertCertified(result)
        report = audit_same_problem(h @ h.T, b, list(range(5)))
        self.assertTrue(report["pass"])
        self.assertFalse(report["alpha_equality_required"])

    def test_zero_rows_and_strict_positive_zero_rhs(self):
        direction, result = from_rows([[0, 0], [0, 0], [1, 0]], [0, -4, 2])
        np.testing.assert_allclose(direction, [2, 0])
        self.assertEqual(result["spectrum"]["zero_rows"], [0, 1])
        self.assertCertified(result)
        for rhs in (1, 1e-30):
            with self.assertRaises(QPZeroRowInfeasible):
                from_rows([[0, 0]], [rhs])

    def test_opposing_rows_infeasible_not_solver_failure(self):
        for order in ("full", "gss", "most_violation"):
            with self.assertRaises(QPInfeasible) as ctx:
                from_rows([[1, 0], [-1, 0]], [1, 1], order)
            self.assertGreater(ctx.exception.receipt["b_dot_witness"], 0)
            self.assertLessEqual(ctx.exception.receipt["full_gram_witness_inf"], 2e-12)
        # Contradiction can involve three rows rather than an opposite pair.
        with self.assertRaises(QPInfeasible):
            from_rows([[1, 0], [0, 1], [-1, -1]], [1, 1, 1])

    def test_rank_deficient_feasible_equality_cone(self):
        direction, result = from_rows([[1, 0], [-1, 0], [0, 1], [0, 2]], [0, 0, 1, 2])
        np.testing.assert_allclose(direction, [0, 1], atol=1e-12)
        self.assertCertified(result)

    def test_scaling_preserves_direction_and_original_multiplier_units(self):
        h = np.array([[1, 0], [-1, 1], [0, 2]], dtype=float)
        b = np.array([1, -.25, -1.])
        direction, _ = from_rows(h, b)
        scale = np.array([1e-12, 1e12, 1e-7])
        other, result = from_rows(h * scale[:, None], b * scale)
        np.testing.assert_allclose(other, direction, atol=2e-13)
        self.assertCertified(result)
        tiny, result = from_rows([[1e-150]], [1e-150])
        np.testing.assert_allclose(tiny, [1], atol=1e-13)
        self.assertCertified(result)

    def test_shape_id_order_nonfinite_psd_and_symmetry_failures(self):
        cases = [([[1]], [1, 2], [0, 1]),
                 ([[np.nan]], [1], [0]), ([[1]], [np.inf], [0]),
                 ([[1]], [1], []), (np.eye(2), [1, 1], [0, 0]),
                 ([[1 + 1j]], [1], [0]),
                 ([[-1]], [1], [0]), ([[0, 1], [1, 1]], [0, 0], [0, 1]),
                 ([[1, 2], [2, 1]], [1, 1], [0, 1]),
                 ([[1, .1], [.2, 1]], [1, 1], [0, 1])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(QPInputError):
                solve(*args)
        with self.assertRaises(QPInputError):
            solve([[1]], [1], [0], "buffer_subsampling")
        with self.assertRaises(QPInputError):
            solve(np.eye(1025), np.ones(1025), list(range(1025)))

    def test_empty_problem(self):
        result = solve(np.empty((0, 0)), [], [])
        self.assertEqual(result["alpha"], [])
        self.assertCertified(result)

    def test_iteration_limit_is_typed_technical_failure(self):
        policy = replace(DEFAULT_QP_POLICY, max_pivots=1)
        with self.assertRaises(QPSolverFailure) as ctx:
            from_rows([[1, 0], [-1, 1]], [1, -.25], policy=policy)
        self.assertEqual(ctx.exception.code, "QP_ITERATION_LIMIT")
        self.assertNotIsInstance(ctx.exception, QPInfeasible)

    def test_nearly_opposing_feasible_rows_are_not_called_infeasible(self):
        # A huge but finite correction exists; there is no scientific norm cap.
        h = np.array([[1., 0.], [-1., 1e-7]])
        try:
            result = solve(h @ h.T, [1, 1], [0, 1])
        except QPSolverFailure as exc:
            self.assertIn("UNRESOLVED", exc.code)
        else:
            self.assertCertified(result)
        h = np.array([[1., 0.], [-1., 1e-3]])
        direction, result = from_rows(h, [1, 1])
        self.assertGreater(np.linalg.norm(direction), 1000)
        self.assertCertified(result)

    def test_uncertified_subsolver_cannot_false_pass(self):
        with patch(f"{solve.__module__}._working_solve", return_value=(np.zeros(1), {})):
            with self.assertRaises(QPSolverFailure) as ctx:
                from_rows([[1]], [1])
        self.assertEqual(ctx.exception.code, "QP_WORKING_KKT_UNRESOLVED")
        self.assertEqual(ctx.exception.receipt["status"], "QP_WORKING_KKT_UNRESOLVED")

    def test_stable_id_seed_and_direction_diversity(self):
        # Positive violation seed must win even though almost parallel rows
        # share its direction. The opposite safe row is maximally diverse.
        h = np.vstack((np.tile([1., 0., 0.], (35, 1)), [-1., 0., 0.], [0., 1., 0.]))
        b = np.full(len(h), -.2)
        b[2] = b[3] = 1
        b[35] = -2
        ids = [f"id-{i:03}" for i in range(len(h))]
        ids[2], ids[3] = "seed-z", "seed-a"
        result = solve(h @ h.T, b, ids)
        added = result["working_set_history"][0]["added_rows"]
        self.assertEqual(added[0], 3)
        self.assertEqual(added[1], 35)
        self.assertEqual(added[2], 36)
        self.assertEqual(len(added), 32)
        self.assertCertified(result)

    def test_all_initially_safe_rows_rechecked_after_every_block(self):
        # The initial 32 orthogonal positives pull coordinate 0 above an
        # initially safe row, whose almost parallel normal is picked later.
        h = np.vstack((np.eye(40), np.r_[-1., np.zeros(39),][None, :]))
        h[-1, 39] = 1
        b = np.r_[np.ones(40), -.25]
        result = solve(h @ h.T, b, list(range(41)))
        self.assertCertified(result)
        self.assertTrue(all(row["all_rows_checked"] == 41 for row in result["working_set_history"]))
        direction = np.asarray(result["alpha"]) @ h
        np.testing.assert_allclose(direction, np.ones(40), atol=1e-12)

    def test_random_small_against_independent_enumeration(self):
        rng = np.random.default_rng(20260918)
        for case in range(30):
            h = rng.normal(size=(7, 3))
            feasible = rng.normal(size=3)
            b = h @ feasible - rng.uniform(0, 1, len(h))
            expected = enumerated_projection(h, b)
            self.assertIsNotNone(expected)
            direction, result = from_rows(h, b)
            np.testing.assert_allclose(direction, expected, atol=2e-8, err_msg=str(case))
            self.assertCertified(result)

    def test_random_512_rank_deficient_against_primal_slsqp(self):
        rng = np.random.default_rng(2026091801)
        h = rng.normal(size=(512, 24))
        feasible = rng.normal(size=24)
        b = h @ feasible - rng.uniform(.01, 3, 512)
        report = audit_same_problem(h @ h.T, b, [f"row-{i:04}" for i in range(512)])
        self.assertTrue(report["pass"])
        direction = np.asarray(report["results"]["gss"]["alpha"]) @ h
        reference = minimize(lambda d: .5 * (d @ d), feasible,
                             jac=lambda d: d, method="SLSQP",
                             constraints=[LinearConstraint(h, b, np.inf)],
                             options={"ftol": 1e-12, "maxiter": 500})
        self.assertTrue(reference.success, reference.message)
        np.testing.assert_allclose(direction, reference.x, atol=2e-7)

    def test_large_512_all_binding_orthogonal_full_bank(self):
        gram = np.eye(512)
        b = np.linspace(.2, 2, 512)
        report = audit_same_problem(gram, b, [f"p-{i:04}" for i in range(512)])
        self.assertTrue(report["pass"])
        for result in report["results"].values():
            np.testing.assert_allclose(result["alpha"], b, atol=1e-12)
            self.assertEqual(len(result["working_rows"]), 512)
            self.assertCertified(result)
        history = report["results"]["gss"]["working_set_history"]
        self.assertEqual(len(history), 16)
        self.assertTrue(all(len(x["added_rows"]) <= 32 for x in history))

    def test_1024_rows_no_bank_truncation(self):
        # Repeated constraints admit nonunique alpha but a unique direction.
        h = np.tile(np.eye(16), (64, 1))
        b = np.tile(np.linspace(.2, 2, 16), 64)
        report = audit_same_problem(h @ h.T, b, list(range(1024)))
        self.assertTrue(report["pass"])
        for result in report["results"].values():
            np.testing.assert_allclose(np.asarray(result["alpha"]) @ h,
                                       np.linspace(.2, 2, 16), atol=1e-10)
            self.assertCertified(result)


if __name__ == "__main__":
    unittest.main()
