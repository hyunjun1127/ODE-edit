"""CPU synthetic checks; not actual model or scientific efficacy evidence."""
import unittest
from unittest.mock import patch

import numpy as np

from .basis import (BasisTechnicalError, build_functional_basis,
                    covariance_action, append_covariance_direction, contract_factors)
from .solver import (SolverTechnicalError, solve_coefficients,
                     history_backtracking_plan, proposal_radius, risk)


class BasisTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260919)

    def test_exact_residual_reconstructs_high_rank_input(self):
        h = self.rng.normal(size=(31, 43))
        b = build_functional_basis(h)
        self.assertEqual(b.rank, 4)
        self.assertLess(b.receipt["gradient_span_relative_error"], 1e-13)
        self.assertLess(b.receipt["orthonormal_max_abs_error"], 1e-13)
        self.assertEqual(b.receipt["seed"], 20260919)
        np.testing.assert_allclose(b.reconstruct(b.receipt["gradient_span_coefficients"]), h,
                                   rtol=1e-12, atol=1e-12)

    def test_repeat_is_deterministic(self):
        h = self.rng.normal(size=(13, 19))
        one, two = build_functional_basis(h), build_functional_basis(h)
        for a, b in zip(one.directions, two.directions):
            np.testing.assert_array_equal(a, b)

    def test_no_full_svd(self):
        h = self.rng.normal(size=(31, 43))
        from scipy import linalg
        original = linalg.svd
        seen = []
        def limited(matrix, *args, **kwargs):
            seen.append(matrix.shape)
            self.assertLessEqual(matrix.shape[0], 8)
            self.assertFalse(kwargs["full_matrices"])
            return original(matrix, *args, **kwargs)
        with patch("project.run_scripts.single_layer_mechanism_first.basis.linalg.svd", limited):
            build_functional_basis(h)
        self.assertEqual(seen, [(8, 43)])

    def test_rank_one_is_not_called_four_directions(self):
        h = np.outer(self.rng.normal(size=17), self.rng.normal(size=25))
        b = build_functional_basis(h)
        self.assertEqual(b.rank, 1)
        self.assertLess(b.receipt["gradient_span_relative_error"], 1e-13)

    def test_project_reorthonormalize(self):
        projector = np.diag([1, 1, 1, 0, 0, 0])
        h = self.rng.normal(size=(9, 6)) @ projector
        b = build_functional_basis(h, project=lambda d: d @ projector)
        for d in b.directions:
            np.testing.assert_array_equal(d[:, 3:], 0)
        self.assertLess(b.receipt["gradient_span_relative_error"], 1e-13)

    def test_inconsistent_projector_is_not_success(self):
        with self.assertRaises(BasisTechnicalError):
            build_functional_basis(np.eye(4), project=lambda d: np.zeros_like(d))

    def test_zero_and_nonfinite_are_distinct(self):
        b = build_functional_basis(np.zeros((4, 6)))
        self.assertEqual(b.rank, 0)
        self.assertEqual(b.receipt["status"], "NO_PROJECTED_DIRECTION")
        with self.assertRaises(BasisTechnicalError):
            build_functional_basis(np.full((4, 6), np.nan))

    def test_covariance_document_means_not_token_pool(self):
        e = self.rng.normal(size=(3, 5))
        keys = [self.rng.normal(size=(5, n)) for n in [1, 3, 11]]
        expected = -e @ sum(k @ k.T / k.shape[1] for k in keys) / len(keys)
        actual, receipt = covariance_action(e, iter(keys), expected_documents=3)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        self.assertEqual(receipt["valid_input_tokens"], 15)
        self.assertFalse(receipt["dense_covariance_created"])

    def test_covariance_completeness(self):
        for count in [1, 3]:
            with self.assertRaises(BasisTechnicalError):
                covariance_action(np.eye(4), [np.eye(4)] * count, expected_documents=2)

    def test_covariance_rank_at_most_five(self):
        h, e = self.rng.normal(size=(13, 17)), self.rng.normal(size=(13, 17))
        base = build_functional_basis(h)
        result = append_covariance_direction(base, e, [self.rng.normal(size=(17, 3))],
                                             expected_documents=1)
        self.assertEqual(result.rank, 5)
        self.assertLess(result.receipt["orthonormal_max_abs_error"], 1e-13)
        with self.assertRaises(BasisTechnicalError):
            append_covariance_direction(result, e, [np.eye(17)], expected_documents=1)

    def test_step_cum_b1_identity(self):
        e = self.rng.normal(size=(5, 8))
        keys = [self.rng.normal(size=(8, n)) for n in [3, 7]]
        step, _ = covariance_action(e, keys, expected_documents=2)
        cum, _ = covariance_action(e.copy(), keys, expected_documents=2)
        np.testing.assert_array_equal(step, cum)

    def test_factor_contraction_against_dense_gradient(self):
        a, k = self.rng.normal(size=(5, 9)), self.rng.normal(size=(8, 9))
        b = build_functional_basis(self.rng.normal(size=(5, 8)))
        actual = contract_factors(a, k, b)
        dense = a @ k.T
        expected = np.array([np.sum(dense * d) for d in b.directions])
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


class SolverTests(unittest.TestCase):
    def assert_solved(self, result):
        self.assertEqual(result.status, "LOCAL_TWO_PHASE_SOLVED", result.receipt)
        self.assertIsNotNone(result.coefficients)
        self.assertTrue(result.receipt["phase1"]["primal"]["passed"])
        self.assertTrue(result.receipt["phase2"]["primal"]["passed"])
        self.assertTrue(result.receipt["phase2"]["risk_constraint_pass"])

    def test_one_dimensional_closed_form_phase2(self):
        result = solve_coefficients([-1.0], [[1.0]], 2.0)
        self.assert_solved(result)
        # f*=0, risk tolerance=1e-12; min norm a=1-sqrt(1e-12).
        self.assertAlmostEqual(result.coefficients[0], 1.0-1e-6, places=8)

    def test_ball_active_two_dimensional_closed_form(self):
        result = solve_coefficients([-2., -1.], np.eye(2), 1.0)
        self.assert_solved(result)
        expected = np.array([2., 1.])/np.sqrt(5)
        np.testing.assert_allclose(result.coefficients, expected, atol=2e-7, rtol=0)
        value, _ = risk(np.array([-2.,-1.]), np.eye(2), result.coefficients)
        self.assertAlmostEqual(value, ((np.sqrt(5)-1)**2)/2, places=7)

    def test_safe_reference_is_hard_row(self):
        result = solve_coefficients([-1., .1], [[1.], [-1.]], 2.0)
        self.assert_solved(result)
        self.assertLessEqual(result.coefficients[0], .1+1e-8)
        self.assertGreater(result.coefficients[0], .09999)

    def test_unsafe_nondegradation_can_force_zero(self):
        result = solve_coefficients([-1., -1.], [[1.], [-1.]], 2.0)
        self.assert_solved(result)
        np.testing.assert_allclose(result.coefficients, 0, atol=1e-10)

    def test_history_repair_feasible(self):
        result = solve_coefficients([-1.], [[1.]], 2.0,
                                     history_slack=[-.25], history_J=[[1.]])
        self.assert_solved(result)
        self.assertGreaterEqual(result.coefficients[0], .25-1e-8)

    def test_history_ball_infeasible_needs_certificate(self):
        result = solve_coefficients([1.], [[1.]], 1.0,
                                     history_slack=[-2.], history_J=[[1.]])
        self.assertEqual(result.status, "LOCAL_HISTORY_CONSTRAINT_INFEASIBLE")
        cert = result.receipt["infeasibility_certificate"]
        self.assertTrue(cert["verified"])
        self.assertGreater(cert["strict_contradiction_gap"], cert["roundoff_envelope"])

    def test_history_contradictory_rows_certificate(self):
        result = solve_coefficients([-1.], [[0.]], 10.0,
                                     history_slack=[-1., -1.], history_J=[[1.], [-1.]])
        self.assertEqual(result.status, "LOCAL_HISTORY_CONSTRAINT_INFEASIBLE")
        self.assertTrue(result.receipt["infeasibility_certificate"]["verified"])

    def test_joint_rows_vs_ball_certificate(self):
        result = solve_coefficients([-1.], [[0., 0.]], 1.0,
                                     history_slack=[-.8, -.8], history_J=np.eye(2))
        self.assertEqual(result.status, "LOCAL_HISTORY_CONSTRAINT_INFEASIBLE")
        self.assertTrue(result.receipt["infeasibility_certificate"]["verified"])

    def test_finite_unresolved_not_infeasible(self):
        with patch("project.run_scripts.single_layer_mechanism_first.solver._dual_audit",
                   return_value={"convex_lower_bound": -10., "dual_gap_upper_bound": 20.}):
            result = solve_coefficients([-2.], [[1.]], .5)
        self.assertEqual(result.status, "FINITE_SOLVER_UNRESOLVED")
        self.assertNotIn("infeasibility_certificate", result.receipt)

    def test_nonfinite_is_technical(self):
        with self.assertRaises(SolverTechnicalError):
            solve_coefficients([np.nan], [[1.]], 1.)
        with self.assertRaises(SolverTechnicalError):
            solve_coefficients([-1.], [[1.]], np.inf)

    def test_512_rows_five_dimensions(self):
        # Repeated complete rows retain all512; analytic symmetric solution.
        j = np.zeros((512, 5))
        j[:, 0] = 1
        result = solve_coefficients(-np.ones(512), j, 2.)
        self.assert_solved(result)
        self.assertEqual(result.receipt["reference_rows"], 512)
        self.assertEqual(len(result.receipt["row_labels"]), 512)
        np.testing.assert_allclose(result.coefficients, [1.-1e-6,0,0,0,0], atol=1e-8)

    def test_gradient_analytic_finite_difference(self):
        mu = np.array([-2., -.5, 1.])
        j = np.array([[1.,2.],[3.,-1.],[1.,0.]])
        a = np.array([.02,-.04])
        _, actual = risk(mu, j, a)
        numerical = []
        for axis in np.eye(2):
            numerical.append((risk(mu,j,a+1e-6*axis)[0]-risk(mu,j,a-1e-6*axis)[0])/2e-6)
        np.testing.assert_allclose(actual, numerical, atol=1e-9)

    def test_fixed_nonzero_512row_five_direction_stress(self):
        rng = np.random.default_rng(893)
        for _ in range(20):
            mu = -rng.uniform(.05, 2, size=512)
            j = rng.normal(size=(512, 5))
            j[:,0] = np.abs(j[:,0]) + .2
            result = solve_coefficients(mu, j, float(rng.uniform(.1, 2)))
            self.assert_solved(result)
            phase1 = result.receipt["phase1"]
            self.assertLessEqual(phase1["dual_audit"]["dual_gap_upper_bound"],
                                  phase1["certification_gap_tolerance"])

    def test_full_900_history_rows_retained(self):
        rng = np.random.default_rng(894)
        mu = -rng.uniform(.2, 1, size=512)
        j = np.zeros((512, 5)); j[:,0] = 1
        history_j = rng.normal(size=(900, 5))
        history_h = 10 + np.linalg.norm(history_j, axis=1)
        result = solve_coefficients(mu, j, 1., history_slack=history_h, history_J=history_j)
        self.assert_solved(result)
        self.assertEqual(len(result.receipt["row_labels"]), 1412)
        self.assertEqual(result.receipt["row_labels"][-1], "history:899")

    def test_history_skip_lower_bound(self):
        plan = history_backtracking_plan([1.], [-.3], [[1.]])
        self.assertEqual([x["skip_before_model"] for x in plan], [False,False,True,True])
        self.assertFalse(plan[0]["nonlinear_feasibility_certificate"])

    def test_zero_rank_and_current_violation(self):
        yes = solve_coefficients([-1.], np.empty((1,0)), 1.)
        self.assertEqual(yes.status, "ZERO_RADIUS_OR_BASIS")
        no = solve_coefficients([-1.], np.empty((1,0)), 1.,
                                current_slack=[-.1], current_J=np.empty((1,0)))
        self.assertEqual(no.status, "LOCAL_HISTORY_CONSTRAINT_INFEASIBLE")

    def test_risk_zero_small_and_projected_zero_are_distinct(self):
        self.assertEqual(proposal_radius(0., 0., 0.)["status"], "NO_OP_ZERO_RISK")
        self.assertEqual(proposal_radius(0., 0., 0., tie_mismatches=1)["status"], "TIE_ONLY_NO_DIRECTION")
        self.assertEqual(proposal_radius(1e-12, 1., 1.)["status"], "BELOW_RISK_RESOLUTION")
        self.assertEqual(proposal_radius(1., 1e-14, 1.)["status"], "NO_PROJECTED_DIRECTION")
        self.assertEqual(proposal_radius(1., 2., 2.)["radius"], .5)


if __name__ == "__main__":
    unittest.main()
