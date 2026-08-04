from __future__ import annotations

import hashlib
import unittest

import numpy as np
import torch

from project.run_scripts.ode_bf.contracts import FieldIdentity, ODEBFContractError
from project.run_scripts.ode_bf.routing import (
    CoupledField,
    QuadraticBarrier,
    RoutingProblem,
    RoutingStatus,
    project_bf_velocity,
    solve_raw_velocity,
    verify_backtracked_candidate,
)
from project.run_scripts.ode_bf.woodbury import (
    ProjectorCertificate,
    WoodburyMethod,
    full_projector_certificate,
    solve_alpha_woodbury,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class WoodburyTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(23)
        self.dimension = 14
        self.current = torch.randn((14, 10), generator=generator, dtype=torch.float64)
        self.history = torch.randn((14, 3), generator=generator, dtype=torch.float64)
        self.regularization = 2.5

    def _dense(self, projector: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((self.current, history), dim=1)
        matrix = self.regularization * torch.eye(self.dimension, dtype=torch.float64)
        matrix = matrix + projector @ combined @ combined.T
        return torch.linalg.solve(matrix, projector @ self.current)

    def test_symmetric_idempotent_cholesky_matches_dense(self) -> None:
        generator = torch.Generator().manual_seed(31)
        basis, _ = torch.linalg.qr(torch.randn((14, 8), generator=generator, dtype=torch.float64))
        projector = basis @ basis.T
        certificate = full_projector_certificate(
            projector,
            source_sha256=_digest("symmetric-projector"),
        )
        self.assertTrue(certificate.exact_symmetric_idempotent)
        result = solve_alpha_woodbury(
            projector,
            self.current,
            history_keys=self.history,
            regularization=self.regularization,
            projector_certificate=certificate,
        )
        self.assertEqual(result.certificate.method, WoodburyMethod.SYMMETRIC_CHOLESKY)
        self.assertTrue(torch.allclose(result.q, self._dense(projector, self.history), rtol=1e-10, atol=1e-10))

    def test_general_projector_uses_raw_x_contractions_and_matches_dense(self) -> None:
        generator = torch.Generator().manual_seed(37)
        projector = 0.15 * torch.randn((14, 14), generator=generator, dtype=torch.float64)
        projector += torch.eye(14, dtype=torch.float64)
        certificate = ProjectorCertificate(
            _digest("general-projector"),
            0.4,
            0.3,
            "artifact-unverified",
            1e-10,
        )
        result = solve_alpha_woodbury(
            projector,
            self.current,
            history_keys=self.history,
            regularization=self.regularization,
            projector_certificate=certificate,
        )
        self.assertIn(result.certificate.method, (WoodburyMethod.GENERAL_LU, WoodburyMethod.GENERAL_QR))
        expected = self._dense(projector, self.history)
        self.assertTrue(torch.allclose(result.q, expected, rtol=1e-10, atol=1e-10))
        combined = torch.cat((self.current, self.history), dim=1)
        z = projector @ combined
        g = projector @ self.current
        wrong_small = self.regularization * torch.eye(combined.shape[1], dtype=torch.float64) + z.T @ z
        wrong_rhs = z.T @ g
        wrong = (g - z @ torch.linalg.solve(wrong_small, wrong_rhs)) / self.regularization
        self.assertGreater(float(torch.linalg.norm(wrong - expected)), 1.0e-4)

    def test_history_column_permutation_and_qr_fallback_are_invariant(self) -> None:
        generator = torch.Generator().manual_seed(41)
        projector = torch.eye(14, dtype=torch.float64) + 0.03 * torch.randn((14, 14), generator=generator, dtype=torch.float64)
        certificate = ProjectorCertificate(_digest("general"), 0.1, 0.1, "action-probe", 1e-10)
        first = solve_alpha_woodbury(
            projector,
            self.current,
            history_keys=self.history,
            regularization=self.regularization,
            projector_certificate=certificate,
            maximum_condition=1.0,
        )
        permutation = torch.tensor([2, 0, 1])
        second = solve_alpha_woodbury(
            projector,
            self.current,
            history_keys=self.history[:, permutation],
            regularization=self.regularization,
            projector_certificate=certificate,
            maximum_condition=1.0,
        )
        self.assertEqual(first.certificate.method, WoodburyMethod.GENERAL_QR)
        self.assertTrue(torch.allclose(first.q, second.q, rtol=1e-9, atol=1e-9))

    def test_wrong_joint_key_shape_fails(self) -> None:
        with self.assertRaisesRegex(ODEBFContractError, "shape"):
            solve_alpha_woodbury(
                torch.eye(14, dtype=torch.float64),
                self.current[:, :1],
                history_keys=None,
                regularization=1.0,
                projector_certificate=ProjectorCertificate(_digest("p"), 0.0, 0.0, "full-matrix", 1e-10),
            )


def _problem(*, tight: bool = False) -> RoutingProblem:
    progress = np.array([1.0, 0.8, -0.4, 0.55, 0.25], dtype=np.float64)
    capacity = np.diag([1.0, 1.2, 0.9, 1.5, 2.0])
    trust = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])
    h_gram = np.diag([3.0, 0.1, 0.1, 0.1, 0.1])
    p_gram = np.diag([0.2, 0.2, 0.1, 1.5, 0.2])
    budget = 0.003 if tight else 0.35
    historical = QuadraticBarrier("historical", 0.0, np.zeros(5), h_gram, budget, "layer-local-diagonal")
    pretrained = QuadraticBarrier("pretrained", 0.0, np.zeros(5), p_gram, budget, "layer-local-diagonal")
    return RoutingProblem(
        progress,
        capacity,
        trust,
        1.5,
        np.ones(5),
        0.45 if tight else 0.6,
        0.4,
        historical,
        pretrained,
    )


class RoutingTests(unittest.TestCase):
    def test_signed_nonpositive_direction_is_excluded_and_bf_projects_same_raw(self) -> None:
        problem = _problem()
        raw = solve_raw_velocity(problem)
        self.assertEqual(raw.values[2], 0.0)
        projected = project_bf_velocity(problem, raw)
        self.assertEqual(projected.status, RoutingStatus.FEASIBLE)
        self.assertEqual(projected.raw_velocity_identity, raw.velocity_identity)
        assert projected.values is not None
        self.assertGreaterEqual(float(problem.signed_progress @ projected.values), problem.requested_progress - 1e-8)
        self.assertLessEqual(problem.historical.value(projected.values), problem.historical.budget + 1e-8)
        self.assertGreater(float(np.linalg.norm(projected.values - raw.values)), 0.0)

    def test_progress_infeasible_is_explicit_and_never_underwritten(self) -> None:
        problem = _problem(tight=True)
        raw = solve_raw_velocity(problem)
        projected = project_bf_velocity(problem, raw)
        self.assertEqual(projected.status, RoutingStatus.PROGRESS_INFEASIBLE)
        self.assertIsNone(projected.values)

    def test_permutation_invariance(self) -> None:
        problem = _problem()
        raw = solve_raw_velocity(problem)
        projected = project_bf_velocity(problem, raw)
        assert projected.values is not None
        permutation = np.array([3, 0, 4, 1, 2])
        inverse = np.argsort(permutation)
        h = problem.historical
        p = problem.pretrained
        permuted = RoutingProblem(
            problem.signed_progress[permutation],
            problem.capacity_metric[np.ix_(permutation, permutation)],
            problem.trust_metric[np.ix_(permutation, permutation)],
            problem.trust_radius,
            problem.layer_caps[permutation],
            problem.requested_progress,
            problem.minimum_progress,
            QuadraticBarrier("historical", h.offset, h.linear[permutation], h.gram[np.ix_(permutation, permutation)], h.budget, h.approximation),
            QuadraticBarrier("pretrained", p.offset, p.linear[permutation], p.gram[np.ix_(permutation, permutation)], p.budget, p.approximation),
        )
        permuted_raw = solve_raw_velocity(permuted)
        permuted_projected = project_bf_velocity(permuted, permuted_raw)
        assert permuted_projected.values is not None
        self.assertTrue(np.allclose(projected.values, permuted_projected.values[inverse], rtol=1e-6, atol=1e-7))

    def test_backtracking_uses_beta_scaled_prediction_and_trust_ratio(self) -> None:
        problem = _problem()
        raw = solve_raw_velocity(problem)
        raw_prediction = float(problem.signed_progress @ raw.values)
        verdicts = [
            verify_backtracked_candidate(
                problem,
                raw.values,
                beta=beta,
                actual_signed_progress=beta * raw_prediction * 0.2,
                functional_h_pass=True,
                functional_p_pass=True,
                authoritative_bf16_pass=True,
            )
            for beta in (1.0, 0.5, 0.25)
        ]
        self.assertEqual(
            [item.predicted_beta_progress for item in verdicts],
            [raw_prediction, 0.5 * raw_prediction, 0.25 * raw_prediction],
        )
        self.assertTrue(all(item.progress_pass for item in verdicts))
        self.assertTrue(all(abs((item.trust_ratio or 0.0) - 0.2) < 1.0e-12 for item in verdicts))
        self.assertLess(
            verdicts[-1].actual_signed_progress,
            problem.requested_progress,
        )

        low_ratio = verify_backtracked_candidate(
            problem,
            raw.values,
            beta=0.5,
            actual_signed_progress=0.5 * raw_prediction * 0.05,
            functional_h_pass=True,
            functional_p_pass=True,
            authoritative_bf16_pass=True,
        )
        self.assertFalse(low_ratio.progress_pass)
        self.assertEqual(low_ratio.first_rejecting_gate, "progress")

    def test_backtracking_rejects_nonpositive_or_nonfinite_prediction(self) -> None:
        problem = _problem()
        raw = solve_raw_velocity(problem)
        common = {
            "beta": 0.5,
            "actual_signed_progress": 0.1,
            "functional_h_pass": True,
            "functional_p_pass": True,
            "authoritative_bf16_pass": True,
        }
        negative = verify_backtracked_candidate(problem, -np.abs(raw.values), **common)
        zero = verify_backtracked_candidate(problem, np.zeros_like(raw.values), **common)
        nonfinite_values = raw.values.copy()
        nonfinite_values[0] = np.nan
        nonfinite = verify_backtracked_candidate(problem, nonfinite_values, **common)
        for verdict in (negative, zero, nonfinite):
            self.assertFalse(verdict.progress_pass)
            self.assertFalse(verdict.accepted)
            self.assertEqual(verdict.first_rejecting_gate, "progress")
        self.assertEqual(negative.predicted_beta_progress, 0.0)
        self.assertEqual(zero.predicted_beta_progress, 0.0)
        self.assertIsNone(nonfinite.predicted_beta_progress)

    def test_acceptance_hierarchy_reports_first_independent_rejecting_gate(self) -> None:
        problem = _problem()
        raw = solve_raw_velocity(problem)
        projected = project_bf_velocity(problem, raw)
        assert projected.values is not None
        predicted = float(problem.signed_progress @ projected.values)
        verdict = verify_backtracked_candidate(
            problem,
            projected.values,
            beta=1.0,
            actual_signed_progress=0.2 * predicted,
            functional_h_pass=True,
            functional_p_pass=False,
            authoritative_bf16_pass=True,
        )
        self.assertTrue(verdict.progress_pass)
        self.assertTrue(verdict.structural_h_pass)
        self.assertTrue(verdict.structural_p_pass)
        self.assertTrue(verdict.trust_pass)
        self.assertEqual(verdict.first_rejecting_gate, "functional_p")
        self.assertFalse(verdict.accepted)

    def test_coupled_field_uses_one_beta_and_target_backward_receipt(self) -> None:
        field = CoupledField(
            FieldIdentity("batch", 0, 0, 0, 0, 0),
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0, 5.0]),
            1,
        )
        write, target = field.beta_scaled(0.25)
        self.assertTrue(np.array_equal(write, np.array([0.25, 0.5])))
        self.assertTrue(np.array_equal(target, np.array([0.75, 1.0, 1.25])))

    def test_psd_and_diagonal_approximation_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(ODEBFContractError, "PSD"):
            QuadraticBarrier(
                "historical",
                0.0,
                np.zeros(2),
                np.array([[1.0, 0.0], [0.0, -1.0]]),
                1.0,
                "common-functional-gram",
                _digest("space"),
            )
        with self.assertRaisesRegex(ODEBFContractError, "not diagonal"):
            QuadraticBarrier(
                "pretrained",
                0.0,
                np.zeros(2),
                np.array([[1.0, 0.1], [0.1, 1.0]]),
                1.0,
                "layer-local-diagonal",
            )


if __name__ == "__main__":
    unittest.main()
