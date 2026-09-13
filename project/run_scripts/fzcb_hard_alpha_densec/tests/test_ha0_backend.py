from __future__ import annotations

import math
import unittest

import torch

from project.run_scripts.fzcb_hard_alpha_densec.cache_transaction import DenseCacheTransaction
from project.run_scripts.fzcb_hard_alpha_densec.contracts import NumericalLock, TrajectoryBoundary
from project.run_scripts.fzcb_hard_alpha_densec.controller import (
    analytic_barrier_correction,
    envelope_directional_derivative,
    minimum_action,
    null_projection,
    suffix_value,
)
from project.run_scripts.fzcb_hard_alpha_densec.dense_backend import (
    ContextIncidence,
    DenseEntryFactor,
    match_update_to_weight,
)
from project.run_scripts.fzcb_hard_alpha_densec.matrix_free import LinearOperator
from project.run_scripts.fzcb_hard_alpha_densec.official_compat import install_official_trace_kwargs_adapter


DTYPE = torch.float64


def orthogonal_projector(size: int, rank: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260901 + size + rank)
    raw = torch.randn((size, rank), dtype=DTYPE, generator=generator)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    return q @ q.T


class DenseBackendTest(unittest.TestCase):
    def test_official_trace_kwargs_adapter_preserves_output(self) -> None:
        from easyeditor.util import nethook

        receipt = install_official_trace_kwargs_adapter()
        class HiddenIdentity(torch.nn.Module):
            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return hidden_states

        layer = HiddenIdentity()
        value = torch.randn((2, 3), dtype=DTYPE)
        with nethook.Trace(layer, retain_input=True, retain_output=True) as trace:
            output = layer(hidden_states=value)
        self.assertTrue(receipt["installed"])
        self.assertTrue(torch.equal(output, value))
        self.assertTrue(torch.equal(trace.input, value))
        self.assertTrue(torch.equal(trace.output, value))

    def exercise(self, history_scale: float) -> tuple[float, float]:
        generator = torch.Generator().manual_seed(19)
        size, requests = 13, 4
        projector = orthogonal_projector(size, 7)
        raw = torch.randn((size, size), dtype=DTYPE, generator=generator)
        history = history_scale * (raw @ raw.T)
        keys = torch.randn((size, requests), dtype=DTYPE, generator=generator)
        factor = DenseEntryFactor.build(projector, history, 0.7)
        if history_scale:
            self.assertFalse(torch.equal(factor.entry_matrix, factor.entry_matrix.T))
        woodbury = factor.woodbury(keys)
        direct = factor.direct(keys)
        relative = float(woodbury.direct_relative(direct).item())
        residual = float(woodbury.solve_relative.item())
        self.assertLess(relative, 2e-12)
        self.assertLess(residual, 2e-12)
        return relative, residual

    def test_zero_history_direct_woodbury_parity(self) -> None:
        self.exercise(0.0)

    def test_nonzero_history_direct_woodbury_parity_and_nonsymmetric_lu(self) -> None:
        self.exercise(0.03)

    def test_identity_projector_reduction(self) -> None:
        generator = torch.Generator().manual_seed(31)
        size, requests = 9, 3
        projector = torch.eye(size, dtype=DTYPE)
        history = torch.zeros((size, size), dtype=DTYPE)
        keys = torch.randn((size, requests), dtype=DTYPE, generator=generator)
        factor = DenseEntryFactor.build(projector, history, 1.3)
        actual = factor.woodbury(keys).d
        expected = torch.linalg.solve(keys @ keys.T + 1.3 * projector, keys)
        self.assertLess(float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)), 2e-12)

    def test_hard_range_and_update_orientation(self) -> None:
        generator = torch.Generator().manual_seed(47)
        size, requests, out = 11, 3, 5
        projector = orthogonal_projector(size, 6)
        keys = torch.randn((size, requests), dtype=DTYPE, generator=generator)
        residual = torch.randn((out, requests), dtype=DTYPE, generator=generator)
        solve = DenseEntryFactor.build(projector, torch.zeros_like(projector), 0.4).woodbury(keys)
        update = solve.update(residual)
        self.assertEqual(update.shape, (out, size))
        self.assertLess(float(solve.coefficient_leakage(projector).item()), 2e-12)
        self.assertLess(float(torch.linalg.vector_norm(update @ (torch.eye(size, dtype=DTYPE) - projector))), 2e-11)

    def test_single_shape_adapter_matches_official(self) -> None:
        from easyeditor.models.alphaedit.AlphaEdit_main import upd_matrix_match_shape

        update = torch.arange(15, dtype=DTYPE).reshape(3, 5)
        ours_direct, transposed_direct = match_update_to_weight(update, torch.Size((3, 5)))
        ours_transposed, transposed = match_update_to_weight(update, torch.Size((5, 3)))
        self.assertFalse(transposed_direct)
        self.assertTrue(transposed)
        self.assertTrue(torch.equal(ours_direct, upd_matrix_match_shape(update, torch.Size((3, 5)))))
        self.assertTrue(torch.equal(ours_transposed, upd_matrix_match_shape(update, torch.Size((5, 3)))))

    def test_request_context_incidence_parity(self) -> None:
        generator = torch.Generator().manual_seed(61)
        request_ids = (4, 9, 12)
        context_ids = (4, 4, 9, 9, 12, 12)
        incidence = ContextIncidence.from_context_request_ids(
            request_ids, context_ids, dtype=DTYPE, device=torch.device("cpu")
        )
        self.assertEqual(incidence.repeat_factor, 2)
        d = torch.randn((8, 6), dtype=DTYPE, generator=generator)
        residual = torch.randn((7, 3), dtype=DTYPE, generator=generator)
        self.assertTrue(torch.allclose(incidence.direct_update(residual, d), incidence.effective_update(residual, d)))
        identity = ContextIncidence.from_context_request_ids(
            request_ids, request_ids, dtype=DTYPE, device=torch.device("cpu")
        )
        self.assertTrue(torch.equal(identity.matrix, torch.eye(3, dtype=DTYPE)))


class MatrixFreeControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = NumericalLock(lsqr_max_iterations=200)
        generator = torch.Generator().manual_seed(73)
        self.matrix = torch.randn((5, 8), dtype=DTYPE, generator=generator)
        self.operator = LinearOperator(
            forward=lambda value: self.matrix @ value,
            adjoint=lambda value: self.matrix.T @ value,
            domain_shape=(8,),
            range_shape=(5,),
            dtype=DTYPE,
            device=torch.device("cpu"),
        )

    def test_adjoint_and_minimum_action(self) -> None:
        self.assertLess(self.operator.check_adjoint(), 2e-14)
        rhs = torch.arange(1, 6, dtype=DTYPE)
        solved = minimum_action(self.operator, rhs, self.lock)
        reference = torch.linalg.lstsq(self.matrix, rhs, driver="gelsd").solution
        self.assertLess(float(torch.linalg.vector_norm(solved.coefficient - reference)), 2e-11)
        self.assertLess(solved.range_receipt.relative_residual, self.lock.fp64_relative_tolerance)

    def test_full_null_projection(self) -> None:
        vector = torch.linspace(-2, 2, 8, dtype=DTYPE)
        free, receipt = null_projection(self.operator, vector, self.lock)
        self.assertTrue(receipt.converged)
        self.assertLess(float(torch.linalg.vector_norm(self.matrix @ free)), 2e-11)
        projected = vector - free
        self.assertLess(float(torch.dot(projected, free).abs()), 2e-11)

    def test_terminal_limit_semantics(self) -> None:
        zero = torch.zeros(5, dtype=DTYPE)
        value = suffix_value(self.operator, zero, 0.0, self.lock, terminal_relative_residual=0.0)
        self.assertTrue(value.terminal)
        self.assertEqual(value.value, 0.0)
        with self.assertRaises(TrajectoryBoundary):
            suffix_value(self.operator, torch.ones(5, dtype=DTYPE), 0.0, self.lock, terminal_relative_residual=0.0)

    def test_envelope_derivative_matches_central_fd(self) -> None:
        generator = torch.Generator().manual_seed(89)
        c0 = torch.randn((3, 5), dtype=DTYPE, generator=generator)
        cd = 0.07 * torch.randn((3, 5), dtype=DTYPE, generator=generator)
        h0_raw = torch.randn((5, 5), dtype=DTYPE, generator=generator)
        hd_raw = torch.randn((5, 5), dtype=DTYPE, generator=generator)
        h0 = h0_raw.T @ h0_raw + 0.7 * torch.eye(5, dtype=DTYPE)
        hd = 0.03 * (hd_raw + hd_raw.T)
        b0 = torch.randn(3, dtype=DTYPE, generator=generator)
        bd = 0.04 * torch.randn(3, dtype=DTYPE, generator=generator)
        r0, rd = 0.8, -0.13

        def solve_value(t: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
            c = c0 + t * cd
            h = h0 + t * hd
            b = b0 + t * bd
            remaining = r0 + t * rd
            hinv_ct = torch.linalg.solve(h, c.T)
            gram = c @ hinv_ct
            lam = torch.linalg.solve(gram, b) / remaining
            q = remaining * hinv_ct @ lam
            value = float((0.5 / remaining * q @ h @ q).item())
            return q, lam, h, value

        q, lam, h, _ = solve_value(0.0)
        analytic = envelope_directional_derivative(
            displacement=q,
            metric_apply=lambda value: h @ value,
            metric_direction_apply=lambda value: hd @ value,
            multiplier=lam,
            rhs_direction=bd,
            operator_direction_displacement=cd @ q,
            remaining=r0,
            remaining_direction=rd,
        )
        epsilon = 1e-5
        finite = (solve_value(epsilon)[3] - solve_value(-epsilon)[3]) / (2 * epsilon)
        self.assertLess(abs(analytic - finite), 2e-8)

    def test_analytic_alpha_matches_first_feasible_root(self) -> None:
        psi0, guide, tolerance = 0.42, 1.25, 0.03
        answer = analytic_barrier_correction(
            psi0=psi0,
            guide_norm_sq=guide,
            cbf_tolerance=tolerance,
            guide_tolerance=1e-14,
        )
        expected = 1.0 - math.sqrt(1.0 - 2.0 * (psi0 - tolerance) / guide)
        self.assertAlmostEqual(answer.alpha, expected, places=14)
        self.assertLessEqual(answer.predicted, tolerance + 2e-15)
        self.assertGreater(psi0 - (answer.alpha - 1e-6) * guide + 0.5 * (answer.alpha - 1e-6) ** 2 * guide, tolerance)


class DenseCacheTransactionTest(unittest.TestCase):
    def test_append_once_reject_zero_and_replay(self) -> None:
        initial = {4: torch.zeros((5, 5)), 5: torch.zeros((4, 4))}
        cache = DenseCacheTransaction(initial)
        version, _ = cache.begin_batch()
        self.assertEqual(version, 0)
        self.assertEqual(cache.observe_trial(), 0)
        cache.reject_batch()
        self.assertEqual(cache.version, 0)
        self.assertEqual(cache.trial_mutation_count, 0)
        cache.begin_batch()
        keys = {4: torch.arange(10, dtype=torch.float32).reshape(5, 2), 5: torch.arange(8, dtype=torch.float32).reshape(4, 2)}
        append = cache.commit_batch(keys, (101, 102))
        self.assertEqual((append.version_before, append.version_after), (0, 1))
        actual = cache.materialize()
        replay = cache.replay_from_checkpoint()
        for layer in initial:
            expected = keys[layer] @ keys[layer].T
            self.assertTrue(torch.equal(actual[layer], expected))
            self.assertTrue(torch.equal(replay[layer], expected))
        self.assertEqual(len(cache.wal), 1)
        self.assertEqual(cache.version, 1)


if __name__ == "__main__":
    unittest.main()
