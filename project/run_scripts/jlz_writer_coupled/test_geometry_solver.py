"""CPU-only algebra and state-machine qualification, not a GPU claim."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import numpy as np
import torch

from .geometry import build_prior_from_npz, solve_basis, solve_layer_from_npz
from .solver import SolverConfig, group_prox, norm_penalty, solve


DTYPE = torch.float64


class GeometryTests(unittest.TestCase):
    def fixture(self):
        generator = torch.Generator().manual_seed(2102)
        keys = torch.randn(7, 6, dtype=DTYPE, generator=generator)
        square = torch.randn(7, 7, dtype=DTYPE, generator=generator)
        A = square @ square.T + 0.5 * torch.eye(7, dtype=DTYPE)
        # Variable counts and interleaved order; actual B is 3, not a profile B.
        index = torch.tensor([0, 2, 0, 1, 2, 0])
        alpha = torch.tensor([0.5, 0.3, 0.25, 1.0, 0.7, 0.25], dtype=DTYPE)
        return keys, alpha, index, A

    def test_full_context_dual_primal_and_direct_agree(self):
        keys, alpha, index, A = self.fixture()
        prior = A.clone()
        kbar = torch.zeros(7, 3, dtype=DTYPE).index_add_(1, index, keys * alpha)
        direct = torch.linalg.solve(A + (keys * alpha) @ keys.T, kbar)
        for backend in ("dual", "primal", "auto"):
            P, metadata = solve_basis(keys, alpha, index, 3, A, backend=backend,
                                      block_size=2, residual_tolerance=1e-10)
            torch.testing.assert_close(P, direct, atol=1e-12, rtol=1e-12)
            self.assertEqual(P.shape, (7, 3))
            self.assertFalse(P.requires_grad)
            self.assertLess(metadata["scaled_residual"], 1e-10)
            self.assertEqual(metadata["context_columns"], 6)
            self.assertTrue(metadata["full_context_second_moment"])
            self.assertFalse(metadata["same_M_reference_used"])
            self.assertFalse(any(isinstance(v, torch.Tensor) for v in metadata.values()))
        self.assertTrue(torch.equal(prior, A))

    def test_context_cancellation_is_not_pooled_away(self):
        A = torch.eye(2, dtype=DTYPE)
        keys = torch.tensor([[1., 1., 1.], [4., -4., 1.]], dtype=DTYPE)
        alpha = torch.tensor([.5, .5, 1.], dtype=DTYPE)
        index = torch.tensor([0, 0, 1])
        kbar = torch.tensor([[1., 1.], [0., 1.]], dtype=DTYPE)
        P, _ = solve_basis(keys, alpha, index, 2, A, backend="dual")
        pooled = torch.linalg.solve(A + kbar @ kbar.T, kbar)
        self.assertGreater(float((P - pooled).norm()), .2)
        canceled_keys = keys.clone()
        canceled_keys[1, :2] = 0.0
        canceled, _ = solve_basis(canceled_keys, alpha, index, 2, A)
        torch.testing.assert_close(canceled, pooled)
        # Request 1's column changes because request 0's canceled contexts
        # remain in the shared system: not two independent B=1 fits.
        self.assertGreater(float((P[:, 1] - canceled[:, 1]).norm()), .1)

    def test_rank_deficient_keys_and_B1(self):
        for B in (1, 2):
            keys = torch.tensor([[1., 2., 1., 2.], [2., 4., 2., 4.]], dtype=DTYPE)[:, :2 * B]
            alpha = torch.full((2 * B,), .5, dtype=torch.float32)
            index = torch.arange(B).repeat_interleave(2)
            A = torch.eye(2, dtype=DTYPE)
            dual, _ = solve_basis(keys, alpha, index, B, A, backend="dual")
            primal, meta = solve_basis(keys, alpha, index, B, A, backend="primal", block_size=1)
            torch.testing.assert_close(dual, primal, atol=1e-12, rtol=1e-12)
            self.assertEqual(meta["actual_B"], B)

    def test_auto_uses_context_count_not_request_count(self):
        keys, alpha, index, A = self.fixture()
        _, meta = solve_basis(keys, alpha, index, 3, A, dual_context_limit=4)
        self.assertEqual(meta["backend"], "equivalent_primal_blocked")

    def test_prior_fp32_division_before_fp64_and_no_history_mutation(self):
        raw = np.array([[1., .5], [.5, 17.]], dtype=np.float32)
        history = torch.tensor([[.125, .25], [.25, 3.]], dtype=torch.float32)
        original = history.clone()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stats.npz"
            np.savez(path, **{"mom2.mom2": raw, "mom2.count": 7})
            A = build_prior_from_npz(path, history, block_size=1)
            expected = (torch.from_numpy(raw) / 7).double() * 15000 + history.double()
            self.assertTrue(torch.equal(A, expected))
            self.assertFalse(torch.equal(A, torch.from_numpy(raw).double() / 7 * 15000 + history.double()))
            self.assertTrue(torch.equal(history, original))
            P, metadata = solve_layer_from_npz(path, history, torch.ones(2, 1),
                                               torch.ones(1), torch.zeros(1, dtype=torch.long), 1)
            self.assertEqual(P.shape, (2, 1))
            self.assertEqual(metadata["lambda_c"], 15000)
            self.assertEqual(metadata["C0_normalization"], "FP32_mom2_div_count_then_FP64")
            with self.assertRaisesRegex(ValueError, "FP32"):
                build_prior_from_npz(path, history.double())
            np.savez(path, **{"mom2.mom2": raw, "mom2.count": 0})
            with self.assertRaisesRegex(ValueError, "positive integer"):
                build_prior_from_npz(path, history)

    def test_rejects_indefinite_prior_without_jitter_and_invalid_inputs(self):
        keys, alpha, index, A = self.fixture()
        bad_A = A.clone()
        bad_A[0, 0] = -100.0
        for backend in ("dual", "primal"):
            with self.assertRaisesRegex(ValueError, "not SPD; no jitter"):
                solve_basis(keys, alpha, index, 3, bad_A, backend=backend)
        bad_A = A.clone()
        bad_A[0, 1] += 0.01
        with self.assertRaisesRegex(ValueError, "symmetric"):
            solve_basis(keys, alpha, index, 3, bad_A)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            solve_basis(keys, alpha * .5, index, 3, A)
        with self.assertRaisesRegex(ValueError, "integer"):
            solve_basis(keys, alpha, index.float(), 3, A)
        with self.assertRaisesRegex(ValueError, "actual batch"):
            solve_basis(keys, alpha, index + 1, 3, A)
        nan_keys = keys.clone()
        nan_keys[0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            solve_basis(nan_keys, alpha, index, 3, A)

    def test_residual_failure_has_one_same_M_reference_without_widening(self):
        keys, alpha, index, A = self.fixture()
        original = torch.cholesky_solve
        with mock.patch.object(torch, "cholesky_solve", side_effect=lambda *args, **kw: original(*args, **kw) + 1e-4):
            P, meta = solve_basis(keys, alpha, index, 3, A, backend="dual", residual_tolerance=1e-10)
        self.assertTrue(meta["same_M_reference_used"])
        self.assertGreater(meta["initial_scaled_residual"], 1e-10)
        self.assertLess(meta["scaled_residual"], 1e-10)
        self.assertEqual(meta["residual_tolerance"], 1e-10)
        self.assertTrue(bool(torch.isfinite(P).all()))
        with mock.patch.object(torch.linalg, "solve", side_effect=lambda matrix, rhs: torch.zeros_like(rhs)):
            with mock.patch.object(torch, "cholesky_solve", side_effect=lambda *args, **kw: original(*args, **kw) + 1e-4):
                with self.assertRaisesRegex(FloatingPointError, "same-M reference residual"):
                    solve_basis(keys, alpha, index, 3, A, backend="dual", residual_tolerance=1e-10)


class Payload:
    def __init__(self, evaluation):
        self.evaluation = evaluation


class SolverTests(unittest.TestCase):
    def initial(self, B=1):
        return {"layer": torch.zeros(1, B, dtype=DTYPE)}, {"layer": torch.ones(B, dtype=DTYPE)}

    def test_exact_group_prox_zero_shrink_and_ball(self):
        v = {"layer": torch.zeros(2, 4, dtype=DTYPE)}
        grad = {"layer": torch.tensor([[0., -.3, -3., -30.], [0., -.4, -4., -40.]], dtype=DTYPE)}
        anchors = {"layer": torch.tensor([1., 1., 2., 1.], dtype=DTYPE)}
        answer = group_prox(v, grad, anchors, .2, lambda_norm=1., native_clamp=.75)["layer"]
        self.assertTrue(torch.equal(answer[:, :2], torch.zeros(2, 2, dtype=DTYPE)))
        torch.testing.assert_close(answer[:, 2], torch.tensor([.45, .6], dtype=DTYPE))
        torch.testing.assert_close(answer[:, 3], torch.tensor([.45, .6], dtype=DTYPE))
        self.assertLessEqual(float(answer.norm(dim=0).max()), .75 + torch.finfo(DTYPE).eps)
        self.assertAlmostEqual(norm_penalty({"layer": answer}, anchors, 1.), 1.125)

    def test_step_is_global_not_per_group_normalized(self):
        v, anchors = self.initial(2)
        grad = {"layer": torch.tensor([[-1., -2.]], dtype=DTYPE)}
        result = group_prox(v, grad, anchors, .1, lambda_norm=0., native_clamp=.75)
        torch.testing.assert_close(result["layer"], torch.tensor([[.1, .2]], dtype=DTYPE))

    def test_zero_is_differentiable_and_zero_mapping_stops(self):
        v, anchors = self.initial()
        calls = []

        def oracle(point, backward):
            calls.append(backward)
            self.assertTrue(point["layer"].is_leaf)
            self.assertTrue(point["layer"].requires_grad)
            smooth = point["layer"].square().sum()
            grad = torch.autograd.grad(smooth, point["layer"])[0]
            return dict(smooth=float(smooth.detach()), grad={"layer": grad}, payload="zero")

        result = solve(oracle, v, anchors, SolverConfig(epsilon_num=0.))
        self.assertEqual(calls, [True])
        self.assertEqual(result.stop_reason, "zero_proximal_mapping")
        self.assertEqual(result.accepted_evaluation, 1)
        self.assertTrue(result.no_change)

    def test_norm_subgradient_zero_mapping_stops_without_extra_evaluation(self):
        v, anchors = self.initial()

        def oracle(point, backward):
            return dict(smooth=0., grad={"layer": torch.full_like(point["layer"], .25)}, payload="zero")

        result = solve(oracle, v, anchors, SolverConfig(epsilon_num=0., lambda_norm=.5))
        self.assertEqual(result.calls, 1)
        self.assertEqual(result.stop_reason, "identical_v")
        self.assertTrue(result.no_change)

    def test_rejected_trials_halve_active_tau_and_return_initial_payload(self):
        v, anchors = self.initial()
        payloads = []

        def oracle(point, backward):
            payload = Payload(len(payloads) + 1)
            payloads.append(payload)
            return dict(smooth=0. if len(payloads) == 1 else 100.,
                        grad={"layer": -torch.ones_like(point["layer"])} if backward else None,
                        payload=payload)

        result = solve(oracle, v, anchors, SolverConfig(epsilon_num=0., lambda_norm=0.))
        self.assertEqual(result.calls, 25)
        self.assertEqual(result.backward_calls, 24)
        self.assertEqual(result.rejected_trials, 24)
        self.assertEqual(result.accepted_updates, 0)
        self.assertIs(result.payload, payloads[0])
        self.assertTrue(result.no_change)
        taus = [row["tau"] for row in result.ledger[1:]]
        for before, after in zip(taus, taus[1:]):
            self.assertEqual(after, before / 2.)
        self.assertFalse(result.ledger[-1]["backward"])

    def test_final_rejection_retains_exact_accepted_v_f_g_payload(self):
        v, anchors = self.initial()
        points, payloads = [], []

        def oracle(point, backward):
            points.append(point["layer"].detach().clone())
            number = len(points)
            payload = Payload(number)
            payloads.append(payload)
            f = {1: 0., 2: -1.}.get(number, 100.)
            g = {1: -1., 2: -2.}.get(number, 999.)
            return dict(smooth=f, grad={"layer": torch.full_like(point["layer"], g)} if backward else None,
                        payload=payload)

        result = solve(oracle, v, anchors, SolverConfig(epsilon_num=0., lambda_norm=0., candidate_cap=5))
        self.assertEqual(result.calls, 5)
        self.assertEqual(result.backward_calls, 4)
        self.assertEqual(result.accepted_evaluation, 2)
        self.assertIs(result.payload, payloads[1])
        self.assertEqual(result.smooth, -1.)
        self.assertEqual(float(result.v["layer"].detach()), .1875)
        self.assertEqual(float(result.grad["layer"]), -2.)
        self.assertEqual([float(point) for point in points], [0., .1875, .375, .28125, .234375])
        self.assertEqual([row["tau"] for row in result.ledger[1:]], [.1875, .09375, .046875, .0234375])

    def test_high_curvature_actual_objective_budget_and_final_forward_accept(self):
        v, anchors = self.initial()
        forward_flags = []

        def oracle(point, backward):
            forward_flags.append(backward)
            x = point["layer"]
            f = .5 * 100. * x.square().sum() - x.sum()
            grad = {"layer": torch.autograd.grad(f, x)[0]} if backward else None
            # All FP32 materialized weights could round identically: the
            # solver must not use this payload as a stationarity test.
            return dict(smooth=float(f.detach()), grad=grad, payload=torch.zeros(1, dtype=torch.float32))

        result = solve(oracle, v, anchors, SolverConfig(epsilon_num=0., lambda_norm=0.))
        self.assertEqual(result.calls, 25)
        self.assertEqual(result.backward_calls, 24)
        self.assertEqual(forward_flags, [True] * 24 + [False])
        self.assertGreater(result.rejected_trials, 0)
        self.assertGreater(result.accepted_updates, 0)
        self.assertLessEqual(result.accepted_updates, 24)
        self.assertEqual(result.accepted_evaluation, 25)
        self.assertIsNone(result.grad)
        accepted = [row["composite"] for row in result.ledger if row["accepted"]]
        self.assertTrue(all(after <= before for before, after in zip(accepted, accepted[1:])))

    def test_initially_inactive_group_reactivates(self):
        v = {"lower": torch.zeros(1, 1, dtype=DTYPE), "upper": torch.zeros(1, 1, dtype=DTYPE)}
        anchors = {layer: torch.ones(1, dtype=DTYPE) for layer in v}
        points = []

        def oracle(point, backward):
            points.append({layer: float(value.detach()) for layer, value in point.items()})
            x, y = point["lower"], point["upper"]
            loss = .5 * (x - 1.).square().sum() + .5 * (y - x).square().sum()
            grad = dict(zip(point, torch.autograd.grad(loss, tuple(point.values())))) if backward else None
            return dict(smooth=float(loss.detach()), grad=grad, payload=None)

        result = solve(oracle, v, anchors, SolverConfig(epsilon_num=0., lambda_norm=.1, candidate_cap=4))
        self.assertGreater(points[1]["lower"], 0.)
        self.assertEqual(points[1]["upper"], 0.)
        self.assertGreater(points[2]["upper"], 0.)
        self.assertGreater(float(result.v["upper"].detach()), 0.)

    def test_nonzero_accepted_zero_gradient_reuses_tau_and_evaluates_zero_trial(self):
        # Scripted state-machine oracle: a zero *trial vector* is not the
        # same as a zero proximal-gradient mapping at a nonzero accepted v.
        v, anchors = self.initial()
        points = []

        def oracle(point, backward):
            points.append(float(point["layer"].detach()))
            number = len(points)
            return dict(smooth=-.1 * (number - 1),
                        grad={"layer": torch.full_like(point["layer"], -1. if number == 1 else 0.)},
                        payload=number)

        result = solve(oracle, v, anchors, SolverConfig(epsilon_num=0., lambda_norm=.5, candidate_cap=4))
        self.assertEqual(points, [0., .09375, 0.])
        self.assertEqual(result.calls, 3)
        self.assertEqual(result.accepted_updates, 2)
        self.assertEqual(result.payload, 3)
        self.assertEqual([row["tau"] for row in result.ledger[1:]], [.1875, .1875])
        self.assertEqual(result.stop_reason, "identical_v")

    def test_minimum_budget_one_backward_one_forward(self):
        v, anchors = self.initial()
        flags = []

        def oracle(point, backward):
            flags.append(backward)
            self.assertEqual(point["layer"].requires_grad, backward)
            return dict(smooth=-float(point["layer"].detach().sum()),
                        grad={"layer": -torch.ones_like(point["layer"])} if backward else None, payload=None)

        result = solve(oracle, v, anchors, SolverConfig(epsilon_num=0., lambda_norm=0., candidate_cap=2))
        self.assertEqual(flags, [True, False])
        self.assertEqual(result.calls, 2)
        self.assertEqual(result.backward_calls, 1)
        self.assertEqual(result.accepted_updates, 1)

    def test_A_B_empty_reference_B1_and_partial_batch_generic_shapes(self):
        for B in (1, 3):
            initial = {"wide": torch.zeros(3, B, dtype=DTYPE), 19: torch.zeros(2, B, dtype=DTYPE)}
            anchors = {layer: torch.arange(1, B + 1, dtype=DTYPE) for layer in initial}

            def make_oracle(eta):
                def oracle(point, backward):
                    f = sum(.5 * (value - .2).square().sum() for value in point.values()) + B * eta * 0.
                    g = dict(zip(point, torch.autograd.grad(f, tuple(point.values())))) if backward else None
                    return dict(smooth=float(f.detach()), grad=g, payload=eta)
                return oracle

            config = SolverConfig(epsilon_num=0., lambda_norm=.01, candidate_cap=5)
            A = solve(make_oracle(0), initial, anchors, config)
            B_arm = solve(make_oracle(1), initial, anchors, config)
            self.assertEqual(A.ledger, B_arm.ledger)
            self.assertEqual(A.smooth, B_arm.smooth)
            for layer in initial:
                self.assertTrue(torch.equal(A.v[layer], B_arm.v[layer]))
                self.assertEqual(A.v[layer].shape[1], B)

    def test_rejected_payloads_are_not_retained(self):
        v, anchors = self.initial()
        weak_payloads = []

        def oracle(point, backward):
            # Only the accepted payload from earlier evaluations is live.
            self.assertLessEqual(sum(ref() is not None for ref in weak_payloads), 1)
            payload = Payload(len(weak_payloads) + 1)
            weak_payloads.append(weakref.ref(payload))
            return dict(smooth=0. if len(weak_payloads) == 1 else 100.,
                        grad={"layer": -torch.ones_like(point["layer"])} if backward else None, payload=payload)

        result = solve(oracle, v, anchors, SolverConfig(epsilon_num=0., lambda_norm=0., candidate_cap=6))
        self.assertEqual(sum(ref() is not None for ref in weak_payloads), 1)
        self.assertIs(weak_payloads[0](), result.payload)
        self.assertFalse(any(isinstance(value, (torch.Tensor, Payload))
                             for row in result.ledger for value in row.values()))

    def test_rejected_candidate_scalar_cost_is_preserved_not_weights(self):
        v, anchors = self.initial()
        count = 0
        stats_references = []

        def oracle(point, backward):
            nonlocal count
            count += 1
            stats = dict(seconds=count / 10., prediction_tokens=17,
                         whole_gradient_norm={'layer': 1.})
            stats_references.append(stats)
            return dict(smooth=0. if count == 1 else 100.,
                        grad={"layer": -torch.ones_like(point["layer"])} if backward else None,
                        payload=dict(stats=stats, weights={'layer': torch.ones(2, 2)}, keys=torch.ones(2, 1)))

        result = solve(oracle, v, anchors, SolverConfig(epsilon_num=0., lambda_norm=0., candidate_cap=4))
        self.assertEqual([row['stats']['seconds'] for row in result.ledger], [.1, .2, .3, .4])
        self.assertEqual(sum(row['stats']['prediction_tokens'] for row in result.ledger), 68)
        stats_references[0]['whole_gradient_norm']['layer'] = 500.
        self.assertEqual(result.ledger[0]['stats']['whole_gradient_norm']['layer'], 1.)
        self.assertTrue(all(set(row['stats']) == {'seconds', 'prediction_tokens', 'whole_gradient_norm'} for row in result.ledger))

    def test_required_immutable_tolerance_and_error_propagation(self):
        with self.assertRaises(TypeError):
            SolverConfig()
        config = SolverConfig(epsilon_num=1e-7)
        with self.assertRaises(FrozenInstanceError):
            config.epsilon_num = 1.
        for bad_cap in (1, 26, True):
            with self.assertRaises(ValueError):
                SolverConfig(epsilon_num=0., candidate_cap=bad_cap)
        v, anchors = self.initial()
        error = RuntimeError("physical oracle failed")

        def broken(point, backward):
            raise error

        with self.assertRaises(RuntimeError) as caught:
            solve(broken, v, anchors, config)
        self.assertIs(caught.exception, error)
        with self.assertRaises(FloatingPointError):
            solve(lambda *_: dict(smooth=float("nan"), grad=None, payload=None), v, anchors, config)
        with self.assertRaisesRegex(ValueError, "every eligible"):
            solve(lambda *_: dict(smooth=0., grad={}, payload=None), v, anchors, config)
        with self.assertRaisesRegex(ValueError, "initializes every eligible group at zero"):
            solve(broken, {"layer": torch.ones(1, 1, dtype=DTYPE)}, anchors, config)

    def test_incremental_trial_receipt_and_callback_errors_propagate(self):
        v, anchors = self.initial()
        rows = []

        def oracle(point, backward):
            return dict(smooth=-float(point["layer"].detach().sum()),
                        grad={"layer": -torch.ones_like(point["layer"])} if backward else None, payload=None)

        def receipt(row):
            rows.append(dict(row))
            row["evaluation"] = -1  # A receipt writer cannot mutate the ledger.

        result = solve(oracle, v, anchors, SolverConfig(epsilon_num=0., candidate_cap=3), on_trial=receipt)
        self.assertEqual(rows, result.ledger)
        self.assertEqual([row["evaluation"] for row in result.ledger], [1, 2, 3])

        def broken_receipt(row):
            raise OSError("receipt storage failure")

        with self.assertRaisesRegex(OSError, "receipt storage failure"):
            solve(oracle, v, anchors, SolverConfig(epsilon_num=0.), on_trial=broken_receipt)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
