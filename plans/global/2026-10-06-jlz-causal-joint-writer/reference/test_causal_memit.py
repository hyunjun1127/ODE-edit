"""Synthetic causal-forward and full-gradient qualification witnesses."""

import unittest

import numpy as np

from causal_memit import (ToyModel, calibration_ratio, candidate, candidate_jvp,
                         committed_forward, gradient_norm, gradients, logsoftmax,
                         memit_writer, native_mean_operator, norm_loss,
                         project_requested_budget, task_loss, writer_jvp)


SEED = 20261006
MEASUREMENTS = {}


def fixture(counts=(2, 3), layers=3, hidden=4, seed=SEED):
    rng = np.random.default_rng(seed)
    owners, mean_weights, rewrite_weights, kl_weights, canonical = [], [], [], [], []
    for owner, count in enumerate(counts):
        canonical.append(len(owners))
        owners.extend([owner] * (count + 1))  # Last row is the owner's KL row.
        mean_weights.extend(([1.] if count == 1 else [.5] + [.5 / (count - 1)] * (count - 1)) + [0.])
        rewrite_weights.extend([1. / count] * count + [0.])
        kl_weights.extend([0.] * count + [1.])
    owners = np.array(owners, dtype=np.int64)
    X = rng.normal(size=(hidden, len(owners)))
    weights = [.7 * np.eye(hidden) + .15 * rng.normal(size=(hidden, hidden)) for _ in range(layers)]
    metrics = []
    for _ in range(layers):
        Z = rng.normal(size=(hidden, hidden))
        metrics.append(Z @ Z.T + .5 * np.eye(hidden))
    head = rng.normal(size=(hidden + 2, hidden))
    M = native_mean_operator(owners, mean_weights, len(counts))
    model = ToyModel(X, weights, metrics, head, M, owners,
                     np.array(rewrite_weights), np.array(kl_weights), owners % head.shape[0],
                     [], np.ones(len(counts)), np.zeros((head.shape[0], len(owners))))
    logits, hidden_states = committed_forward(model, [np.zeros_like(W) for W in weights])
    model.anchors = [np.linalg.norm(H[:, canonical], axis=0) for H in hidden_states]
    model.anchor_star = model.anchors[-1].copy()
    model.teacher_logprobs = logsoftmax(logits)
    blocks = [.025 * rng.normal(size=(hidden, len(counts))) for _ in weights]
    return rng, model, blocks


def finite_difference(fun, blocks, epsilon=1e-6):
    result = [np.zeros_like(u) for u in blocks]
    for layer, u in enumerate(blocks):
        for index in np.ndindex(u.shape):
            plus, minus = [x.copy() for x in blocks], [x.copy() for x in blocks]
            plus[layer][index] += epsilon
            minus[layer][index] -= epsilon
            result[layer][index] = (fun(plus) - fun(minus)) / (2 * epsilon)
    return result


def max_error(left, right):
    return max(float(np.max(abs(x - y))) for x, y in zip(left, right))


class CausalMemitTests(unittest.TestCase):
    def assertClose(self, actual, expected, atol=1e-9, rtol=1e-9):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

    def test_native_mean_support_is_not_all_context_equality(self):
        _, model, blocks = fixture()
        first = candidate(model, blocks)["layers"][0]["writer"]
        self.assertEqual(first["K"].shape[1], len(model.anchor_star))
        self.assertGreater(model.inputs.shape[1], first["K"].shape[1])
        self.assertClose(first["K"], model.inputs @ model.mean_operator)
        # Even the exact mean-target solution need not realize all context targets.
        exact_mean = first["R"] @ np.linalg.solve(first["K"].T @ first["K"], first["K"].T)
        self.assertClose(exact_mean @ first["K"], first["R"])
        context_target = first["R"][:, model.owners]
        self.assertGreater(np.linalg.norm(exact_mean @ model.inputs - context_target), .01)
        # KL rows are present in the task but absent from native rewrite-key means.
        self.assertClose(model.mean_operator[model.kl_weights > 0], 0)

    def test_ridge_response_and_fixed_support_energy(self):
        _, model, blocks = fixture()
        w = candidate(model, blocks)["layers"][0]["writer"]
        inverse_K = np.linalg.solve(w["A"], w["K"])
        S = w["K"].T @ inverse_K
        M = np.linalg.solve(np.eye(S.shape[0]) + S, S)
        self.assertClose(w["response"], w["R"] @ M)
        self.assertClose(w["Q_dense"], w["Q_compact"])
        self.assertGreater(w["residual_energy"], 0)
        self.assertTrue(np.all(np.linalg.eigvalsh(M) < 1))

    def test_writer_jvp_value_and_compact_gradient(self):
        rng, model, blocks = fixture(seed=SEED + 1)
        w = candidate(model, blocks)["layers"][0]["writer"]
        dR, dK = rng.normal(size=w["R"].shape), rng.normal(size=w["K"].shape)
        tangent = writer_jvp(w, dR, dK)
        eps = 1e-6
        plus = memit_writer(w["R"] + eps * dR, w["K"] + eps * dK, w["A"])
        minus = memit_writer(w["R"] - eps * dR, w["K"] - eps * dK, w["A"])
        self.assertClose(tangent["dU"], (plus["U"] - minus["U"]) / (2 * eps), atol=2e-8)
        self.assertClose(tangent["dQ_dense"], tangent["dQ_compact"])
        self.assertClose(tangent["dQ_dense"], (plus["Q_dense"] - minus["Q_dense"]) / (2 * eps), atol=2e-8)

    def test_full_causal_actual_loss_and_Q_gradients(self):
        _, model, blocks = fixture(seed=SEED + 2)
        result = gradients(model, blocks)
        numeric_task = finite_difference(lambda u: task_loss(model, candidate(model, u)["logits"])["native_sum"], blocks)
        numeric_Q = finite_difference(lambda u: candidate(model, u)["Q_dense"], blocks)
        task_error = max_error(result["native_gradient"], numeric_task)
        q_error = max_error(result["Q_gradient"], numeric_Q)
        self.assertLess(task_error, 2e-7)
        self.assertLess(q_error, 2e-8)
        self.assertLess(max_error(result["Q_gradient"], result["Q_compact_gradient"]), 1e-10)
        MEASUREMENTS["causal_native_gradient_FD_max_error"] = task_error
        MEASUREMENTS["causal_Q_gradient_FD_max_error"] = q_error
        MEASUREMENTS["causal_dense_compact_gradient_max_error"] = max_error(result["Q_gradient"], result["Q_compact_gradient"])

    def test_missing_upper_key_derivative_is_detected(self):
        _, model, blocks = fixture(seed=SEED + 3)
        full = gradients(model, blocks)
        frozen = gradients(model, blocks, differentiate_upper_keys=False)
        lower_q_error = float(np.linalg.norm(full["Q_gradient"][0] - frozen["Q_gradient"][0]))
        lower_task_error = float(np.linalg.norm(full["native_gradient"][0] - frozen["native_gradient"][0]))
        self.assertGreater(lower_q_error, 1e-7)
        self.assertGreater(lower_task_error, 1e-6)
        # There are no later key paths after the last eligible layer.
        self.assertClose(full["Q_gradient"][-1], frozen["Q_gradient"][-1])
        MEASUREMENTS["frozen_upper_key_lower_Q_gradient_error_norm"] = lower_q_error
        MEASUREMENTS["frozen_upper_key_lower_native_gradient_error_norm"] = lower_task_error

    def test_functional_and_committed_candidate_states_match(self):
        _, model, blocks = fixture(seed=SEED + 4)
        cached = candidate(model, blocks)
        updates = [layer["writer"]["U"] for layer in cached["layers"]]
        logits, hidden = committed_forward(model, updates)
        self.assertClose(logits, cached["logits"], atol=0, rtol=0)
        for state, layer in zip(hidden, cached["layers"]):
            self.assertClose(state, layer["H"], atol=0, rtol=0)
        self.assertEqual(task_loss(model, logits)["native_sum"], task_loss(model, cached["logits"])["native_sum"])
        MEASUREMENTS["functional_commit_FP64_logit_max_error"] = float(np.max(abs(logits - cached["logits"])))

    def test_zero_candidate_and_single_calibration_ratio(self):
        _, model, blocks = fixture(seed=SEED + 5)
        zero = [np.zeros_like(u) for u in blocks]
        at_zero = gradients(model, zero)
        self.assertEqual(at_zero["cached"]["Q_dense"], 0)
        self.assertEqual(gradient_norm(at_zero["Q_gradient"]), 0)
        self.assertIsNone(calibration_ratio(model, zero))
        self.assertGreater(gradient_norm(at_zero["native_gradient"]), 0)
        coefficient = calibration_ratio(model, blocks)
        _, norm_gradient = norm_loss(model, blocks)
        q_gradient = gradients(model, blocks)["Q_gradient"]
        self.assertGreater(coefficient, 0)
        self.assertClose(coefficient * gradient_norm(q_gradient), gradient_norm(norm_gradient))
        MEASUREMENTS["synthetic_calibration_ratio_not_experiment_value"] = coefficient

    def test_zero_layer_can_reenter_shared_budget(self):
        _, model, blocks = fixture(seed=SEED + 6)
        blocks[1][:] = 0
        gradient = gradients(model, blocks)["native_gradient"]
        self.assertGreater(np.linalg.norm(gradient[1]), 1e-8)
        updated = project_requested_budget([u - .001 * g for u, g in zip(blocks, gradient)])
        self.assertEqual(len(updated), len(model.weights))
        self.assertGreater(np.linalg.norm(updated[1]), 0)
        budgets = sum(np.linalg.norm(u, axis=0) for u in updated)
        self.assertTrue(np.all(budgets <= .75 + 1e-12))
        # This checks graph/reentry and feasibility, not production Adam parity.

    def test_arbitrary_logical_batches_and_variable_context_counts(self):
        for counts in ((1,), (1, 3), (1, 3, 2, 4), (2, 1, 3)):
            with self.subTest(counts=counts):
                _, model, blocks = fixture(counts=counts, hidden=3, layers=2, seed=SEED + len(counts))
                result = gradients(model, blocks)
                self.assertClose(result["cached"]["Q_dense"], result["cached"]["Q_compact"])
                self.assertTrue(np.isfinite(gradient_norm(result["Q_gradient"])))
                self.assertEqual(result["cached"]["layers"][0]["writer"]["K"].shape[1], len(counts))

    def test_microbatch_native_adjoints_and_regularizer_once(self):
        _, model, blocks = fixture(counts=(1, 3, 2, 4, 1), layers=2, hidden=3, seed=SEED + 7)
        full = gradients(model, blocks)
        norm_value, norm_gradient = norm_loss(model, blocks)
        coefficient = .23
        expected_value = full["task"]["native_sum"] + norm_value + coefficient * full["cached"]["Q_dense"]
        expected_gradient = [n + r + coefficient * q for n, r, q in zip(full["native_gradient"], norm_gradient, full["Q_gradient"])]
        errors = {}
        for width in (1, 2, 4):
            value = 0.0
            native_gradient = [np.zeros_like(u) for u in blocks]
            sizes = []
            for start in range(0, len(model.anchor_star), width):
                stop = min(start + width, len(model.anchor_star))
                sizes.append(stop - start)
                rows = np.flatnonzero((model.owners >= start) & (model.owners < stop))
                # A whole-logical-batch writer is held fixed; only loss rows are sliced.
                partial = gradients(model, blocks, selected_rows=rows)
                value += partial["task"]["native_sum"]
                for g, add in zip(native_gradient, partial["native_gradient"]):
                    g += add
            value += norm_value + coefficient * full["cached"]["Q_dense"]
            got = [g + r + coefficient * q for g, r, q in zip(native_gradient, norm_gradient, full["Q_gradient"])]
            self.assertClose(value, expected_value)
            self.assertLess(max_error(got, expected_gradient), 1e-10)
            if width > 1:
                self.assertLess(sizes[-1], width)
            errors[str(width)] = max_error(got, expected_gradient)
        MEASUREMENTS["microbatch_1_2_4_tail_gradient_errors"] = errors

    def test_pre_writer_norm_budget_not_realized_action_budget(self):
        _, model, blocks = fixture(seed=SEED + 8)
        cached = candidate(model, blocks)
        requested = sum(np.linalg.norm(u, axis=0) for u in blocks)
        realized = sum(np.linalg.norm(layer["writer"]["response"], axis=0) / a
                       for layer, a in zip(cached["layers"], model.anchors))
        self.assertGreater(np.linalg.norm(requested - realized), .01)
        norm_value, norm_gradient = norm_loss(model, blocks)
        self.assertClose(norm_value, np.sum(.5 / model.anchor_star * requested))
        numeric = finite_difference(lambda u: norm_loss(model, u)[0], blocks)
        self.assertLess(max_error(norm_gradient, numeric), 1e-7)

    def test_invalid_metric_is_not_silently_repaired(self):
        K, R = np.eye(2), np.ones((3, 2))
        with self.assertRaises(ValueError):
            memit_writer(R, K, np.array([[1., .1], [0., 1.]]))
        with self.assertRaises(np.linalg.LinAlgError):
            memit_writer(R, K, np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            native_mean_operator(np.array([0, 0]), np.array([.2, .2]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
