"""Meaningful algebra and execution-contract checks; no model/data loading."""

import unittest

import numpy as np

from geometry import (action_pullback, build_geometry, projected_action,
                      realization_diagnostics, writer_cost)


MEASUREMENTS = {}
SEED = 20261005


def fixture(counts=(2, 3), output_dim=4, seed=SEED, previous=True):
    rng = np.random.default_rng(seed)
    owners = np.repeat(np.arange(len(counts)), counts)
    n, m = len(owners) + 3, len(owners)
    Z = rng.normal(size=(n, n))
    C0 = Z @ Z.T + np.eye(n)
    history_keys = rng.normal(size=(n, 3))
    lambda_cov = 1.7  # Toy algebra value, not an experimental proposal.
    A = lambda_cov * C0 + history_keys @ history_keys.T
    K = rng.normal(size=(n, m))
    old = rng.normal(size=(output_dim, n)) if previous else np.zeros((output_dim, n))
    geometry = build_geometry(K, owners, A, C0, old, lambda_cov, svd_rtol=1e-12)
    return rng, geometry, rng.normal(size=(output_dim, len(counts)))


def finite_difference(fun, value, step=1e-6):
    derivative = np.zeros_like(value)
    for index in np.ndindex(value.shape):
        offset = np.zeros_like(value)
        offset[index] = step
        derivative[index] = (fun(value + offset) - fun(value - offset)) / (2 * step)
    return derivative


def truncated_geometry(geometry, retained_rank):
    X = np.linalg.solve(geometry.L, geometry.keys) * np.sqrt(geometry.weights)
    values = np.linalg.svd(X, compute_uv=False)
    threshold = float((values[retained_rank - 1] + values[retained_rank]) / 2)
    return build_geometry(geometry.keys, geometry.owners, geometry.A, geometry.C0,
                          geometry.previous_delta, geometry.lambda_cov,
                          svd_rtol=0, svd_atol=threshold, weights=geometry.weights)


class ReferenceTests(unittest.TestCase):
    def assertClose(self, actual, expected, atol=1e-9, rtol=1e-9):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)

    def test_full_rank_exact_writer_and_dense_gradient(self):
        _, g, D = fixture()
        U = D @ g.Bmap
        self.assertEqual(g.retained_rank, g.keys.shape[1])
        self.assertClose(U @ g.keys, D @ g.E)
        self.assertClose(U @ g.keys, projected_action(D, g))
        self.assertClose(g.F, g.Bmap @ g.A @ g.Bmap.T)
        result = writer_cost(D, g, alpha=1, allocation_price=0.37)
        dense_q = float(np.sum((U @ g.L) ** 2))
        dense_cross = float(np.sum((g.previous_delta @ g.C0) * U))
        dense_grad = (2 * U @ g.A + 2 * g.lambda_cov * np.sign(dense_cross)
                      * g.previous_delta @ g.C0) @ g.Bmap.T
        self.assertClose(result["Q"], dense_q)
        self.assertClose(result["cross"], dense_cross)
        self.assertClose(result["gradient"], 0.37 * dense_grad)
        MEASUREMENTS["dense_Q_absolute_error"] = abs(result["Q"] - dense_q)
        MEASUREMENTS["dense_gradient_max_absolute_error"] = float(
            np.max(abs(result["gradient"] - 0.37 * dense_grad)))

    def test_cost_and_action_finite_difference(self):
        rng, g, D = fixture(seed=SEED + 1)
        g = truncated_geometry(g, 3)
        action_cotangent = rng.normal(size=(D.shape[0], g.keys.shape[1]))
        price = 0.29
        fun = lambda target: (writer_cost(target, g, alpha=1,
                                         allocation_price=price)["weighted_cost"]
                              + float(np.sum(projected_action(target, g) * action_cotangent)))
        analytic = (writer_cost(D, g, alpha=1, allocation_price=price)["gradient"]
                    + action_pullback(action_cotangent, g))
        numeric = finite_difference(fun, D)
        self.assertClose(analytic, numeric, atol=2e-7, rtol=2e-7)
        MEASUREMENTS["finite_difference_max_absolute_error"] = float(np.max(abs(analytic - numeric)))

    def test_rank_deficient_compatible_and_incompatible_targets(self):
        rng = np.random.default_rng(SEED + 2)
        a, b = np.eye(5)[:, 0], np.eye(5)[:, 1]
        K = np.column_stack((a, a, b, b, a))
        owners = np.array([0, 0, 1, 1, 2])
        g = build_geometry(K, owners, 2 * np.eye(5), np.eye(5),
                           np.zeros((3, 5)), 2, svd_rtol=1e-12)
        self.assertEqual(g.retained_rank, 2)
        D = rng.normal(size=(3, 3))
        D[:, 2] = D[:, 0]
        self.assertClose(projected_action(D, g), D @ g.E)
        self.assertLess(realization_diagnostics(D, g)["discarded_target_relative"], 1e-12)
        D[:, 2] += 1
        diagnosis = realization_diagnostics(D, g)
        self.assertGreater(diagnosis["discarded_target_relative"], 0.1)
        self.assertClose((D @ g.Bmap) @ g.keys, projected_action(D, g))
        MEASUREMENTS["rank_deficient_incompatible_target_relative_error"] = diagnosis["discarded_target_relative"]

    def test_retained_nullspace_no_free_virtual_action(self):
        # The third owner lies entirely in a deliberately discarded direction.
        K = np.diag([4.0, 1.0, 0.01])
        g = build_geometry(K, np.arange(3), 2 * np.eye(3), np.eye(3),
                           np.ones((2, 3)), 2, svd_rtol=0.1)
        self.assertEqual(g.retained_rank, 2)
        D = np.array([[0., 0., 3.], [0., 0., -2.]])
        self.assertGreater(np.linalg.norm(D @ g.E), 0)
        self.assertClose(projected_action(D, g), 0)
        self.assertEqual(writer_cost(D, g, alpha=1)["Pi"], 0)
        self.assertClose(realization_diagnostics(D, g)["discarded_target_relative"], 1)
        live = D.copy()
        live[0, 0] = 1
        self.assertGreater(np.linalg.norm(projected_action(live, g)), 0)
        self.assertGreater(writer_cost(live, g, alpha=0)["Q"], 0)
        # The all-zero key rank is also explicit, rather than a pseudoinverse fallback.
        zero = build_geometry(np.zeros((3, 3)), np.arange(3), 2 * np.eye(3),
                              np.eye(3), np.zeros((2, 3)), 2, svd_rtol=1e-12)
        self.assertEqual(zero.retained_rank, 0)
        self.assertClose(projected_action(D, zero), 0)
        self.assertEqual(writer_cost(D, zero, alpha=1)["Pi"], 0)

    def test_absolute_drift_upper_bound_and_no_reset_reward(self):
        _, g, D = fixture(seed=SEED + 3)
        U = D @ g.Bmap
        drift = lambda update: float(np.sum((update @ g.C0) * update))
        q = drift(U)
        qh = float(np.sum((U @ g.H) * U))
        cross = float(np.sum((g.previous_delta @ g.C0) * U))
        change = drift(g.previous_delta + U) - drift(g.previous_delta)
        self.assertClose(change, q + 2 * cross)
        cost = writer_cost(D, g, alpha=1)["Pi"]
        slack = cost - (g.lambda_cov * abs(change) + qh)
        self.assertGreaterEqual(slack, -1e-9)
        # Make the cumulative update exactly cancellable in this writer space.
        reset_geometry = build_geometry(g.keys, g.owners, g.A, g.C0, U,
                                        g.lambda_cov, svd_rtol=1e-12, weights=g.weights)
        reset = writer_cost(-D, reset_geometry, alpha=1)
        self.assertLess(reset["cross"], 0)
        self.assertGreater(reset["Pi"], 0)
        self.assertEqual(writer_cost(np.zeros_like(D), reset_geometry, alpha=1)["Pi"], 0)
        self.assertClose(U - D @ reset_geometry.Bmap, 0)
        MEASUREMENTS["absolute_drift_upper_bound_slack"] = slack
        MEASUREMENTS["reset_penalty_positive"] = reset["Pi"]

    def test_first_batch_arm_identity_and_zero_cross_subgradient(self):
        _, g, D = fixture(previous=False)
        baseline = writer_cost(D, g, alpha=0)
        cumulative = writer_cost(D, g, alpha=1)
        self.assertEqual(baseline["weighted_cost"], cumulative["weighted_cost"])
        np.testing.assert_array_equal(baseline["gradient"], cumulative["gradient"])
        self.assertEqual(cumulative["cross"], 0)
        self.assertEqual(cumulative["cumulative"], 0)
        _, nonzero_g, D = fixture(seed=SEED + 4)
        at_zero = writer_cost(np.zeros_like(D), nonzero_g, alpha=1)
        self.assertClose(at_zero["gradient"], 0)
        # Zero is a valid selected subgradient at the abs kink (no claim of differentiability).
        self.assertGreater(writer_cost(D, nonzero_g, alpha=1)["weighted_cost"], 0)

    def test_variable_owners_and_logical_batch_sizes(self):
        ranks = {}
        for counts in ((3,), (1, 4), (1, 3, 2, 4), (2, 1, 3)):
            with self.subTest(counts=counts):
                _, g, D = fixture(counts=counts, output_dim=3, seed=SEED + len(counts))
                B = len(counts)
                self.assertEqual(g.request_count, B)
                for owner, count in enumerate(counts):
                    self.assertClose(g.weights[g.owners == owner], 1 / (B * count))
                self.assertClose(np.sum(g.weights), 1)
                self.assertClose(projected_action(D, g), D @ g.E)
                result = writer_cost(D, g, alpha=1, allocation_price=0.43)
                # A reporting mean divides the complete cost and gradient alike.
                self.assertClose(result["weighted_cost"] / B,
                                 0.43 * result["Pi"] / B)
                ranks[str(B)] = g.retained_rank
        MEASUREMENTS["logical_batch_ranks"] = ranks

    def test_microbatch_pullback_accumulation_1_2_4_and_tail(self):
        rng, g, D = fixture(counts=(1, 3, 2, 4, 1, 2, 3), output_dim=3, seed=SEED + 5)
        g = truncated_geometry(g, 8)
        self.assertEqual(g.retained_rank, 8)
        other_owner_entries = np.arange(g.request_count)[:, None] != g.owners[None, :]
        self.assertGreater(np.linalg.norm(g.S[other_owner_entries]), 0.01)
        target = rng.normal(size=(D.shape[0], g.keys.shape[1]))
        action = projected_action(D, g)
        native_weights = 1.0 / np.bincount(g.owners)[g.owners]
        residual = action - target
        task_loss = .5 * float(np.sum(residual ** 2 * native_weights))
        task_gradient = action_pullback(residual * native_weights, g)
        owner_only_gradient = np.zeros_like(D)
        for owner in range(g.request_count):
            columns = g.owners == owner
            owner_only_gradient[:, owner] = ((residual[:, columns] * native_weights[columns])
                                             @ g.S[owner, columns])
        self.assertGreater(np.linalg.norm(task_gradient - owner_only_gradient), 0.01)
        regularizer = writer_cost(D, g, alpha=1, allocation_price=0.17)
        errors = {}
        for chunk in (1, 2, 4):
            loss = 0.0
            gradient = np.zeros_like(D)
            sizes = []
            for start in range(0, g.request_count, chunk):
                stop = min(start + chunk, g.request_count)
                sizes.append(stop - start)
                selected = (g.owners >= start) & (g.owners < stop)
                local_residual = D @ g.S[:, selected] - target[:, selected]
                loss += .5 * float(np.sum(local_residual ** 2 * native_weights[selected]))
                # Each context's loss may affect ALL owner variables through S.
                gradient += ((local_residual * native_weights[selected])
                             @ g.S[:, selected].T)
            loss += regularizer["weighted_cost"]
            gradient += regularizer["gradient"]  # Exactly once per logical candidate.
            self.assertClose(loss, task_loss + regularizer["weighted_cost"])
            self.assertClose(gradient, task_gradient + regularizer["gradient"])
            if chunk > 1:
                self.assertLess(sizes[-1], chunk)
            errors[str(chunk)] = float(np.max(abs(gradient - task_gradient - regularizer["gradient"])))
        MEASUREMENTS["microbatch_gradient_max_errors"] = errors
        MEASUREMENTS["incorrect_owner_only_gradient_error_norm"] = float(
            np.linalg.norm(task_gradient - owner_only_gradient))

    def test_global_weight_scale_and_duplicate_context_invariance(self):
        _, g, D = fixture(seed=SEED + 6)
        scaled = build_geometry(g.keys, g.owners, g.A, g.C0, g.previous_delta,
                                g.lambda_cov, svd_rtol=1e-12, weights=7 * g.weights)
        self.assertEqual(scaled.retained_rank, g.retained_rank)
        self.assertClose(scaled.F, g.F)
        self.assertClose(scaled.Bmap, g.Bmap)
        self.assertClose(scaled.S, g.S)
        duplicate = build_geometry(np.repeat(g.keys, 2, axis=1), np.repeat(g.owners, 2),
                                   g.A, g.C0, g.previous_delta, g.lambda_cov, svd_rtol=1e-12)
        self.assertClose(duplicate.F, g.F)
        self.assertClose(duplicate.Bmap, g.Bmap)
        self.assertClose(projected_action(D, duplicate), np.repeat(projected_action(D, g), 2, axis=1))
        # These claims require the same retained subspace; absolute rank cutoffs
        # can invalidate global weight-scale invariance.

    def test_weighted_mean_contrast_orthogonal_reparameterization(self):
        _, g, D = fixture(counts=(3, 4), seed=SEED + 7)
        Q = np.zeros((g.keys.shape[1], g.keys.shape[1]))
        for owner in range(g.request_count):
            indices = np.flatnonzero(g.owners == owner)
            direction = np.sqrt(g.weights[indices])
            direction /= np.linalg.norm(direction)
            basis, _ = np.linalg.qr(np.column_stack((direction, np.eye(len(indices))[:, :-1])))
            Q[np.ix_(indices, indices)] = basis
        self.assertClose(Q.T @ Q, np.eye(Q.shape[0]))
        X = np.linalg.solve(g.L, g.keys) * np.sqrt(g.weights)
        Tw = (D @ g.E) * np.sqrt(g.weights)
        rotated_targets = Tw @ Q
        for owner in range(g.request_count):
            indices = np.flatnonzero(g.owners == owner)
            self.assertClose(rotated_targets[:, indices[1:]], 0)
        left, values, right_t = np.linalg.svd(X @ Q, full_matrices=False)
        self.assertClose(values, np.linalg.svd(X, compute_uv=False))
        keep = values > g.rank_threshold
        rotated_whitened = ((rotated_targets @ right_t[keep].T) / values[keep]) @ left[:, keep].T
        rotated_update = np.linalg.solve(g.L.T, rotated_whitened.T).T
        self.assertClose(rotated_update, D @ g.Bmap)

    def test_invalid_inputs_fail_without_silent_repairs(self):
        _, g, _ = fixture()
        base = dict(keys=g.keys, owners=g.owners, A=g.A, C0=g.C0,
                    previous_delta=g.previous_delta, lambda_cov=g.lambda_cov, svd_rtol=1e-12)
        invalid = [dict(weights=np.zeros_like(g.weights)),
                   dict(owners=np.zeros(len(g.owners), dtype=float)),
                   dict(owners=np.full(len(g.owners), 2)),
                   dict(svd_rtol=-1), dict(lambda_cov=0),
                   dict(previous_delta=np.zeros((2, 1))),
                   dict(A=np.zeros_like(g.A))]
        for replacement in invalid:
            with self.subTest(replacement=list(replacement)):
                args = dict(base); args.update(replacement)
                with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                    build_geometry(**args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
