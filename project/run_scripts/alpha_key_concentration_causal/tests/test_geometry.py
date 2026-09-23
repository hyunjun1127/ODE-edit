"""Deterministic CPU algebra tests; not real-model/native parity evidence."""
import json
import unittest

import numpy as np

from project.run_scripts.alpha_key_concentration_causal.geometry import (
    GeometryError, _eigh_psd, compare_keys, context_geometry, geometry_summary,
    history_coverage, writer_modes,
)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260923)

    def test_identity_and_rank_one(self):
        identity = geometry_summary(np.eye(7))
        self.assertAlmostEqual(identity['raw']['participation_ratio'], 7)
        self.assertAlmostEqual(identity['centered']['participation_ratio'], 6)
        self.assertAlmostEqual(identity['raw']['top1_energy_share'], 1 / 7)
        one = geometry_summary(np.ones((8, 3)))
        self.assertAlmostEqual(one['raw']['participation_ratio'], 1)
        self.assertIsNone(one['centered']['participation_ratio'])
        self.assertAlmostEqual(one['mean_energy_fraction'], 1)

    def test_zero_policy(self):
        result = geometry_summary(np.zeros((5, 2), dtype=np.float32))
        self.assertEqual(result['unit_n'], 0)
        self.assertIsNone(result['raw']['participation_ratio'])
        self.assertEqual(result['unit_raw']['status'], 'NO_NONZERO_SAMPLES')
        json.dumps(result, allow_nan=False)
        mixed = geometry_summary(np.array([[0., 0.], [2., 0.], [0., 3.]]))
        self.assertEqual(mixed['zero_norm_indices'], [0])
        self.assertEqual(mixed['unit_n'], 2)
        self.assertAlmostEqual(mixed['unit_raw']['participation_ratio'], 2)

    def test_energy_ess_top_samples(self):
        keys = np.zeros((100, 2))
        keys[0, 0] = 3
        keys[1, 0] = 4
        result = geometry_summary(keys)
        self.assertAlmostEqual(result['norm_energy_ess'], 625 / 337)
        self.assertAlmostEqual(result['top1pct_sample_energy_share'], 16 / 25)
        self.assertAlmostEqual(result['top5pct_sample_energy_share'], 1)

    def test_raw_gram_agrees_feature_spectrum(self):
        keys = self.rng.normal(size=(13, 5))
        result = geometry_summary(keys)
        expected = np.linalg.eigvalsh(keys.T @ keys)[::-1]
        np.testing.assert_allclose(result['raw']['eigenvalues_descending'][:5], expected, rtol=1e-12)
        self.assertAlmostEqual(result['raw']['trace'], float(np.sum(keys * keys)))

    def test_mean_shift_centered_invariance(self):
        a = self.rng.normal(size=(12, 4))
        b = a + np.array([10., 20., -4., 8.])
        ga, gb = geometry_summary(a), geometry_summary(b)
        self.assertAlmostEqual(ga['centered']['participation_ratio'], gb['centered']['participation_ratio'])
        self.assertNotAlmostEqual(ga['raw']['participation_ratio'], gb['raw']['participation_ratio'])

    def test_roundoff_negative_and_substantial_negative(self):
        values, _, info = _eigh_psd(np.diag([1., -1e-15]))
        self.assertEqual(info['negative_roundoff_count'], 1)
        self.assertEqual(values[-1], 0)
        with self.assertRaises(GeometryError):
            _eigh_psd(np.diag([1., -0.01]))
        with self.assertRaises(GeometryError):
            _eigh_psd(np.array([[1., .5], [.3, 1.]]))

    def test_invalid_inputs(self):
        for value in [np.ones((2, 2), dtype=int), np.empty((0, 2)), np.ones((1001, 2)), np.array([[float('nan')]])]:
            with self.assertRaises(GeometryError):
                geometry_summary(value)
        with self.assertRaises(GeometryError):
            geometry_summary(np.array([[1e308]]))

    def test_pairing_identity_scale_and_zeros(self):
        keys = self.rng.normal(size=(9, 5))
        keys[0] = 0
        result = compare_keys(keys, keys * 2, request_ids_a=list(range(9)), request_ids_b=list(range(9)))
        self.assertEqual(result['identity'], 'EXACT_ORDERED_IDS')
        self.assertIsNone(result['cosine_per_request'][0])
        np.testing.assert_allclose(result['cosine_per_request'][1:], 1)
        np.testing.assert_allclose(result['norm_ratio_b_over_a'][1:], 2)
        self.assertAlmostEqual(result['raw_gram_alignment'], 1)
        self.assertAlmostEqual(result['centered_gram_alignment'], 1)
        self.assertEqual(result['pairwise_distances']['unique_unordered_pairs'], 36)
        with self.assertRaises(GeometryError):
            compare_keys(keys, keys, request_ids_a=list(range(9)), request_ids_b=list(reversed(range(9))))
        with self.assertRaises(GeometryError):
            compare_keys(keys, keys, request_ids_a=[0] * 9, request_ids_b=[0] * 9)

    def test_context_group_mean_not_equal_six_mean(self):
        context = np.zeros((4, 6, 2), dtype=np.float32)
        context[:, 0] = 10
        context[:, 1:] = 2
        actual = np.full((4, 2), 6, dtype=np.float32)
        result = context_geometry(context, layer=6, writer_mean=actual)
        self.assertEqual(result['context_weights'], [.5, .1, .1, .1, .1, .1])
        self.assertEqual(result['reconstruction_max_abs_difference'], 0)
        self.assertAlmostEqual(result['writer_mean']['raw']['trace'], 4 * 2 * 36)
        self.assertEqual(result['writer_mean_source'], 'CAPTURED_NATIVE')

    def test_writer_ideal_response_identity(self):
        d, n, output = 9, 5, 3
        U, _ = np.linalg.qr(self.rng.normal(size=(d, 4)))
        raw_m = self.rng.normal(size=(d, 11))
        M = raw_m @ raw_m.T
        K, R = self.rng.normal(size=(d, n)), self.rng.normal(size=(output, n))
        Z = U.T @ K
        A = 10 * np.eye(4) + U.T @ M @ U
        H = Z.T @ np.linalg.solve(A, Z)
        delta = R @ Z.T @ np.linalg.solve(A + Z @ Z.T, U.T)
        result = writer_modes(K, R, delta, history_whitened_gram=H)
        self.assertLess(result['actual_vs_ideal_response_frobenius'], 1e-12)
        self.assertAlmostEqual(result['ideal_remaining_error_formula'], result['actual_remaining_error_energy'])
        self.assertAlmostEqual(result['ideal_remaining_error_direct'], result['ideal_remaining_error_formula'])
        self.assertFalse(result['native_solver_replaced'])
        json.dumps(result, allow_nan=False)

    def test_writer_actual_mismatch_not_hidden(self):
        K, R = np.eye(3), np.eye(3)
        delta = np.eye(3) * .2
        result = writer_modes(K, R, delta, history_whitened_gram=np.eye(3))
        self.assertGreater(result['actual_vs_ideal_response_frobenius'], .5)
        with self.assertRaises(GeometryError):
            writer_modes(K, R[:, :2], delta)

    def test_history_penalty_and_trace_coverage(self):
        allkeys = self.rng.normal(size=(7, 13))
        stamp = allkeys[:, :5]
        current = stamp * 2
        M = allkeys @ allkeys.T
        P = np.diag([1., 1., 0., 1., 0., 0., 0.])
        delta = self.rng.normal(size=(3, 7))
        result = history_coverage(stamp, current, delta, M=M, P=P, block_size=2)
        np.testing.assert_allclose(result['projected_history_trace'], np.trace(P @ M @ P.T))
        np.testing.assert_allclose(result['native_write_history_penalty'], np.trace(delta @ M @ delta.T))
        self.assertAlmostEqual(result['current_penalty'], 4 * result['stamp_penalty'])
        self.assertLess(result['native_write_penalty_coverage'], 1)
        self.assertFalse(result['superseded_excluded_from_statistics'])
        reused = history_coverage(stamp, current, delta, P=P, totals={
            'projected_history_trace': result['projected_history_trace'],
            'native_write_history_penalty': result['native_write_history_penalty'],
        })
        self.assertAlmostEqual(reused['projected_trace_coverage'], result['projected_trace_coverage'])
        self.assertEqual(reused['total_identity'], 'CALLER_BOUND')

    def test_operands_not_mutated(self):
        keys = self.rng.normal(size=(5, 3)).astype(np.float32)
        original = keys.copy()
        keys.flags.writeable = False
        geometry_summary(keys)
        compare_keys(keys, keys)
        np.testing.assert_array_equal(keys, original)


if __name__ == '__main__':
    unittest.main()
