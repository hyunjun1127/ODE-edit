"""Tiny CPU checks for the bounded diagnostic route; no GPU execution."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from project.run_scripts.alpha_key_concentration_causal.writer_diagnostics import (
    WriterDiagnosticError, projector_diagnostics, quadratic_trace, summarize,
)


class WriterDiagnosticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        generator = torch.Generator().manual_seed(73)
        self.K = torch.randn(6, 100, generator=generator)
        self.R = torch.randn(4, 100, generator=generator)
        self.P = torch.eye(6).repeat(5, 1, 1)
        matrix = torch.randn(6, 11, generator=generator)
        self.M = (matrix@matrix.T + torch.eye(6)).repeat(5, 1, 1)
        A = self.P[0]@(self.K@self.K.T+self.M[0])+10*torch.eye(6)
        C = torch.linalg.solve(A, self.P[0]@self.K)
        delta = torch.linalg.solve(A, self.P[0]@self.K@self.R.T).T
        G = self.K.double().T@C.double()
        H = torch.linalg.solve((torch.eye(100)-G).T, G.T).T
        self.factor = dict(K=self.K, R=self.R, delta=delta, C=C, G=G, H=H,
                           reconstruction_max_abs=float((self.R@C.T-delta).abs().max()),
                           reconstruction_relative_frobenius=float(torch.linalg.norm(self.R@C.T-delta)/torch.linalg.norm(delta)),
                           artifact={'path': 'local-only-factors.pt', 'sha256': 'fixture'})
        self.stamp = {4: torch.randn(6, 512, generator=generator)}
        self.current = {4: self.stamp[4]+.1*torch.randn(6, 512, generator=generator)}
        self.rt = SimpleNamespace(P=self.P, weights={4: torch.zeros(4, 6)})

    def test_quadratic_trace_matches_dense_algebra_in_blocks(self):
        left = self.factor['delta'].double()
        middle = self.M[0].double()
        actual = quadratic_trace(left, middle, block_rows=2)
        expected = float(torch.trace(left@middle@left.T))
        self.assertAlmostEqual(actual, expected, places=12)

    def test_projector_numerical_is_measurement_not_native_replacement(self):
        P = torch.tensor([[1., .1], [0., 0.]], dtype=torch.float64)
        original = P.clone()
        result = projector_diagnostics(P)
        self.assertGreater(result['symmetry_max_abs'], 0)
        self.assertEqual(result['ideal_exact_projector'], 'NOT_ESTABLISHED')
        self.assertFalse(result['orthogonalization'])
        self.assertTrue(torch.equal(P, original))

    def test_summary_full512_and_dense_reference_penalties(self):
        result = summarize(self.rt, {4: self.factor}, self.stamp, self.current, self.M, {})['4']
        h = result['history_coverage']
        delta = self.factor['delta'].double()
        self.assertEqual(h['bank_n'], 512)
        self.assertFalse(h['superseded_excluded_from_statistics'])
        self.assertAlmostEqual(h['native_write_history_penalty'],
                               float(torch.trace(delta@self.M[0].double()@delta.T)), places=10)
        self.assertAlmostEqual(h['stamp_penalty'], float((delta@self.stamp[4].double()).square().sum()), places=10)
        self.assertAlmostEqual(h['projected_history_trace'], float(torch.trace(self.M[0].double())), places=10)
        self.assertEqual(result['raw_native_H']['ideal_orthogonal_P_identity'], 'NOT_ESTABLISHED')
        self.assertFalse(result['diagnostic_execution']['GPU_actual'])
        self.assertFalse(result['diagnostic_execution']['CPU_d_cubed_matmul'])

    def test_raw_nonsymmetry_is_preserved_and_sym_modes_labeled_approximation(self):
        self.factor['H'] = torch.eye(100, dtype=torch.float64)*.3
        self.factor['H'][0, 1] = .1
        before = self.factor['H'].clone()
        result = summarize(self.rt, {4: self.factor}, self.stamp, self.current, self.M, {})['4']
        self.assertGreater(result['raw_native_H']['raw_asymmetry_max_abs'], 0)
        self.assertEqual(result['history_whitened_modes_status'], 'SYMMETRIZED_NATIVE_H_APPROXIMATION_NOT_IDEAL_P')
        self.assertEqual(result['ideal_formula_interpretation'], 'DIAGNOSTIC_APPROXIMATION_ONLY')
        self.assertTrue(torch.equal(before, self.factor['H']))

    def test_negative_H_is_not_applicable_not_native_failure_or_clipping(self):
        self.factor['H'] = torch.eye(100, dtype=torch.float64)
        self.factor['H'][0, 0] = -.25
        result = summarize(self.rt, {4: self.factor}, self.stamp, self.current, self.M, {})['4']
        self.assertEqual(result['history_whitened_modes_status'], 'NOT_APPLICABLE')
        self.assertEqual(result['raw_native_H']['reason'], 'SYMMETRIC_PART_NOT_PSD')
        self.assertEqual(result['raw_native_H']['symmetric_minimum_eigenvalue'], -.25)
        self.assertGreater(result['actual_response_energy'], 0)
        self.assertEqual(float(self.factor['H'][0, 0]), -.25)

    def test_cache_hits_bound_to_value_hash_not_alias_pointer_version(self):
        cache = {}
        first = summarize(self.rt, {4: self.factor}, self.stamp, self.current, self.M, cache)['4']
        second = summarize(self.rt, {4: self.factor}, self.stamp, self.current, self.M, cache)['4']
        self.assertFalse(any(first['diagnostic_execution']['cache_hits'].values()))
        self.assertTrue(all(second['diagnostic_execution']['cache_hits'].values()))
        # NumPy alias mutation can bypass Tensor._version: SHA must catch it.
        old_version = self.P._version
        self.P.numpy()[0, 0, 0] += .01
        self.assertEqual(self.P._version, old_version)
        third = summarize(self.rt, {4: self.factor}, self.stamp, self.current, self.M, cache)['4']
        self.assertFalse(any(third['diagnostic_execution']['cache_hits'].values()))
        self.M.numpy()[0, 0, 0] += .02
        fourth = summarize(self.rt, {4: self.factor}, self.stamp, self.current, self.M, cache)['4']
        self.assertTrue(fourth['diagnostic_execution']['cache_hits']['P_numerical'])
        self.assertFalse(fourth['diagnostic_execution']['cache_hits']['projected_history_trace'])

    def test_inputs_unchanged_and_no_tensors_in_cache(self):
        copies = {name: value.clone() for name, value in self.factor.items() if isinstance(value, torch.Tensor)}
        P, M, stamp, current = self.P.clone(), self.M.clone(), self.stamp[4].clone(), self.current[4].clone()
        cache = {}
        summarize(self.rt, {4: self.factor}, self.stamp, self.current, self.M, cache)
        for key, old in copies.items():
            self.assertTrue(torch.equal(old, self.factor[key]))
        for a, b in [(P, self.P), (M, self.M), (stamp, self.stamp[4]), (current, self.current[4])]:
            self.assertTrue(torch.equal(a, b))
        def tensor_nested(item):
            if isinstance(item, torch.Tensor):
                return True
            if isinstance(item, dict):
                return any(tensor_nested(x) for x in item.values())
            return False
        self.assertFalse(tensor_nested(cache))

    def test_json_finite_and_missing_history_fail_closed(self):
        import json
        result = summarize(self.rt, {4: self.factor}, self.stamp, self.current, self.M, {})
        json.dumps(result, allow_nan=False)
        with self.assertRaisesRegex(WriterDiagnosticError, 'full H512'):
            summarize(self.rt, {4: self.factor}, self.stamp, {4:self.current[4][:, :500]}, self.M, {})
        with self.assertRaises(WriterDiagnosticError):
            quadratic_trace(self.factor['delta'], self.M[0])


if __name__ == '__main__':
    unittest.main()
