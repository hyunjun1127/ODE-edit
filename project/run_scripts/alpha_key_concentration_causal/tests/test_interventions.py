"""Small CPU algebra/ownership checks. These are NOT actual model parity."""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from project.run_scripts.alpha_key_concentration_causal.interventions import (
    InterventionError, PromptActionMean, action_component_hook,
    calibration_key_basis, component_action, history_operand, history_penalty,
    interchange_key_hook, interchange_keys, kr_cross_solve, native_solve,
    weight_ablation,
)


class InterventionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.generator = torch.Generator().manual_seed(20260923)
        self.K = torch.randn(5, 7, generator=self.generator)
        self.R = torch.randn(3, 7, generator=self.generator)
        self.P = torch.eye(5)
        history = torch.randn(5, 14, generator=self.generator)
        self.M = history @ history.T
        self.stamp = torch.randn(5, 512, generator=self.generator)
        self.current = self.stamp + .1 * torch.randn(5, 512, generator=self.generator)
        self.Mh = self.M + self.stamp @ self.stamp.T
        self.ids = tuple(range(512))
        self.bank_kwargs = dict(bank_ids=self.ids, expected_bank_ids=self.ids,
                                timestamp_validated=True)

    def test_native_exact_expression_and_orientation(self):
        # A nonsymmetric P catches accidental symmetrization/transposition.
        projector = torch.rand(5, 5, generator=self.generator) * .1
        expected = torch.linalg.solve(
            projector @ (self.K @ self.K.T + self.M) + 10 * torch.eye(5),
            projector @ self.K @ self.R.T).T
        result = native_solve(self.K, self.R, self.M, projector)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(result.shape, (3, 5))

    def test_native_no_input_mutation(self):
        copies = [x.clone() for x in (self.K, self.R, self.M, self.P)]
        native_solve(self.K, self.R, self.M, self.P)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(copies, (self.K, self.R, self.M, self.P))))

    def test_native_rejects_non_native_ridge_precision_shape_and_nonfinite(self):
        calls = (
            lambda: native_solve(self.K, self.R, self.M, self.P, ridge=1),
            lambda: native_solve(self.K.double(), self.R, self.M, self.P),
            lambda: native_solve(self.K[:, :-1], self.R, self.M, self.P),
            lambda: native_solve(self.K, self.R * float('nan'), self.M, self.P),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaises(InterventionError):
                call()

    def test_native_propagates_solve_failure_not_fallback(self):
        with patch('torch.linalg.solve', side_effect=RuntimeError('singular mock')):
            with self.assertRaisesRegex(RuntimeError, 'singular mock'):
                native_solve(self.K, self.R, self.M, self.P)

    def test_history_refresh_all512_native_preserved(self):
        native = self.Mh.clone()
        result = history_operand(self.Mh, self.stamp, self.current,
                                 branch='H5', layer=5, **self.bank_kwargs)
        expected = self.Mh - self.stamp @ self.stamp.T + self.current @ self.current.T
        self.assertTrue(torch.equal(result.M, expected))
        self.assertTrue(torch.equal(native, self.Mh))
        self.assertEqual(result.receipt['bank_n'], 512)
        self.assertTrue(result.receipt['includes_superseded'])
        self.assertEqual(result.receipt['history_appends'], 0)

    def test_sham_executes_subtract_add_instead_of_copy(self):
        # Adversarial cancellation fixture, not a valid native-history example.
        # It makes an implementation that returns a clone observably incorrect.
        stamp = torch.ones(1, 512) * 1e10
        M = torch.ones(1, 1)
        result = history_operand(M, stamp, stamp, branch='SHAM', layer=4, **self.bank_kwargs)
        expected = M - stamp @ stamp.T + stamp @ stamp.T
        self.assertTrue(torch.equal(result.M, expected))
        self.assertFalse(torch.equal(result.M, M))
        self.assertTrue(result.receipt['sham_subtract_readd_executed'])

    def test_history_layer_dispatch_and_no_alias(self):
        for branch, active in [('NATIVE', ()), ('H5', (5,)), ('H6', (6,)),
                               ('H56', (5, 6)), ('MASS56', (5, 6)),
                               ('SHAM', (4, 5, 6, 7, 8))]:
            for layer in range(4, 9):
                with self.subTest(branch=branch, layer=layer):
                    result = history_operand(self.Mh, self.stamp, self.current,
                                             branch=branch, layer=layer, **self.bank_kwargs)
                    self.assertEqual(result.receipt['applied'], layer in active)
                    self.assertNotEqual(result.M.data_ptr(), self.Mh.data_ptr())

    def test_mass_matches_refresh_trace_not_spectrum(self):
        result = history_operand(self.Mh, self.stamp, self.current,
                                 branch='MASS56', layer=6, **self.bank_kwargs)
        refresh = self.Mh - self.stamp @ self.stamp.T + self.current @ self.current.T
        scale = torch.trace(refresh) / torch.trace(self.Mh)
        self.assertTrue(torch.equal(result.M, self.Mh * scale))
        self.assertAlmostEqual(float(torch.trace(result.M)), float(torch.trace(refresh)), places=3)
        self.assertFalse(torch.equal(result.M, refresh))

    def test_history_rejects_unvalidated_missing_filtered_or_reordered_bank(self):
        kwargs_list = [dict(self.bank_kwargs, timestamp_validated=False),
                       dict(self.bank_kwargs, bank_ids=self.ids[::-1]),
                       dict(self.bank_kwargs, bank_ids=(1,) * 512),
                       dict(self.bank_kwargs, expected_bank_ids=None)]
        for kwargs in kwargs_list:
            with self.assertRaises(InterventionError):
                history_operand(self.Mh, self.stamp, self.current, branch='H5', layer=5, **kwargs)
        with self.assertRaises(InterventionError):
            history_operand(self.Mh, self.stamp[:, :511], self.current[:, :511],
                            branch='H5', layer=5, **self.bank_kwargs)

    def test_no_psd_clipping(self):
        tiny = torch.eye(5)
        result = history_operand(tiny, self.stamp, torch.zeros_like(self.current),
                                 branch='H6', layer=6, **self.bank_kwargs)
        self.assertLess(float(torch.linalg.eigvalsh(result.M).min()), 0)
        self.assertFalse(result.receipt['psd_clipping'])
        with self.assertRaises(InterventionError):
            history_operand(tiny, self.stamp, torch.zeros_like(self.current),
                            branch='MASS56', layer=6, **self.bank_kwargs)

    def test_penalty_signed_column_difference(self):
        delta = native_solve(self.K, self.R, self.M, self.P)
        result = history_penalty(delta, self.stamp, self.current)
        expected = (delta @ self.current).double().square().sum(0) - (
            delta @ self.stamp).double().square().sum(0)
        self.assertTrue(torch.equal(result['signed_current_minus_stamp'], expected))
        self.assertEqual(result['bank_n'], 512)

    def test_prompt_mean_not_token_weighted_and_pad_excluded(self):
        accumulator = PromptActionMean()
        accumulator.add(torch.tensor([[2., 4.], [99., 99.]]), torch.tensor([True, False]))
        accumulator.add(torch.tensor([[8., 10.]]).repeat(3, 1), torch.ones(3, dtype=torch.bool))
        mean, receipt = accumulator.finish()
        self.assertTrue(torch.equal(mean, torch.tensor([5., 7.])))
        self.assertEqual(receipt['valid_tokens'], 4)
        self.assertEqual(receipt['prompts'], 2)

    def test_prompt_mean_rejects_observer_and_empty_prompt(self):
        for role in ['N512', 'current_P', 'assessment3488']:
            with self.assertRaises(InterventionError):
                PromptActionMean(panel_role=role)
        with self.assertRaises(InterventionError):
            PromptActionMean().add(torch.ones(2, 3), torch.zeros(2, dtype=torch.bool))
        with self.assertRaises(InterventionError):
            PromptActionMean().finish()

    def test_activation_component_sum(self):
        keys = self.K.T.unsqueeze(0)
        delta = torch.randn(3, 5, generator=self.generator)
        mean = torch.randn(3, generator=self.generator)
        full = component_action(keys, delta, mean, 'full')
        centered = component_action(keys, delta, mean, 'centered')
        mu = component_action(keys, delta, mean, 'mean')
        torch.testing.assert_close(centered + mu, full)
        self.assertEqual(int(torch.count_nonzero(component_action(keys, delta, mean, 'no'))), 0)

    def test_hook_all_valid_tokens_tuple_and_input_alias_ownership(self):
        weight = torch.randn(3, 5, generator=self.generator)
        delta = torch.randn(3, 5, generator=self.generator) * .01
        mean = torch.tensor([1., 2., 3.])
        mask = torch.tensor([[True, True, False]])
        keys = torch.randn(1, 3, 5, generator=self.generator)
        expected = F.linear(keys, weight) + torch.where(
            mask.unsqueeze(-1), F.linear(keys, delta), torch.zeros(1, 3, 3))
        hook = action_component_hook(weight, delta, mean, 'full', mask)
        weight.zero_()
        delta.zero_()
        mean.zero_()
        mask.zero_()
        sentinel = object()
        output = hook(None, (keys,), (torch.zeros(1, 3, 3), sentinel))
        self.assertTrue(torch.equal(output[0], expected))
        self.assertIs(output[1], sentinel)

    def test_full_hook_vs_physical_write_cpu_numerical_only(self):
        weight = torch.randn(3, 5, generator=self.generator)
        delta = torch.randn(3, 5, generator=self.generator) * .01
        keys = torch.randn(2, 4, 5, generator=self.generator)
        hook = action_component_hook(weight, delta, torch.zeros(3), 'full',
                                     torch.ones(2, 4, dtype=torch.bool))
        actual = hook(None, (keys,), F.linear(keys, weight))
        physical = F.linear(keys, weight + delta)
        torch.testing.assert_close(actual, physical, rtol=1e-5, atol=1e-6)
        # This small CPU check deliberately does not establish actual GPU parity.

    def test_hook_rejects_mask_shape_and_does_not_infer_subject_mask(self):
        hook = action_component_hook(torch.zeros(3, 5), torch.zeros(3, 5),
                                     torch.zeros(3), 'mean', torch.ones(2, dtype=torch.bool))
        with self.assertRaises(InterventionError):
            hook(None, (torch.zeros(1, 2, 5),), torch.zeros(1, 2, 3))

    def test_weight_ablation_sum_null_direction_and_norm_control(self):
        delta = torch.randn(3, 5, generator=self.generator)
        mean = torch.tensor([1., -2., 3.])
        result = weight_ablation(delta, mean)
        self.assertEqual(result.status, 'DEFINED')
        torch.testing.assert_close(result.parallel + result.perpendicular, delta)
        torch.testing.assert_close(result.direction @ result.perpendicular, torch.zeros(5), atol=1e-6, rtol=0)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(result.norm_control)),
                               float(torch.linalg.vector_norm(result.perpendicular)), places=5)
        # Right-nullspace preservation under left projection is algebraic.
        delta[:, -1] = 0
        result = weight_ablation(delta, mean)
        self.assertTrue(torch.equal(result.perpendicular[:, -1], torch.zeros(3)))

    def test_zero_mean_is_not_applicable_no_replacement_direction(self):
        result = weight_ablation(torch.ones(3, 5), torch.zeros(3))
        self.assertEqual(result.status, 'NOT_APPLICABLE')
        self.assertEqual(result.reason, 'ZERO_CALIBRATION_ACTION_MEAN')
        self.assertIsNone(result.direction)
        self.assertIsNone(result.perpendicular)
        with self.assertRaises(InterventionError):
            weight_ablation(torch.ones(3, 5), torch.ones(3), panel_role='N512')

    def test_kr_four_updates_one_receiving_state_native_direct(self):
        K_b = self.K + .2
        R_b = self.R - .3
        before = self.M.clone()
        updates, receipt = kr_cross_solve(self.K, K_b, self.R, R_b, self.M, self.P,
                                         receiving_state_id='fixed-W_b-SHA')
        self.assertEqual(list(updates), ['aa', 'ab', 'ba', 'bb'])
        for key, K, R in [('aa', self.K, self.R), ('ab', self.K, R_b),
                          ('ba', K_b, self.R), ('bb', K_b, R_b)]:
            self.assertTrue(torch.equal(updates[key], native_solve(K, R, self.M, self.P)))
        self.assertTrue(torch.equal(self.M, before))
        self.assertEqual(receipt['history_appends'], 0)
        self.assertEqual(receipt['receiving_state_id'], 'fixed-W_b-SHA')
        with self.assertRaises(InterventionError):
            kr_cross_solve(self.K, K_b, self.R, R_b, self.M, self.P, receiving_state_id='')

    def test_calibration_basis_rank_control_and_interchange(self):
        result = calibration_key_basis(self.K, 2)
        self.assertEqual(result.status, 'DEFINED')
        basis = result.vectors
        torch.testing.assert_close(basis.T @ basis, torch.eye(2), rtol=1e-5, atol=1e-6)
        base = torch.randn(2, 3, 5, generator=self.generator)
        donor = torch.randn(2, 3, 5, generator=self.generator)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        out = interchange_keys(base, donor, basis, mask)
        expected = base + torch.where(mask.unsqueeze(-1),
                                     ((donor-base) @ basis) @ basis.T, torch.zeros_like(base))
        self.assertTrue(torch.equal(out, expected))
        self.assertTrue(torch.equal(out[~mask], base[~mask]))
        hook = interchange_key_hook(donor, basis, mask)
        self.assertTrue(torch.equal(hook(None, (base, 'extra'))[0], out))
        self.assertEqual(hook(None, (base, 'extra'))[1], 'extra')

    def test_basis_zero_rank_and_observer_rejection(self):
        result = calibration_key_basis(torch.zeros(5, 10), 1)
        self.assertEqual(result.status, 'NOT_APPLICABLE')
        self.assertIsNone(result.vectors)
        for rank, role in [(3, 'calibration512'), (1, 'N512'), (1, 'assessment3488')]:
            with self.assertRaises(InterventionError):
                calibration_key_basis(self.K, rank, panel_role=role)


if __name__ == '__main__':
    unittest.main()
