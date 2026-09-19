"""Synthetic harsh-case regressions; these are not actual Llama gate evidence."""
import json
import unittest

import torch

from .validation import (elementwise_gate, evaluation_rows_gate, repeat_spread_gate,
                         response_gate, solve_residual_gate, update_gate)


def d(value):
    return torch.tensor(value, dtype=torch.float64)


class ValidationTests(unittest.TestCase):
    def test_one_bad_key_element_not_hidden_by_average(self):
        ref = torch.zeros(10000)
        actual = ref.clone(); actual[7999] = 1.1e-6
        receipt = elementwise_gate(actual, ref)
        self.assertFalse(receipt['passed'])
        self.assertEqual(receipt['failed_elements'], 1)
        self.assertEqual(receipt['worst_index'], [7999])

    def test_elementwise_relative_to_reference_not_actual(self):
        self.assertTrue(elementwise_gate(d([100.0005]), d([100]))['passed'])
        self.assertFalse(elementwise_gate(d([100.002]), d([100]))['passed'])

    def test_nonsymmetric_solve_and_input_nonmutation(self):
        a = d([[1, 2], [0, 3]])
        b = d([[1, 5], [6, 2]])
        x = torch.linalg.solve(a, b)
        originals = [t.clone() for t in (a, x, b)]
        out = solve_residual_gate(a, x, b, row_chunk=1)
        self.assertTrue(out['passed'])
        self.assertLess(out['max_relative_error'], 1e-14)
        for left, right in zip((a, x, b), originals):
            self.assertTrue(torch.equal(left, right))

    def test_bad_rhs_column_cannot_hide_in_frobenius(self):
        a = torch.eye(2, dtype=torch.float64)
        b = d([[1e9, 1], [0, 0]])
        x = b.clone(); x[0, 1] += 1e-3
        out = solve_residual_gate(a, x, b)
        self.assertFalse(out['passed'])
        self.assertEqual(out['failed_indices'], [1])

    def test_solve_uses_rhs_not_backward_error_denominator(self):
        a = d([[1e8, -1e8], [0, 1]])
        x = d([[1.00000001], [1]])
        b = d([[0.5], [1]])
        out = solve_residual_gate(a, x, b)
        self.assertFalse(out['passed'])
        self.assertGreater(out['max_relative_error'], 0.1)

    def test_zero_rhs_absolute_rule(self):
        a = torch.eye(2, dtype=torch.float64); b = torch.zeros((2, 2), dtype=torch.float64)
        x = d([[0.5e-7, 1.1e-7], [0, 0]])
        out = solve_residual_gate(a, x, b)
        self.assertFalse(out['passed'])
        self.assertEqual(out['relative_errors'], [None, None])
        self.assertEqual(out['zero_reference_count'], 2)
        self.assertEqual(out['failed_indices'], [1])

    def test_near_zero_threshold_inclusive(self):
        out = update_gate(d([[1e-8]]), d([[1e-12]]))
        self.assertTrue(out['passed']); self.assertEqual(out['zero_reference_count'], 1)
        out = update_gate(d([[1e-8]]), d([[1.01e-12]]))
        self.assertFalse(out['passed']); self.assertEqual(out['zero_reference_count'], 0)

    def test_delta_denominator_is_actual(self):
        # If reconstructed norm were the denominator, this would pass .001.
        out = update_gate(d([[1.0010005]]), d([[1]]))
        self.assertFalse(out['passed'])
        self.assertAlmostEqual(out['max_relative_error'], .0010005)

    def test_response_bad_request_not_average(self):
        actual = torch.eye(2, dtype=torch.float64)
        reconstructed = actual.clone(); reconstructed[1, 1] += .002
        keys = d([[1e8, 0], [0, 1]])
        out = response_gate(reconstructed, actual, keys)
        self.assertFalse(out['passed']); self.assertEqual(out['failed_indices'], [1])

    def test_response_zero_request(self):
        out = response_gate(d([[0, 2e-7]]), d([[0, 0]]), torch.eye(2, dtype=torch.float64))
        self.assertFalse(out['passed']); self.assertEqual(out['zero_reference_count'], 2)

    def test_nll_single_row_outlier(self):
        ref = torch.ones(1000, dtype=torch.float64)
        actual = ref.clone(); actual[711] += .00011
        out = evaluation_rows_gate(actual, ref + 1, ref, ref + 1, ['RS'] * 1000)
        self.assertFalse(out['passed']); self.assertEqual(out['failed_rows'], [711])

    def test_margin_difference_not_only_each_nll(self):
        out = evaluation_rows_gate(d([1.00006]), d([1.99994]), d([1]), d([2]), ['PS'])
        self.assertFalse(out['passed'])
        self.assertLess(out['max_nll_absolute_error'], 1e-4)
        self.assertGreater(out['max_margin_absolute_error'], 1e-4)

    def test_tie_fails_and_both_near_changed_bit_is_reported(self):
        out = evaluation_rows_gate(d([1]), d([1]), d([1]), d([1.00005]), ['RS'])
        self.assertTrue(out['passed']); self.assertEqual(out['status'], 'NUMERIC_BOUNDARY')
        self.assertEqual(out['actual_successes'], 0)
        self.assertEqual(out['numeric_boundary_count'], 1)
        self.assertEqual(out['changed_success_bits'], 1)

    def test_only_one_near_band_is_robust_failure(self):
        out = evaluation_rows_gate(d([1]), d([.99999]), d([1]), d([1.00011]), ['RS'])
        self.assertFalse(out['passed']); self.assertEqual(out['robust_success_bit_mismatches'], 1)
        self.assertEqual(out['numeric_boundary_count'], 0)

    def test_nli_not_accepted_as_metric_tag(self):
        with self.assertRaises(ValueError):
            evaluation_rows_gate(d([1]), d([2]), d([1]), d([2]), ['NLI'])

    def test_ns_safety_sign_and_tie(self):
        out = evaluation_rows_gate(d([2, 1]), d([1, 1]), d([2, 1]), d([1, 1]), ['NS', 'NS'])
        self.assertTrue(out['passed']); self.assertEqual(out['actual_successes'], 1)
        self.assertEqual(out['actual_ties'], 1)

    def test_repeat_range_not_sd_or_first_difference(self):
        out = repeat_spread_gate(d([[0], [-.000006], [.000006]]), 1e-4)
        self.assertFalse(out['passed']); self.assertEqual(out['failed_elements'], 1)
        self.assertAlmostEqual(out['max_tolerance_ratio'], 1.2)

    def test_repeat_elementwise_threshold(self):
        out = repeat_spread_gate(d([[0, 0], [1e-8, 1e-6]]), d([1e-6, 1e-4]))
        self.assertTrue(out['passed'])

    def test_nonfinite_fail_and_json_safe(self):
        for fn in (elementwise_gate, update_gate):
            out = fn(d([[float('nan')]]), d([[0]]))
            self.assertFalse(out['passed']); self.assertEqual(out['reason'], 'NONFINITE_INPUT')
            json.dumps(out, allow_nan=False)

    def test_shape_broadcast_rejected(self):
        with self.assertRaises(ValueError):
            elementwise_gate(d([[1, 2]]), d([1, 2]))

    def test_empty_gate_not_pass(self):
        with self.assertRaises(ValueError):
            elementwise_gate(d([]), d([]))

    def test_no_finite_relative_for_all_zero_receipt(self):
        out = update_gate(d([[0]]), d([[0]]))
        self.assertTrue(out['passed']); self.assertIsNone(out['max_relative_error'])
        json.dumps(out, allow_nan=False)

    def test_fp32_input_fp64_residual(self):
        a = torch.eye(2, dtype=torch.float32); b = torch.ones((2, 1), dtype=torch.float32)
        out = solve_residual_gate(a, b, b)
        self.assertEqual(out['compute_dtype'], 'torch.float64')
        self.assertEqual(out['input_dtypes'], ['torch.float32'] * 3)

    def test_finite_inputs_overflow_computation_fail_json_safe(self):
        receipts = [
            elementwise_gate(d([1e308]), d([-1e308])),
            solve_residual_gate(d([[1e200]]), d([[1e200]]), d([[1]])),
            update_gate(d([[1e308]]), d([[-1e308]])),
            response_gate(d([[1e200]]), d([[1]]), d([[1e200]])),
            evaluation_rows_gate(d([1e308]), d([-1e308]), d([1]), d([2]), ['RS']),
            repeat_spread_gate(d([[1e308], [-1e308]]), 1e-4),
        ]
        for receipt in receipts:
            self.assertFalse(receipt['passed'])
            self.assertEqual(receipt['reason'], 'NONFINITE_COMPUTATION')
            json.dumps(receipt, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
