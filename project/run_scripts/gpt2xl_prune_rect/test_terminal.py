"""Contract-authorized CPU algebra controls; no target fit/GPU/model pilot."""
import unittest

import torch

from .terminal import apply_terminal, compressed_delta


class TerminalTests(unittest.TestCase):
    def test_terminal_hash_exact_existing_commit_state_convention(self):
        from .terminal import tensor_sha
        from project.run_scripts.jlz_realization.common import tensor_sha as state_sha
        value = torch.diag(torch.tensor([2., 1.], dtype=torch.float32))
        self.assertEqual(tensor_sha(value), state_sha(value))
        self.assertNotEqual(tensor_sha(value), tensor_sha(value.double()))

    def test_zero_delta_returns_same_cold_W0(self):
        cold = torch.diag(torch.tensor([2., 1.], dtype=torch.float32))
        delta, row = compressed_delta(cold, cold.clone())
        self.assertTrue(torch.equal(delta, torch.zeros_like(cold)))
        self.assertEqual(row['transformed_singular_count'], 0)

    def test_uncompressed_small_delta_is_dense_once_not_doubleinsert(self):
        cold = torch.diag(torch.tensor([2., 1.], dtype=torch.float32))
        dense = cold + torch.diag(torch.tensor([.25, .125], dtype=torch.float32))
        weights = {f'weight{i}': dense.clone() for i in range(5)}
        originals = {name: cold.clone() for name in weights}
        result = apply_terminal(weights, originals, expected_shape=(2, 2))
        self.assertTrue(all(torch.equal(value, dense) for value in weights.values()))
        self.assertFalse(torch.equal(weights['weight0'], dense + (dense - cold)))
        self.assertEqual(result['repair'], 'PRUNE_TERMINAL_BASE_FIX')
        self.assertEqual(result['final_base'], 'SAVED_COLD_W0')
        self.assertEqual(result['terminal_transforms'], 1)
        self.assertFalse(result['upstream_bitwise_equivalence'])

    def test_native_log_spectral_expression_preserved(self):
        cold = torch.diag(torch.tensor([2., 1.], dtype=torch.float32))
        dense = cold + torch.diag(torch.tensor([4., .125], dtype=torch.float32))
        delta, row = compressed_delta(cold, dense)
        expected = torch.log(torch.tensor(4.)) - torch.log(torch.tensor(2.)) + 2
        self.assertTrue(torch.equal(delta[0, 0], expected))
        self.assertEqual(delta[1, 1].item(), .125)
        self.assertEqual(row['transformed_singular_count'], 1)
        self.assertTrue(torch.isfinite(delta).all())

    def test_shape_dtype_nonfinite_and_zero_original_spectrum_blocked(self):
        cold = torch.eye(2, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, 'SHAPE_DTYPE'):
            compressed_delta(cold.double(), cold.double())
        with self.assertRaisesRegex(RuntimeError, 'SHAPE_DTYPE'):
            compressed_delta(cold, torch.ones(3, 2))
        with self.assertRaisesRegex(RuntimeError, 'SHAPE_DTYPE'):
            compressed_delta(cold, torch.full_like(cold, float('nan')))
        with self.assertRaisesRegex(RuntimeError, 'SPECTRUM_POSITIVE'):
            compressed_delta(torch.zeros_like(cold), torch.zeros_like(cold))

    def test_five_selected_shape_and_cold_mapping_guard(self):
        cold = torch.eye(2)
        with self.assertRaisesRegex(RuntimeError, 'FIVE_COLD'):
            apply_terminal({'one': cold}, {'one': cold}, expected_shape=(2, 2))


if __name__ == '__main__':
    unittest.main()
