"""Tiny CPU bias/output fixtures only: no pretrained model, fit, GPU or pilot."""
import unittest
from unittest.mock import patch

import torch

from . import native_cake_blue as native
from .test_cake_blue import mock_engine, requests


class NativeBiasGuardTests(unittest.TestCase):
    def setUp(self):
        self.dims = patch.multiple(native, HIDDEN=4, INTERMEDIATE=8)
        self.dims.start()
        self.module = torch.nn.Linear(8, 4, dtype=torch.float32)

    def tearDown(self):
        self.dims.stop()

    def test_stock_affine_bias_contribution_and_output_are_preserved(self):
        with torch.no_grad():
            self.module.weight.zero_()
            self.module.bias.copy_(torch.tensor([1., -2., 3., -4.]))
        bias = self.module.bias
        before = (bias.data_ptr(), bias._version, bias.detach().clone())
        inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        guard = native.NativeFCOutBiasGuard({3: self.module})
        handles = guard.observe()
        try:
            output = self.module(inputs)
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(torch.equal(output, before[2].expand(2, 4)))
        self.assertIs(self.module.bias, bias)
        self.assertEqual((bias.data_ptr(), bias._version), before[:2])
        self.assertTrue(torch.equal(bias, before[2]))
        self.assertFalse(self.module._forward_hooks)
        guard.check()
        receipt = guard.receipt()
        self.assertEqual(receipt['stock_linear_forward']['expression'],
                         'F.linear(input, self.weight, self.bias)')
        self.assertEqual(receipt['B1_outputs'][0]['output_shape'], [2, 4])
        self.assertEqual(receipt['affine_numerical_parity'], 'NOT_MEASURED')
        self.assertEqual(receipt['extra_model_forwards'], 0)
        self.assertEqual(receipt['extra_linear_computations'], 0)

    def test_missing_bias_and_invalid_shape_dtype_nonfinite_are_rejected(self):
        for bias in (None, torch.ones(3), torch.ones(4, dtype=torch.float64),
                     torch.tensor([0., 0., 0., float('nan')]),
                     torch.tensor([0., 0., 0., float('inf')])):
            with self.subTest(bias=bias):
                self.module.bias = None if bias is None else torch.nn.Parameter(bias)
                with self.assertRaisesRegex(RuntimeError, 'BIAS_SHAPE_DTYPE_NONFINITE'):
                    native.NativeFCOutBiasGuard({3: self.module})

    def test_bias_replacement_storage_and_in_place_changes_are_rejected(self):
        for change in ('parameter', 'storage', 'in_place'):
            with self.subTest(change=change):
                module = torch.nn.Linear(8, 4, dtype=torch.float32)
                guard = native.NativeFCOutBiasGuard({3: module})
                with torch.no_grad():
                    if change == 'parameter':
                        module.bias = torch.nn.Parameter(module.bias.detach().clone())
                    elif change == 'storage':
                        module.bias.data = module.bias.detach().clone()
                    else:
                        module.bias.add_(1.)
                with self.assertRaisesRegex(RuntimeError, 'FC_OUT_BIAS_MODIFIED'):
                    guard.check()

    def test_invalid_observed_output_shape_dtype_nonfinite_are_rejected(self):
        changes = (lambda output: output[..., :-1], lambda output: output.double(),
                   lambda output: torch.full_like(output, float('nan')))
        for change in changes:
            with self.subTest(change=change):
                guard = native.NativeFCOutBiasGuard({3: self.module})
                earlier = self.module.register_forward_hook(
                    lambda module, inputs, output: change(output))
                handles = guard.observe()
                try:
                    with self.assertRaisesRegex(RuntimeError, 'OUTPUT_SHAPE_DTYPE_NONFINITE'):
                        self.module(torch.ones((2, 8), dtype=torch.float32))
                finally:
                    for handle in handles:
                        handle.remove()
                    earlier.remove()
                self.assertFalse(self.module._forward_hooks)

    def test_receipt_requires_actual_existing_forward_observation(self):
        guard = native.NativeFCOutBiasGuard({3: self.module})
        with self.assertRaisesRegex(RuntimeError, 'B1_FC_OUT_OUTPUT_NOT_OBSERVED'):
            guard.receipt()

    def test_public_apply_removes_temporary_hooks_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), patch('torch.linalg.solve',
                                               return_value=torch.zeros((8, 4))):
                engine, _ = mock_engine('CAKE')
                engine.prepare_contexts()
                engine.fc_out_bias_guard = native.NativeFCOutBiasGuard({3: self.module})
                original = engine.native_apply

                def public(*args, **kwargs):
                    self.module(torch.ones((1, 8), dtype=torch.float32))
                    if fail:
                        raise RuntimeError('FIXTURE_PUBLIC_APPLY_FAILED')
                    return original(*args, **kwargs)

                engine.native_apply = public
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'FIXTURE_PUBLIC_APPLY_FAILED'):
                        engine.apply(requests(), 1)
                else:
                    _, receipt = engine.apply(requests(), 1)
                    self.assertTrue(receipt['fc_out_native_bias']['pointer_version_unchanged'])
                    engine.apply(requests(), 2)
                self.assertFalse(self.module._forward_hooks)
                engine.fc_out_bias_guard.check()


if __name__ == '__main__':
    unittest.main()
