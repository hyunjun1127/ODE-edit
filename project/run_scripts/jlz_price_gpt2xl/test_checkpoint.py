"""Actual production checkpoint closure regressions using CPU operators only.

No model, scientific fit, input dataset, evaluator, GPU, or Slurm is used.
The failed frozen source is read-only provenance, not a modified reference.
"""
import hashlib
import importlib.util
from pathlib import Path
import unittest

import torch
from torch.utils.checkpoint import CheckpointError

from .adapter import Adapter


FROZEN_ADAPTER = Path(
    '/mnt/raid5/janghj/ODE-edit/local/jlz-price-gpt2xl-2k/'
    'attempt-execution-r1/source/project/run_scripts/jlz_price_gpt2xl/adapter.py'
)
FROZEN_ADAPTER_SHA256 = (
    'd4052ecfac877fc4caa474051a8a90c9a41e8635293dd258cd39e8e7d9650f01'
)


def frozen_adapter_class():
    """Import the exact failed Adapter without editing or copying its bytes."""
    if hashlib.sha256(FROZEN_ADAPTER.read_bytes()).hexdigest() != FROZEN_ADAPTER_SHA256:
        raise AssertionError('FAILED_FROZEN_ADAPTER_IDENTITY_CHANGED')
    spec = importlib.util.spec_from_file_location(
        'jlz_gpt2_failed_checkpoint_adapter_fixture', FROZEN_ADAPTER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Adapter


def make_fixture(adapter_class, checkpoint_enabled):
    """Seven owner rows and pure operators exercise the real masked method."""
    adapter = object.__new__(adapter_class)
    adapter.device = torch.device('cpu')
    adapter.first = 13
    adapter.sites = (13, 14, 15, 16, 17)
    adapter.checkpoint_enabled = checkpoint_enabled
    adapter.weights = {layer: torch.eye(2) for layer in adapter.sites}
    adapter.blocks = [None] * 19
    adapter.blocks[18] = lambda value, **kwargs: (torch.sin(value),)
    adapter.pre_projection = lambda layer, value, kwargs: (torch.sin(value), .2 * value)
    adapter.compose = lambda layer, key, residual, weight: residual + key @ weight
    rows = [
        dict(
            lookup=index % 3,
            global_row=index,
            kind='rewrite' if index < 6 else 'kl',
            target=(torch.tensor([-100, 1, 2, -100]) if index < 6
                    else torch.full((4,), -100)),
        )
        for index in range(7)
    ]
    group = dict(
        rows=rows,
        cache=dict(
            key=torch.arange(56, dtype=torch.float32).reshape(7, 4, 2) / 60,
            residual=torch.zeros(7, 4, 2),
            kwargs={},
        ),
    )
    # This is only a no-grad pure-operator reference for the C0 observation
    # guard, not a model entry, native fit, or checkpoint/state dump.
    with torch.no_grad():
        zero = {layer: torch.zeros(7, 2) for layer in adapter.sites}
        reference = adapter.masked(group, zero, capture=True)[0]
    group['native_c0_selected'] = [
        reference[index, (
            torch.nonzero(row['target'] != -100).flatten()
            if row['kind'] == 'rewrite' else torch.tensor([row['lookup']])
        )].clone()
        for index, row in enumerate(rows)
    ]
    group['native_c0_pending'] = True
    adapter.physical_calls = dict(checkpoint_wrappers=0, function_invocations=0)
    increments = {
        layer: torch.zeros(7, 2, requires_grad=True) for layer in adapter.sites
    }
    return adapter, group, increments


def evaluated_fixture(adapter_class, checkpoint_enabled):
    adapter, group, increments = make_fixture(adapter_class, checkpoint_enabled)
    lookup_before = tuple(row['lookup'] for row in group['rows'])
    targets_before = [row['target'].clone() for row in group['rows']]
    output, final, keys, bases = adapter.masked(group, increments, capture=True)
    gradients = torch.autograd.grad(output.square().sum(), tuple(increments.values()))
    if group['native_c0_pending'] or group['native_c0_error_max'] != 0:
        raise AssertionError('CPU_C0_GUARD_NOT_EXACT')
    if tuple(row['lookup'] for row in group['rows']) != lookup_before:
        raise AssertionError('CPU_LOOKUP_ROWS_MUTATED')
    if any(not torch.equal(row['target'], old)
           for row, old in zip(group['rows'], targets_before)):
        raise AssertionError('CPU_TARGET_ROWS_MUTATED')
    return dict(
        output=output.detach(), final=final.detach(),
        keys={layer: value.detach() for layer, value in keys.items()},
        bases={layer: value.detach() for layer, value in bases.items()},
        gradients=gradients, physical_calls=dict(adapter.physical_calls),
    )


class ProductionCheckpointTests(unittest.TestCase):
    def test_frozen_c0_observation_rebind_reproduces_saved_shape_failure(self):
        adapter, group, increments = make_fixture(frozen_adapter_class(), True)
        output = adapter.masked(group, increments, capture=True)[0]
        self.assertFalse(group['native_c0_pending'])
        with self.assertRaises(CheckpointError) as observed:
            torch.autograd.grad(output.square().sum(), tuple(increments.values()))
        message = str(observed.exception)
        self.assertIn('Recomputed values', message)
        self.assertIn('torch.Size([7])', message)
        self.assertIn('torch.Size([1])', message)
        self.assertFalse(torch.cuda.is_initialized())

    def test_repaired_production_checkpoint_matches_disabled_reference(self):
        reference = evaluated_fixture(Adapter, False)
        actual = evaluated_fixture(Adapter, True)
        for field in ('output', 'final'):
            torch.testing.assert_close(actual[field], reference[field], atol=0, rtol=0)
        for field in ('keys', 'bases'):
            self.assertEqual(set(actual[field]), {13, 14, 15, 16, 17})
            for layer in actual[field]:
                torch.testing.assert_close(
                    actual[field][layer], reference[field][layer], atol=0, rtol=0
                )
        self.assertEqual(len(actual['gradients']), 5)
        for actual_gradient, reference_gradient in zip(
            actual['gradients'], reference['gradients']
        ):
            self.assertTrue(bool(torch.isfinite(actual_gradient).all()))
            self.assertTrue(bool((actual_gradient != 0).any()))
            torch.testing.assert_close(
                actual_gradient, reference_gradient, atol=0, rtol=0
            )
        self.assertGreater(
            actual['physical_calls']['function_invocations'],
            reference['physical_calls']['function_invocations'],
        )
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
