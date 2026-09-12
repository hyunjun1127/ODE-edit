import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from project.run_scripts.baseline_mechanism_first.fixtures import (
    FixtureBoundary, FixtureTransaction, SingletonSpec, capture_rng,
    restore_checkpoint, restore_rng, select_projector, tensor_sha,
)


def toy():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(3, 2, bias=False) for _ in range(9)])
    return model


class FixtureTests(unittest.TestCase):
    def test_all_physical_projector_indices_and_local_zero(self):
        p = torch.stack([torch.eye(3) * i for i in range(5)])
        for layer in range(4, 9):
            with self.subTest(layer=layer):
                selected, receipt = select_projector(p, SingletonSpec(layer))
                self.assertTrue(torch.equal(selected[0], p[layer - 4]))
                self.assertEqual(receipt['local_index'], 0)
                self.assertNotEqual(selected.data_ptr(), p.data_ptr())

    def test_rng_replays_all_cpu_streams(self):
        state = capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng(state)
        self.assertEqual(expected[0], random.random())
        self.assertEqual(expected[1], np.random.rand())
        self.assertTrue(torch.equal(expected[2], torch.rand(3)))

    def test_cuda_missing_no_reseed(self):
        if torch.cuda.is_initialized():
            self.skipTest('CPU contract fixture')
        saved = capture_rng(); saved['cuda'] = [[1, 2, 3]]
        before = capture_rng()
        with self.assertRaisesRegex(FixtureBoundary, 'CUDA_RNG_DEVICE_MAPPING_UNVERIFIED'):
            restore_rng(saved)
        self.assertEqual(capture_rng(), before)

    def test_transaction_restores_selected_nonselected_globals_hooks_and_rng(self):
        model = toy(); history = torch.zeros(1, 3, 3)
        module = SimpleNamespace(CONTEXT_TEMPLATES_CACHE=[['{}']], COV_CACHE={'x': torch.ones(2)})
        original = {k: (v, tensor_sha(v)) for k, v in model.named_parameters()}
        covariance = module.COV_CACHE['x']; rng = capture_rng()
        tx = FixtureTransaction(model, module, history)
        with self.assertRaisesRegex(ValueError, 'forced'):
            with tx:
                with torch.no_grad():
                    model.layers[4].weight.add_(1)
                    model.layers[0].weight.mul_(2)
                    history.add_(3); covariance.zero_()
                model.layers[1].weight = torch.nn.Parameter(torch.zeros(2, 3))
                model.layers[2].weight.data = torch.zeros(2, 3)
                model.requires_grad_(False)
                torch.backends.cudnn.benchmark = not tx.backend_flags[2]
                model.layers[0].register_forward_hook(lambda *a: None)
                module.CONTEXT_TEMPLATES_CACHE[0].append('changed')
                module.COV_CACHE = {}
                random.random(); torch.rand(4)
                raise ValueError('forced')
        for k, v in model.named_parameters():
            self.assertIs(v, original[k][0]); self.assertEqual(tensor_sha(v), original[k][1])
            self.assertTrue(v.requires_grad)
        self.assertEqual(torch.backends.cudnn.benchmark, tx.backend_flags[2])
        self.assertEqual(module.CONTEXT_TEMPLATES_CACHE, [['{}']])
        self.assertIs(module.COV_CACHE['x'], covariance)
        self.assertTrue(torch.equal(covariance, torch.ones(2)))
        self.assertEqual(len(model.layers[0]._forward_hooks), 0)
        self.assertEqual(capture_rng(), rng)
        self.assertTrue(tx.receipt['pointer_bytes_exact'])
        self.assertIn('NOT_CLAIMED', tx.receipt['version_restore'])

    def checkpoint(self):
        model = toy(); spec = SingletonSpec(4, 'layers.{}')
        history = torch.zeros(1, 3, 3)
        weight = torch.ones(2, 3)
        metadata = dict(batch=10, seen_ids=list(range(1000)), base_model_revision='pinned',
                        contexts=[['{}']], rng=capture_rng(), covariance={},
                        state=dict(weights={spec.weight_name: tensor_sha(weight)}, cache=tensor_sha(history)))
        cp = dict(weights={spec.weight_name: weight}, cache_c=history.clone(), metadata=metadata)
        return model, spec, history, cp

    def test_warm_restore_and_next_index(self):
        model, spec, history, cp = self.checkpoint()
        module = SimpleNamespace(CONTEXT_TEMPLATES_CACHE=None, COV_CACHE={})
        receipt = restore_checkpoint(model, module, history, cp, spec,
                                     expected_model_revision='pinned', expected_seen_ids=list(range(1000)))
        self.assertEqual(receipt['next_batch_index'], 11)
        self.assertTrue(torch.equal(model.layers[4].weight, cp['weights'][spec.weight_name]))
        self.assertFalse(receipt['C0_restored'])

    def test_missing_rng_and_wrong_layer_fail(self):
        for bad in ('rng', 'layer', 'hash'):
            with self.subTest(bad=bad):
                model, spec, history, cp = self.checkpoint()
                if bad == 'rng': del cp['metadata']['rng']
                if bad == 'layer': cp['weights'] = {'layers.5.weight': torch.ones(2, 3)}
                if bad == 'hash': cp['cache_c'].add_(1)
                module = SimpleNamespace(CONTEXT_TEMPLATES_CACHE=None, COV_CACHE={})
                with self.assertRaises(FixtureBoundary):
                    restore_checkpoint(model, module, history, cp, spec,
                                       expected_model_revision='pinned', expected_seen_ids=list(range(1000)))


if __name__ == '__main__':
    unittest.main()
