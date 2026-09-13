from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import torch

from project.run_scripts.baseline_mechanism_first.fixtures import FixtureBoundary, FixtureTransaction, SingletonSpec
from project.run_scripts.baseline_mechanism_first.native_runner import run_native_batch
from .test_fixtures import toy


class NativeRunnerTests(unittest.TestCase):
    def setup_native(self):
        model = toy(); spec = SingletonSpec(4, 'layers.{}')
        hp = SimpleNamespace(layers=[4], L2=1, blue=True, rewrite_module_tmp='layers.{}')
        tok = SimpleNamespace(padding_side='right')
        history = torch.zeros(1, 3, 3); p = torch.eye(3)[None]
        key = torch.tensor([[1., 2.], [2., -1.], [0., 1.]])
        residual = torch.tensor([[1., -2.], [3., 1.]])
        calls = []
        module = SimpleNamespace(CONTEXT_TEMPLATES_CACHE=None, COV_CACHE={})
        def apply(model, tok, requests, hp, *, cache_template, cache_c, P):
            calls.append('native_dense')
            if module.CONTEXT_TEMPLATES_CACHE is None:
                module.CONTEXT_TEMPLATES_CACHE = [['{}'], ['generated {}']]
                torch.rand(1)
            update = torch.linalg.solve(P[0] @ (key @ key.T + cache_c[0]) + torch.eye(3),
                                        P[0] @ key @ residual.T).T
            with torch.no_grad():
                model.layers[4].weight.add_(update)
                cache_c[0].add_(key @ key.T)
            return model, cache_c
        module.apply_AlphaEdit_to_model = apply
        return model, tok, module, hp, history, p, spec, calls

    def test_dense_native_and_observer_order_then_exact_rollback(self):
        model, tok, module, hp, history, p, spec, calls = self.setup_native()
        before = model.layers[4].weight.detach().clone()
        stages = []
        @contextmanager
        def observer(*args):
            calls.append('observer_enter')
            yield {'source_observer': True}
            calls.append('observer_exit')
        with FixtureTransaction(model, module, history) as tx:
            result = run_native_batch(model, tok, module, hp, history, p,
                                      [{'case_id': 7}], spec, observer=observer,
                                      on_stage=lambda label, receipt: stages.append(label))
            self.assertGreater(result['receipt']['actual_delta_norm'], 0)
            self.assertTrue(torch.equal(result['weight'], model.layers[4].weight))
            self.assertEqual(result['receipt']['dense_native_entrypoint_called'], 1)
            self.assertIsNotNone(result['contexts'])
        self.assertTrue(torch.equal(before, model.layers[4].weight))
        self.assertTrue(tx.receipt['pointer_bytes_exact'])
        self.assertEqual(calls, ['observer_enter', 'native_dense', 'observer_exit'])
        self.assertEqual(stages, ['NATIVE_ENTRY_BOUND', 'NATIVE_ENDPOINT_CAPTURED'])

    def test_b0_noop(self):
        model, tok, module, hp, history, p, spec, calls = self.setup_native()
        result = run_native_batch(model, tok, module, hp, history, p, [], spec)
        self.assertEqual(result['receipt']['status'], 'EMPTY_BATCH_NOOP')
        self.assertEqual(result['receipt']['actual_delta_norm'], 0)
        self.assertEqual(calls, [])

    def test_nonselected_mutation_rejected_and_restored(self):
        model, tok, module, hp, history, p, spec, calls = self.setup_native()
        old = module.apply_AlphaEdit_to_model
        def bad(*args, **kwargs):
            result = old(*args, **kwargs)
            with torch.no_grad(): model.layers[1].weight.add_(1)
            return result
        module.apply_AlphaEdit_to_model = bad
        with FixtureTransaction(model, module, history) as tx:
            with self.assertRaisesRegex(FixtureBoundary, 'NONSELECTED_PARAMETER_MUTATION'):
                run_native_batch(model, tok, module, hp, history, p, [{'case_id': 1}], spec)
        self.assertTrue(tx.receipt['pointer_bytes_exact'])

    def test_wrong_contract_rejected_before_native(self):
        for bad in ('padding', 'layer', 'ridge'):
            with self.subTest(bad=bad):
                model, tok, module, hp, history, p, spec, calls = self.setup_native()
                if bad == 'padding': tok.padding_side = 'left'
                if bad == 'layer': hp.layers = [8]
                if bad == 'ridge': hp.L2 = 10
                with self.assertRaises(FixtureBoundary):
                    run_native_batch(model, tok, module, hp, history, p, [{'case_id': 1}], spec)
                self.assertFalse(calls)


if __name__ == '__main__':
    unittest.main()
