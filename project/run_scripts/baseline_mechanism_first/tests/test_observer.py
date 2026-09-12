from copy import deepcopy
from types import ModuleType, SimpleNamespace
import unittest

import torch

from project.run_scripts.baseline_mechanism_first.fixtures import (
    FixtureBoundary, SingletonSpec, capture_rng, restore_rng,
)
from project.run_scripts.baseline_mechanism_first.native_runner import run_native_batch
from project.run_scripts.baseline_mechanism_first.observer import observe_native
from .test_fixtures import toy


# Synthetic original-style native call: the observer sees the actual dense
# RHS/solve local tensors, not an independently factorized writer surrogate.
SOURCE = '''
import torch
from copy import deepcopy
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
def compute_z(model, tok, request, hp, layer, contexts):
    return torch.rand(2) + request['case_id'] / 100
def compute_ks(model, tok, requests, hp, layer, contexts):
    return torch.tensor([[1., 2., 0.], [2., -1., 1.]])[:len(requests)]
def upd_matrix_match_shape(value, shape):
    if value.shape == shape: return value
    if value.T.shape == shape: return value.T
    raise ValueError('shape')
def apply_AlphaEdit_to_model(model, tok, requests, hp, cache_template=None, cache_c=None, P=None):
    global CONTEXT_TEMPLATES_CACHE
    requests = deepcopy(requests)
    for row in requests: row['target_new']['str'] = ' ' + row['target_new']['str']
    if CONTEXT_TEMPLATES_CACHE is None:
        torch.rand(3)
        CONTEXT_TEMPLATES_CACHE = [['{}'], ['generated {}']]
    layer = hp.layers[0]
    zs = torch.stack([compute_z(model,tok,r,hp,layer,CONTEXT_TEMPLATES_CACHE) for r in requests],dim=1)
    layer_ks = compute_ks(model,tok,requests,hp,layer,CONTEXT_TEMPLATES_CACHE).T
    resid = zs - model.layers[layer].weight @ layer_ks
    upd_matrix = torch.linalg.solve(P[0] @ (layer_ks @ layer_ks.T + cache_c[0]) + hp.L2*torch.eye(3), P[0] @ layer_ks @ resid.T)
    upd_matrix = upd_matrix_match_shape(upd_matrix, model.layers[layer].weight.shape)
    with torch.no_grad(): model.layers[layer].weight[...] = model.layers[layer].weight + upd_matrix
    layer_ks = compute_ks(model,tok,requests,hp,layer,CONTEXT_TEMPLATES_CACHE).T
    cache_c[0] += layer_ks.cpu() @ layer_ks.cpu().T
    return model, cache_c
'''


def module():
    native = ModuleType('toy_native')
    exec(SOURCE, native.__dict__)
    return native


class ObserverTests(unittest.TestCase):
    def arguments(self):
        return dict(tok=SimpleNamespace(padding_side='right'),
                    hparams=SimpleNamespace(layers=[4], L2=1, blue=True, rewrite_module_tmp='layers.{}'),
                    projector=torch.eye(3)[None],
                    requests=[dict(case_id=i, target_new=dict(str='test')) for i in (1, 2)],
                    spec=SingletonSpec(4, 'layers.{}'))

    def test_wrapped_unwrapped_exact_write_rng_input_and_dense_capture(self):
        base = toy(); original = module(); wrapped = module(); kwargs = self.arguments()
        requests_before = deepcopy(kwargs['requests']); entry = capture_rng()
        direct_state = torch.zeros(1, 3, 3)
        direct = run_native_batch(deepcopy(base), module=original, history=direct_state, **kwargs)
        direct_rng = capture_rng()
        restore_rng(entry)
        observed_state = torch.zeros(1, 3, 3)
        global_torch_solve = torch.linalg.solve
        observed = run_native_batch(deepcopy(base), module=wrapped, history=observed_state,
                                    observer=observe_native, **kwargs)
        self.assertTrue(torch.equal(direct['weight'], observed['weight']))
        self.assertTrue(torch.equal(direct['history'], observed['history']))
        self.assertEqual(capture_rng(), direct_rng)
        self.assertEqual(kwargs['requests'], requests_before)
        self.assertIs(torch.linalg.solve, global_torch_solve)
        self.assertIs(wrapped.torch, torch)
        record = observed['observer']
        self.assertTrue(record['wrappers_restored'])
        self.assertTrue(record['history_append_exact'])
        self.assertEqual((record['compute_z'], record['solve_calls'], record['key_calls']), (2, 1, 2))
        self.assertEqual(record['compute_z_final_training_nll'], 'NOT_OBSERVED')
        solve = record['_tensors']['solves'][0]
        expected_r = torch.stack(record['_tensors']['targets'], dim=1) - base.layers[4].weight @ solve['K']
        self.assertTrue(torch.equal(expected_r, solve['R']))
        self.assertTrue(torch.equal(solve['solution'].T, record['_tensors']['physical_updates'][0]))
        self.assertEqual(record['history_verification_extra_cpu_gram_calls'], 1)

    def test_wrappers_restored_after_original_error(self):
        model, native, kwargs = toy(), module(), self.arguments()
        def failing(*args, **kwargs):
            raise ValueError('original compute_z failure')
        native.compute_z = failing
        originals = {k: getattr(native, k) for k in ('torch', 'compute_z', 'compute_ks', 'upd_matrix_match_shape')}
        with self.assertRaisesRegex(ValueError, 'original compute_z failure'):
            with observe_native(native, kwargs['hparams'], {}, torch.zeros(1, 3, 3), kwargs['requests'], model) as receipt:
                native.apply_AlphaEdit_to_model(model, kwargs['tok'], kwargs['requests'], kwargs['hparams'],
                                               cache_c=torch.zeros(1, 3, 3), P=kwargs['projector'])
        self.assertEqual(receipt['status'], 'OBSERVATION_FAILED')
        self.assertTrue(receipt['wrappers_restored'])
        for name, original in originals.items(): self.assertIs(getattr(native, name), original)
        self.assertFalse(hasattr(native, '_e01_observer_active'))

    def test_missing_native_history_append_detected(self):
        model, native, kwargs = toy(), module(), self.arguments()
        state = torch.zeros(1, 3, 3)
        with self.assertRaisesRegex(FixtureBoundary, 'OBSERVER_HISTORY_APPEND_MISMATCH'):
            with observe_native(native, kwargs['hparams'], {}, state, kwargs['requests'], model):
                native.apply_AlphaEdit_to_model(model, kwargs['tok'], kwargs['requests'], kwargs['hparams'],
                                               cache_c=state, P=kwargs['projector'])
                state.zero_()
        self.assertIs(native.torch, torch)

    def test_empty_batch_noop(self):
        kwargs = self.arguments(); kwargs['requests'] = []
        result = run_native_batch(toy(), module=module(), history=torch.zeros(1, 3, 3),
                                  observer=observe_native, **kwargs)
        self.assertEqual(result['observer']['compute_z'], 0)
        self.assertEqual(result['observer']['history_append_passes'], 0)
        self.assertTrue(result['observer']['history_append_exact'])


if __name__ == '__main__':
    unittest.main()
