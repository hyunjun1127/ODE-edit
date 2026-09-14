"""CPU writer-binding fixtures; NOT model-level GPU/native parity evidence."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch

from .fitting import FitBoundary, materialize_alpha, split_native_source
from .write_refresh_policy import ExternalTargetSingletonFitter, POLICIES


FIXTURE = '''def apply_AlphaEdit_to_model(model, tok, requests, hparams, cache_template=None, cache_c=None, P=None):
    """Native-shaped CPU fixture, not a replacement model writer."""
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    context_templates = get_context_templates(model, tok)
    for i, layer in enumerate(hparams.layers):
        zs = torch.stack([compute_z(model, tok, request, hparams, layer, context_templates) for request in requests], dim=1)
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cur_zs = get_module_input_output_at_words(model, tok, layer, requests=requests)[1].T
        targets = zs - cur_zs
        resid = targets
        upd_matrix = torch.linalg.solve(P[i,:,:] @ (layer_ks @ layer_ks.T + cache_c[i,:,:]) + hparams.L2*torch.eye(layer_ks.shape[0]), P[i,:,:] @ layer_ks @ resid.T)
        with torch.no_grad():
            model.weight[...] = model.weight + upd_matrix.T
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T
    print("done")
    return model, cache_c
'''


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[.1, .2], [-.1, .3]]), requires_grad=False)
        self.eval()

    def named_parameters(self, *args, **kwargs):
        yield 'layer4.weight', self.weight


def fixture(n=2):
    model = TinyModel()
    tok = types.SimpleNamespace(padding_side='right')
    hp = types.SimpleNamespace(layers=[4], blue=True, L2=1, v_num_grad_steps=25,
                               rewrite_module_tmp='layer{}')
    requests = [dict(case_id=i, prompt='{} fact', subject=f's{i}', target_new={'str': f'x{i}'}) for i in range(n)]
    keys = torch.stack([torch.ones(n), torch.linspace(.1, 1., n)])
    module = types.ModuleType('native_cpu_fixture')
    module.__file__ = '<native_cpu_fixture>'
    module.torch = torch
    module.deepcopy = deepcopy
    module.CONTEXT_TEMPLATES_CACHE = [['{}']]
    module.get_context_templates = lambda *args: [['{}']]
    module.compute_ks = lambda *args: keys.T.clone()
    module.compute_z = lambda model, tok, request, *args: torch.tensor([1.+request['case_id']*.01, -.4])
    module.get_module_input_output_at_words = lambda model, *args, **kwargs: (keys.T, (model.weight @ keys).T)
    source_sha = hashlib.sha256(FIXTURE.encode()).hexdigest()
    with patch('project.run_scripts.low_cost_write_donor_pilot.fitting.inspect.getsource', return_value=FIXTURE):
        fitter = ExternalTargetSingletonFitter(module, expected_source_sha256=source_sha, contexts=[['{}']])
    return fitter, model, tok, hp, requests, torch.zeros(1, 2, 2), torch.eye(2)[None]


class WriterTests(unittest.TestCase):
    def test_native_vs_external_same_targets_and_once_history(self):
        f, model, tok, hp, rows, history, projector = fixture(100)
        before = model.weight.detach().clone()
        module_z = f.module.compute_z
        native = f.fit(model, tok, hp, history, projector, rows, layer=4, capture=True)
        self.assertTrue(torch.equal(history, torch.zeros_like(history)))
        model.weight.copy_(before)
        targets = f.bind_targets(rows, native['captures']['compute_z'])
        external = f.fit_targets(model, tok, hp, history, projector, rows, targets)
        self.assertTrue(torch.equal(native['weight'], external['weight']))
        self.assertEqual(external['receipt']['target_supply'], 100)
        self.assertEqual(external['receipt']['compute_z'], 0)
        self.assertEqual(external['receipt']['solve'], 1)
        self.assertTrue(torch.equal(history, torch.zeros_like(history)))
        self.assertIs(f.module.compute_z, module_z)
        self.assertEqual(rows[0]['target_new']['str'], 'x0')
        final = f.finalize(model, tok, rows, [(4, hp, history, projector)])
        self.assertEqual(final[0]['history_append'], 1)
        self.assertEqual(final[0]['compute_ks'], 1)
        self.assertGreater(float(history.norm()), 0.)
        with self.assertRaises(FitBoundary):
            f.finalize(model, tok, rows, [(4, hp, history, projector)] * 2)

    def test_frozen_reuses_absolute_z_but_reads_fresh_y(self):
        f, model, tok, hp, rows, history, projector = fixture()
        values = [torch.tensor([1., -.4]), torch.tensor([1.01, -.4])]
        bound = f.bind_targets(rows, values)
        entry = model.weight.detach().clone()
        first = f.fit_targets(model, tok, hp, history, projector, rows, bound)
        materialize_alpha(model.weight, entry, first['weight'], .75)
        second = f.fit_targets(model, tok, hp, history, projector, rows, bound)
        self.assertTrue(torch.equal(first['absolute_z'], second['absolute_z']))
        self.assertFalse(torch.equal(first['current_y'], second['current_y']))
        self.assertFalse(torch.equal(first['residual'], second['residual']))
        self.assertTrue(torch.equal(second['residual'], second['absolute_z']-second['current_y']))
        self.assertTrue(torch.equal(history, torch.zeros_like(history)))
        self.assertEqual(second['receipt']['compute_ks'], 1)
        self.assertEqual(second['receipt']['get_module_input_output_at_words'], 1)

    def test_order_target_count_and_bytes_fail_before_write(self):
        f, model, tok, hp, rows, history, projector = fixture()
        bound = f.bind_targets(rows, [torch.ones(2), torch.zeros(2)])
        before = model.weight.clone()
        for altered in (dict(bound, members=bound['members'][::-1]),
                        dict(bound, members=bound['members'][:1])):
            with self.assertRaises(FitBoundary):
                f.fit_targets(model, tok, hp, history, projector, rows, altered)
            self.assertTrue(torch.equal(model.weight, before))
        bound['members'][0]['z'].add_(.1)
        with self.assertRaises(FitBoundary):
            f.fit_targets(model, tok, hp, history, projector, rows, bound)
        self.assertTrue(torch.equal(model.weight, before))

    def test_bound_vectors_are_own_detached_snapshots(self):
        f, model, tok, hp, rows, history, projector = fixture()
        vector = torch.ones(2, requires_grad=True)
        bound = f.bind_targets(rows, [vector, vector])
        with torch.no_grad():
            vector.add_(7)
        self.assertTrue(torch.equal(bound['members'][0]['z'], torch.ones(2)))
        self.assertFalse(bound['members'][0]['z'].requires_grad)
        f.fit_targets(model, tok, hp, history, projector, rows, bound)

    def test_policy_caps_and_fixed_materialization(self):
        self.assertEqual(POLICIES['I4'], ((6, 6, 6, 6), (.75, .75, .75, 1.)))
        self.assertEqual(POLICIES['FROZEN2'], ((24, 0), (.75, 1.)))
        self.assertEqual(sum(POLICIES['I2'][0]), 24)
        self.assertEqual(sum(POLICIES['REFIT4'][0]), 48)

    def test_original_native_ast_math_unchanged(self):
        path = Path('/data/janghj/ODE-edit/local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source/AlphaEdit/AlphaEdit_main.py')
        if not path.is_file():
            self.skipTest('locked native source not local')
        source = path.read_text()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),
                         '79da927aad5ab817fd008c5958768adcd00556989a8efbaa2c4bdc80d8fc842e')
        tree, evidence = split_native_source(source)
        original = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
                        and node.name == 'apply_AlphaEdit_to_model')
        restored = deepcopy(tree.body[0])
        restored.name = original.name
        restored.body.insert(-2, deepcopy(original.body[-3]))
        self.assertEqual(ast.dump(restored), ast.dump(original))
        self.assertEqual(evidence['optimizer_or_solve_equation_changes'], 0)


if __name__ == '__main__':
    unittest.main()
