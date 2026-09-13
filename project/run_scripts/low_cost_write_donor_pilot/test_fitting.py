"""Small CPU tests; these are not GPU/model-level fidelity evidence."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

import torch

from .fitting import FitBoundary, materialize_alpha, select_projector, split_native_source


FIXTURE = '''def apply_AlphaEdit_to_model(model, tok, requests, hparams, cache_template=None, cache_c=None, P=None):
    """Native-shaped synthetic fixture."""
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    context_templates = get_context_templates(model, tok)
    model.add_(2)
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T
    print("done")
    return model, cache_c
'''


class FittingTests(unittest.TestCase):
    def test_split_preserves_native_fit_and_exact_history(self):
        tree, evidence = split_native_source(FIXTURE)
        ns = dict(deepcopy=deepcopy, get_context_templates=lambda *_: [['{}']],
                  compute_ks=lambda model, *args: model.clone())
        exec(compile(tree, '<fixture>', 'exec'), ns)
        hp = type('HP', (), dict(layers=[4]))()
        model, history = torch.ones(2, 2), torch.zeros(1, 2, 2)
        rows = [{'target_new': {'str': 'x'}}]
        ns['_pilot_fit_without_history'](model, None, rows, hp, cache_c=history)
        self.assertTrue(torch.equal(history, torch.zeros_like(history)))
        self.assertTrue(torch.equal(model, torch.full_like(model, 3)))
        ns['_pilot_endpoint_history'](model, None, rows, hp, cache_c=history)
        self.assertTrue(torch.equal(model, torch.full_like(model, 3)))
        self.assertTrue(torch.equal(history, torch.full_like(history, 18)))
        self.assertEqual(rows[0]['target_new']['str'], 'x')
        self.assertEqual(evidence['optimizer_or_solve_equation_changes'], 0)

    def test_final_history_source_drift_rejected(self):
        with self.assertRaises(FitBoundary):
            split_native_source(FIXTURE.replace('cache_c[i,:,:] +=', 'cache_c[i,:,:] ='))

    def test_projector_physical_mapping(self):
        full = torch.stack([torch.eye(2)*i for i in range(5)])
        p4, r4 = select_projector(full, 4)
        p8, r8 = select_projector(full, 8)
        self.assertEqual(r4['source_index'], 0)
        self.assertEqual(r8['source_index'], 4)
        self.assertTrue(torch.equal(p4[0], full[0]))
        self.assertTrue(torch.equal(p8[0], full[4]))
        self.assertNotEqual(p8.data_ptr(), full.data_ptr())

    def test_materialization_endpoints_and_registered_alphas(self):
        e = torch.tensor([1e8, .3, -.25], dtype=torch.float32)
        n = torch.tensor([-1e8, .7, .125], dtype=torch.float32)
        w = torch.zeros_like(e)
        for alpha in (0., .75, .875, 1.):
            materialize_alpha(w, e, n, alpha)
            expected = e if alpha == 0 else n if alpha == 1 else e+alpha*(n-e)
            self.assertTrue(torch.equal(w, expected))
        with self.assertRaises(FitBoundary):
            materialize_alpha(w, e, n, .5)

    def test_original_source_ast_split_if_available(self):
        path = Path('/data/janghj/BLUE/AlphaEdit/AlphaEdit_main.py')
        if not path.exists():
            self.skipTest('pinned source filesystem is unavailable')
        source = path.read_text()
        tree, _ = split_native_source(source)
        orig = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                    and n.name == 'apply_AlphaEdit_to_model')
        extracted = deepcopy(tree.body[0])
        extracted.name = orig.name
        extracted.body.insert(-2, deepcopy(orig.body[-3]))
        self.assertEqual(ast.dump(extracted), ast.dump(orig))


if __name__ == '__main__':
    unittest.main()
