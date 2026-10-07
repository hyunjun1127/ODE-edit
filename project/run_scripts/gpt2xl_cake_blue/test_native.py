"""CPU-only native source/parser and mocked public-call regressions.

No model loading, target fitting, GPU, scheduler, network or numerical toy fit.
The public-call fixture replaces computations with counters/shape sentinels.
"""
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .native import (NATIVE_SPECS, NativeEngine, adopt_native, closure, digest,
                     effective_source, load_native, native_requests, parse_hparams,
                     source_files)


def request_fixture():
    return [dict(case_id=2000 - i * 2, requested_rewrite=dict(
        prompt='{} works in', subject='subject' + str(i),
        target_new={'str': ' target', 'id': 'fixture-new'},
        target_true={'str': 'old', 'id': 'fixture-true'})) for i in range(100)]


def mock_bundle(arm, *, key_rows=100, target_finite=True):
    """No original mathematics executed; validate production API composition."""
    module = types.SimpleNamespace(CONTEXT_TEMPLATES_CACHE=None, COV_CACHE={})
    model = torch.nn.Module()

    def repr_at_idxs(model, tok, contexts, idxs, layer, module_template, track='in'):
        tensor = torch.zeros((len(contexts), 1))
        return (tensor, tensor) if track == 'both' else tensor

    def compute_z(*args, **kwargs):
        return torch.zeros(1600) if target_finite else torch.full((1600,), float('nan'))

    module.compute_z = compute_z
    module.compute_ks = lambda *args, **kwargs: torch.zeros((key_rows, 6400))
    module.get_context_templates = lambda *args: _prepare_context(module)
    z_module = types.SimpleNamespace(repr_tools=types.SimpleNamespace(get_reprs_at_idxs=repr_at_idxs))
    hp = types.SimpleNamespace(layers=NATIVE_SPECS[arm]['layers'])
    # asdict in the production receipt requires an actual dataclass.
    from dataclasses import make_dataclass
    hp = make_dataclass('FixtureHParams', [('layers', list)])(hp.layers)
    captured = []

    def public(model, tok, requests, hp, **kwargs):
        captured.append(dict(requests=copy.deepcopy(requests), kwargs=dict(kwargs)))
        require_dict = all(isinstance(r['target_new'], dict) for r in requests)
        if not require_dict:
            raise RuntimeError('FIXTURE_DICT_SCHEMA')
        contexts = module.get_context_templates(model, tok)
        for layer in hp.layers if arm == 'ALPHAEDIT_BLUE' else [hp.layers[-1]]:
            for request in requests:
                module.compute_z(model, tok, request, hp, layer, contexts)
        for layer in hp.layers:
            module.compute_ks(model, tok, requests, hp, layer, contexts)
            module.torch.linalg.solve(None, None)
        for index, layer in enumerate(hp.layers):
            module.compute_ks(model, tok, requests, hp, layer, contexts)
            kwargs['cache_c'][index, :, :] += 1
        return model, kwargs['cache_c']

    if arm == 'CAKE':
        module.apply_Cake_to_model = public
    else:
        module.apply_AlphaEdit_to_model = public
    bundle = types.SimpleNamespace(arm=arm, module=module, z_module=z_module)
    # Small RAM sentinels are schema-only mocks, never numerical fitting.
    H = torch.zeros((len(hp.layers), 1, 1))
    P = object()
    return NativeEngine(bundle, hp, model, None, P, H), captured


def _prepare_context(module):
    if module.CONTEXT_TEMPLATES_CACHE is None:
        module.CONTEXT_TEMPLATES_CACHE = [['{}'], [f'c{i}. {{}}' for i in range(5)]]
    return module.CONTEXT_TEMPLATES_CACHE


class NativeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='cake-blue-source-')
        cls.bindings = adopt_native(Path(cls.temp.name) / 'adopted')
        cls.bundles = {arm: load_native(binding, arm) for arm, binding in cls.bindings.items()}

    @classmethod
    def tearDownClass(cls):
        for spec in NATIVE_SPECS.values():
            for name in list(sys.modules):
                if name == spec['namespace'] or name.startswith(spec['namespace'] + '.'):
                    del sys.modules[name]
        cls.temp.cleanup()

    def test_actual_native_parser_fields(self):
        cake = parse_hparams(self.bundles['CAKE'])
        blue = parse_hparams(self.bundles['ALPHAEDIT_BLUE'])
        self.assertEqual(cake.L2, 40)
        self.assertEqual(cake.layers, [13, 14, 15, 16, 17])
        self.assertEqual(blue.layers, [13, 17])
        self.assertTrue(blue.blue)
        self.assertEqual(blue.L2, 80)
        self.assertEqual(blue.nullspace_threshold, .02)

    def test_exact_imported_closure_and_disjoint_namespaces(self):
        self.assertNotEqual(self.bundles['CAKE'].namespace, self.bundles['ALPHAEDIT_BLUE'].namespace)
        for arm, bundle in self.bundles.items():
            self.assertEqual(len(self.bindings[arm]['files']), 15)
            self.assertEqual({r['relative'] for r in closure(bundle)},
                             {p for p in source_files(arm) if p.endswith('.py')})

    def test_scientific_functions_unchanged_after_import_repair(self):
        import ast
        for arm, binding in self.bindings.items():
            for row in binding['files']:
                if row['relative'].endswith('.py'):
                    original = ast.parse(Path(row['original']['path']).read_text())
                    actual = ast.parse(Path(row['effective']['path']).read_text())
                    funcs = lambda tree: {n.name: ast.dump(n, include_attributes=False)
                                          for n in tree.body
                                          if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                    self.assertEqual(funcs(original), funcs(actual), row['relative'])

    def test_hparams_and_generator_bytes_are_not_scientifically_changed(self):
        for binding in self.bindings.values():
            for row in binding['files']:
                if row['relative'].endswith('.json') or row['relative'] == 'globals.yml':
                    self.assertEqual(row['original']['sha256'], row['effective']['sha256'])
        cake = next(r for r in self.bindings['CAKE']['files'] if r['relative'] == 'util/generate.py')
        self.assertIn('attention_mask=attention_mask[:, :current_pos]',
                      Path(cake['effective']['path']).read_text())

    def test_adopt_create_once(self):
        with self.assertRaisesRegex(RuntimeError, 'CREATE_ONCE'):
            adopt_native(Path(self.temp.name) / 'adopted')

    def test_no_native_recompute_or_model_load_from_adoption(self):
        for binding in self.bindings.values():
            self.assertTrue(binding['original_clean'])
            self.assertFalse(binding['original_package_initializers_executed'])
        self.assertFalse(torch.cuda.is_initialized())


class NativeCompositionTests(unittest.TestCase):
    def test_dict_request_schema_preserves_payload_and_order(self):
        records = request_fixture()
        before = copy.deepcopy(records)
        actual = native_requests(records)
        self.assertEqual([r['case_id'] for r in actual], [r['case_id'] for r in records])
        self.assertEqual(actual[0]['target_new'], records[0]['requested_rewrite']['target_new'])
        self.assertEqual(actual[0]['target_true'], records[0]['requested_rewrite']['target_true'])
        self.assertEqual(records, before)
        self.assertEqual(native_requests(actual), actual)

    def test_plain_string_and_missing_requests_rejected(self):
        records = request_fixture()
        records[0]['requested_rewrite']['target_new'] = 'wrong adapter'
        with self.assertRaisesRegex(RuntimeError, 'DICT_TARGET'):
            native_requests(records)
        with self.assertRaisesRegex(RuntimeError, 'BS100'):
            native_requests(request_fixture()[:-1])

    def test_blue_exact_public_kwargs_two_layers_native_history(self):
        engine, captured = mock_bundle('ALPHAEDIT_BLUE')
        with patch('torch.linalg.solve', return_value=None):
            model, result = engine.apply(request_fixture(), 1)
        self.assertIs(model, engine.model)
        self.assertEqual(set(captured[0]['kwargs']), {'cache_template', 'cache_c', 'P'})
        self.assertIs(captured[0]['kwargs']['cache_c'], engine.H)
        self.assertEqual(result['delta'], dict(native_z=200, write_keys=2, history_keys=2,
                                              solves=2, history_appends=2))
        self.assertTrue(result['native_has_history'])
        self.assertEqual(len(captured), 1)

    def test_cake_native_H_return_and_exactly_once_final_append(self):
        engine, captured = mock_bundle('CAKE')
        before_pointer = engine.H.data_ptr()
        with patch('torch.linalg.solve', return_value=None):
            _, first = engine.apply(request_fixture(), 1)
            _, second = engine.apply(request_fixture(), 2)
        self.assertIs(captured[0]['kwargs']['cache_c'], engine.H)
        self.assertIs(captured[1]['kwargs']['cache_c'], engine.H)
        self.assertEqual(engine.H.data_ptr(), before_pointer)
        self.assertTrue(torch.equal(engine.H, torch.full_like(engine.H, 2)))
        self.assertEqual(first['delta'], dict(native_z=100, write_keys=5, history_keys=5,
                                             solves=5, history_appends=5))
        self.assertEqual(second['cumulative']['history_appends'], 10)
        self.assertEqual(set(captured[0]['kwargs']), {'cache_template', 'cache_c', 'P'})

    def test_batch_sequence_and_batch21_blocked(self):
        engine, _ = mock_bundle('ALPHAEDIT_BLUE')
        with self.assertRaisesRegex(RuntimeError, 'BATCH_SEQUENCE'):
            engine.apply(request_fixture(), 2)
        engine.next_batch = 21
        with self.assertRaisesRegex(RuntimeError, 'NO21'):
            engine.apply(request_fixture(), 21)

    def test_nonfinite_native_target_and_silent_key_drop_blocked(self):
        engine, _ = mock_bundle('ALPHAEDIT_BLUE', target_finite=False)
        with patch('torch.linalg.solve', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'TARGET_NONFINITE'):
                engine.apply(request_fixture(), 1)
        engine, _ = mock_bundle('ALPHAEDIT_BLUE', key_rows=99)
        with patch('torch.linalg.solve', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'CARDINALITY'):
                engine.apply(request_fixture(), 1)

    def test_native_context_prepare_once_and_restore(self):
        engine, _ = mock_bundle('ALPHAEDIT_BLUE')
        contexts = engine.prepare_contexts()
        snapshot = engine.context_snapshot()
        self.assertEqual(contexts, snapshot)
        self.assertEqual(engine.counts['native_z'], 0)
        self.assertEqual(engine.counts['public_applies'], 0)
        with self.assertRaisesRegex(RuntimeError, 'PREPARE_ONCE'):
            engine.prepare_contexts()
        engine.restore_context(None)
        self.assertIsNone(engine.contexts())
        engine.restore_context(snapshot)
        self.assertEqual(engine.contexts(), contexts)

    def test_Adam_scalar_proxy_preserves_original_call(self):
        from .native import _OptimCalls
        engine, _ = mock_bundle('ALPHAEDIT_BLUE')
        engine.current = dict(batch=1, **{k: 0 for k in engine.cumulative},
                              seconds={k: 0. for k in engine.cumulative})
        seen = []
        original = types.SimpleNamespace(Adam=lambda *a, **k: types.SimpleNamespace(
            step=lambda *a, **k: seen.append((a, k)) or 'actual-return'))
        proxy = _OptimCalls(original, engine)
        optimizer = proxy.Adam('delta', lr=.5)
        self.assertEqual(optimizer.step('closure'), 'actual-return')
        self.assertEqual(seen, [(('closure',), {})])
        self.assertEqual(engine.current['fit_updates'], 1)


if __name__ == '__main__':
    unittest.main()
