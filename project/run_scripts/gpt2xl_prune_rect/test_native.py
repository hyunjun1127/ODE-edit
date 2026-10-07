"""CPU source/parser/API sentinels, no native fit/model/GPU/Slurm/network."""
import ast
import copy
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass, make_dataclass
from pathlib import Path
from unittest.mock import patch

import torch

from .native import (LAYERS, NATIVE_SPECS, NativeEngine, adopt_native, closure,
                     load_native, native_requests, parse_hparams, source_files)


def fixtures():
    return [dict(case_id=5000 - 2 * i, requested_rewrite=dict(prompt='{} works in',
        subject='subject' + str(i), target_new={'str': ' target', 'id': 'fixture'},
        target_true={'str': 'old'})) for i in range(100)]


@dataclass
class WeightSentinel:
    """Metadata-only selected-weight clone; no actual model tensor allocation."""
    tag: str
    shape: tuple = (6400, 1600)
    dtype: object = torch.float32
    _version: int = 0

    def detach(self): return self
    def cpu(self): return self
    def clone(self): return WeightSentinel(self.tag)
    def data_ptr(self): return id(self)


def mocked_engine(arm, *, key_rows=100, finite_target=True):
    module = types.SimpleNamespace(CONTEXT_TEMPLATES_CACHE=None, COV_CACHE={})
    model = torch.nn.Module()
    selected = {f'transformer.h.{l}.mlp.c_proj.weight': WeightSentinel('cold' + str(l)) for l in LAYERS}
    module.nethook = types.SimpleNamespace(get_parameter=lambda model, name: selected[name])
    module.compute_z = lambda *a, **k: (torch.zeros(1600) if finite_target else torch.full((1600,), float('nan')))
    module.compute_ks = lambda *a, **k: torch.zeros((key_rows, 6400))

    def getter(model, tokenizer):
        if module.CONTEXT_TEMPLATES_CACHE is None:
            module.CONTEXT_TEMPLATES_CACHE = [['{}'], [f'c{i}. {{}}' for i in range(5)]]
        return module.CONTEXT_TEMPLATES_CACHE

    module.get_context_templates = getter

    def repr_at_idxs(model, tok, contexts, idxs, layer, module_template, track='in'):
        value = torch.zeros((len(contexts), 1))
        return (value, value) if track == 'both' else value

    z = types.SimpleNamespace(repr_tools=types.SimpleNamespace(get_reprs_at_idxs=repr_at_idxs))
    hp = make_dataclass('FixtureHP', [('layers', list), ('rewrite_module_tmp', str)])(
        LAYERS, 'transformer.h.{}.mlp.c_proj')
    calls = []

    def execute(model, tok, requests, hp, cache_template=None):
        contexts = module.get_context_templates(model, tok)
        for request in requests:
            module.compute_z(model, tok, request, hp, 17, contexts)
        for layer in hp.layers:
            module.compute_ks(model, tok, requests, hp, layer, contexts)
            module.torch.linalg.solve(None, None)
        return {}

    module.execute_memit = execute

    def public(model, tok, requests, hp, **kwargs):
        calls.append(dict(requests=copy.deepcopy(requests), kwargs=dict(kwargs)))
        module.execute_memit(model, tok, requests, hp, cache_template=kwargs['cache_template'])
        if arm == 'RECT':
            for name in selected:
                module._ODEEDIT_RECT_MASK_OBSERVER(name, None, None, None)
        return model, ({name: value.clone() for name, value in selected.items()}
                       if kwargs['return_orig_weights'] else {})

    if arm == 'PRUNE':
        module.apply_memit_to_model = public
    else:
        module.apply_memit_rect_to_model = public
    bundle = types.SimpleNamespace(arm=arm, module=module, z_module=z)
    engine = NativeEngine(bundle, hp, model, None)
    if arm == 'RECT':
        def mask_count(*unused):
            engine.current['rect_masks'] += 1
            engine.cumulative['rect_masks'] += 1
        module._ODEEDIT_RECT_MASK_OBSERVER = mask_count
    return engine, calls


def fake_finite(original):
    return lambda value, *a, **k: torch.tensor(True) if isinstance(value, WeightSentinel) else original(value, *a, **k)


class RemoveObserver(ast.NodeTransformer):
    def visit_If(self, node):
        if '_ODEEDIT_RECT_MASK_OBSERVER' in ast.dump(node.test):
            return None
        return self.generic_visit(node)


class NativeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='prune-rect-source-')
        cls.adopted = adopt_native(Path(cls.temp.name) / 'sources')
        cls.bundles = {arm: load_native(row, arm) for arm, row in cls.adopted.items()}

    @classmethod
    def tearDownClass(cls):
        for spec in NATIVE_SPECS.values():
            for name in list(sys.modules):
                if name == spec['namespace'] or name.startswith(spec['namespace'] + '.'):
                    del sys.modules[name]
        cls.temp.cleanup()

    def test_actual_parser_blue_false_five_layers_20000(self):
        for bundle in self.bundles.values():
            hp = parse_hparams(bundle)
            self.assertEqual(hp.layers, LAYERS)
            self.assertFalse(hp.blue)
            self.assertEqual(hp.mom2_update_weight, 20000)
            self.assertEqual((hp.v_lr, hp.v_num_grad_steps), (.5, 20))
            self.assertFalse(hasattr(hp, 'L2'))

    def test_actual_execution_closure_disjoint_and_prune_provenance(self):
        for arm, bundle in self.bundles.items():
            self.assertEqual({r['relative'] for r in closure(bundle)},
                {p for p in source_files(arm) if p.endswith('.py') and p != 'experiments/evaluate.py'})
        self.assertEqual(len(self.adopted['PRUNE']['files']), 16)
        self.assertEqual(len(self.adopted['RECT']['files']), 15)
        self.assertNotEqual(self.bundles['PRUNE'].namespace, self.bundles['RECT'].namespace)
        evidence = next(r for r in self.adopted['PRUNE']['files'] if r['relative'] == 'experiments/evaluate.py')
        self.assertEqual(evidence['original']['sha256'], evidence['effective']['sha256'])
        self.assertFalse(evidence['execution'])

    def test_scientific_AST_unchanged_except_observer_only_callback(self):
        for binding in self.adopted.values():
            for row in binding['files']:
                if not row['relative'].endswith('.py'):
                    continue
                original = ast.parse(Path(row['original']['path']).read_text())
                actual = RemoveObserver().visit(ast.parse(Path(row['effective']['path']).read_text()))
                functions = lambda t: {n.name: ast.dump(n, include_attributes=False) for n in t.body
                                       if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                self.assertEqual(functions(original), functions(actual), row['relative'])

    def test_rect_original_mask_and_dense_planning_expressions_preserved(self):
        text = Path(next(r for r in self.adopted['RECT']['files']
                         if r['relative'] == 'memit/memit_rect_main.py')['effective']['path']).read_text()
        for exact in ('delta = torch.abs(upd_matrix / (w + epsilon))',
                      'mask = delta >= threshold', 'w[mask] += upd_matrix[mask].float()',
                      'resid = targets / (len(hparams.layers) - i)',
                      'weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()'):
            self.assertIn(exact, text)
        self.assertEqual(text.count('_ODEEDIT_RECT_MASK_OBSERVER(w_name, delta, threshold, mask)'), 1)

    def test_no_model_gpu_or_P_load_during_source_adoption(self):
        self.assertFalse(torch.cuda.is_initialized())
        for binding in self.adopted.values():
            self.assertFalse(binding['original_package_initializers_executed'])
            self.assertTrue(binding['original_clean'])


class CompositionTests(unittest.TestCase):
    def test_dict_requested_rewrite_schema_exact_order(self):
        records = fixtures()
        previous = copy.deepcopy(records)
        requests = native_requests(records)
        self.assertEqual([r['case_id'] for r in requests], [r['case_id'] for r in records])
        self.assertEqual(requests[0]['target_new'], records[0]['requested_rewrite']['target_new'])
        self.assertEqual(records, previous)
        self.assertEqual(native_requests(requests), requests)

    def test_plain_string_and_missing_rows_block(self):
        records = fixtures()
        records[0]['requested_rewrite']['target_new'] = 'not-native'
        with self.assertRaisesRegex(RuntimeError, 'DICT_TARGET'):
            native_requests(records)
        with self.assertRaisesRegex(RuntimeError, 'BS100'):
            native_requests(fixtures()[:-1])

    def test_prune_B1_selected_W0_capture_only_once_same_native_calls(self):
        engine, calls = mocked_engine('PRUNE')
        original_finite = torch.isfinite
        with patch('torch.linalg.solve', return_value=None), \
             patch('torch.isfinite', side_effect=fake_finite(original_finite)), \
             patch('project.run_scripts.gpt2xl_prune_rect.native.tensor_sha', side_effect=lambda value: value.tag):
            _, first = engine.apply(fixtures(), 1)
            _, second = engine.apply(fixtures(), 2)
        self.assertEqual(calls[0]['kwargs'], dict(copy=False, return_orig_weights=True, cache_template=None))
        self.assertEqual(calls[1]['kwargs'], dict(copy=False, return_orig_weights=False, cache_template=None))
        self.assertEqual(first['delta'], dict(native_z=100, write_keys=5, history_keys=0, solves=5, history_appends=0))
        self.assertEqual(second['cumulative']['native_z'], 200)
        self.assertEqual(len(engine.cold_W0), 5)
        self.assertFalse(engine.prune_applied)
        self.assertEqual(engine.history(), {})

    def test_rect_exact_public_call_noH_noP_no_original_copies(self):
        engine, calls = mocked_engine('RECT')
        with patch('torch.linalg.solve', return_value=None), \
             patch('project.run_scripts.gpt2xl_prune_rect.native.tensor_sha', side_effect=lambda value: value.tag):
            _, receipt = engine.apply(fixtures(), 1)
        self.assertEqual(calls[0]['kwargs'], dict(copy=False, return_orig_weights=False, cache_template=None))
        self.assertEqual(receipt['counts']['rect_masks'], 5)
        self.assertEqual(engine.history(), {})
        self.assertFalse(engine.cold_W0)
        self.assertFalse(engine.finish_batch(1)['prune_applied'])

    def test_context_native_prepare_once_and_rollback_signature(self):
        engine, _ = mocked_engine('RECT')
        engine.prepare_contexts()
        before = engine.native_state_signature()
        saved = engine.snapshot_native_state()
        engine.next_batch, engine.prune_applied = 8, True
        engine.restore_context(None)
        engine.restore_native_state(saved)
        self.assertEqual(engine.native_state_signature(), before)
        with self.assertRaisesRegex(RuntimeError, 'ONCE'):
            engine.prepare_contexts()

    def test_nonfinite_target_and_missing_key_blocks(self):
        engine, _ = mocked_engine('RECT', finite_target=False)
        with patch('project.run_scripts.gpt2xl_prune_rect.native.tensor_sha', side_effect=lambda value: value.tag):
            with self.assertRaisesRegex(RuntimeError, 'Z_SHAPE_FINITE'):
                engine.apply(fixtures(), 1)
        engine, _ = mocked_engine('RECT', key_rows=99)
        with patch('project.run_scripts.gpt2xl_prune_rect.native.tensor_sha', side_effect=lambda value: value.tag):
            with self.assertRaisesRegex(RuntimeError, 'FULL_B100'):
                engine.apply(fixtures(), 1)

    def test_prune_terminal_requires20_and_exactlyonce(self):
        engine, _ = mocked_engine('PRUNE')
        with self.assertRaisesRegex(RuntimeError, 'AFTER20'):
            engine.terminal_prune()
        engine.next_batch, engine.prune_applied = 21, True
        engine.cold_W0 = {str(i): object() for i in range(5)}
        with self.assertRaisesRegex(RuntimeError, 'ONCE'):
            engine.terminal_prune()

    def test_RECT_ties_actualmask_support_can_exceed40pct(self):
        engine, _ = mocked_engine('RECT')
        engine.current = dict(batch=1, rect_masks=0, rect_mask_rows=[])
        scores = torch.zeros((6400, 1600), dtype=torch.float64)
        threshold = torch.tensor(0., dtype=torch.float64)
        engine._rect_mask(next(iter(engine.selected())), scores, threshold, scores >= threshold)
        row = engine.current['rect_mask_rows'][0]
        self.assertEqual(row['support_count'], row['denominator'])
        self.assertEqual(row['support_pct'], 100.)
        self.assertEqual(row['threshold_ties'], row['denominator'])
        self.assertEqual(row['native_comparison'], '>=')
        self.assertEqual(row['kth_index'], int(row['denominator'] * .6))
        scores[0, 0] = float('inf')
        with self.assertRaisesRegex(RuntimeError, 'NONFINITE_OR_SHAPE'):
            engine._rect_mask(next(iter(engine.selected())), scores, threshold, scores >= threshold)


if __name__ == '__main__':
    unittest.main()
