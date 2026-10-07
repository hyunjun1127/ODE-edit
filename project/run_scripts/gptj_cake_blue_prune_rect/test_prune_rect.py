"""CPU-only source/parser and terminal/mask controls; no model or target fit."""
import ast
import copy
import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import native_prune_rect as native


def request_fixture():
    return [dict(case_id=2000 - index * 2, requested_rewrite=dict(
        prompt='{} works in', subject='subject' + str(index),
        target_new={'str': ' target', 'id': 'fixture-new'},
        target_true={'str': 'old', 'id': 'fixture-true'})) for index in range(100)]


class CPUFactor:
    """Mock native factor transfer only; no CUDA allocation or model exists."""
    def __init__(self, value):
        self.value = value

    def to(self, device):
        if device != 'cuda':
            raise AssertionError('Unexpected native transfer')
        return self.value


class NativeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='gptj-prune-rect-source-')
        cls.bindings = native.adopt_native(Path(cls.temp.name) / 'adopted')
        cls.bundles = {arm: native.load_native(binding, arm)
                       for arm, binding in cls.bindings.items()}

    @classmethod
    def tearDownClass(cls):
        for spec in native.NATIVE_SPECS.values():
            for name in list(sys.modules):
                if name == spec['namespace'] or name.startswith(spec['namespace'] + '.'):
                    del sys.modules[name]
        cls.temp.cleanup()

    def test_actual_parser_six_layers_lr_readout_covariance_and_no_history(self):
        for arm, bundle in self.bundles.items():
            hp = native.parse_hparams(bundle)
            self.assertEqual(hp.layers, [3, 4, 5, 6, 7, 8])
            self.assertEqual((hp.v_lr, hp.v_num_grad_steps, hp.v_loss_layer), (.5, 25, 27))
            self.assertEqual(hp.mom2_update_weight, 15000)
            self.assertFalse(hp.blue)
            self.assertEqual(hp.rewrite_module_tmp, 'transformer.h.{}.mlp.fc_out')
            self.assertEqual(len(self.bindings[arm]['files']), 15)
            self.assertEqual({row['relative'] for row in native.closure(bundle)},
                             {relative for relative in native.source_files(arm) if relative.endswith('.py')})
        self.assertNotEqual(self.bundles['PRUNE'].namespace, self.bundles['RECT'].namespace)
        self.assertFalse(torch.cuda.is_initialized())

    def test_import_only_diff_preserves_all_scientific_functions(self):
        for binding in self.bindings.values():
            for row in binding['files']:
                if not row['relative'].endswith('.py'):
                    continue
                original = ast.parse(Path(row['original']['path']).read_text())
                effective = ast.parse(Path(row['effective']['path']).read_text())

                def functions(tree):
                    return {node.name: ast.dump(node, include_attributes=False)
                            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}

                self.assertEqual(functions(original), functions(effective), row['relative'])
        self.assertTrue(all(row['compatibility_diff'] == '' for binding in self.bindings.values()
                            for row in binding['files'] if row['relative'] in (native.HPARAMS, 'globals.yml')))

    def test_gptj_extra_block8_capture_is_preserved_in_native_source(self):
        for bundle in self.bundles.values():
            source = inspect.getsource(bundle.z_module.repr_tools.get_reprs_at_idxs)
            self.assertIn("isinstance(model, GPTJForCausalLM) and module_name == 'transformer.h.8'", source)
            self.assertIn("layer=module_name + '.ln_1'", source)
            self.assertIn('tr.input = tr2.input', source)
            self.assertEqual(source.count('model(**contexts_tok)'), 2)

    def test_rect_dense_provisional_plan_then_restore_before_masked_public_commit(self):
        module = self.bundles['RECT'].module
        planner = inspect.getsource(module.execute_memit)
        public = inspect.getsource(module.apply_memit_rect_to_model)
        self.assertIn('resid = targets / (len(hparams.layers) - i)', planner)
        dense_write = 'weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()'
        restore = 'v[...] = weights_copy[k]'
        self.assertLess(planner.index(dense_write), planner.index(restore))
        self.assertLess(planner.index(restore), planner.index('return deltas'))
        self.assertNotIn('mask =', planner)
        self.assertNotIn('kthvalue', planner)
        self.assertLess(public.index('deltas = execute_memit('), public.index('delta = torch.abs('))
        self.assertIn('w[mask] += upd_matrix[mask].float()', public)

    def test_actual_rect_public_mask_retains_threshold_ties(self):
        module = self.bundles['RECT'].module
        scores = torch.tensor([0., 1., 2., 3., 4., 4., 4., 7., 8., 9.]).reshape(2, 5)
        # Native epsilon rounds exactly into this FP32 denominator of 1.
        weight = torch.ones_like(scores)
        key, val = CPUFactor(torch.eye(2)), CPUFactor(scores.T)
        planned = []

        def fake_plan(model, tok, requests, hp, cache_template=None):
            planned.append(weight.clone())
            return {'weight': (key, val)}

        with patch.object(module, 'execute_memit', fake_plan), \
                patch.object(module, 'nethook', types.SimpleNamespace(get_parameter=lambda model, name: weight)):
            result, old = module.apply_memit_rect_to_model(
                object(), None, request_fixture(), types.SimpleNamespace(blue=False), cache_template=None)
        self.assertEqual(len(planned), 1)
        self.assertEqual(old, {})
        expected = torch.ones_like(scores)
        expected[scores >= 4] += scores[scores >= 4]
        torch.testing.assert_close(weight, expected, atol=0, rtol=0)
        stats = native.native_rect_mask_stats(scores.reshape(-1), 6, torch.tensor(4.))
        self.assertEqual((stats['retained_count'], stats['threshold_tie_count'], stats['masked_count']), (6, 3, 4))
        self.assertEqual(stats['retained_pct'], 60.)
        self.assertFalse(torch.cuda.is_initialized())

    def test_terminal_repair_has_explicit_one_line_diff_not_upstream_equivalence(self):
        terminal = self.bindings['PRUNE']['terminal_source']
        self.assertEqual(terminal['sha256'], native.TERMINAL_SOURCE_SHA)
        diff = terminal['base_fix_exactdiff']
        changed = [line for line in diff.splitlines() if line.startswith(('+', '-'))
                   and not line.startswith(('+++', '---'))]
        self.assertEqual(len(changed), 2)
        self.assertIn('original_weight + upd_matrix[k]', changed[0])
        self.assertIn('weights_copy[k].to(original_weight.device) + upd_matrix[k]', changed[1])
        self.assertEqual(self.bindings['PRUNE']['explicit_repair'], native.BASE_FIX)
        self.assertIsNone(self.bindings['RECT']['explicit_repair'])


class NativeTerminalControls(unittest.TestCase):
    def test_no_compression_identity_does_not_double_dense_delta(self):
        cold = torch.tensor([[4., 0., 0.], [0., 3., 0.]])
        delta = torch.tensor([[.1, 0., 0.], [0., .2, 0.]])
        dense = cold + delta
        before = dense.clone()
        final, stats = native.native_terminal_compression(cold, dense)
        torch.testing.assert_close(final, dense, atol=1e-6, rtol=0)
        torch.testing.assert_close(dense, before, atol=0, rtol=0)
        self.assertEqual(stats['singular_values_compressed'], 0)
        self.assertEqual(stats['native_svd_calls'], 2)
        self.assertFalse(torch.allclose(final, dense + delta))

    def test_zero_delta_is_identity_and_no_nan(self):
        cold = torch.tensor([[4., 0., 0.], [0., 3., 0.]])
        final, stats = native.native_terminal_compression(cold, cold.clone())
        torch.testing.assert_close(final, cold, atol=0, rtol=0)
        self.assertEqual(stats['dense_delta_norm'], 0.)
        self.assertEqual(stats['compressed_delta_norm'], 0.)
        self.assertEqual(stats['singular_values_compressed'], 0)

    def test_native_strict_threshold_and_log_formula(self):
        cold = torch.diag(torch.tensor([4., 3.]))
        dense = cold + torch.diag(torch.tensor([10., 4.]))
        final, stats = native.native_terminal_compression(cold, dense)
        expected = cold + torch.diag(torch.tensor([10., 4.]).where(
            torch.tensor([10., 4.]) <= 4.,
            torch.log(torch.tensor([10., 4.])) - torch.log(torch.tensor(4.)) + 4.))
        torch.testing.assert_close(final, expected, atol=1e-6, rtol=0)
        self.assertEqual(stats['singular_values_compressed'], 1)
        self.assertEqual(stats['max_sigma_cold'], 4.)

    def test_dict_request_conversion_no_space_or_record_mutation(self):
        records = request_fixture()
        records[0]['requested_rewrite']['target_new']['str'] = 'target'
        before = copy.deepcopy(records)
        normalized = native.native_requests(records)
        self.assertEqual(records, before)
        self.assertEqual(normalized[0]['target_new']['str'], 'target')
        self.assertEqual(native.native_requests(normalized), normalized)
        self.assertIs(native.normalize_requests, native.native_requests)
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_BS100'):
            native.native_requests(records[:-1])

    def test_no_checkpoint_or_reconstruction_payload_writes_in_engine(self):
        tree = ast.parse(inspect.getsource(native.PruneRectEngine))
        forbidden = {'save', 'savez', 'savez_compressed', 'save_pretrained', 'dump', 'dumps',
                     'write', 'write_bytes', 'write_text', 'open'}
        called = {node.func.attr for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(called & forbidden)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
