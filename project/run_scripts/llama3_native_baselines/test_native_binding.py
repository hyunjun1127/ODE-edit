"""CPU import/schema/base-fix mechanics only, not an actual native fit PASS."""
import ast
import copy
import inspect
from pathlib import Path
import unittest

import torch

from .native_binding import (METHODS, SPECS, build_bindings, effective_source,
                             native_function_ast, source_files)
from .native import _TupleNethook, load_native, native_terminal_compression, normalize_requests


class _FakeDecoder(torch.nn.Module):
    """One identity arithmetic hook fixture, not an LM/fit implementation."""
    def __init__(self, tuple_output=False):
        super().__init__()
        self.tuple_output = tuple_output

    def forward(self, x):
        y = x * 1
        return (y, None) if self.tuple_output else y


class _FakeDecoderOwner(torch.nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleDict({
            '8': _FakeDecoder(tuple_output), '31': _FakeDecoder(tuple_output)})
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        hidden = self.model.layers['8'](x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.model.layers['31'](hidden)


class NativeBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bindings = build_bindings()

    def test_exact_native_parsers_selected_layers_and_coefficients(self):
        expected = {
            'MEMIT': ([4, 5, 6, 7, 8], .75, .5, None),
            'PRUNE': ([4, 5, 6, 7, 8], .75, .5, None),
            'RECT': ([4, 5, 6, 7, 8], .75, .5, None),
            'ALPHAEDIT': ([4, 5, 6, 7, 8], .75, .5, 1),
            'ALPHAEDIT_BLUE': ([4, 8], .75, .5, 1),
            'CAKE': ([4, 5, 6, 7, 8], .5, .4, 10),
        }
        self.assertEqual(set(self.bindings), set(METHODS))
        for method, binding in self.bindings.items():
            bundle = load_native(binding)
            hp = bundle.hp
            self.assertEqual((hp.layers, hp.clamp_norm_factor, hp.v_weight_decay,
                              getattr(hp, 'L2', None)), expected[method])
            self.assertEqual((hp.v_lr, hp.v_num_grad_steps, hp.v_loss_layer,
                              hp.kl_factor, hp.mom2_update_weight), (.1, 25, 31, .0625, 15000))
            self.assertEqual(len(bundle.imported_files), 13)
        self.assertFalse(torch.cuda.is_initialized())

    def test_import_plumbing_preserves_native_numerical_function_asts(self):
        for method, spec in SPECS.items():
            for relative in source_files(method):
                source = (spec['root'] / relative).read_bytes()
                original = ast.parse(source)
                effective = ast.parse(effective_source(relative, source, '_fixture_native'))
                before = [ast.dump(node, include_attributes=False) for node in original.body
                          if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
                after = [ast.dump(node, include_attributes=False) for node in effective.body
                         if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
                self.assertEqual(before, after, (method, relative))

    def test_occurrence_conversion_is_nonmutating_and_keeps_dict_native_targets(self):
        records = [dict(case_id=i // 2, requested_rewrite=dict(prompt='{} is',
            subject='subject', target_new={'str': 'target', 'id': 'target-id'})) for i in range(100)]
        before = copy.deepcopy(records)
        converted = normalize_requests(records)
        self.assertEqual(records, before)
        self.assertEqual(len(converted), 100)
        self.assertEqual(converted[0]['target_new'], {'str': 'target', 'id': 'target-id'})
        self.assertEqual(converted[0]['case_id'], converted[1]['case_id'])
        self.assertEqual(normalize_requests(converted), converted)
        with self.assertRaisesRegex(RuntimeError, 'BS100'):
            normalize_requests(records[:-1])

    def test_native_history_only_and_blue_physical_projector_slots(self):
        self.assertEqual(self.bindings['ALPHAEDIT_BLUE']['projector_slots'], [0, 4])
        for method in ('MEMIT', 'PRUNE', 'RECT'):
            self.assertFalse(self.bindings[method]['native_has_history'])
            self.assertNotIn('projector', self.bindings[method])
        for method in ('ALPHAEDIT', 'ALPHAEDIT_BLUE', 'CAKE'):
            self.assertTrue(self.bindings[method]['native_has_history'])
            self.assertEqual(self.bindings[method]['projector']['physical_layers'], [4, 5, 6, 7, 8])

    def test_prune_cold_base_fix_no_double_add_and_input_preserved(self):
        # Tiny operator unit check, not a model/fit/quality experiment.
        cold = torch.diag(torch.tensor([2., 1.], dtype=torch.float32))
        dense = cold + torch.diag(torch.tensor([.4, .2], dtype=torch.float32))
        before = (cold.clone(), dense.clone())
        final, receipt = native_terminal_compression(cold, dense)
        torch.testing.assert_close(final, dense, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(cold, before[0], atol=0, rtol=0)
        torch.testing.assert_close(dense, before[1], atol=0, rtol=0)
        self.assertFalse(receipt['double_add'])
        self.assertEqual(receipt['base'], 'RAM_COLD_W0')
        self.assertEqual(receipt['native_svd_calls'], 2)
        text = inspect.getsource(native_terminal_compression)
        self.assertIn('final = cold_weight.to(dense_weight.device) + compressed', text)
        self.assertNotIn('final = dense_weight + compressed', text)
        self.assertFalse(torch.cuda.is_initialized())

    def test_preserved_context_is_same_declared_native_getter_not_auto_generation(self):
        get_context_asts = set()
        for method, spec in SPECS.items():
            path = spec['root'] / (spec['package'].replace('.', '/') + '/' + spec['main'] + '.py')
            get_context_asts.add(native_function_ast(path, 'get_context_templates'))
            context = self.bindings[method]['context_reuse']
            self.assertEqual(context['contexts']['sha256'],
                '33cec0eef9ec130f26c2f0e17f7c8be39e93c717ebb47eaa5e9f1c88b263524e')
            self.assertFalse(context['cross_native_generator_bitwise_parity_claim'])
        self.assertEqual(len(get_context_asts), 1)

    def test_foreign_trace_shim_same_tensor_subject_edit_retention_and_gradient(self):
        bundle = load_native(self.bindings['CAKE'])
        original_nethook = bundle.z_module.nethook
        proxy = _TupleNethook(original_nethook, ['model.layers.8', 'model.layers.31'])
        model = _FakeDecoderOwner()
        hooks_before = {name: tuple(m._forward_hooks) for name, m in model.named_modules()}
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).requires_grad_()
        delta = torch.ones(4, requires_grad=True)
        observed = []
        def native_tuple_edit(output, layer):
            self.assertIsInstance(output, tuple)
            observed.append((layer, output[0].data_ptr()))
            if layer == 'model.layers.8':
                for i in range(2):
                    output[0][i, 1, :] += delta
            return output
        with proxy.TraceDict(model, layers=['model.layers.31', 'model.layers.8', 'model.layers.8'],
                retain_input=False, retain_output=True, edit_output=native_tuple_edit) as traces:
            result = model(x)
            self.assertIsInstance(result, torch.Tensor)
            self.assertIsInstance(traces['model.layers.31'].output, tuple)
            self.assertEqual(traces['model.layers.31'].output[0].data_ptr(), result.data_ptr())
            expected = x.detach().clone()
            expected[:, 1] += 1
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
            result.sum().backward()
        torch.testing.assert_close(delta.grad, torch.full((4,), 2.), atol=0, rtol=0)
        self.assertEqual(model.calls, 1)
        self.assertEqual(proxy.counters['boxed_decoder_outputs'], 2)
        self.assertEqual(proxy.counters['unboxed_decoder_outputs'], 2)
        self.assertEqual({name: tuple(m._forward_hooks) for name, m in model.named_modules()}, hooks_before)
        self.assertIs(bundle.module.nethook, original_nethook)
        self.assertIs(bundle.z_module.nethook, original_nethook)  # test never patches source module.
        self.assertFalse(torch.cuda.is_initialized())

    def test_foreign_trace_shim_keeps_original_tuple_and_closes_on_exception(self):
        bundle = load_native(self.bindings['ALPHAEDIT_BLUE'])
        proxy = _TupleNethook(bundle.z_module.nethook, ['model.layers.8', 'model.layers.31'])
        model = _FakeDecoderOwner(tuple_output=True)
        with proxy.TraceDict(model, ['model.layers.8', 'model.layers.31'], edit_output=lambda output, layer: output) as traces:
            result = model(torch.zeros(2, 3, 4))
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            self.assertIs(traces['model.layers.31'].output, result)
        self.assertEqual(proxy.counters['boxed_decoder_outputs'], 0)
        self.assertEqual(proxy.counters['native_tuple_outputs'], 2)
        def failing(output, layer):
            raise ValueError('hook fixture error')
        with self.assertRaisesRegex(ValueError, 'hook fixture error'):
            with proxy.TraceDict(model, ['model.layers.8'], edit_output=failing):
                model(torch.zeros(2, 3, 4))
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))
        with self.assertRaisesRegex(RuntimeError, 'DECODER_LAYERS_ONLY'):
            proxy.TraceDict(model, ['model'])
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))


if __name__ == '__main__':
    unittest.main()
