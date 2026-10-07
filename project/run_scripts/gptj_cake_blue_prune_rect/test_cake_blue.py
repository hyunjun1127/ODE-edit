"""Narrow CPU source/parser/API composition tests; never a scientific pilot.

No pretrained model, numerical fitting, GPU, scheduler, network or durable
P/H tensors. Small shape sentinels test routing, counters and hook behavior.
Actual GPT-J fitting/materialization guards run only in the sealed main B1.
"""
import ast
import copy
import random
import sys
import tempfile
import types
import unittest
from dataclasses import make_dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import native_cake_blue as native


def requests():
    return [dict(case_id=2000 - 2 * i, requested_rewrite=dict(
        prompt='{} works in', subject='subject' + str(i),
        target_new={'str': ' target', 'id': 'fixture-new'},
        target_true={'str': 'old', 'id': 'fixture-true'})) for i in range(100)]


class APISentinel(torch.nn.Module):
    """No model calculation, just an observable API forward completion."""
    def forward(self, *unused, **kwargs):
        return None


def mock_engine(arm, *, rows=100, finite=True, bad_history=False, extra_append=False):
    module = types.SimpleNamespace(CONTEXT_TEMPLATES_CACHE=None, COV_CACHE={})
    model = APISentinel()

    class Trace:
        def __init__(self, module, layer=None, **kwargs):
            self.layer = layer

    def representations(model, tok, contexts, idxs, layer, module_template, track='in'):
        model()
        value = torch.zeros((len(contexts), 4))
        return (value, value) if track == 'both' else value

    # Return-local telemetry is observed exactly as it is in the native source.
    def compute_z(model, tok, request, hparams, layer, context_templates):
        it = 0
        loss = nll_loss = kl_loss = weight_decay = torch.tensor(0.)
        model()
        return torch.zeros(4) if finite else torch.full((4,), float('nan'))

    module.compute_z = compute_z
    module.compute_ks = lambda *a, **k: torch.zeros((rows, 8))
    module.nethook = types.SimpleNamespace(Trace=Trace)
    module.upd_matrix_match_shape = lambda matrix, shape: matrix.T

    def generator(model, tok, prompts, n_gen_per_prompt, max_out_len):
        random.random()
        np.random.random()
        torch.rand(1)
        return [f'generated-{i}' for i in range(5)]

    def context_getter(model, tok):
        if module.CONTEXT_TEMPLATES_CACHE is None:
            module.CONTEXT_TEMPLATES_CACHE = [['{}'], [f + '. {}' for f in
                module.generate_fast(model, tok, ['The', 'Therefore', 'Because', 'I', 'You'],
                                     n_gen_per_prompt=1, max_out_len=10)]]
        return module.CONTEXT_TEMPLATES_CACHE

    module.generate_fast = generator
    module.get_context_templates = context_getter
    z_module = types.SimpleNamespace(repr_tools=types.SimpleNamespace(
        get_reprs_at_idxs=representations, GPTJForCausalLM=type('NotSentinel', (), {})))
    hp = make_dataclass('FixtureHParams', [('layers', list)])(native.NATIVE_SPECS[arm]['layers'])
    captured = []

    def public(model, tok, requests, hp, **kwargs):
        captured.append(dict(requests=copy.deepcopy(requests), kwargs=dict(kwargs)))
        contexts = module.get_context_templates(model, tok)
        for layer in hp.layers if arm == 'ALPHAEDIT_BLUE' else [hp.layers[-1]]:
            for request in requests:
                module.compute_z(model, tok, request, hp, layer, contexts)
        for layer in hp.layers:
            module.compute_ks(model, tok, requests, hp, layer, contexts)
            delta = module.torch.linalg.solve(None, None)
            module.upd_matrix_match_shape(delta, (4, 8))
        for index, layer in enumerate(hp.layers):
            module.compute_ks(model, tok, requests, hp, layer, contexts)
            kwargs['cache_c'][index, :, :] += 1
        if extra_append:
            kwargs['cache_c'][0, :, :] += 1
        return model, kwargs['cache_c'].clone() if bad_history else kwargs['cache_c']

    if arm == 'CAKE':
        module.apply_Cake_to_model = public
    else:
        module.apply_AlphaEdit_to_model = public
    bundle = types.SimpleNamespace(arm=arm, module=module, z_module=z_module,
                                  root=Path(native.NATIVE_SPECS[arm]['root']))
    H = torch.zeros((len(hp.layers), 8, 8))
    full = torch.stack([torch.full((8, 8), float(i)) for i in range(6)])
    P = native.ProjectorSlots(full, native.NATIVE_SPECS[arm]['P_slots'])
    engine = native.NativeEngine(bundle, hp, model, None, P, H, {'seed': 20261002})
    return engine, captured


class NativeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='gptj-cake-blue-source-')
        cls.bindings = native.adopt_native(Path(cls.temp.name) / 'adopted')
        cls.bundles = {arm: native.load_native(binding, arm) for arm, binding in cls.bindings.items()}

    @classmethod
    def tearDownClass(cls):
        for spec in native.NATIVE_SPECS.values():
            for name in list(sys.modules):
                if name == spec['namespace'] or name.startswith(spec['namespace'] + '.'):
                    del sys.modules[name]
        cls.temp.cleanup()

    def test_native_gptj_parser_unchanged_defaults(self):
        cake = native.parse_hparams(self.bundles['CAKE'])
        blue = native.parse_hparams(self.bundles['ALPHAEDIT_BLUE'])
        self.assertEqual(cake.layers, [3, 4, 5, 6, 7, 8])
        self.assertEqual(cake.L2, 30)
        self.assertEqual(blue.layers, [3, 8])
        self.assertEqual(blue.L2, 95)
        self.assertTrue(blue.blue)
        self.assertEqual((cake.v_num_grad_steps, blue.v_num_grad_steps), (25, 25))
        self.assertEqual((cake.v_loss_layer, blue.v_loss_layer), (27, 27))

    def test_exact_small_closure_disjoint_namespaces_no_gpu(self):
        self.assertNotEqual(self.bundles['CAKE'].namespace, self.bundles['ALPHAEDIT_BLUE'].namespace)
        for arm, bundle in self.bundles.items():
            self.assertEqual(len(self.bindings[arm]['files']), 15)
            self.assertEqual({row['relative'] for row in native.closure(bundle)},
                             {p for p in native.source_files(arm) if p.endswith('.py')})
            self.assertFalse(self.bindings[arm]['original_package_initializers_executed'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_scientific_function_and_class_asts_are_original(self):
        for binding in self.bindings.values():
            for row in binding['files']:
                if row['relative'].endswith('.py'):
                    funcs = lambda text: {node.name: ast.dump(node, include_attributes=False)
                        for node in ast.parse(text).body
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
                    self.assertEqual(funcs(Path(row['original']['path']).read_text()),
                                     funcs(Path(row['effective']['path']).read_text()), row['relative'])
                else:
                    self.assertEqual(row['original']['sha256'], row['effective']['sha256'])

    def test_native_gptj_repr_branches_preserved(self):
        cake = Path(self.bundles['CAKE'].root / 'rome/repr_tools.py').read_text()
        blue = Path(self.bundles['ALPHAEDIT_BLUE'].root / 'rome/repr_tools.py').read_text()
        self.assertIn('if layer_num in [3, 4, 5, 6, 7, 8]:', cake)
        self.assertIn("module_name == 'transformer.h.8'", blue)
        for source in (cake, blue):
            self.assertIn("layer=module_name + '.ln_1'", source)
            self.assertIn('tr.input = tr2.input', source)

    def test_sha_mismatch_blocked_before_import(self):
        binding = copy.deepcopy(self.bindings['CAKE'])
        binding['files'][0]['effective']['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'MEMBER_IDENTITY'):
            native.load_native(binding, 'CAKE')

    def test_adoption_create_once(self):
        with self.assertRaisesRegex(RuntimeError, 'CREATE_ONCE'):
            native.adopt_native(Path(self.temp.name) / 'adopted')

    def test_native_generator_head_and_context_reduction_preserved(self):
        for arm, bundle in self.bundles.items():
            generator = Path(bundle.root / 'util/generate.py').read_text()
            native_mask = ('attention_mask=attention_mask[:, :current_pos]' if arm == 'CAKE'
                           else 'attention_mask=attention_mask[:, cur_context]')
            self.assertIn(native_mask, generator)
            compute = Path(bundle.root / f'{native.NATIVE_SPECS[arm]["package"]}/compute_z.py').read_text()
            self.assertIn('lm_b.to(full_repr.device)', compute)
            self.assertIn('ln_f(full_repr) @ lm_w.to(full_repr.device)', compute)
            ks = Path(bundle.root / f'{native.NATIVE_SPECS[arm]["package"]}/compute_ks.py').read_text()
            self.assertIn('torch.stack(tmp, 0).mean(0)', ks)

    def test_original_repr_extra_forward_observed_without_trace_mutation(self):
        """Exercise the existing hook branch only, not a target fit or toy model."""
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ln_1 = torch.nn.Identity()

            def forward(self, value):
                return (self.ln_1(value),)

        class HookSentinel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.transformer = torch.nn.Module()
                self.transformer.h = torch.nn.ModuleList([Block() for _ in range(9)])

            def forward(self, input_ids, attention_mask):
                value = torch.zeros((*input_ids.shape, 4))
                for block in self.transformer.h:
                    value = block(value)[0]
                return value

        class Batch(dict):
            def to(self, device):
                return self

        def tokenizer(contexts, **kwargs):
            return Batch(input_ids=torch.zeros((len(contexts), 2), dtype=torch.long),
                         attention_mask=torch.ones((len(contexts), 2), dtype=torch.long))

        for arm, layer, extra in [('CAKE', 3, 1), ('CAKE', 8, 1),
                                  ('ALPHAEDIT_BLUE', 3, 0), ('ALPHAEDIT_BLUE', 8, 1)]:
            bundle = self.bundles[arm]
            original_tools = bundle.z_module.repr_tools
            tools = types.SimpleNamespace(get_reprs_at_idxs=original_tools.get_reprs_at_idxs,
                                          GPTJForCausalLM=HookSentinel)
            engine = native.NativeEngine.__new__(native.NativeEngine)
            engine.arm, engine.model, engine.module = arm, HookSentinel(), bundle.module
            engine.bundle = types.SimpleNamespace(z_module=types.SimpleNamespace(repr_tools=tools))
            engine._timer_stack = []
            engine.cumulative = {key: 0 for key in native.COUNTERS}
            engine.current = dict(**engine.cumulative,
                                 seconds={key: 0. for key in native.COUNTERS},
                                 exclusive_seconds={key: 0. for key in native.COUNTERS})
            original_trace = bundle.module.nethook.Trace
            native.NativeEngine._bind_representations(engine)
            with patch.object(original_tools, 'GPTJForCausalLM', HookSentinel):
                result = tools.get_reprs_at_idxs(engine.model, tokenizer, ['one', 'two'],
                    [[0], [1]], layer, 'transformer.h.{}', 'both')
            self.assertEqual(tuple(result[0].shape), (2, 4))
            self.assertEqual(engine.current['repr_forwards'], 1 + extra)
            self.assertEqual(engine.current['ln_1_extra_forwards'], extra)
            self.assertEqual(engine.current['repr_calls'], 1)
            self.assertIs(bundle.module.nethook.Trace, original_trace)
            self.assertFalse(engine.model._forward_hooks)
            self.assertTrue(all(not module._forward_hooks for module in engine.model.modules()))


class NativeCompositionTests(unittest.TestCase):
    def setUp(self):
        # Schema sentinels only: never weaken production dimensions/defaults.
        self.dims = patch.multiple(native, HIDDEN=4, INTERMEDIATE=8)
        self.dims.start()
        self.solve = patch('torch.linalg.solve', return_value=torch.zeros((8, 4)))
        self.solve.start()

    def tearDown(self):
        self.solve.stop()
        self.dims.stop()

    def ready(self, arm, **kwargs):
        engine, captured = mock_engine(arm, **kwargs)
        engine.prepare_contexts()
        return engine, captured

    def test_request_schema_order_and_native_leading_space(self):
        records = requests()
        records[0]['requested_rewrite']['target_new']['str'] = 'no leading space'
        before = copy.deepcopy(records)
        normalized = native.native_requests(records)
        self.assertEqual(records, before)
        self.assertEqual(normalized[0]['target_new']['str'], 'no leading space')
        self.assertEqual(normalized[1]['target_new']['str'], ' target')
        self.assertEqual([r['case_id'] for r in normalized], [r['case_id'] for r in records])
        self.assertEqual(native.native_requests(normalized), normalized)

    def test_wrong_schema_short_batch_and_duplicates_block(self):
        records = requests()
        records[0]['requested_rewrite']['target_new'] = 'wrong string schema'
        with self.assertRaisesRegex(RuntimeError, 'DICT_TARGET'):
            native.native_requests(records)
        with self.assertRaisesRegex(RuntimeError, 'BS100'):
            native.native_requests(requests()[:-1])
        records = requests()
        records[1]['case_id'] = records[0]['case_id']
        with self.assertRaisesRegex(RuntimeError, 'OCCURRENCE_IDENTITY'):
            native.native_requests(records)

    def test_blue_virtual_projector_mapping_and_independent_history(self):
        engine, captured = self.ready('ALPHAEDIT_BLUE')
        self.assertTrue(torch.equal(engine.P[0, :, :], torch.zeros((8, 8))))
        self.assertTrue(torch.equal(engine.P[1, :, :], torch.full((8, 8), 5.)))
        self.assertEqual(engine.P[1, :, :].data_ptr(), engine.P.full[5].data_ptr())
        model, receipt = engine.apply(requests(), 1)
        self.assertIs(model, engine.model)
        self.assertEqual(receipt['delta'], native.expected_counts('ALPHAEDIT_BLUE'))
        self.assertEqual(set(captured[0]['kwargs']), {'cache_template', 'cache_c', 'P'})
        self.assertIsNone(captured[0]['kwargs']['cache_template'])
        self.assertIs(captured[0]['kwargs']['P'], engine.P)
        self.assertIs(captured[0]['kwargs']['cache_c'], engine.H)
        self.assertEqual(set(engine.history()), {3, 8})
        self.assertEqual(receipt['counts']['fit_forwards'], 200)
        self.assertEqual(receipt['counts']['fit_updates'], 0)

    def test_cake_exactly_six_native_history_appends_and_b2_continuity(self):
        engine, captured = self.ready('CAKE')
        pointer = engine.H.data_ptr()
        _, first = engine.apply(requests(), 1)
        _, second = engine.apply(requests(), 2)
        self.assertEqual(first['delta'], native.expected_counts('CAKE'))
        self.assertEqual(second['cumulative']['history_appends'], 12)
        self.assertEqual(second['history_continuity']['tensor_version_before'],
                         first['history_continuity']['tensor_version_after'])
        self.assertEqual(engine.H.data_ptr(), pointer)
        self.assertTrue(torch.equal(engine.H, torch.full_like(engine.H, 2.)))
        self.assertIs(captured[0]['kwargs']['cache_c'], captured[1]['kwargs']['cache_c'])

    def test_returned_H_identity_and_double_append_blocked(self):
        engine, _ = self.ready('CAKE', bad_history=True)
        with self.assertRaisesRegex(RuntimeError, 'RETURNED_HISTORY_IDENTITY'):
            engine.apply(requests(), 1)
        engine, _ = self.ready('ALPHAEDIT_BLUE', extra_append=True)
        with self.assertRaisesRegex(RuntimeError, 'EXACT_PUBLIC_CALL_AND_HISTORY_COUNTS'):
            engine.apply(requests(), 1)

    def test_nonfinite_target_and_missing_key_blocked(self):
        engine, _ = self.ready('CAKE', finite=False)
        with self.assertRaisesRegex(RuntimeError, 'TARGET_SHAPE_NONFINITE'):
            engine.apply(requests(), 1)
        engine, _ = self.ready('ALPHAEDIT_BLUE', rows=99)
        with self.assertRaisesRegex(RuntimeError, 'KEY_FULL_REQUEST_CARDINALITY'):
            engine.apply(requests(), 1)

    def test_context_actual_generator_once_restores_all_cpu_rng(self):
        engine, _ = mock_engine('CAKE')
        before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
        contexts = engine.prepare_contexts()
        self.assertEqual(random.getstate(), before[0])
        self.assertEqual(np.random.get_state()[0], before[1][0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
        self.assertEqual(np.random.get_state()[2:], before[1][2:])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before[2]))
        self.assertEqual(engine.counts['native_z'], 0)
        self.assertEqual(engine.counts['history_appends'], 0)
        self.assertTrue(engine.context_receipt['RNG_restored'])
        self.assertEqual(engine.context_receipt['native_generator_calls'], 1)
        with self.assertRaisesRegex(RuntimeError, 'PREPARE_ONCE'):
            engine.prepare_contexts()
        engine.restore_context(None)
        self.assertIsNone(engine.contexts())
        engine.restore_context(contexts)
        self.assertEqual(engine.context_snapshot(), contexts)

    def test_native_fit_return_trace_keeps_existing_profiler_and_hooks(self):
        engine, _ = self.ready('CAKE')
        prior = sys.getprofile()
        visited = []
        profiler = lambda frame, event, arg: visited.append(event) if frame.f_code.co_name == 'compute_z' else None
        sys.setprofile(profiler)
        try:
            _, receipt = engine.apply(requests(), 1)
            self.assertIs(sys.getprofile(), profiler)
        finally:
            sys.setprofile(prior)
        self.assertEqual(len(engine.model._forward_hooks), 0)
        self.assertTrue(visited)
        self.assertEqual(receipt['counts']['fit_trace'][0]['evaluations'], 1)
        self.assertEqual(receipt['counts']['fit_trace'][0]['Adam_updates'], 0)
        self.assertEqual(receipt['counts']['fit_trace'][0]['site'], 8)

    def test_history_ram_rollback_and_batch21_guard(self):
        engine, _ = self.ready('CAKE')
        entries = {layer: value.clone() for layer, value in engine.history().items()}
        engine.apply(requests(), 1)
        engine.restore_history(entries)
        self.assertTrue(torch.equal(engine.H, torch.zeros_like(engine.H)))
        with self.assertRaisesRegex(RuntimeError, 'BATCH_SEQUENCE'):
            engine.apply(requests(), 1)
        engine.next_batch = 21
        with self.assertRaisesRegex(RuntimeError, 'NO21'):
            engine.apply(requests(), 21)

    def test_optimizer_proxy_preserves_arguments_and_return(self):
        engine, _ = self.ready('CAKE')
        engine.current = dict(**{key: 0 for key in native.COUNTERS},
                              seconds={key: 0. for key in native.COUNTERS},
                              exclusive_seconds={key: 0. for key in native.COUNTERS})
        seen = []
        original = types.SimpleNamespace(Adam=lambda *a, **k: types.SimpleNamespace(
            step=lambda *a, **k: seen.append((a, k)) or 'native-return'))
        optimizer = native._OptimCalls(original, engine).Adam('delta', lr=.5)
        self.assertEqual(optimizer.step('closure'), 'native-return')
        self.assertEqual(seen, [(('closure',), {})])
        self.assertEqual(engine.current['fit_updates'], 1)


if __name__ == '__main__':
    unittest.main()
