"""CPU fixture executes the actual frozen apply function with tiny model stubs.

Tensor.cuda/torch.eye are CPU-routed only inside these tests.  This is not an
actual Llama/native-GPU PASS and does not establish numerical model parity.
"""
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from project.run_scripts.alpha_key_concentration_causal.common import digest, rng_get, rng_set, tensor_sha
from project.run_scripts.alpha_key_concentration_causal.native_writer import (
    NativeWriterError, _rng_sha, write,
)


DESIGN = Path(os.environ.get("ALPHA_CAUSAL_DESIGN_ROOT",
    "/data/janghj/ODE-edit/local/alpha-key-concentration-causal/20260923-r1/inputs/design"))


class CpuTorch:
    def __getattr__(self, name):
        return getattr(torch, name)

    def eye(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        return torch.eye(*args, **kwargs)


class TinyRuntime:
    def __init__(self):
        torch.manual_seed(123)
        self.native = ModuleType("frozen_native_fixture")
        source = (DESIGN / "source/AlphaEdit_main.py").read_text()
        tree = ast.parse(source)
        needed = {"apply_AlphaEdit_to_model", "upd_matrix_match_shape"}
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in needed]
        self.native.__dict__.update(torch=CpuTorch(), deepcopy=copy.deepcopy, Path=Path,
                                    CONTEXT_TEMPLATES_CACHE=[["{}"], [f"c{i} {{}}" for i in range(5)]])
        code = compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__",
                       names=[ast.alias(name="annotations")], level=0)] + functions,
                       type_ignores=[])), "frozen-native-AlphaEdit_main.py", "exec")
        exec(code, self.native.__dict__)
        self.weights = {layer: torch.nn.Parameter(torch.randn(3, 5) * .01,
                                                  requires_grad=False) for layer in range(4, 9)}
        self.model = SimpleNamespace(weights=self.weights)
        self.tok = object()
        self.contexts = copy.deepcopy(self.native.CONTEXT_TEMPLATES_CACHE)
        self.hp = SimpleNamespace(layers=[4, 5, 6, 7, 8], blue=False, L2=10,
            rewrite_module_tmp="model.layers.{}.mlp.down_proj", layer_module_tmp="model.layers.{}",
            fact_token="subject_last", clamp_norm_factor=.75)
        self.M = torch.eye(5).repeat(5, 1, 1) * 4
        self.P = torch.eye(5).repeat(5, 1, 1)
        self.calls = dict(z=0, keys=[], residual=0)
        self.batch = 50
        self.timestamp_keys_validated = True
        self.native.nethook = SimpleNamespace(get_parameter=lambda model, name: self.weights[int(name.split('.')[2])])
        self.native.get_context_templates = lambda model, tok: copy.deepcopy(self.contexts)

        def key(model, tok, requests, hp, layer, contexts):
            self.calls['keys'].append(layer)
            ids = torch.tensor([int(r['case_id']) for r in requests], dtype=torch.float32)
            base = torch.stack([torch.sin(ids*.01 + j) for j in range(5)], dim=1)
            lower = sum(float(self.weights[l].sum()) for l in range(4, layer))
            return base + lower*.01

        def z(model, tok, request, hp, layer, contexts):
            self.calls['z'] += 1
            i = int(request['case_id'])
            return torch.tensor([.1, .2, .3]) + i*.0001 + torch.rand(3)*.001

        def residual(model, tok, layer, context_templates, words, module_template, fact_token_strategy):
            self.calls['residual'] += 1
            requests = [dict(case_id=int(word)) for word in words]
            total = torch.zeros(len(words), 3)
            for physical in range(4, 9):
                keys = key(model, tok, requests, self.hp, physical, self.contexts)
                self.calls['keys'].pop()  # Internal stub work is not native compute_ks calls.
                total += keys @ self.weights[physical].T
            return torch.zeros(len(words), 5), total

        self.native.compute_z = z
        self.native.compute_ks = key
        self.native.get_module_input_output_at_words = residual

    def signature(self):
        return dict(weights={str(l): tensor_sha(w) for l, w in self.weights.items()},
                    history=tensor_sha(self.M), context=digest(self.contexts), cursor=self.batch)

    def assert_nonselected(self):
        pass

    def snapshot(self):
        return dict(weights={l: w.detach().clone() for l, w in self.weights.items()},
                    M=self.M.clone(), rng=rng_get())

    def restore(self, snap):
        for layer, weight in self.weights.items():
            weight.data.copy_(snap['weights'][layer])
        self.M.copy_(snap['M'])
        rng_set(snap['rng'])


@contextlib.contextmanager
def cpu_native():
    with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self), \
         patch.object(torch.cuda, 'empty_cache', lambda: None), \
         contextlib.redirect_stdout(io.StringIO()):
        yield


class NativeWriterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="alpha-native-writer-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rt = TinyRuntime()
        self.requests = [dict(case_id=i, prompt="{} knows", subject=str(i),
                              target_new={"str": "New"}) for i in range(100)]

    def test_original_expression_physical_result_and_history_exact_once(self):
        entry = self.rt.snapshot()
        with cpu_native():
            self.rt.native.apply_AlphaEdit_to_model(self.rt.model, self.rt.tok,
                self.requests, self.rt.hp, cache_template=None, cache_c=self.rt.M, P=self.rt.P)
        baseline = self.rt.snapshot()
        self.rt.restore(entry)
        stages = []
        def observer(stage, rt, info):
            stages.append(stage)
            torch.rand(17)  # Callback RNG changes must not affect native state.
        with cpu_native():
            result = write(self.rt, self.requests, self.root/'native', stage_callback=observer)
        self.assertEqual(stages, ['entry', 'z', 'W4', 'W5', 'W6', 'W7', 'W8', 'history'])
        for layer in range(4, 9):
            self.assertTrue(torch.equal(self.rt.weights[layer], baseline['weights'][layer]))
            K = result['finalkeys'][layer]
            self.assertTrue(torch.equal(self.rt.M[layer-4], entry['M'][layer-4] + K @ K.T))
            factor = result['factors'][layer]
            self.assertEqual(factor['divisor'], 9-layer)
            self.assertTrue(factor['physical_write_applied'])
            self.assertEqual(factor['K'].shape, (5, 100))
            self.assertEqual(factor['R'].shape, (3, 100))
            self.assertEqual(factor['C'].shape, (5, 100))
        self.assertTrue(torch.equal(self.rt.M, baseline['M']))
        self.assertEqual(_rng_sha(rng_get()), _rng_sha(baseline['rng']))
        self.assertEqual(result['z']['targets'].shape, (3, 100))
        self.assertEqual(result['receipts']['counters']['history_appends'], 5)
        self.assertEqual(result['receipts']['counters']['diagnostic_solves'], 5)
        self.assertEqual(result['receipts']['counters']['target_new'], 100)
        self.assertTrue((self.root/'native/receipt.json').exists())
        # No persisted payload contains a pre-weight or a complete resume state.
        for path in (self.root/'native').rglob('*.pt'):
            payload = torch.load(path, weights_only=True)
            self.assertFalse({'pre_weight', 'weights', 'M', 'rng'} & set(payload))

    def test_shared_z_restores_post_z_rng_and_has_no_new_fit(self):
        entry = self.rt.snapshot()
        with cpu_native():
            original = write(self.rt, self.requests, self.root/'first')
        selected = self.rt.snapshot()
        self.rt.restore(entry)
        self.rt.calls['z'] = 0
        with cpu_native():
            replay = write(self.rt, self.requests, self.root/'replay', shared_z=original['z'])
        self.assertEqual(self.rt.calls['z'], 0)
        self.assertEqual(replay['receipts']['counters']['target_reused'], 100)
        self.assertEqual(replay['receipts']['counters']['target_new'], 0)
        self.assertEqual(_rng_sha(rng_get()), _rng_sha(selected['rng']))
        for layer in range(4, 9):
            self.assertTrue(torch.equal(self.rt.weights[layer], selected['weights'][layer]))
        self.assertFalse((self.root/'replay/native-z.pt').exists())
        self.assertTrue((self.root/'replay/native-z-reuse.json').exists())

    def test_shared_z_rejects_changed_entry_request_and_rng(self):
        entry = self.rt.snapshot()
        with cpu_native():
            first = write(self.rt, self.requests, self.root/'first')
        for scenario in ['entry', 'request', 'rng']:
            self.rt.restore(entry)
            requests = copy.deepcopy(self.requests)
            if scenario == 'entry':
                self.rt.weights[4].data.add_(.1)
            elif scenario == 'request':
                requests[0]['target_new']['str'] = 'Changed'
            else:
                torch.rand(1)
            with self.subTest(scenario=scenario), self.assertRaisesRegex(NativeWriterError, 'SHARED_Z'):
                with cpu_native():
                    write(self.rt, requests, self.root/scenario, shared_z=first['z'])

    def test_h56_current_once_per_prefix_and_persistent_history_native(self):
        entry = self.rt.snapshot()
        with cpu_native():
            first = write(self.rt, self.requests, self.root/'native')
        self.rt.restore(entry)
        stamp = {layer: torch.ones(5, 512)*.001 for layer in range(4, 9)}
        calls = []
        def current(layer, rt):
            calls.append(layer)
            return stamp[layer] + float(rt.weights[4].sum())*.0001
        with cpu_native():
            result = write(self.rt, self.requests, self.root/'h56', branch='H56',
                           shared_z=first['z'], stamp=stamp, current=current, bank_ids=tuple(range(512)))
        self.assertEqual(calls, [5, 6])
        for layer in range(4, 9):
            K = result['finalkeys'][layer]
            self.assertTrue(torch.equal(self.rt.M[layer-4], entry['M'][layer-4]+K@K.T))
            self.assertEqual(result['factors'][layer]['operand']['applied'], layer in (5, 6))

    def test_sham_subtract_readd_and_no_current_bank_refitting(self):
        entry = self.rt.snapshot()
        with cpu_native():
            first = write(self.rt, self.requests, self.root/'native')
        self.rt.restore(entry)
        stamp = {layer: torch.ones(5, 512)*.001 for layer in range(4, 9)}
        with cpu_native():
            result = write(self.rt, self.requests, self.root/'sham', branch='SHAM',
                           shared_z=first['z'], stamp=stamp,
                           current=lambda *args: self.fail('SHAM must not compute current bank'),
                           bank_ids=tuple(range(512)))
        self.assertTrue(all(f['operand']['sham_subtract_readd_executed'] for f in result['factors'].values()))
        self.assertEqual(result['receipts']['counters']['history_appends'], 5)

    def test_diagnostic_failure_preserves_physical_stage_and_restores_wrappers(self):
        originals = {k: getattr(self.rt.native, k) for k in
                     ('torch', 'compute_z', 'compute_ks', 'get_module_input_output_at_words')}
        solve = torch.linalg.solve
        count = 0
        def fail_second(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError('diagnostic test exception')
            return solve(*args, **kwargs)
        with patch.object(torch.linalg, 'solve', fail_second), cpu_native():
            with self.assertRaisesRegex(NativeWriterError, 'DIAGNOSTIC_RESPONSE_FACTOR_FAILED'):
                write(self.rt, self.requests, self.root/'failed')
        for name, value in originals.items():
            self.assertIs(getattr(self.rt.native, name), value)
        failure = json.loads((self.root/'failed/failure.json').read_text())
        self.assertEqual(failure['physical_write_layers'], [4])
        self.assertEqual(failure['counters']['native_solves'], 1)
        self.assertEqual(failure['counters']['history_appends'], 0)
        self.assertEqual(failure['cause_message'], 'diagnostic test exception')
        self.assertFalse(failure['automatic_fallback'])

    def test_rng_value_hash_clone_stability(self):
        value = rng_get()
        self.assertEqual(_rng_sha(value), _rng_sha(copy.deepcopy(value)))

    def test_incomplete_batch_and_nonempty_attempt_fail_closed(self):
        with self.assertRaisesRegex(NativeWriterError, '100_UNIQUE'):
            write(self.rt, self.requests[:2], self.root/'short')
        (self.root/'nonempty').mkdir()
        (self.root/'nonempty/existing').touch()
        with self.assertRaisesRegex(NativeWriterError, 'CREATE_ONCE'):
            write(self.rt, self.requests, self.root/'nonempty')


if __name__ == '__main__':
    unittest.main()
