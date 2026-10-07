"""CPU-only native import/schema/counter fixtures; no model or fitting run."""
import copy
from dataclasses import dataclass
import types
import unittest

import torch

from .native import NativeEngine, NATIVE_FILES, closure, load_native, native_requests, normalize_requests


@dataclass
class FixtureHParams:
    layers: tuple = (3, 4, 5, 6, 7, 8)


def records():
    return [dict(case_id=i, requested_rewrite=dict(prompt='{} is', subject='subject',
                 target_new={'str': 'target'})) for i in range(100)]


def native_fixture(alpha):
    """A callable mock of apply/execute, not a small instantiated model."""
    module = types.SimpleNamespace(torch=torch, compute_z=lambda *args, **kwargs: None,
                                   compute_ks=lambda *args, **kwargs: None,
                                   cache_c_new=False)
    def compute_z(*args,**kwargs):
        it=2;loss=torch.tensor(.01);nll_loss=loss;kl_loss=loss*0;weight_decay=loss*0
        return torch.zeros(2)
    module.compute_z=compute_z
    reset_calls = []
    model, tokenizer = object(), object()

    def execute(current_model, tok, requests, hp, cache_template=None):
        for request in requests:
            module.compute_z(current_model, tok, request, hp)
        for layer in hp.layers:
            module.compute_ks(current_model, tok, requests, hp, layer)
            module.torch.linalg.solve(torch.eye(2), torch.ones(2, 1))
        if alpha:
            for index, layer in enumerate(hp.layers):
                module.compute_ks(current_model, tok, requests, hp, layer)
                module.cache_c[index, :, :] += torch.eye(2)
        return {}

    def apply(current_model, tok, requests, hp, copy=False,
              return_orig_weights=False, cache_template=None, reset_cache=False):
        if copy or return_orig_weights or cache_template is not None:
            raise AssertionError('Unexpected native apply options')
        if alpha:
            reset_calls.append(reset_cache)
            if reset_cache:
                module.cache_c_new = False
            if not module.cache_c_new:
                module.cache_c = torch.zeros(6, 2, 2)
                module.cache_c_new = True
            module.execute_AlphaEdit(current_model, tok, requests, hp, cache_template=cache_template)
        else:
            module.execute_memit(current_model, tok, requests, hp, cache_template=cache_template)
        return current_model, {}

    module.execute_memit = module.execute_AlphaEdit = execute
    module.apply_memit_to_model = module.apply_AlphaEdit_to_model = apply
    engine = NativeEngine(module, FixtureHParams(), model, tokenizer,
                          'BASE_ALPHAEDIT' if alpha else 'BASE_MEMIT',
                          dict(status='CPU_MOCK_CONTEXT_NOT_ACTUAL_QUALIFICATION'))
    return engine, module, model, reset_calls


class NativeBindingTests(unittest.TestCase):
    def test_direct_native_imports_and_actual_hparams_parser(self):
        bundle = load_native()
        rows = closure(bundle)
        self.assertEqual({row['relative'] for row in rows}, set(NATIVE_FILES))
        self.assertEqual(len(rows), 17)
        for module, name, yaml in (
            (bundle.memit, 'MEMITHyperParams', '/mnt/raid5/janghj/EasyEdit/hparams/MEMIT/gpt-j-6B.yaml'),
            (bundle.alphaedit, 'AlphaEditHyperParams', '/mnt/raid5/janghj/EasyEdit/hparams/AlphaEdit/gpt-j-6B.yaml'),
        ):
            hp = getattr(module, name).from_hparams(yaml)
            self.assertEqual((hp.v_lr, hp.v_num_grad_steps, hp.v_loss_layer), (.5, 25, 27))
            self.assertEqual(hp.layers, [3, 4, 5, 6, 7, 8])
        self.assertEqual(bundle.memit.MEMITHyperParams.from_hparams(
            '/mnt/raid5/janghj/EasyEdit/hparams/MEMIT/gpt-j-6B.yaml').mom2_update_weight, 15000)
        self.assertEqual(bundle.alphaedit.AlphaEditHyperParams.from_hparams(
            '/mnt/raid5/janghj/EasyEdit/hparams/AlphaEdit/gpt-j-6B.yaml').L2, 10)
        self.assertFalse(torch.cuda.is_initialized())

    def test_request_conversion_does_not_change_input_or_add_native_space(self):
        original = records()
        before = copy.deepcopy(original)
        result = native_requests(original)
        self.assertEqual(original, before)
        self.assertEqual(result[0], dict(case_id=0, prompt='{} is',
                                        subject='subject', target_new='target'))
        self.assertEqual(native_requests(result), result)
        self.assertIs(normalize_requests, native_requests)
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_BS100'):
            native_requests(original[:-1])

    def test_memit_direct_apply_same_model_no_history_and_original_solve(self):
        engine, module, model, resets = native_fixture(False)
        original_solve = torch.linalg.solve
        result, receipt = engine.apply(records(), 1)
        self.assertIs(result, model)
        self.assertIs(torch.linalg.solve, original_solve)
        self.assertEqual(engine.history(), {})
        self.assertEqual(resets, [])
        counts = receipt['counts']
        self.assertEqual((counts['native_z'], counts['write_keys'], counts['solves']), (100, 6, 6))
        self.assertEqual((counts['history_keys'], counts['history_appends']), (0, 0))
        self.assertEqual(receipt['delta'], dict(native_z=100, write_keys=6,
                         history_keys=0, solves=6, history_appends=0))
        self.assertEqual(engine.counts, receipt['delta'])
        self.assertFalse(receipt['native_z_disk_cache'])
        self.assertFalse(torch.cuda.is_initialized())

    def test_alpha_first_reset_once_native_history_not_caller_append(self):
        engine, module, model, resets = native_fixture(True)
        self.assertEqual(engine.history(), {})
        _, first = engine.apply(records(), 1)
        history_pointer = module.cache_c.data_ptr()
        first_history = module.cache_c.clone()
        _, second = engine.apply(records(), 2)
        self.assertEqual(resets, [True, False])
        self.assertEqual(module.cache_c.data_ptr(), history_pointer)
        torch.testing.assert_close(module.cache_c, 2 * first_history, atol=0, rtol=0)
        self.assertEqual(set(engine.history()), {3, 4, 5, 6, 7, 8})
        for receipt in (first, second):
            counts = receipt['counts']
            self.assertEqual((counts['native_z'], counts['write_keys'], counts['history_keys'],
                              counts['solves'], counts['history_appends']), (100, 6, 6, 6, 6))
            self.assertEqual(counts['history_tensor_version_delta'], 12)
            self.assertEqual(receipt['caller_history_appends'], 0)
        self.assertEqual(second['cumulative']['history_appends'], 12)
        self.assertEqual(second['delta']['history_appends'], 6)

    def test_native_progress_actual_completed_z_axis_and_ram_first_entry_rollback(self):
        engine, module, model, resets = native_fixture(True)
        seen = []
        engine.progress = seen.append
        self.assertIs(engine.progress, engine.on_progress)
        engine.apply(records(), 1)
        engine.apply(records(), 2)
        self.assertEqual(len(seen), 200)
        self.assertEqual([row['native_z'] for row in seen], list(range(1, 201)))
        self.assertEqual([row['request_index'] for row in seen], list(range(1, 101)) * 2)
        self.assertEqual([row['batch'] for row in seen], [1] * 100 + [2] * 100)
        self.assertTrue(all(type(row['seconds']) is float and row['seconds'] >= 0 for row in seen))
        counts = engine.counts.copy()
        engine.reset_history_to_uninitialized()
        self.assertEqual(engine.history(), {})
        self.assertIsNone(module.cache_c)
        self.assertFalse(module.cache_c_new)
        self.assertEqual(engine.counts, counts)  # observed failed/rolled back work is still cost.

    def test_progress_callback_failure_cannot_change_native_apply(self):
        engine, module, model, resets = native_fixture(False)
        def broken(_):
            raise RuntimeError('CPU_FAKE_LOGGING_FAILURE')
        engine.progress = broken
        returned, receipt = engine.apply(records(), 1)
        self.assertIs(returned, model)
        self.assertEqual(receipt['counts']['logging_callback_errors'], 100)
        self.assertEqual(receipt['counts']['native_z'], 100)
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_BATCH_SEQUENCE'):
            engine.apply(records(), 1)


if __name__ == '__main__':
    unittest.main()
