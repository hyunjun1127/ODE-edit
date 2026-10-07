"""CPU fake-model software checks only; no pretrained/GPU qualification."""
import random
import types
import unittest
from unittest import mock

import numpy as np
import torch

from .common import case_seed, GenerationError
from .generator import generate_row, native_eos, normalize_decode, rng_snapshot, rng_equal, isolated_rng


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def __init__(self):
        self.calls = []
        self.decode_calls = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        ids = [1] * len(text.split())
        return dict(input_ids=torch.tensor([ids], dtype=torch.long),
                    attention_mask=torch.ones((1, len(ids)), dtype=torch.long))

    def decode(self, ids, skip_special_tokens=False):
        self.decode_calls.append((list(ids), skip_special_tokens))
        return ' '.join('a' if x == 1 else 'b' for x in ids if not (skip_special_tokens and x in (0, 7, 9)))


class FakeModel(torch.nn.Module):
    def __init__(self, chosen=7):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = types.SimpleNamespace(eos_token_id=9, use_cache=True)
        self.generation_config = types.SimpleNamespace(eos_token_id=[7, 9])
        self.chosen = chosen
        self.calls = []
        self.failure = None
        self.mutate = False
        self.consume_random = False
        self.eval()

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        if self.failure:
            random.random(); np.random.random(); torch.rand(1)
            raise self.failure
        if self.consume_random:
            random.random(); np.random.random(); torch.rand(1)
        if self.mutate:
            self.weight.add_(1)
        self.calls.append(dict(ids=input_ids.clone(), mask=attention_mask.clone(),
            cache=use_cache, grad=torch.is_grad_enabled()))
        logits = torch.full((1, input_ids.shape[1], 10), -1000., dtype=torch.float32)
        logits[:, -1, self.chosen] = 0
        return types.SimpleNamespace(logits=logits)


def generated(model=None, tok=None, prompt='a a', **kwargs):
    return generate_row(model or FakeModel(), tok or FakeTokenizer(), prompt,
        model_identity={'model':'fake', 'revision':'sealed'}, occurrence=1,
        prompt_index=0, **kwargs)


class GeneratorTests(unittest.TestCase):
    def test_single_row_eos_masks_and_no_cache_no_grad(self):
        model, tok = FakeModel(), FakeTokenizer()
        result = generated(model, tok)
        self.assertEqual(result['continuation_token_ids'], [7])
        self.assertEqual(result['stop_reason'], 'eos')
        self.assertEqual(result['full_token_ids'], [1, 1, 7])
        self.assertEqual(result['text'], 'a a')
        self.assertEqual(result['eos_ids'], [7, 9])
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(model.calls[0]['cache']); self.assertFalse(model.calls[0]['grad'])
        self.assertTrue(torch.equal(model.calls[0]['mask'], torch.ones((1,2), dtype=torch.long)))
        self.assertTrue(tok.decode_calls[0][1])
        self.assertFalse(tok.calls[0]['padding']); self.assertFalse(tok.calls[0]['truncation'])

    def test_total_limit_preserves_full_long_input_and_independent_short_row(self):
        model, tok = FakeModel(), FakeTokenizer()
        for length in (100, 103):
            result = generated(model, tok, ' '.join(['a']*length))
            self.assertEqual(len(result['full_token_ids']), length)
            self.assertEqual(result['continuation_token_count'], 0)
            self.assertEqual(result['stop_reason'], 'length_cap_no_continuation')
        self.assertEqual(model.calls, [])
        self.assertEqual(generated(model, tok)['continuation_token_count'], 1)

    def test_99_prompt_one_continuation_total100(self):
        model = FakeModel(chosen=2)
        result = generated(model, prompt=' '.join(['a']*99))
        self.assertEqual(len(result['full_token_ids']),100)
        self.assertEqual(result['continuation_token_count'],1)
        self.assertEqual(result['stop_reason'],'length_cap')

    def test_full_prefix_coordinates_each_step_not_query_only(self):
        model = FakeModel(chosen=2)
        result = generated(model, prompt=' '.join(['a']*97))
        self.assertEqual([r['ids'].shape[1] for r in model.calls], [97,98,99])
        self.assertTrue(all(torch.equal(r['mask'],torch.ones_like(r['ids'])) for r in model.calls))
        self.assertEqual(result['input_token_ids'],[1]*97)

    def test_rng_all_cpu_domains_finally_success_and_exception(self):
        saved = rng_snapshot()
        model = FakeModel(); model.consume_random = True
        generated(model)
        self.assertTrue(rng_equal(saved))
        model.failure = ValueError('original-error')
        with self.assertRaisesRegex(ValueError, 'original-error'):
            generated(model)
        self.assertTrue(rng_equal(saved))

    def test_rng_tokenization_and_decode_exception_also_restore(self):
        class BadTokenizer(FakeTokenizer):
            def decode(self, *args, **kwargs):
                random.random(); np.random.random(); torch.rand(1)
                raise ValueError('decode-original')
        saved = rng_snapshot()
        with self.assertRaisesRegex(ValueError, 'decode-original'):
            generated(tok=BadTokenizer())
        self.assertTrue(rng_equal(saved))

    def test_all_relevant_cuda_rng_states_finally_restore_using_fake_cuda(self):
        values=[torch.tensor([1,2],dtype=torch.uint8),torch.tensor([3,4],dtype=torch.uint8)]
        def setter(states):values[:]=[x.clone() for x in states]
        def seeded(seed):setter([torch.tensor([8,9],dtype=torch.uint8)]*2)
        with mock.patch.object(torch.cuda,'is_initialized',return_value=True), \
             mock.patch.object(torch.cuda,'get_rng_state_all',side_effect=lambda:[x.clone() for x in values]), \
             mock.patch.object(torch.cuda,'set_rng_state_all',side_effect=setter), \
             mock.patch.object(torch.cuda,'manual_seed_all',side_effect=seeded):
            saved=rng_snapshot()
            with self.assertRaisesRegex(ValueError,'failure'):
                with isolated_rng(5):
                    self.assertFalse(rng_equal(saved))
                    raise ValueError('failure')
            self.assertTrue(rng_equal(saved))

    def test_cpu_rng_isolation_does_not_initialize_or_query_cuda(self):
        with mock.patch.object(torch.cuda,'is_initialized',return_value=False), \
             mock.patch.object(torch.cuda,'get_rng_state_all',side_effect=AssertionError('CUDA queried')), \
             mock.patch.object(torch.cuda,'manual_seed_all',side_effect=AssertionError('CUDA seeded')):
            with isolated_rng(3):torch.rand(2)

    def test_seed_excludes_arm_job_and_distinguishes_occurrence_prompt(self):
        model = {'revision':'same','payload':'identity'}
        first = case_seed(model,1,0)
        self.assertEqual(first,case_seed(dict(payload='identity',revision='same'),1,0))
        self.assertNotEqual(first,case_seed(model,2,0))
        self.assertNotEqual(first,case_seed(model,1,1))
        with self.assertRaisesRegex(GenerationError,'ARM_JOB_IDENTITY_FORBIDDEN'):
            case_seed(dict(model,job_id='dummy'),1,0)
        with self.assertRaisesRegex(GenerationError,'ARM_JOB_IDENTITY_FORBIDDEN'):
            case_seed(dict(model,runtime=dict(arm='CAKE')),1,0)

    def test_nonfinite_logit_and_training_and_padding_are_hard_failures(self):
        class BadModel(FakeModel):
            def forward(self, *args, **kwargs):
                result=super().forward(*args,**kwargs);result.logits[0,-1,0]=float('nan');return result
        with self.assertRaisesRegex(GenerationError,'NONFINITE_LOGITS'):generated(BadModel())
        model=FakeModel().train()
        with self.assertRaisesRegex(GenerationError,'NOT_EVAL'):generated(model)
        class Padded(FakeTokenizer):
            def __call__(self,*args,**kwargs):
                result=super().__call__(*args,**kwargs);result['attention_mask'][0,0]=0;return result
        with self.assertRaisesRegex(GenerationError,'NOT_UNPADDED'):generated(tok=Padded())

    def test_declared_nfkd_double_newline_normalization(self):
        self.assertEqual(normalize_decode('é\n\nx'), 'e\u0301 x')
        model=FakeModel();model.generation_config.eos_token_id=[7,9]
        self.assertEqual(native_eos(model,FakeTokenizer())[0],[7,9])


if __name__ == '__main__':
    unittest.main()
