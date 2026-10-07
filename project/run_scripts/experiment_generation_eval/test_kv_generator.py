"""CPU fake-cache regressions; these are NOT GPT2/GPT-J GPU parity evidence."""
import tempfile
import types
import unittest

import torch

from .common import GenerationError, digest
from .generator import (generate_row, generate_rows, REFERENCE_ROUTE, SINGLETON_ROUTE,
    BATCH_ROUTE, rng_snapshot, rng_equal)
from .kv_qualification import build_qualification_plan, run_qualification, verify_actual_receipt


class Tokenizer:
    eos_token_id = 9
    def __call__(self, text, **kwargs):
        ids = [int(x) for x in text.split()]
        return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones((1, len(ids)), dtype=torch.long))
    def decode(self, ids, skip_special_tokens=False):
        return ' '.join(str(x) for x in ids if not (skip_special_tokens and x == 9))


class CacheModel(torch.nn.Module):
    def __init__(self, *, divergent=False, failing=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = types.SimpleNamespace(model_type='gpt2', eos_token_id=9, use_cache=False)
        self.generation_config = types.SimpleNamespace(eos_token_id=9)
        self.calls = []
        self.divergent, self.failing = divergent, failing
        self.eval()
    def forward(self, input_ids, attention_mask, use_cache, return_dict,
                position_ids=None, past_key_values=None):
        if self.failing and use_cache:
            raise RuntimeError('fake native cache error')
        previous = None if past_key_values is None else past_key_values[0][0][:, 0, :, 0].long()
        full = input_ids if previous is None else torch.cat((previous, input_ids), 1)
        assert attention_mask.shape == full.shape
        if position_ids is not None:
            start = 0 if previous is None else previous.shape[1]
            assert torch.equal(position_ids, torch.arange(start, full.shape[1]).expand(input_ids.shape[0], -1))
        self.calls.append(dict(batch=input_ids.shape[0], query=input_ids.shape[1],
                               full=full.clone(), use_cache=use_cache))
        logits = torch.full((*input_ids.shape, 10), -1000., dtype=torch.float32)
        # First row EOS at full length3, second length5: exercises active gather.
        for row in range(full.shape[0]):
            stop_at = 3 if int(full[row, 0]) == 1 else 5
            chosen = 9 if full.shape[1] >= stop_at else 3
            if self.divergent and use_cache:
                chosen = 4
            logits[row, -1, chosen] = 0
        cache = full.float().view(full.shape[0], 1, full.shape[1], 1)
        return types.SimpleNamespace(logits=logits, past_key_values=((cache, cache.clone()),))


IDENTITY = dict(model='fixture', revision='fixed')
def request(prompt, occurrence, prompt_index=0):
    return dict(prompt=prompt, occurrence=occurrence, prompt_index=prompt_index, model_identity=IDENTITY)


class KVGeneratorTests(unittest.TestCase):
    def test_singleton_native_mask_position_and_reference_token_parity(self):
        reference = generate_rows(CacheModel(), Tokenizer(), [request('1 2', 0)], route=REFERENCE_ROUTE)
        model = CacheModel()
        saved = rng_snapshot()
        cached = generate_rows(model, Tokenizer(), [request('1 2', 0)], route=SINGLETON_ROUTE)
        self.assertTrue(rng_equal(saved))
        for key in ('seed', 'full_token_ids', 'stop_reason', 'text'):
            self.assertEqual(reference[0][key], cached[0][key])
        self.assertEqual([x['query'] for x in model.calls], [2, 1])
        self.assertFalse(model.config.use_cache)
        self.assertEqual(cached[0]['prefill_query_tokens'], 2)
        self.assertEqual(cached[0]['decode_query_tokens'], 1)

    def test_equal_length_eos_gather_order_and_physical_counters(self):
        requests = [request('1 2', 0), request('2 2', 1), request('1 2 2', 2)]
        reference = generate_rows(CacheModel(), Tokenizer(), requests, route=REFERENCE_ROUTE)
        model = CacheModel()
        rows = generate_rows(model, Tokenizer(), requests, route=BATCH_ROUTE, microbatch=4)
        self.assertEqual([x['full_token_ids'] for x in rows], [x['full_token_ids'] for x in reference])
        self.assertEqual([x['occurrence'] for x in rows], [0, 1, 2])
        self.assertEqual([(x['batch'], x['query']) for x in model.calls], [(2,2),(2,1),(1,1),(1,1),(1,3)])
        self.assertEqual(sum(x['physical_forward_calls'] for x in rows), len(model.calls))
        self.assertEqual(sum(x['prefill_query_tokens'] for x in rows), 7)
        self.assertEqual(sum(x['decode_query_tokens'] for x in rows), 4)

    def test_total100_and_long_input_no_truncate(self):
        requests = [request(' '.join(['2']*100), 0), request(' '.join(['2']*103), 1)]
        model = CacheModel()
        rows = generate_rows(model, Tokenizer(), requests, route=BATCH_ROUTE, microbatch=8)
        self.assertEqual([x['continuation_token_count'] for x in rows], [0,0])
        self.assertEqual([x['input_token_count'] for x in rows], [100,103])
        self.assertEqual(model.calls, [])

    def test_actual_native_dynamic_cache_selector_and_cache_position(self):
        from transformers.cache_utils import DynamicCache
        class DynamicModel(CacheModel):
            def forward(self, input_ids, attention_mask, use_cache, return_dict,
                        position_ids=None, past_key_values=None, cache_position=None):
                old = None if past_key_values is None else tuple(past_key_values)
                result = super().forward(input_ids, attention_mask, use_cache, return_dict,
                                         position_ids, old)
                if cache_position is not None:
                    self.assert_positions = cache_position.tolist()
                    self.assert_shape = input_ids.shape[1]
                    assert self.assert_shape == len(self.assert_positions)
                    assert self.assert_positions[-1] == attention_mask.shape[1]-1
                key, value = result.past_key_values[0]
                result.past_key_values = DynamicCache(ddp_cache_data=[(key, value)])
                return result
        model = DynamicModel()
        requests = [request('1 2', 0), request('2 2', 1)]
        rows = generate_rows(model, Tokenizer(), requests, route=BATCH_ROUTE, microbatch=4)
        self.assertEqual([x['full_token_ids'] for x in rows], [[1,2,3,9], [2,2,3,3,3,9]])
        self.assertEqual(rows[0]['cache_class'], 'transformers.cache_utils.DynamicCache')
        self.assertTrue(rows[0]['cache_position_explicit'])
        self.assertFalse(model.config.use_cache)

    def test_independent_private_rng_matches_reference_sampling(self):
        class RandomCache(CacheModel):
            def forward(self, *args, **kwargs):
                result = super().forward(*args, **kwargs)
                result.logits[..., :] = torch.tensor([-.2,.1,.2,.3,.4,-3,-5,-6,-7,-8])
                return result
        requests = [request('1 2', 0), request('2 2', 1), request('1 2', 2)]
        reference = generate_rows(RandomCache(), Tokenizer(), requests)
        cached = generate_rows(RandomCache(), Tokenizer(), requests, route=BATCH_ROUTE, microbatch=4)
        self.assertEqual([x['continuation_token_ids'] for x in reference],
                         [x['continuation_token_ids'] for x in cached])
        self.assertNotEqual(cached[0]['continuation_token_ids'], cached[2]['continuation_token_ids'])

    def test_failure_restores_rng_and_does_not_retry(self):
        before = rng_snapshot()
        with self.assertRaisesRegex(RuntimeError, 'fake native cache error'):
            generate_rows(CacheModel(failing=True), Tokenizer(), [request('1 2', 0)], route=SINGLETON_ROUTE)
        self.assertTrue(rng_equal(before))
        with self.assertRaisesRegex(GenerationError, 'MICROBATCH_UNQUALIFIED'):
            generate_rows(CacheModel(), Tokenizer(), [request('1 2', 0)], route=BATCH_ROUTE, microbatch=2)

    def test_plan_width8_is_not_pass_from_partial_batch(self):
        records = []
        for index, length in enumerate([1,2,2,2,2,2,2,2,2,7,100]):
            records.append(dict(ordered_occurrence=index, case_id=index,
                requested_rewrite=dict(target_new={'id':'fixed'}, relation_id='fixed'),
                generation_prompts=[' '.join(['1']*length)]))
        plan = build_qualification_plan(Tokenizer(), records, model_identity=IDENTITY)
        self.assertLess(plan['coverage']['tested_max_batch_width'], 8)
        self.assertTrue(plan['coverage']['at_or_above100'])
        with tempfile.TemporaryDirectory() as out:
            actual = run_qualification(CacheModel(), Tokenizer(), None, plan, out=out)
            self.assertEqual(actual['selected_route'], SINGLETON_ROUTE)
            self.assertFalse(actual['route_results'][BATCH_ROUTE]['passed'])
            self.assertEqual(actual['route_results'][BATCH_ROUTE]['failure_reason'],
                             'ACTUAL_BATCH_WIDTH_UNVERIFIED')
            verified = verify_actual_receipt(actual['member'], expected_plan_sha256=digest(plan), allow_cpu_fixture=True)
            self.assertFalse(verified['pretrained_GPU_PASS'])
            with self.assertRaisesRegex(GenerationError, 'PLAN_SHA'):
                verify_actual_receipt(actual['member'], expected_plan_sha256='wrong', allow_cpu_fixture=True)
            with self.assertRaisesRegex(GenerationError, 'NATIVE_GPU_QUALIFICATION_REQUIRED'):
                verify_actual_receipt(actual['member'])

    def test_predeclared_mb4_full_width_and_fallback_cache_token_gate(self):
        records = [dict(ordered_occurrence=index, requested_rewrite=dict(target_new={}),
                       generation_prompts=[prompt]) for index, prompt in enumerate(
                       ['1','1 2','1 2','2 2','2 2','1 2 2 2 2 2 2', ' '.join(['1']*100)])]
        with self.assertRaisesRegex(GenerationError, 'MEMORY_ALTERNATIVE_REQUIRED'):
            build_qualification_plan(Tokenizer(), records, model_identity=IDENTITY, microbatch=4)
        plan = build_qualification_plan(Tokenizer(), records, model_identity=IDENTITY,
                                        microbatch=4, memory_alternative=True)
        self.assertTrue(plan['coverage']['batch_width_covered'])
        with tempfile.TemporaryDirectory() as out:
            actual = run_qualification(CacheModel(), Tokenizer(), None, plan, out=out)
            self.assertEqual(actual['selected_route'], BATCH_ROUTE)
            self.assertEqual(actual['fixed_microbatch'], 4)
        with tempfile.TemporaryDirectory() as out:
            actual = run_qualification(CacheModel(divergent=True), Tokenizer(), None, plan, out=out)
            self.assertEqual(actual['selected_route'], REFERENCE_ROUTE)
            self.assertFalse(actual['route_results'][SINGLETON_ROUTE]['token_sequence_exact'])
            self.assertFalse(actual['route_results'][BATCH_ROUTE]['executed'])


if __name__ == '__main__':
    unittest.main()
