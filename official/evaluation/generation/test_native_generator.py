"""CPU fake-model checks only: no model assets/GPU/Slurm/network qualification."""
import random
import types
import unittest

import numpy as np
import torch

from .common import EVAL_SEED, GenerationError
from .generator import isolated_rng, rng_snapshot, rng_equal
from .native_generator import generate_case


class Tokenizer:
    pad_token_id = 9
    eos_token_id = 9
    padding_side = 'right'
    def __init__(self):
        self.calls, self.decode_calls = [], []
    def __call__(self, prompts, **kwargs):
        self.calls.append((list(prompts), kwargs))
        tokens = [[int(value) for value in prompt.split()] for prompt in prompts]
        width = max(map(len, tokens))
        return dict(input_ids=torch.tensor([row+[9]*(width-len(row)) for row in tokens]),
            attention_mask=torch.tensor([[1]*len(row)+[0]*(width-len(row)) for row in tokens]))
    def decode(self, tokens, **kwargs):
        self.decode_calls.append((list(tokens), kwargs))
        return ' '.join('<|endoftext|>' if token == 9 else str(token) for token in tokens)


class Model(torch.nn.Module):
    def __init__(self, *, stochastic=False, chosen=9, family='gpt2'):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = types.SimpleNamespace(model_type=family, use_cache=False, eos_token_id=9)
        self.calls, self.failure, self.mutate = [], False, False
        self.stochastic, self.chosen = stochastic, chosen
        self.eval()
    def forward(self, input_ids, attention_mask, past_key_values, use_cache):
        if self.failure:
            random.random(); np.random.random(); torch.rand(1)
            raise ValueError('native-forward-original')
        if self.mutate:
            self.weight.add_(1)
        previous = None if past_key_values is None else past_key_values[0][0][:, 0, :, 0].long()
        full = input_ids if previous is None else torch.cat([previous, input_ids], dim=1)
        assert use_cache is True and attention_mask.shape == full.shape
        self.calls.append(dict(query=input_ids.clone(), mask=attention_mask.clone(),
            past_length=0 if previous is None else previous.shape[1], grad=torch.is_grad_enabled()))
        logits = torch.full((*input_ids.shape, 10), -1000., dtype=torch.float32)
        if self.stochastic:
            logits[:] = torch.tensor([.0,.1,.2,.3,.4,.5,.6,.7,.8,.9])
        else:
            logits[:, -1, self.chosen] = 0
        cache = full.float().view(full.shape[0], 1, full.shape[1], 1)
        return types.SimpleNamespace(logits=logits, past_key_values=((cache, cache.clone()),))


def prompt(length, token=1):
    return ' '.join([str(token)]*length)


def original_generate_fast(model, tok, prompts):
    """Frozen CAKE generate_fast core oracle, with only max_out_len=100 fixed."""
    encoded = tok(prompts, padding=True, return_tensors='pt')
    ids, mask = encoded['input_ids'], encoded['attention_mask']
    batch = ids.size(0)
    past, context = None, slice(0, mask.sum(1).min().item())
    with torch.no_grad():
        while ids.size(1) < 100:
            output = model(input_ids=ids[:, context], attention_mask=mask[:, :context.stop],
                past_key_values=past, use_cache=True)
            logits, past = output.logits, output.past_key_values
            probabilities = torch.nn.functional.softmax(logits[:, -1, :], dim=1)
            selected = torch.topk(probabilities, 5, dim=1).indices
            probabilities = torch.gather(probabilities, 1, selected)
            probabilities = probabilities / probabilities.sum(1)[:, None]
            samples = torch.multinomial(probabilities, 1)
            tokens = torch.gather(selected, 1, samples)
            if context.stop == ids.size(1):
                mask = torch.cat([mask, mask.new_zeros(batch, 1)], dim=1)
                ids = torch.cat([ids, ids.new_ones(batch, 1)*tok.pad_token_id], dim=1)
            last_non_masked = mask.sum(1)-1
            for index in range(batch):
                new_index = last_non_masked[index]+1
                if last_non_masked[index].item()+1 != context.stop:
                    continue
                if new_index < 100:
                    ids[index][new_index] = tokens[index]
                    mask[index][new_index] = 1
            context = slice(context.stop, context.stop+1)
    import unicodedata
    padded = ids.tolist()
    texts = [unicodedata.normalize('NFKD', tok.decode(row)).replace('\n\n',' ')
             .replace('<|endoftext|>','') for row in padded]
    return padded, texts


class NativeGeneratorTests(unittest.TestCase):
    def test_variable_length_whole_case_original_oracle_and_incremental_mask(self):
        prompts = [prompt(96), prompt(99, 2), prompt(98, 3)]
        original, native = Model(stochastic=True), Model(stochastic=True)
        with isolated_rng(EVAL_SEED):
            expected_tokens, expected_text = original_generate_fast(original, Tokenizer(), prompts)
        saved = rng_snapshot()
        tok = Tokenizer()
        with isolated_rng(EVAL_SEED):
            rows = generate_case(native, tok, prompts, occurrence=1)
        self.assertTrue(rng_equal(saved))
        self.assertEqual([row['padded_decode_token_ids'] for row in rows], expected_tokens)
        self.assertEqual([row['text'] for row in rows], expected_text)
        self.assertEqual([row['query'].shape[1] for row in native.calls], [96,1,1,1])
        self.assertEqual([row['mask'].shape[1] for row in native.calls], [96,97,98,99])
        self.assertEqual([row['past_length'] for row in native.calls], [0,96,97,98])
        self.assertTrue(all(row['query'].shape[0] == 3 and not row['grad'] for row in native.calls))
        self.assertEqual(sum(row['physical_forward_calls'] for row in rows), len(native.calls))
        self.assertEqual(sum(row['prefill_query_tokens'] for row in rows), 3*96)
        self.assertEqual(sum(row['decode_query_tokens'] for row in rows), 3*3)
        self.assertEqual(tok.calls, [(prompts, dict(padding=True, return_tensors='pt'))])
        self.assertTrue(all(kwargs == {} for _, kwargs in tok.decode_calls))

    def test_no_eos_stop_total100_no_extra_forward(self):
        model = Model(chosen=9)
        with isolated_rng(EVAL_SEED):
            rows = generate_case(model, Tokenizer(), [prompt(97)], occurrence=1)
        self.assertEqual(rows[0]['continuation_token_ids'], [9,9,9])
        self.assertEqual(rows[0]['stop_reason'], 'length_cap')
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(len(rows[0]['full_token_ids']), 100)
        self.assertFalse(model.config.use_cache)

    def test_longest_prompt_blocks_whole_case_without_truncation_or_tokenizer_mutation(self):
        model, tok = Model(), Tokenizer()
        with isolated_rng(EVAL_SEED):
            rows = generate_case(model, tok, [prompt(98), prompt(103)], occurrence=1)
        self.assertEqual(model.calls, [])
        self.assertEqual([row['continuation_token_count'] for row in rows], [0,0])
        self.assertEqual([row['input_token_count'] for row in rows], [98,103])
        self.assertEqual([len(row['padded_decode_token_ids']) for row in rows], [103,103])
        self.assertEqual(tok.padding_side, 'right')

    def test_global_stream_advances_across_cases_not_perprompt_seed(self):
        with isolated_rng(EVAL_SEED):
            first = generate_case(Model(stochastic=True), Tokenizer(), [prompt(95)]*4, occurrence=1)
            second = generate_case(Model(stochastic=True), Tokenizer(), [prompt(95)]*4, occurrence=2)
        with isolated_rng(EVAL_SEED):
            reset = generate_case(Model(stochastic=True), Tokenizer(), [prompt(95)]*4, occurrence=2)
        self.assertEqual([row['full_token_ids'] for row in first], [row['full_token_ids'] for row in reset])
        self.assertNotEqual([row['full_token_ids'] for row in second], [row['full_token_ids'] for row in reset])

    def test_empty_no_forward_and_native_cache_failure_is_typed_without_fallback(self):
        model, tok = Model(), Tokenizer()
        self.assertEqual(generate_case(model,tok,[],occurrence=1), [])
        self.assertEqual(tok.calls, [])
        model.failure = True
        with self.assertRaisesRegex(GenerationError, 'FORWARD_COMPATIBILITY:ValueError'):
            with isolated_rng(EVAL_SEED):
                generate_case(model,tok,[prompt(99)],occurrence=1)
        self.assertEqual(model.calls, [])

    def test_nonfinite_missing_cache_and_left_padding_are_typed_failures(self):
        class Nonfinite(Model):
            def forward(self, *args, **kwargs):
                result = super().forward(*args, **kwargs)
                result.logits[0,-1,0] = float('nan')
                return result
        class NoCache(Model):
            def forward(self, *args, **kwargs):
                result = super().forward(*args, **kwargs)
                result.past_key_values = None
                return result
        class LeftPad(Tokenizer):
            def __call__(self, *args, **kwargs):
                result = super().__call__(*args, **kwargs)
                result['attention_mask'][0] = result['attention_mask'][0].flip(0)
                return result
        for model, tok, reason in [(Nonfinite(),Tokenizer(),'NONFINITE_LOGITS'),
                (NoCache(),Tokenizer(),'KV_CACHE_MISSING'), (Model(),LeftPad(),'RIGHT_PADDED')]:
            with self.assertRaisesRegex(GenerationError,reason):
                with isolated_rng(EVAL_SEED):
                    generate_case(model,tok,[prompt(98),prompt(99)],occurrence=1)


if __name__ == '__main__':
    unittest.main()
