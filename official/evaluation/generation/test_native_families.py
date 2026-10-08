"""Random tiny CPU models only; not pretrained/GPU generation qualification.

The numerical gates are fixed independently of the generated metrics. Reference
forwards below are diagnostic test work, never part of the production generator.
"""
import copy
import unittest

import torch

from .common import EVAL_SEED, GenerationError
from .generator import isolated_rng, rng_snapshot, rng_equal
from .native_generator import generate_case
from .test_native_generator import Model, Tokenizer, original_generate_fast, prompt


LOGIT_ATOL = 2e-5
LOGIT_RTOL = 2e-4
PROB_ATOL = 2e-6
PROB_RTOL = 2e-4


def tiny_model(family):
    from transformers import (GPT2Config, GPT2LMHeadModel, GPTJConfig, GPTJForCausalLM,
        LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM)
    if family == 'gpt2':
        config, implementation = GPT2Config(vocab_size=16, n_positions=128,
            n_ctx=128, n_embd=16, n_layer=2, n_head=2), GPT2LMHeadModel
    elif family == 'gptj':
        config, implementation = GPTJConfig(vocab_size=16, n_positions=128,
            n_embd=16, n_layer=2, n_head=2, rotary_dim=8), GPTJForCausalLM
    else:
        config_class, implementation = ((LlamaConfig, LlamaForCausalLM)
            if family == 'llama' else (Qwen2Config, Qwen2ForCausalLM))
        config = config_class(vocab_size=16, max_position_embeddings=128,
            hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2)
    config._attn_implementation = 'eager'
    config.pad_token_id = config.eos_token_id = 9
    with isolated_rng(0):
        return implementation(config).float().eval()


class PrefixParityModel(torch.nn.Module):
    """Check each consumed native prefix without influencing sampling RNG."""
    def __init__(self, base):
        super().__init__()
        self.base, self.config = base, base.config
        self.history, self.calls = None, []
        self.eval()

    def forward(self, input_ids, attention_mask, past_key_values, use_cache,
                position_ids=None, cache_position=None):
        full = input_ids if self.history is None else torch.cat([self.history, input_ids], dim=1)
        assert use_cache is True and attention_mask.shape == full.shape
        arguments = dict(input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache)
        if position_ids is not None:
            arguments['position_ids'] = position_ids
        if cache_position is not None:
            arguments['cache_position'] = cache_position
        result = self.base(**arguments)
        reference = self.base(input_ids=full, attention_mask=attention_mask, use_cache=False)
        actual_logits, reference_logits = result.logits[:, -1], reference.logits[:, -1]
        torch.testing.assert_close(actual_logits, reference_logits,
                                   atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
        actual_prob, reference_prob = torch.softmax(actual_logits, -1), torch.softmax(reference_logits, -1)
        actual_top, reference_top = torch.topk(actual_prob, 5, -1), torch.topk(reference_prob, 5, -1)
        assert torch.equal(actual_top.indices, reference_top.indices)
        torch.testing.assert_close(actual_top.values, reference_top.values,
                                   atol=PROB_ATOL, rtol=PROB_RTOL)
        self.calls.append(dict(query=input_ids.clone(), attention=attention_mask.clone(),
            positions=None if position_ids is None else position_ids.clone(),
            cache_position=None if cache_position is None else cache_position.clone(),
            total_length=full.shape[1]))
        self.history = full.clone()
        return result


class NativeFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Fixed tiny native family models use no downloaded model/tokenizer asset.
        try:
            import transformers
        except ImportError as error:
            raise unittest.SkipTest('transformers CPU fixtures unavailable') from error
        cls.transformers_version = transformers.__version__
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_four_native_families_cached_prefix_probabilities_original_tokens_and_state(self):
        prompts = [prompt(96), prompt(99, 2), prompt(98, 3)]
        for family in ('gpt2', 'gptj', 'llama', 'qwen2'):
            with self.subTest(family=family):
                base = tiny_model(family)
                original = copy.deepcopy(base)
                config_before = copy.deepcopy(base.config.to_dict())
                parameters_before = {name: parameter.detach().clone()
                                     for name, parameter in base.named_parameters()}
                hooks_before = {name: (len(module._forward_hooks), len(module._forward_pre_hooks))
                                for name, module in base.named_modules()}
                with isolated_rng(EVAL_SEED):
                    expected_tokens, expected_text = original_generate_fast(original, Tokenizer(), prompts)
                model, trace = PrefixParityModel(base), []
                saved = rng_snapshot()
                with isolated_rng(EVAL_SEED):
                    rows = generate_case(model, Tokenizer(), prompts, occurrence=1, trace=trace.append)
                self.assertTrue(rng_equal(saved))
                self.assertEqual([row['padded_decode_token_ids'] for row in rows], expected_tokens)
                self.assertEqual([row['text'] for row in rows], expected_text)
                self.assertEqual([call['query'].shape[1] for call in model.calls], [96, 1, 1, 1])
                self.assertEqual([call['total_length'] for call in model.calls], [96, 97, 98, 99])
                self.assertEqual([row['past_length'] for row in trace], [0, 96, 97, 98])
                for call, event in zip(model.calls, trace):
                    start, stop = event['context']
                    self.assertEqual(event['cache_position'], list(range(start, stop)))
                    self.assertEqual(event['position_ids'], [list(range(start, stop))])
                    self.assertTrue(bool((call['attention'] == 1).all()))
                    if family in ('llama', 'qwen2'):
                        self.assertTrue(event['cache_position_explicit'])
                        self.assertEqual(call['positions'].tolist(), event['position_ids'])
                        self.assertEqual(call['cache_position'].tolist(), event['cache_position'])
                    else:
                        self.assertFalse(event['cache_position_explicit'])
                        self.assertIsNone(call['positions'])
                        self.assertIsNone(call['cache_position'])
                self.assertEqual(base.config.to_dict(), config_before)
                self.assertEqual(hooks_before, {name: (len(module._forward_hooks), len(module._forward_pre_hooks))
                                               for name, module in base.named_modules()})
                for name, parameter in base.named_parameters():
                    self.assertTrue(torch.equal(parameter, parameters_before[name]))
                self.assertFalse(hasattr(base, '_cache'))
                self.assertEqual(sum(row['physical_forward_calls'] for row in rows), 4)

    def test_cache_position_length_and_unqualified_family_fail_closed(self):
        class WrongCacheLength(Model):
            def forward(self, *args, **kwargs):
                result = super().forward(*args, **kwargs)
                result.past_key_values = tuple((key[:, :, :-1], value[:, :, :-1])
                    for key, value in result.past_key_values)
                return result
        for model, reason in ((WrongCacheLength(), 'KV_CACHE_POSITION'),
                              (Model(family='unqualified'), 'MODEL_FAMILY_UNSUPPORTED'),
                              (Model(family='llama'), 'POSITION_API_UNSUPPORTED')):
            with self.subTest(reason=reason), self.assertRaisesRegex(GenerationError, reason):
                with isolated_rng(EVAL_SEED):
                    generate_case(model, Tokenizer(), [prompt(99)], occurrence=1)


if __name__ == '__main__':
    unittest.main()
