"""CAKE 0b378234 generate_fast, observed without changing its batch semantics.

The endpoint owner seeds and isolates RNG once. This function intentionally
advances that global stream: it never makes per-prompt/device generators. The
CAKE cumulative mask already supports incremental cached queries; unlike BLUE's
query-only mask this needs no modern-transformers mask substitution.
"""
from collections.abc import Mapping
import inspect
import unicodedata

import torch

from .common import EVAL_SEED, GenerationError, require
from .generator import model_device
from .native_profile import PROFILE, ROUTE


def _compatible(call, reason):
    try:
        return call()
    except GenerationError:
        raise
    except Exception as error:
        raise GenerationError(reason + ':' + type(error).__name__) from error


def _cache_length(cache, *, batch_size, device):
    """Validate only call-local full-context caches; never repair or repack one.

    Legacy native tuples remain supported. Modern Llama/Qwen native forwards
    return DynamicCache. Sliding/static/offloaded caches are not silently
    admitted by this full-prefix CAKE route.
    """
    dynamic = not isinstance(cache, (tuple, list))
    if dynamic:
        from transformers.cache_utils import DynamicCache
        require(type(cache) is DynamicCache and not getattr(cache, 'offloading', False),
                'NATIVE_GENERATION_KV_CACHE_CLASS_UNSUPPORTED')
    require(len(cache) > 0, 'NATIVE_GENERATION_KV_CACHE_EMPTY')
    lengths = []
    for layer in cache:
        require(isinstance(layer, (tuple, list)) and len(layer) == 2,
                'NATIVE_GENERATION_KV_LAYER_SCHEMA')
        key, value = layer
        require(all(torch.is_tensor(tensor) and tensor.ndim == 4
            and tensor.shape[0] == batch_size and tensor.device == device
            and tensor.dtype == torch.float32 for tensor in (key, value))
            and key.shape[-2] == value.shape[-2],
            'NATIVE_GENERATION_KV_SHAPE_DEVICE_OR_DTYPE')
        lengths.append(key.shape[-2])
    require(len(set(lengths)) == 1, 'NATIVE_GENERATION_KV_LAYER_LENGTH')
    if dynamic:
        require(cache.get_seq_length() == lengths[0], 'NATIVE_GENERATION_KV_DYNAMIC_LENGTH')
    return lengths[0]


def generate_case(model, tokenizer, prompts, *, occurrence, eval_seed=EVAL_SEED, trace=None):
    """Generate all prompts of one case together, using the caller's RNG stream.

Returns old observation-shaped rows plus native padded-decode evidence. Empty
cases do not tokenize/forward. No cache/no-EOS fallback or extra forward exists.
"""
    require(type(prompts) is list and all(type(prompt) is str for prompt in prompts),
            'NATIVE_GENERATION_PROMPTS_SCHEMA')
    require(type(occurrence) is int and occurrence >= 0, 'NATIVE_GENERATION_OCCURRENCE')
    require(eval_seed == EVAL_SEED, 'NATIVE_GENERATION_SEED_CHANGED')
    if not prompts:
        return []
    require(not model.training, 'NATIVE_GENERATION_MODEL_NOT_EVAL')
    require(all(not p.is_floating_point() or p.dtype == torch.float32 for p in model.parameters()),
            'NATIVE_GENERATION_MODEL_NOT_FP32')
    family = getattr(getattr(model, 'config', None), 'model_type', None)
    require(family in ('gpt2', 'gptj', 'llama', 'qwen2'),
            'NATIVE_GENERATION_MODEL_FAMILY_UNSUPPORTED')
    explicit_positions = family in ('llama', 'qwen2')
    if explicit_positions:
        arguments = inspect.signature(model.forward).parameters
        require('position_ids' in arguments and 'cache_position' in arguments,
                'NATIVE_GENERATION_POSITION_API_UNSUPPORTED')
    device = model_device(model)
    if device.type == 'cuda':
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                'NATIVE_GENERATION_TF32_ENABLED')
    require(type(getattr(tokenizer, 'pad_token_id', None)) is int and tokenizer.pad_token_id >= 0,
            'NATIVE_GENERATION_PAD_BINDING')
    encoded = _compatible(lambda: tokenizer(prompts, padding=True, return_tensors='pt'),
                          'NATIVE_GENERATION_TOKENIZER_COMPATIBILITY')
    require(isinstance(encoded, Mapping) and 'input_ids' in encoded and 'attention_mask' in encoded,
            'NATIVE_GENERATION_INPUT_FIELDS')
    ids, mask = encoded['input_ids'], encoded['attention_mask']
    require(torch.is_tensor(ids) and torch.is_tensor(mask) and ids.ndim == 2
        and ids.shape[0] == len(prompts) and ids.shape[1] > 0 and mask.shape == ids.shape
        and ids.dtype in (torch.int32, torch.int64) and mask.dtype in (torch.int32, torch.int64)
        and bool((ids >= 0).all()) and bool(((mask == 0) | (mask == 1)).all()),
        'NATIVE_GENERATION_INPUT_SHAPE')
    lengths = mask.sum(1)
    positions = torch.arange(ids.shape[1], device=mask.device).unsqueeze(0)
    require(bool((lengths > 0).all()) and bool((mask == (positions < lengths[:, None])).all())
        and bool((ids[mask == 0] == tokenizer.pad_token_id).all()),
        'NATIVE_GENERATION_RIGHT_PADDED_INPUT_REQUIRED')
    input_tokens = [ids[index, :int(length)].tolist() for index, length in enumerate(lengths)]
    padded_inputs = ids.tolist()
    initial_width = ids.shape[1]
    ids, mask = ids.to(device=device, dtype=torch.long), mask.to(device=device, dtype=torch.long)
    batch_size = ids.size(0)
    past, context = None, slice(0, int(lengths.min().item()))
    forwards = prefill = decode = dense_work = 0
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        # Original stopping rule is padded batch width, including long prompts.
        while ids.size(1) < 100:
            current_pos = context.stop
            query, attention = ids[:, context], mask[:, :current_pos]
            history_length = 0 if past is None else _cache_length(
                past, batch_size=batch_size, device=device)
            require(history_length == context.start and attention.shape[1]
                == history_length + query.shape[1], 'NATIVE_GENERATION_CACHE_QUERY_POSITION')
            arguments = dict(input_ids=query, attention_mask=attention,
                past_key_values=past, use_cache=True)
            # The original traversal processes the shared unpadded prefix and
            # then one physical column at a time. Contiguous physical positions
            # therefore equal the native default positions; no cumsum/padding
            # position rewrite or generation helper is introduced.
            positions = (torch.arange(history_length, current_pos, device=device)
                         if explicit_positions or trace is not None else None)
            if explicit_positions:
                arguments.update(position_ids=positions.unsqueeze(0), cache_position=positions)
            output = _compatible(lambda: model(**arguments),
                                 'NATIVE_GENERATION_FORWARD_COMPATIBILITY')
            logits, new_past = getattr(output, 'logits', None), getattr(output, 'past_key_values', None)
            require(torch.is_tensor(logits) and logits.ndim == 3
                and logits.shape[:2] == query.shape and logits.shape[-1] >= 5
                and logits.dtype == torch.float32, 'NATIVE_GENERATION_LOGIT_SHAPE_OR_DTYPE')
            require(new_past is not None, 'NATIVE_GENERATION_KV_CACHE_MISSING')
            require(_cache_length(new_past, batch_size=batch_size, device=device) == current_pos,
                    'NATIVE_GENERATION_KV_CACHE_POSITION')
            last = logits[:, -1, :]
            require(bool(torch.isfinite(last).all()), 'NATIVE_GENERATION_NONFINITE_LOGITS')
            softmax = torch.nn.functional.softmax(last, dim=1)
            selected = torch.topk(softmax, 5, dim=1).indices
            probabilities = torch.gather(softmax, 1, selected)
            probabilities = probabilities / probabilities.sum(1)[:, None]
            require(bool(torch.isfinite(probabilities).all()) and bool((probabilities.sum(1) > 0).all()),
                    'NATIVE_GENERATION_NONFINITE_SAMPLING')
            samples = _compatible(lambda: torch.multinomial(probabilities, 1),
                                  'NATIVE_GENERATION_SAMPLING_COMPATIBILITY')
            new_tokens = torch.gather(selected, 1, samples)
            forwards += 1
            work = query.numel()
            if past is None:
                prefill += work
            else:
                decode += work
            dense_work += batch_size * current_pos
            if trace is not None:
                trace(dict(query_input_ids=query.detach().cpu().tolist(),
                    attention_mask=attention.detach().cpu().tolist(),
                    sampled_tokens=new_tokens.detach().cpu().tolist(),
                    context=[context.start, context.stop], use_cache=True,
                    past_present=past is not None, past_length=history_length,
                    position_ids=positions.unsqueeze(0).detach().cpu().tolist(),
                    cache_position=positions.detach().cpu().tolist(),
                    cache_position_explicit=explicit_positions))
            past = new_past
            if context.stop == ids.size(1):
                mask = torch.cat([mask, mask.new_zeros(batch_size, 1)], dim=1)
                ids = torch.cat([ids, ids.new_ones(batch_size, 1) * tokenizer.pad_token_id], dim=1)
            last_non_masked = mask.sum(1) - 1
            for index in range(batch_size):
                new_index = last_non_masked[index] + 1
                if last_non_masked[index].item() + 1 != context.stop:
                    continue
                if new_index < 100:
                    ids[index][new_index] = new_tokens[index]
                    mask[index][new_index] = 1
            context = slice(context.stop, context.stop + 1)
    padded_output, output_lengths = ids.detach().cpu().tolist(), mask.sum(1).detach().cpu().tolist()
    result = []
    for index, (prompt, inputs, padded, final_length) in enumerate(
            zip(prompts, input_tokens, padded_output, output_lengths)):
        full = padded[:final_length]
        require(full[:len(inputs)] == inputs, 'NATIVE_GENERATION_INPUT_OVERWRITTEN')
        continuation = full[len(inputs):]
        text = _compatible(lambda: tokenizer.decode(padded), 'NATIVE_GENERATION_DECODE_COMPATIBILITY')
        require(type(text) is str, 'NATIVE_GENERATION_DECODE_TYPE')
        # Exact CAKE decode cleanup; BLUE differs in skip_special_tokens=True.
        text = unicodedata.normalize('NFKD', text).replace('\n\n', ' ').replace('<|endoftext|>', '')
        # Attribute physical calls once per case; per-prompt logical forwards are
        # retained separately. Query token counts partition the batch exactly.
        result.append(dict(profile=PROFILE, route=ROUTE, seed=eval_seed,
            sampling_scope='ENDPOINT_GLOBAL_BATCH_STREAM', occurrence=occurrence, prompt_index=index,
            prompt=prompt, input_token_ids=inputs, continuation_token_ids=continuation,
            full_token_ids=full, padded_input_token_ids=padded_inputs[index],
            padded_decode_token_ids=padded, initial_batch_width=initial_width,
            case_batch_prompt_count=batch_size, text=text, input_token_count=len(inputs),
            continuation_token_count=len(continuation),
            stop_reason='length_cap_no_continuation' if initial_width >= 100 else 'length_cap',
            EOS_stop=False, model_forwards=forwards,
            physical_forward_calls=forwards if index == 0 else 0,
            prefill_query_tokens=prefill // batch_size,
            decode_query_tokens=decode // batch_size,
            full_prefix_token_work=dense_work // batch_size,
            sampling=dict(top_k=5, temperature=1, top_p=1, max_total_tokens=100,
                          n_gen_per_prompt=1), endpoint_RNG_restore_guard_required=True))
    return result
