"""Unpadded FP32 reference and explicitly qualified temporary KV routes.

The reference remains the old semantic profile. Optimized routes do not edit
model config and use one independently advancing device generator per prompt.
"""
from contextlib import contextmanager
import copy
import inspect
import random
import unicodedata

import numpy as np
import torch

from .common import EVAL_SEED, PROFILE, case_seed, require

REFERENCE_ROUTE = 'UNPADDED_FULL_PREFIX_NO_CACHE'
SINGLETON_ROUTE = 'UNPADDED_KV_SINGLETON'
BATCH_ROUTE = 'EQUAL_LENGTH_KV_BATCH'
ALLOWED_ROUTES = (REFERENCE_ROUTE, SINGLETON_ROUTE, BATCH_ROUTE)


def rng_snapshot():
    return dict(python=random.getstate(), numpy=copy.deepcopy(np.random.get_state()),
        cpu=torch.random.get_rng_state().clone(),
        cuda=[state.clone() for state in torch.cuda.get_rng_state_all()]
             if torch.cuda.is_initialized() else None)


def rng_restore(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def rng_equal(state):
    now = rng_snapshot()
    return (now['python'] == state['python'] and now['numpy'][0] == state['numpy'][0]
        and np.array_equal(now['numpy'][1], state['numpy'][1])
        and now['numpy'][2:] == state['numpy'][2:]
        and torch.equal(now['cpu'], state['cpu'])
        and ((now['cuda'] is None and state['cuda'] is None) or
             (now['cuda'] is not None and state['cuda'] is not None
              and len(now['cuda']) == len(state['cuda'])
              and all(torch.equal(a, b) for a, b in zip(now['cuda'], state['cuda'])))))


@contextmanager
def isolated_rng(seed=None):
    saved = rng_snapshot()
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            # Do not initialize a CUDA context merely for CPU scoring/tests.
            torch.random.default_generator.manual_seed(seed)
            if saved['cuda'] is not None:
                torch.cuda.manual_seed_all(seed)
        yield
    finally:
        rng_restore(saved)
        require(rng_equal(saved), 'GENERATION_RNG_RESTORE')


def native_eos(model, tokenizer):
    bindings = {}
    for name, owner in [('generation_config', getattr(model, 'generation_config', None)),
                        ('model_config', getattr(model, 'config', None)),
                        ('tokenizer', tokenizer)]:
        value = getattr(owner, 'eos_token_id', None)
        if value is not None:
            values = list(value) if isinstance(value, (list, tuple, set)) else [value]
            require(all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in values),
                    'GENERATION_EOS_BINDING')
            bindings[name] = values
    return sorted({x for values in bindings.values() for x in values}), bindings


def normalize_decode(text):
    return unicodedata.normalize('NFKD', text).replace('\n\n', ' ')


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(getattr(model, 'device', 'cpu'))


def _generate_row(model, tokenizer, prompt, *, model_identity, occurrence,
                 prompt_index, eval_seed=EVAL_SEED, max_total_tokens=100,
                 top_k=5, temperature=1, top_p=1, trace=None):
    """One full prompt + continuation. trace is a CPU qualification callback only.

    Every actual forward sees precisely one unpadded prefix and its full mask.
    No persistent hooks, KV caches, config edits, beam wrappers or chat template.
    """
    require(isinstance(prompt, str), 'GENERATION_PROMPT_TYPE')
    require(max_total_tokens == 100 and top_k == 5 and temperature == 1 and top_p == 1,
            'GENERATION_PROFILE_CHANGED')
    require(not model.training, 'GENERATION_MODEL_NOT_EVAL')
    require(all(not p.is_floating_point() or p.dtype == torch.float32 for p in model.parameters()),
            'GENERATION_MODEL_NOT_FP32')
    device = model_device(model)
    if device.type == 'cuda':
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                'GENERATION_TF32_ENABLED')
    encoded = tokenizer(prompt, return_tensors='pt', padding=False, truncation=False)
    ids = encoded['input_ids']
    mask = encoded.get('attention_mask', torch.ones_like(ids))
    require(ids.ndim == 2 and ids.shape[0] == 1 and ids.shape[1] > 0
        and ids.dtype in (torch.int32, torch.int64), 'GENERATION_INPUT_SHAPE')
    require(mask.shape == ids.shape and bool((mask == 1).all()), 'GENERATION_INPUT_NOT_UNPADDED')
    input_tokens = ids[0].tolist()
    ids = ids.to(device=device, dtype=torch.long)
    eos, eos_binding = native_eos(model, tokenizer)
    seed = case_seed(model_identity, occurrence, prompt_index, eval_seed)
    continuation = []
    forwards = 0
    token_work = 0
    reason = 'length_cap_no_continuation' if len(input_tokens) >= max_total_tokens else 'length_cap'
    with isolated_rng(seed), torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        while ids.shape[1] < max_total_tokens:
            attention = torch.ones_like(ids)
            result = model(input_ids=ids, attention_mask=attention, use_cache=False,
                           return_dict=True)
            logits = result.logits
            require(logits.ndim == 3 and logits.shape[:2] == ids.shape
                and logits.shape[-1] >= top_k and logits.dtype == torch.float32,
                'GENERATION_LOGIT_SHAPE_OR_DTYPE')
            last = logits[:, -1, :]
            require(bool(torch.isfinite(last).all()), 'GENERATION_NONFINITE_LOGITS')
            probs = torch.softmax(last, dim=1)
            # Exact CAKE selection sequence, with the declared per-row repairs.
            selected = torch.topk(probs, top_k, dim=1).indices
            selected_probs = torch.gather(probs, 1, selected)
            selected_probs = selected_probs / selected_probs.sum(1, keepdim=True)
            sampled = torch.multinomial(selected_probs, 1)
            token = int(torch.gather(selected, 1, sampled).item())
            forwards += 1
            token_work += ids.shape[1]
            if trace is not None:
                trace(dict(input_ids=ids.detach().cpu().tolist(),
                    attention_mask=attention.detach().cpu().tolist(),
                    logits=last.detach().cpu().clone(), sampled_token=token,
                    topk_ids=selected.detach().cpu().clone(),
                    topk_probabilities=selected_probs.detach().cpu().clone(),
                    position_ids=torch.arange(ids.shape[1]).view(1, -1),
                    query_input_ids=ids.detach().cpu().tolist(),
                    occurrence=occurrence, prompt_index=prompt_index,
                    use_cache=False))
            continuation.append(token)
            ids = torch.cat([ids, ids.new_tensor([[token]])], dim=1)
            if token in eos:
                reason = 'eos'
                break
    full_tokens = input_tokens + continuation
    require(full_tokens[:len(input_tokens)] == input_tokens, 'GENERATION_INPUT_OVERWRITTEN')
    require(not continuation or len(full_tokens) <= max_total_tokens, 'GENERATION_LENGTH_OVERFLOW')
    text = normalize_decode(tokenizer.decode(full_tokens, skip_special_tokens=True))
    return dict(profile=PROFILE, seed=seed, occurrence=occurrence, prompt_index=prompt_index,
        prompt=prompt, input_token_ids=input_tokens, continuation_token_ids=continuation,
        full_token_ids=full_tokens, text=text, input_token_count=len(input_tokens),
        continuation_token_count=len(continuation), stop_reason=reason,
        eos_ids=eos, eos_binding=eos_binding, model_forwards=forwards,
        full_prefix_token_work=token_work, route='UNPADDED_FULL_PREFIX_NO_CACHE',
        physical_forward_calls=forwards, prefill_query_tokens=token_work,
        decode_query_tokens=0,
        sampling=dict(top_k=top_k, temperature=temperature, top_p=top_p,
                      max_total_tokens=max_total_tokens), RNG_restored=True)


def generate_row(model, tokenizer, prompt, *, model_identity, occurrence,
                 prompt_index, eval_seed=EVAL_SEED, **kwargs):
    # Covers tokenizer/decode exceptions as well as forward/sampling failures.
    seed = case_seed(model_identity, occurrence, prompt_index, eval_seed)
    with isolated_rng(seed):
        return _generate_row(model, tokenizer, prompt, model_identity=model_identity,
            occurrence=occurrence, prompt_index=prompt_index, eval_seed=eval_seed, **kwargs)


def _encoded_tokens(tokenizer, prompt):
    require(isinstance(prompt, str), 'GENERATION_PROMPT_TYPE')
    encoded = tokenizer(prompt, return_tensors='pt', padding=False, truncation=False)
    ids = encoded['input_ids']
    mask = encoded.get('attention_mask', torch.ones_like(ids))
    require(ids.ndim == 2 and ids.shape[0] == 1 and ids.shape[1] > 0
        and ids.dtype in (torch.int32, torch.int64), 'GENERATION_INPUT_SHAPE')
    require(mask.shape == ids.shape and bool((mask == 1).all()), 'GENERATION_INPUT_NOT_UNPADDED')
    return ids[0].tolist()


def _gather_past(past, indices, batch_size):
    """Gather call-local legacy tuples or the actual native DynamicCache.

    The public native DynamicCache selector replaces only its RAM key/value
    tensors; this object is never attached to model config/hooks/buffers.
    """
    dynamic = False
    if not isinstance(past, (tuple, list)):
        from transformers.cache_utils import DynamicCache
        require(type(past) is DynamicCache and not getattr(past, 'offloading', False),
                'GENERATION_KV_CACHE_CLASS_UNQUALIFIED')
        require(callable(getattr(past, 'batch_select_indices', None))
            and callable(getattr(past, 'get_seq_length', None)), 'GENERATION_KV_DYNAMIC_API')
        dynamic = True
    require(len(past) > 0, 'GENERATION_KV_CACHE_SCHEMA')
    result = []
    for layer in past:
        require(isinstance(layer, (tuple, list)) and len(layer) == 2,
                'GENERATION_KV_LAYER_SCHEMA')
        values = []
        for tensor in layer:
            require(torch.is_tensor(tensor) and tensor.ndim == 4
                and tensor.shape[0] == batch_size, 'GENERATION_KV_BATCH_AXIS')
            values.append(tensor if dynamic else tensor.index_select(0, indices.to(tensor.device)))
        result.append(tuple(values))
    if dynamic:
        before_length = past.get_seq_length()
        if indices.numel() != batch_size or not torch.equal(indices, torch.arange(batch_size, device=indices.device)):
            past.batch_select_indices(indices)
        require(past.get_seq_length() == before_length
            and all(key.shape[0] == indices.numel() and value.shape[0] == indices.numel()
                    for key, value in past), 'GENERATION_KV_DYNAMIC_GATHER')
        return past
    return tuple(result)


def _generate_kv_bucket(model, tokenizer, requests, indices, *, route, eval_seed,
                        trace=None, forced_prefixes=None):
    require(not model.training, 'GENERATION_MODEL_NOT_EVAL')
    require(all(not p.is_floating_point() or p.dtype == torch.float32 for p in model.parameters()),
            'GENERATION_MODEL_NOT_FP32')
    require(getattr(getattr(model, 'config', None), 'model_type', None) in ('gpt2', 'gptj'),
            'GENERATION_KV_MODEL_FAMILY_UNQUALIFIED')
    device = model_device(model)
    if device.type == 'cuda':
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                'GENERATION_TF32_ENABLED')
    inputs = [_encoded_tokens(tokenizer, request['prompt']) for request in requests]
    require(len({len(tokens) for tokens in inputs}) == 1, 'GENERATION_KV_UNEQUAL_LENGTH')
    eos, eos_binding = native_eos(model, tokenizer)
    seeds = [case_seed(request['model_identity'], request['occurrence'], request['prompt_index'],
                       eval_seed) for request in requests]
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    samples = [[] for _ in requests]
    actual_prefixes = [list(tokens) for tokens in inputs]
    reasons = ['length_cap_no_continuation' if len(tokens) >= 100 else 'length_cap'
               for tokens in inputs]
    forwards, physical, prefill, decode = ([0] * len(requests) for _ in range(4))
    active = list(range(len(requests))) if len(inputs[0]) < 100 else []
    query = torch.tensor(inputs, dtype=torch.long, device=device) if active else None
    past = None
    cache_class = None
    use_cache_position = 'cache_position' in inspect.signature(model.forward).parameters
    history_length = 0
    # Global edit RNG is restored even if native forward or a callback raises.
    with isolated_rng(), torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        while active:
            batch_size, query_length = query.shape
            total_length = history_length + query_length
            attention = torch.ones((batch_size, total_length), dtype=torch.long, device=device)
            positions = torch.arange(history_length, total_length, device=device).expand(batch_size, -1)
            arguments = dict(input_ids=query, attention_mask=attention, position_ids=positions,
                             past_key_values=past, use_cache=True, return_dict=True)
            if use_cache_position:
                arguments['cache_position'] = torch.arange(history_length, total_length, device=device)
            result = model(**arguments)
            updated_past = getattr(result, 'past_key_values', None)
            cache_class = (type(updated_past).__module__+'.'+type(updated_past).__name__)
            logits = result.logits
            require(logits.ndim == 3 and logits.shape[:2] == query.shape
                and logits.shape[-1] >= 5 and logits.dtype == torch.float32,
                'GENERATION_LOGIT_SHAPE_OR_DTYPE')
            last = logits[:, -1, :]
            require(bool(torch.isfinite(last).all()), 'GENERATION_NONFINITE_LOGITS')
            # This is full-vocabulary softmax, NOT a softmax only over logits top5.
            probabilities = torch.softmax(last, dim=1)
            top_ids = torch.topk(probabilities, 5, dim=1).indices
            top_probs = torch.gather(probabilities, 1, top_ids)
            top_probs = top_probs / top_probs.sum(1, keepdim=True)
            physical[active[0]] += 1
            if past is None:
                prefill[active[0]] += query.numel()
            else:
                decode[active[0]] += query.numel()
            survivors, next_tokens = [], []
            for slot, row in enumerate(active):
                # [1,5] multinomial shape preserves the original per-row operation.
                choice = torch.multinomial(top_probs[slot:slot+1], 1, generator=generators[row])
                sampled = int(torch.gather(top_ids[slot:slot+1], 1, choice).item())
                forwards[row] += 1
                if trace is not None:
                    trace(dict(request_index=indices[row], occurrence=requests[row]['occurrence'],
                        prompt_index=requests[row]['prompt_index'],
                        input_ids=[list(actual_prefixes[row])],
                        query_input_ids=query[slot:slot+1].detach().cpu().tolist(),
                        attention_mask=attention[slot:slot+1].detach().cpu().tolist(),
                        position_ids=positions[slot:slot+1].detach().cpu().clone(),
                        cache_position=torch.arange(history_length, total_length),
                        cache_position_explicit=use_cache_position, cache_class=cache_class,
                        logits=last[slot:slot+1].detach().cpu().clone(),
                        topk_ids=top_ids[slot:slot+1].detach().cpu().clone(),
                        topk_probabilities=top_probs[slot:slot+1].detach().cpu().clone(),
                        sampled_token=sampled, use_cache=True))
                samples[row].append(sampled)
                token = sampled
                if forced_prefixes is not None:
                    forced = forced_prefixes[indices[row]]
                    require(len(samples[row]) <= len(forced), 'GENERATION_FORCED_TRACE_LENGTH')
                    token = int(forced[len(samples[row])-1])
                actual_prefixes[row].append(token)
                if token in eos:
                    reasons[row] = 'eos'
                elif len(actual_prefixes[row]) < 100:
                    survivors.append(slot)
                    next_tokens.append(token)
            if not survivors:
                break
            selection = torch.tensor(survivors, device=device, dtype=torch.long)
            past = _gather_past(updated_past, selection, batch_size)
            active = [active[slot] for slot in survivors]
            history_length = total_length
            query = torch.tensor(next_tokens, dtype=torch.long, device=device).view(-1, 1)
    observations = []
    for row, request in enumerate(requests):
        full = inputs[row] + samples[row]
        require(len(inputs[row]) >= 100 or len(full) <= 100, 'GENERATION_LENGTH_OVERFLOW')
        value = dict(profile=PROFILE, seed=seeds[row], occurrence=request['occurrence'],
            prompt_index=request['prompt_index'], prompt=request['prompt'], input_token_ids=inputs[row],
            continuation_token_ids=samples[row], full_token_ids=full,
            text=normalize_decode(tokenizer.decode(full, skip_special_tokens=True)),
            input_token_count=len(inputs[row]), continuation_token_count=len(samples[row]),
            stop_reason=reasons[row], eos_ids=eos, eos_binding=eos_binding,
            model_forwards=forwards[row], full_prefix_token_work=0, route=route,
            physical_forward_calls=physical[row], prefill_query_tokens=prefill[row],
            decode_query_tokens=decode[row],
            cache_class=cache_class, cache_position_explicit=use_cache_position,
            sampling=dict(top_k=5, temperature=1, top_p=1, max_total_tokens=100), RNG_restored=True)
        if forced_prefixes is not None:
            value['qualification_forced_prefix_only'] = True
            value['forced_prefix_token_ids'] = actual_prefixes[row]
        observations.append(value)
    return observations


def generate_rows(model, tokenizer, requests, *, route=REFERENCE_ROUTE, microbatch=1,
                  eval_seed=EVAL_SEED, trace=None, forced_prefixes=None):
    """Return rows in original request order; grouping changes physical work only.

    Requests contain prompt/model_identity/occurrence/prompt_index. This API does
    not certify a route; the production caller must bind its actual receipt.
    forced_prefixes is permitted solely for one-shot qualification diagnostics.
    """
    requests = list(requests)
    require(route in ALLOWED_ROUTES, 'GENERATION_ROUTE_UNQUALIFIED')
    require(isinstance(microbatch, int) and not isinstance(microbatch, bool)
        and (microbatch == 1 if route != BATCH_ROUTE else microbatch in (4, 8)),
        'GENERATION_MICROBATCH_UNQUALIFIED')
    require(eval_seed == EVAL_SEED, 'GENERATION_SEED_CHANGED')
    if route == REFERENCE_ROUTE:
        require(forced_prefixes is None, 'GENERATION_REFERENCE_FORCED_PREFIX_FORBIDDEN')
        return [generate_row(model, tokenizer, request['prompt'],
            model_identity=request['model_identity'], occurrence=request['occurrence'],
            prompt_index=request['prompt_index'], eval_seed=eval_seed, trace=trace)
            for request in requests]
    # Encode without padding/truncation; do not reorder/deduplicate returned rows.
    buckets = {}
    for index, request in enumerate(requests):
        size = len(_encoded_tokens(tokenizer, request['prompt']))
        key = size if route == BATCH_ROUTE else index
        buckets.setdefault(key, []).append(index)
    output = [None] * len(requests)
    for bucket in buckets.values():
        for offset in range(0, len(bucket), microbatch):
            indices = bucket[offset:offset+microbatch]
            values = _generate_kv_bucket(model, tokenizer, [requests[i] for i in indices], indices,
                route=route, eval_seed=eval_seed, trace=trace, forced_prefixes=forced_prefixes)
            for index, value in zip(indices, values):
                output[index] = value
    return output
