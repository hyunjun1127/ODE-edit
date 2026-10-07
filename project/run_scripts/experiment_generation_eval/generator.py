"""Declared unpadded no-cache FP32 reference, not legacy CAKE byte parity."""
from contextlib import contextmanager
import copy
import random
import unicodedata

import numpy as np
import torch

from .common import EVAL_SEED, PROFILE, case_seed, require


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
        sampling=dict(top_k=top_k, temperature=temperature, top_p=top_p,
                      max_total_tokens=max_total_tokens), RNG_restored=True)


def generate_row(model, tokenizer, prompt, *, model_identity, occurrence,
                 prompt_index, eval_seed=EVAL_SEED, **kwargs):
    # Covers tokenizer/decode exceptions as well as forward/sampling failures.
    seed = case_seed(model_identity, occurrence, prompt_index, eval_seed)
    with isolated_rng(seed):
        return _generate_row(model, tokenizer, prompt, model_identity=model_identity,
            occurrence=occurrence, prompt_index=prompt_index, eval_seed=eval_seed, **kwargs)
