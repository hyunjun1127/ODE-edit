"""Model-scoped native subject-position repair, not an evaluator transform."""
from official.ours.config import require_config

PREFIX = '<|endoftext|>'
PREFIX_IDS = {'qwen2': 151643, 'gptj': 50256}

def policy(config):
    if config is None:
        return None
    cfg = require_config(config)
    model = cfg['model_type']
    if model not in PREFIX_IDS:
        return None
    if cfg.get('price_m1_anchor_guard') is not False:
        raise ValueError('POS0_REQUIRES_NATIVE_ANCHOR_NOT_M1')
    return model

def repair_prompt(tokenizer, prompt, subject, model, lookup):
    if model is None or lookup(tokenizer, prompt, subject) != 0:
        return prompt
    token = PREFIX_IDS[model]
    if tokenizer.encode(PREFIX, add_special_tokens=False) != [token]:
        raise ValueError('POS0_PREFIX_TOKEN_IDENTITY')
    new = PREFIX + prompt
    if tokenizer.encode(new.format(subject)) != [token] + tokenizer.encode(prompt.format(subject)):
        raise ValueError('POS0_PREFIX_RETOKENIZATION')
    if lookup(tokenizer, new, subject) != 1:
        raise ValueError('POS0_SUBJECT_SHIFT')
    # The valid prefix may equal PAD (GPT-J). Never derive masks from IDs.
    return new

def check_entry(profile, pack):
    model = profile['model_type']
    if model not in PREFIX_IDS:
        return
    if profile.get('price_m1_anchor_guard') is not False:
        raise ValueError('POS0_REQUIRES_NATIVE_ANCHOR_NOT_M1')
    if pack.get('pos0_model_type') != model:
        raise ValueError('POS0_PACK_CONFIG_REQUIRED')
    for kind, lookups in (('tokens', pack['lookup']), ('key_tokens', pack['key_lookup'])):
        for row, col in enumerate(lookups):
            if col <= 0 or not bool(pack[kind]['attention_mask'][row, col]):
                raise ValueError('POS0_LOOKUP_OR_MASK')
