"""Method-internal position-0 rule for tokenizers without BOS (Qwen2.5, GPT-J).

A PRICE row whose subject's last token would sit at position 0 (the attention
sink) gets the model's document-boundary token prepended, so the subject is read
at position 1. This covers the canonical "{}" rewrite/key rows and the "{} is a"
KL row. Every row is kept; evaluation prompts (benchmark panels) are untouched.
Llama-3 never triggers the rule because BOS always occupies position 0.
"""
import hashlib
import inspect
from project.run_scripts.jlz_pilot import prompts
from project.run_scripts.jlz_realization import inputs
from project.run_scripts.jlz_realization.common import require

PREFIX='<|endoftext|>'
OLD_BASE='        base_prompts = [context.format(request["prompt"]) for context in flat_contexts]\n'
NEW_BASE=OLD_BASE+('        base_prompts = [_POS0_PREFIX + p if subject_last(tokenizer, p, request["subject"]) == 0'
                   ' else p for p in base_prompts]\n')
OLD_KL='        prompts = rewrite + ["{} is a"]\n'
NEW_KL=('        kl_prompt = "{} is a"\n'
        '        if subject_last(tokenizer, kl_prompt, request["subject"]) == 0:\n'
        '            kl_prompt = _POS0_PREFIX + kl_prompt\n'
        '        prompts = rewrite + [kl_prompt]\n')


def source():
    text=inspect.getsource(prompts.prepare)
    require(text.count(OLD_BASE)==1 and text.count(OLD_KL)==1,'POS0_PATCH_ANCHOR')
    return text.replace(OLD_BASE,NEW_BASE).replace(OLD_KL,NEW_KL).replace('def prepare(','def prepare_pos0(',1)


def build(prefix=PREFIX):
    namespace=dict(vars(prompts));namespace['_POS0_PREFIX']=prefix
    exec(compile(source(),'<price-position0-prepare>','exec'),namespace)
    return namespace['prepare_pos0']


def check_tokenizer(tokenizer,prefix=PREFIX):
    ids=tokenizer(prefix,add_special_tokens=False)['input_ids']
    require(len(ids)==1 and ids[0]!=tokenizer.pad_token_id,'POS0_PREFIX_SINGLE_NONPAD_TOKEN')
    probe=tokenizer(prefix+'Paris',add_special_tokens=False)['input_ids']
    require(probe==ids+tokenizer('Paris',add_special_tokens=False)['input_ids'],'POS0_PREFIX_SPLITS_CLEANLY')
    return ids[0]


def install(tokenizer,prefix=PREFIX):
    """Route every CounterFactAdapter.prepare in this process through the rule."""
    token=check_tokenizer(tokenizer,prefix)
    inputs.native_prepare=build(prefix)
    return dict(rule='SUBJECT_NEVER_AT_POSITION_0',prefix=prefix,prefix_id=token,
        rows='rewrite+key rows and KL row whose subject_last==0',evaluation_prompts='UNCHANGED',
        prepare_source_sha256=hashlib.sha256(source().encode()).hexdigest(),M1='OFF_NOT_NEEDED')
