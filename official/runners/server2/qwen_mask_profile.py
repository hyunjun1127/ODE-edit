"""Explicit new cold attempt overlay; canonical native scientific fields unchanged."""
from copy import deepcopy
from pathlib import Path
from official.experiments.prepare import digest, file_sha, write_new

INSTRUCTION = 'USER-GH-QWEN-BASELINE-MASK-COLD-RERUN-20261010-R1'
GENERATOR_SHA = '35506690c41ecb7d59f11660da41dde50338f5a2ba9613735e045a8c7ab98db4'
DEFERRED = 'DEFERRED_CHECKPOINT_EVALUATION'
METHODS = ('MEMIT', 'ALPHAEDIT', 'MEMIT_FE', 'SPHERE')

def rows():
    from official.runners.server2.qwen_plan import rows as canonical
    result=[]
    for old in canonical():
        if old['config']['method'] not in METHODS: continue
        row=deepcopy(old); c=row['config']; c.pop('config_sha256')
        c['mask_repair_instruction']=INSTRUCTION
        c['context_generator_sha256']=GENERATOR_SHA
        if c['dataset']=='cf': c['generation_schedule']=DEFERRED
        c['config_sha256']=digest(c)
        result.append(row)
    return result

def validate(config):
    if config not in [r['config'] for r in rows()]:
        raise ValueError('MASK_REPAIR_EXACT_CONFIG_REQUIRED')
    return config

def deferred(config):
    return config.get('generation_schedule')==DEFERRED

def cold_guard(config, native, resume):
    validate(config)
    if resume or native.context_snapshot() is not None:
        raise ValueError('MASK_REPAIR_REQUIRES_FRESH_COLD_CONTEXT')
    from official.baselines.easyedit.util import generate
    if file_sha(generate.__file__)!=GENERATOR_SHA:
        raise ValueError('MASK_REPAIR_GENERATOR_SHA')

def context_receipt(out, config, native, source, batch):
    if batch!=1: return
    context=native.context_snapshot()
    if not context: raise ValueError('FIRST_NATIVE_CONTEXT_NOT_RECORDED')
    # Text stays local only; the public report carries its hash, never its text.
    path=Path(out)/'native-context-first-call.json'
    write_new(path,dict(context=context))
    write_new(Path(out)/'native-context-identity.json',dict(
        instruction_id=INSTRUCTION,source_sha=source['code_commit'],
        config_sha256=config['config_sha256'],generator_sha256=GENERATOR_SHA,
        native_module=native.module.__name__,native_module_sha256=file_sha(native.module.__file__),
        path=str(path),context_sha256=file_sha(path),cold_context=True,
        additional_context_generation_calls=0,observed_after_actual_batch=1))
