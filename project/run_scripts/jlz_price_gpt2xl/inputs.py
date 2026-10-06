"""Native EasyEdit context preparation once, then immutable six-cell inputs.

Only the named native functions are loaded, avoiding EasyEdit's unrelated
package imports. Their exact source bytes are bound before submission.
No reference fit, model update, or history append is performed here.
"""
import ast
import contextlib
import io
import json
import time
import typing
import unicodedata
from pathlib import Path
import torch
from .common import *

def function(path,name,namespace):
    source=Path(path).read_text();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

def augment(c,ready):
    for field in ('native_input_alignment','native_full_input_binding','observer_identity','contexts_member'):
        verify(ready[field])
    require(ready['model_asset_identity']==c['model_asset_identity'],'READY_MODEL_IDENTITY')
    require(ready['ordered_ids_sha256']==c['ordered_ids_sha256'],'READY_ORDER')
    c.update({k:ready[k] for k in ('packs','contexts','native_input_alignment',
        'native_full_input_binding','observer_identity','observation_identity')})
    c['input_ready_sha256']=digest(ready)
    return c

def bind(c,attempt,model,tok,records):
    folder=attempt/'inputs';path=folder/'ready.json'
    if path.exists():return augment(c,json.loads(path.read_text()))
    require(c['cell']=='MEMIT_CAP075','NATIVE_CONTEXT_READY_MISSING')
    require(not folder.exists(),'NATIVE_CONTEXT_PARTIAL_NO_AUTORETRY')
    folder.mkdir()
    from project.run_scripts.jlz_realization.writer import rng_snapshot,rng_restore
    from project.run_scripts.jlz_interference_l1.cap_prepare import native_binding
    rng=rng_snapshot();started=time.monotonic();calls=[]
    for row in c['context_sources']:verify(row)
    ns=dict(torch=torch,unicodedata=unicodedata,List=typing.List,Optional=typing.Optional,
            AutoModelForCausalLM=object,AutoTokenizer=object,CONTEXT_TEMPLATES_CACHE=None)
    generate=function(c['context_sources'][0]['path'],'generate_fast',ns)
    def counted(*args,**kwargs):
        calls.append(dict(prompts=5,n_gen_per_prompt=kwargs['n_gen_per_prompt'],max_out_len=kwargs['max_out_len']))
        return generate(*args,**kwargs)
    ns['generate_fast']=counted
    native=function(c['context_sources'][1]['path'],'get_context_templates',ns)
    physical=[0]
    hook=model.register_forward_pre_hook(lambda *_:physical.__setitem__(0,physical[0]+1))
    try:
        with contextlib.redirect_stdout(io.StringIO()):contexts=native(model,tok)
    finally:
        hook.remove();rng_restore(rng)
    require(len(contexts)==2 and contexts[0]==['{}'] and len(contexts[1])==5,'NATIVE_1_PLUS_5_CONTEXTS')
    require(calls==[dict(prompts=5,n_gen_per_prompt=1,max_out_len=10)],'NATIVE_CONTEXT_CALL_BUDGET')
    write(folder/'contexts.json',contexts)
    packs,bindings,identities=native_binding(c['model'],folder/'contexts.json',records,folder)
    require(max(v['max_owner_width'] for v in bindings)<=c['resource_binding']['max_owner_width'],
            'RESOURCE_BLOCKED_NATIVE_PACK_WIDTH_NO_TRUNCATION')
    require(max(v['max_owner_width'] for v in bindings)<=1024,'GPT2_NATIVE_POSITION_LIMIT')
    require(max(v['owner_padded_tokens'] for v in bindings)<=c['resource_binding']['max_owner_padded_tokens'],
            'RESOURCE_BLOCKED_NATIVE_TOTAL_TOKENS_NO_TRUNCATION')
    ready=dict(model_asset_identity=c['model_asset_identity'],ordered_ids_sha256=c['ordered_ids_sha256'],
        contexts=str(folder/'contexts.json'),contexts_member=member(folder/'contexts.json'),packs=packs,
        native_input_alignment=member(folder/'native-input-alignment.json'),
        native_full_input_binding=member(folder/'native-full-input-binding.json'),
        observer_identity=member(folder/'observer-identity.json'),
        observation_identity=digest([c['model_asset_identity'],c['runtime'],identities,c['cold_W0_H0'],
            c['evaluator_sources']]),native_context_physical_forwards=physical[0],
        native_context_calls=calls,input_preparation_seconds=time.monotonic()-started,
        reference_fits=0,updates=0,history_appends=0,RNG_restored=True,raw_context_local_only=True)
    write(path,ready)
    return augment(c,ready)
