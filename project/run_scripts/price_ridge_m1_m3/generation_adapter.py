"""Task-local Llama family adapter over SH1's immutable KV implementation.

Only the family guard is extended (real model_type stays 'llama'). The native
cache/prefill/decode/position/sampling implementation is reused byte-for-byte.
No shared module global or source is modified. Actual post-edit qualification
must pass before this route is used for production observations.
"""
import ast
import hashlib
import inspect
import textwrap
import types

from project.run_scripts.experiment_generation_eval import generator as original
from project.run_scripts.experiment_generation_eval import kv_qualification as qualification
from project.run_scripts.experiment_generation_eval.observer import GenerationObserver as SharedObserver


def clone(function, **overrides):
    namespace=dict(function.__globals__); namespace.update(overrides)
    result=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def llama_bucket():
    source=inspect.getsource(original._generate_kv_bucket)
    tree=ast.parse(source);changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Tuple) and all(isinstance(x,ast.Constant) for x in node.elts):
            if [x.value for x in node.elts]==['gpt2','gptj']:
                node.elts.append(ast.Constant('llama'));changed+=1
    if changed!=1:raise RuntimeError('SHARED_GENERATOR_FAMILY_ADAPTER_SOURCE_CHANGED')
    namespace=dict(original.__dict__)
    exec(compile(ast.fix_missing_locations(tree),__file__+':llama_native_KV','exec'),namespace)
    return namespace['_generate_kv_bucket'],hashlib.sha256(source.encode()).hexdigest()


_bucket,ORIGINAL_BUCKET_SHA256=llama_bucket()
generate_rows=clone(original.generate_rows,_generate_kv_bucket=_bucket)
def qualified_family_adapter():
    tree=ast.parse(inspect.getsource(qualification.run_qualification));changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.keyword) and node.arg=='native_models_scope':
            if ast.literal_eval(node.value)!=['gpt2','gptj']:
                raise RuntimeError('QUALIFICATION_SCOPE_SOURCE_CHANGED')
            node.value.elts.append(ast.Constant('llama'));changed+=1
    if changed!=1:raise RuntimeError('QUALIFICATION_SCOPE_ADAPTER_REQUIRED')
    namespace=dict(qualification.run_qualification.__globals__,generate_rows=generate_rows)
    exec(compile(ast.fix_missing_locations(tree),__file__+':qualification','exec'),namespace)
    return namespace['run_qualification']


run_qualification=qualified_family_adapter()


def observer_method():
    # Shared observe() deliberately imports its backend locally. Rebind just
    # this import in the task-private subclass; do not mutate SH1's module.
    tree=ast.parse(textwrap.dedent(inspect.getsource(SharedObserver.observe)))
    changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module=='generator' and node.level==1:
            if [a.name for a in node.names]==['generate_rows']:
                node.module=__name__;node.level=0;changed+=1
    if changed!=1:raise RuntimeError('SHARED_OBSERVER_IMPORT_ADAPTER_SOURCE_CHANGED')
    namespace=dict(SharedObserver.observe.__globals__)
    exec(compile(ast.fix_missing_locations(tree),__file__+':observe','exec'),namespace)
    return namespace['observe']


def subset_method():
    """Repair only loop-variable shadowing; keep shared identity checks intact."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(SharedObserver.subset)))
    changed=0
    for node in ast.walk(tree):
        if (isinstance(node,ast.For) and isinstance(node.target,ast.Name)
                and node.target.id=='key' and isinstance(node.iter,ast.Tuple)
                and ast.literal_eval(node.iter)==('qualification_receipt_member','compatibility_member')):
            for name in ast.walk(node):
                if isinstance(name,ast.Name) and name.id=='key':name.id='member_key'
            changed+=1
    if changed!=1:raise RuntimeError('SHARED_SUBSET_REPAIR_SOURCE_CHANGED')
    namespace=dict(SharedObserver.subset.__globals__)
    exec(compile(ast.fix_missing_locations(tree),__file__+':subset','exec'),namespace)
    return namespace['subset']


class GenerationObserver(SharedObserver):
    observe=observer_method()
    subset=subset_method()
