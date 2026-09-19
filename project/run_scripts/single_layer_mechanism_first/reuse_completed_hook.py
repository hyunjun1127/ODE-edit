"""Reuse the completed bounded hook checks, not a failed whole T0 PASS.

The only change to the measured numerical function is removal of a second
oracle unregister after Runtime.reset cleared it. The old error stays sealed.
New process W0/input/source identities must match before remaining T0 begins.
"""
import ast
from copy import deepcopy
from pathlib import Path
import torch
from .technical_repair import checked_json,validate_reference,assess_hook
from .z_hook import normalize_requests,prepare_batch
from project.run_scripts.single_layer_edit_preserving_correction.common import sha,write


def assert_cleanup_only_change(old_text,new_text):
    def functions(text):
        return {n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
    old,new=functions(old_text),functions(new_text)
    if set(old)!=set(new):raise ValueError('HOOK_REUSE_FUNCTION_SET_CHANGED')
    before=old['run_hook_repair'];after=new['run_hook_repair']
    legacy=ast.parse('rt.reset();rt.oracles.remove(oracle)').body
    fixed=ast.parse('rt.reset()').body
    dump=lambda value:ast.dump(value,include_attributes=False)
    if ([dump(n) for n in before.body[-1].finalbody]!=[dump(n) for n in legacy] or
            [dump(n) for n in after.body[-1].finalbody]!=[dump(n) for n in fixed]):
        raise ValueError('HOOK_REUSE_CLEANUP_DELTA_NOT_EXACT')
    normalized=deepcopy(before);normalized.body[-1].finalbody=deepcopy(fixed)
    old['run_hook_repair']=normalized
    if any(dump(old[name])!=dump(new[name]) for name in old):
        raise ValueError('HOOK_REUSE_NUMERICAL_CODE_CHANGED')


def check_completed_evidence(lock):
    """CPU-only binding and re-selection of already saved technical checks."""
    reuse=lock['completed_hook_reuse'];previous=checked_json(reuse['execution_lock'])
    if previous['execution']['commit']!=reuse['source'] or reuse['job_id']!='51055':
        raise ValueError('EXACT_COMPLETED_HOOK_SOURCE')
    failure=checked_json(reuse['failure'])
    if (failure['error']!="ValueError('list.remove(x): x not in list')" or
            'rt.reset();rt.oracles.remove(oracle)' not in failure['traceback']):
        raise ValueError('EXACT_CLEANUP_FAILURE_REQUIRED')
    for key in ('hook_repair','hook_gate_policy','numerical','records_digest','sample_order',
                'cold_capsule','config4','projector','editor_sha256','model_revision'):
        if previous[key]!=lock[key]:raise ValueError('COMPLETED_HOOK_INPUT_POLICY_CHANGED:'+key)
    current=Path(__file__).parent;old=Path(previous['execution']['source'])/'project/run_scripts/single_layer_mechanism_first'
    if set(reuse['source_members'])!={'z_hook.py','z_hook_parity.py','config.py','model.py',
                                    'technical.py','technical_repair.py','current.py'}:
        raise ValueError('COMPLETED_HOOK_SOURCE_COVERAGE')
    checks={}
    for name,item in reuse['source_members'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError('COMPLETED_HOOK_SOURCE_MEMBER_CHANGED')
        if name=='technical_repair.py':
            assert_cleanup_only_change(Path(item['path']).read_text(),(current/name).read_text())
            checks[name]='ONLY_DOUBLE_UNREGISTER_REMOVED_AST_CHECKED'
        else:
            if sha(current/name)!=item['sha256']:raise ValueError('COMPLETED_HOOK_IMPLEMENTATION_CHANGED:'+name)
            checks[name]='EXACT_SHA256'
        if Path(item['path']).resolve()!=(old/name).resolve():raise ValueError('COMPLETED_HOOK_SOURCE_PATH')
    summary=checked_json(reuse['summary'])
    assessment=assess_hook(summary['hook']['batch1_comparison'],summary['write']['batch1'],lock['hook_gate_policy'])
    if (assessment!=summary['gate_assessment'] or not assessment['pass_'] or not summary['pass_'] or
            summary['configured_science_batch_size']!=1 or summary['history_appends']!=0):
        raise ValueError('COMPLETED_HOOK_NOT_VALID')
    return previous,summary,checks


def reuse_completed_hook(rt,out):
    previous,summary,checks=check_completed_evidence(rt.lock)
    reused=rt.lock['completed_hook_reuse']
    if checked_json(reused['runtime'])['identity']!=rt.identity:
        raise ValueError('COMPLETED_HOOK_FRESH_RUNTIME_MISMATCH')
    native,records,binding=validate_reference(rt,rt.lock['hook_repair'])
    identities=[prepare_batch(rt.tok,[r],rt.context,rt.hp,rt.module.find_fact_lookup_idx,torch.device('cpu'))['identity']
                for r in normalize_requests(rt.requests(records))]
    if identities!=binding['binding']['batch_input_identities']:
        raise ValueError('COMPLETED_HOOK_ACTUAL_TOKENIZATION_CHANGED')
    # No continuation of the failed process: restore/guard a fresh W0 process.
    restored=rt.reset()
    write(Path(out)/'receipt.json',dict(job_id='51055',source=reused['source'],summary=reused['summary'],
        old_failure_preserved=reused['failure'],code_compatibility=checks,gate_assessment=summary['gate_assessment'],
        actual_write=summary['write']['batch1'],fresh_runtime_identity=rt.identity,restored=restored,
        new_hook_z_calls=0,new_hook_write_solves=0,new_hook_forwards=0,history_appends=0,
        validation_scope='FOUR_REQUEST_CACHE_BATCH1_HOOK_COMPONENT_ONLY; REMAINING_FULL_T0_REQUIRED',
        old_process_cleanup='FAILED_NOT_RETROACTIVELY_PASS',checkpoint_saved=False,exact_resume='NOT_AVAILABLE'))
    return native,records
