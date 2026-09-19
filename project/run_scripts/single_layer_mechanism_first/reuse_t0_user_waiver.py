"""Exact completed T0 reuse with USER-approved FD-only diagnostic status.

This never changes a failed FD result, its grid, threshold or measured values.
All non-FD technical checks and every scientific method guard remain required.
"""
from copy import deepcopy
from pathlib import Path
from .config import NUMERIC
from .technical_repair import checked_json
from project.run_scripts.single_layer_edit_preserving_correction.common import sha, write

FD_WAIVER_ID='USER_FD_DIAGNOSTIC_ONLY_B1_20260920_V1'


def assess(checks,policy):
    if policy.get('id')!=FD_WAIVER_ID or policy.get('user_quote')!='통과할테니 task 이어서 진행해':
        raise ValueError('EXPLICIT_FD_ONLY_USER_APPROVAL_REQUIRED')
    if checks.get('pass_') is not False or checks.get('blocking')!=['pair_factor_AD_FD']:
        raise ValueError('EXACT_FD_ONLY_UNRESOLVED_T0')
    results=checks['results']
    expected={'native_dense_replay','Current_Q_geometry','reference_repeat','panel_gradient_basis',
              'pair_factor_AD_FD','Current_affine_physical_invariant','cross_term_STEP_CUM','restore'}
    if set(results)!=expected or any(v.get('pass_') is not True for k,v in results.items() if k!='pair_factor_AD_FD'):
        raise ValueError('NON_FD_CHECK_CANNOT_BE_WAIVED')
    rows=results['pair_factor_AD_FD']['rows']
    if len(rows)!=4 or {r['index'] for r in rows}!={57,125,332,337}:
        raise ValueError('EXACT_FIXED_REFERENCE_PANEL')
    for row in rows:
        if not (row['gradient_relative']<=NUMERIC['gradient_relative'] and
                row['scalar_absolute_gap']<=NUMERIC['noop_logits'] and
                row['coefficient_relative_gap']<=NUMERIC['gradient_relative'] and
                row['FD']['status'] in ('PASS','NO_RESOLVED_ADJACENT_WINDOW')):
            raise ValueError('NON_FD_PARITY_CANNOT_BE_WAIVED')
    result=deepcopy(checks)
    result.update(status='B1_AUTHORIZED_FD_DIAGNOSTIC_ONLY',pass_=False,
        user_authorized_B1=True,policy=policy,
        full_numerical_validation='NOT_ESTABLISHED',native_replay_actual_pass=True,
        original_T0_status=checks['status'],original_blocking=checks['blocking'],
        method_guards_changed=False,new_T0_GPU_calls=0)
    return result


def validate(lock):
    if lock.get('maximum_batch')!=1 or lock.get('sequential_authorized') is not False:
        raise ValueError('FD_WAIVER_B1_ONLY')
    reuse=lock['completed_T0_reuse'];old=checked_json(reuse['execution_lock'])
    if old['execution']['commit']!='e3f92b43788d2491cad5b77328eb0229ccb69ca7' or reuse['job_id']!='51057':
        raise ValueError('EXACT_COMPLETED_T0_SOURCE')
    failure=checked_json(reuse['failure'])
    if failure['error']!="RuntimeError('T0_FULL_CHECKS_NOT_ESTABLISHED')":
        raise ValueError('T0_NOT_A_SERIALIZATION_OR_RUNTIME_ERROR')
    for key in ('numerical','records_digest','sample_order','cold_capsule','config4','projector',
                'P_star_basis','generated_ready','reference_inputs','editor_sha256','model_revision',
                'torch','numpy','scipy','transformers','transformers_import','external_members'):
        if old[key]!=lock[key]:raise ValueError('T0_REUSE_INPUT_CHANGED:'+key)
    old_source=Path(old['execution']['source']);here=Path(__file__).resolve().parents[3]
    names=('single_layer_mechanism_first/technical_decision.py','single_layer_mechanism_first/basis.py',
           'single_layer_mechanism_first/config.py','single_layer_mechanism_first/model.py',
           'single_layer_mechanism_first/current.py','single_layer_mechanism_first/decision.py',
           'single_layer_mechanism_first/z_hook.py','single_layer_mechanism_first/native.py')
    # Full dependency namespaces used by the measured all-token/geometry path.
    paths=[Path('project/run_scripts')/name for name in names]
    for namespace in ('en_execution_reuse','single_layer_edit_preserving_correction'):
        paths.extend(p.relative_to(old_source) for p in
                     (old_source/'project/run_scripts'/namespace).rglob('*.py'))
    for rel in paths:
        if sha(old_source/rel)!=sha(here/rel):raise ValueError('T0_REUSE_IMPLEMENTATION_CHANGED:'+str(rel))
    return assess(checked_json(reuse['checks']),lock['FD_user_waiver'])


def reuse(rt,out):
    result=validate(rt.lock)
    old_runtime=checked_json(rt.lock['completed_T0_reuse']['runtime'])
    if old_runtime['identity']!=rt.identity:raise ValueError('T0_REUSE_ACTUAL_RUNTIME_MISMATCH')
    result.update(reused_job='51057',original_members=rt.lock['completed_T0_reuse'])
    write(Path(out)/'receipt.json',result)
    return result
