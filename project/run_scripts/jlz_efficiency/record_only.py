"""Explicit USER policy, limited to the fixed B100 E123_MB4 comparison.

Historical comparisons remain unchanged; finite closeness is not certification.
No optimizer, checkpoint, model loading, or hidden reference calls here.
"""
import json
import math
from pathlib import Path

NONCE = 'ODEEDIT-GH-SH1-JLZ-EFFICIENCY-NUMERICAL-RECORD-ONLY-20261001-R1'
AUTHORITY = 'audits/global/2026-10-01-jlz-efficiency-sh1-dispatch/numerical-record-only-r1-instruction.json'
AUTHORITY_SHA = 'dd6430dc2c30fe15c5782cf660a3459a1b3be8eb772a765fc6248b4911c667c7'


def finite(value):
    if isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, (tuple, list)):
        for v in value: finite(v)
    elif isinstance(value, (float, int)) and not math.isfinite(value):
        raise FloatingPointError('RECORD_ONLY_NONFINITE_FATAL')


def hard(ok, reason):
    if not ok: raise RuntimeError('RECORD_ONLY_HARD_' + reason)


def annotate(comparison, errors=None, limits=None):
    finite(comparison)
    errors = comparison.get('errors', {}) if errors is None else errors
    limits = comparison.get('limits', {}) if limits is None else limits
    finite(errors); finite(limits)
    warning = comparison['status'] != 'PASS'
    return comparison | dict(
        original_status=comparison['status'],
        original_verdict='FAIL' if warning else 'PASS',
        numerical_policy='RECORD_ONLY_USER_DIRECTED',
        numerical_certification='NOT_ESTABLISHED',
        effective_action='CONTINUE_WITH_WARNING' if warning else 'CONTINUE',
        authority_nonce=NONCE,
        threshold_records={k: dict(actual=v, threshold=limits[k],
            excess=max(0., v-limits[k]), ratio=v/limits[k] if limits[k] else None,
            original_verdict='FAIL' if v>limits[k] else 'PASS') for k,v in errors.items()},
        location_policy='block index maps layer4..8 then request order; component/request argmax NOT_RECORDED unless supplied')


def entry_decision(comparison):
    # Both paths feed the same reference teacher/active/adj. A different key
    # violates the retained key identity boundary; finite entry errors do not.
    hard(comparison['key_bitwise'], 'KEY_IDENTITY')
    errors={k:comparison[k] for k in ('anchor_maxabs','teacher_maxabs','nll_maxabs')}
    return annotate(comparison, errors, {k:1e-4 for k in errors})


def point_decision(comparison, point):
    finite(point)
    hard(point['feasible'], 'FEASIBILITY')
    hard(comparison.get('same_materialization', comparison.get('reference', False)), 'MATERIALIZATION')
    value=annotate(comparison)
    if comparison.get('block_diff_norm'):
        ratios=[d/max(1.,r) for d,r in zip(comparison['block_diff_norm'],comparison['block_reference_norm'],strict=True)]
        index=max(range(len(ratios)),key=ratios.__getitem__)
        value['largest_block_error']=dict(flat_index=index,layer=4+index//100,request_index=index%100,relative=ratios[index])
    return value


def observer_identity(reference, candidate, expected=None):
    a,b=reference['rows'],candidate['rows']
    hard(len(a)==len(b), 'OBSERVER_DENOMINATOR')
    hard(len({r['identity'] for r in a})==len(a), 'DUPLICATE_REFERENCE')
    hard(len({r['identity'] for r in b})==len(b), 'DUPLICATE_CANDIDATE')
    if expected is not None:
        hard({k:sum(r['kind']==k for r in b) for k in expected}==expected, 'OBSERVER_KIND_DENOMINATOR')
    for x,y in zip(a,b,strict=True):
        hard(all(x[k]==y[k] for k in ('identity','case_id','kind','prompt_index')), 'OBSERVER_IDENTITY')
        finite(x); finite(y)
        for prefix in ('true','new'):
            hard(x[prefix+'_token_count']==y[prefix+'_token_count'], 'TARGET_TOKEN_COUNT')
            hard(len(x[prefix+'_predictions'])==len(y[prefix+'_predictions'])==x[prefix+'_token_count'], 'PREDICTION_SHAPE')


def observer_decision(comparison, reference, candidate):
    observer_identity(reference,candidate,{'R':100,'P':200,'N':1000})
    errors={k:comparison[k] for k in ('nll_maxabs','logit_maxabs','logit_rms')}
    result=annotate(comparison,errors,dict(nll_maxabs=1e-4,logit_maxabs=1e-3,logit_rms=1e-4))
    paired={}
    for kind in ('R','P','N'):
        def success(r): return r['true_nll']<r['new_nll'] if kind=='N' else r['new_nll']<r['true_nll']
        rows=[(a,b) for a,b in zip(reference['rows'],candidate['rows'],strict=True) if a['kind']==kind]
        paired[kind]=dict(denominator=len(rows),lost=sum(success(a) and not success(b) for a,b in rows),
            gained=sum(not success(a) and success(b) for a,b in rows),
            strict_changed=sum(a[('true' if kind=='N' else 'new')+'_strict']!=b[('true' if kind=='N' else 'new')+'_strict'] for a,b in rows))
    return result | dict(paired=paired)


def execute(model,tok,records,contexts,history,cache,budget,out,lock,large,state_hash,write):
    prior=Path(lock['record_only']['root'])
    oldruntime=json.loads((prior/'output/runtime.json').read_text())
    hard(state_hash(model,history)==oldruntime['entry'],'W0_H0_IDENTITY')
    selection=json.loads((prior/'output/selection.json').read_text())
    hard(selection['candidate']=='E123_MB4','FIXED_ROUTE')
    write(out/'reuse-receipt.json',dict(mode='B100_ONLY_NEW_COLD_RAM_PREPARATION_NOT_CRASH_RESUME',
        prior_attempt=str(prior),prior_GPU_seconds=794,new_small_oracles=0,new_native_requests=0,
        new_kernel_cases=0,old_warmups='HISTORICAL_ONLY_NOT_NEW_TIMING',
        reused='fixed/short/native/kernel/small-probe; old UNQUALIFIED unchanged',authority_nonce=NONCE))
    write(out/'selection.json',dict(candidate='E123_MB4',selection='FIXED_USER_OVERRIDE_NO_RESELECTION',numerical_certification='NOT_ESTABLISHED'))
    return large(model,tok,records,contexts,history,cache,budget,'E123_MB4',out,record_only=lock['record_only'])
