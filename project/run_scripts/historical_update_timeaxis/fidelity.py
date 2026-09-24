"""Task-local user-directed numerical diagnostics; structural checks still fail closed."""
import math
from .common import digest, read, sha

POLICY = 'RECORD_ONLY_USER_DIRECTED'
CERTIFICATION = 'NOT_ESTABLISHED'
NONCE = 'ODEEDIT-GH-SH4-HISTORICAL-TIMEAXIS-RECORD-ONLY-20260925-R1'
NLL_LIMIT = 2.5e-4
MARGIN_LIMIT = 5e-4
COMPLETE = ('COMPLETED', 'COMPLETED_WITH_NUMERICAL_WARNINGS')
ROW_IDENTITY = ('case_id', 'kind', 'prompt_index', 'prompt', 'target', 'target_token_ids')


def policy(lock):
    p = lock['numerical_policy']
    assert p['numerical_fidelity_policy'] == POLICY
    assert p['numerical_certification'] == CERTIFICATION and p['instruction_id'] == NONCE
    assert p['nll_limit'] == NLL_LIMIT and p['margin_limit'] == MARGIN_LIMIT
    assert sha(p['waiver']['path']) == p['waiver']['sha256'], 'WAIVER_IDENTITY'
    return dict(numerical_fidelity_policy=POLICY, numerical_certification=CERTIFICATION,
                numerical_policy_sha256=digest(p), waiver_sha256=p['waiver']['sha256'])


def execution_identity(lock):
    return {**{k:lock[k] for k in ('instruction_id', 'attempt', 'source_sha256',
             'contract_sha256', 'T0_sha256', 'token_sha256')}, **policy(lock)}


def check_gate(receipt, identity):
    assert receipt['identity'] == identity and receipt['status'] == 'PASS', 'GATE_IDENTITY'
    assert receipt['pass_semantics'] == 'STRUCTURAL_COMPLETION_NOT_NUMERICAL_CERTIFICATION'
    assert receipt['numerical_certification'] == CERTIFICATION
    assert receipt['numerical_fidelity_policy'] == POLICY
    assert receipt['waiver_sha256'] == identity['waiver_sha256']
    return True  # numerical warnings are deliberately not a failure condition


def completion(warnings):
    return 'COMPLETED_WITH_NUMERICAL_WARNINGS' if warnings else 'COMPLETED'


def json_evidence(value):
    """Retain nonfinite evidence as explicit strings before raising, never coerce to zero."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': repr(value)}
    if isinstance(value, dict):return {k:json_evidence(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):return [json_evidence(v) for v in value]
    return value


def validate_rows(rows):
    keys = [(r['case_id'], r['kind'], r['prompt_index']) for r in rows]
    assert len(keys) == len(set(keys)), 'DUPLICATE_ROWS'
    for r in rows:
        assert math.isfinite(r['nll']), 'NONFINITE_NLL'
        labels, preds = r['target_token_ids'], r['token_predictions']
        assert len(labels) == len(preds) > 0, 'TOKEN_CARDINALITY'
        correct = [a == b for a,b in zip(labels, preds, strict=True)]
        assert correct == r['token_correct'] and all(correct) == r['all_tokens_correct'], 'TF_ROW_INTEGRITY'


def compare(a, b):
    assert len(a) == len(b), 'ROW_CARDINALITY'
    validate_rows(a);validate_rows(b)
    errors = []
    for x,y in zip(a,b,strict=True):
        assert all(x[k] == y[k] for k in ROW_IDENTITY), 'ROW_IDENTITY'
        errors.append(abs(x['nll']-y['nll']))
    return max(errors, default=0.)


def compare_raw(left, right):
    assert set(left) == set(right) == {k+'_target_'+s for k in ('rewrite','rephrase') for s in ('new','true')}, 'RAW_CATEGORIES'
    maxn = maxm = 0.; nr = []; margins = []; flips = []; warnings = []
    for category in left:
        maxn = max(maxn, compare(left[category], right[category]))
        for x,y in zip(left[category],right[category],strict=True):
            error = abs(x['nll']-y['nll'])
            row = dict(category=category,case_id=x['case_id'],prompt_index=x['prompt_index'],
                       left=x['nll'],right=y['nll'],absolute_error=error,limit=NLL_LIMIT,
                       excess=max(0.,error-NLL_LIMIT))
            nr.append(row)
            if error>NLL_LIMIT:warnings.append(dict(code='NLL_FIDELITY',**row))
    for kind in ('rewrite','rephrase'):
        for n,t,nn,tt in zip(left[kind+'_target_new'],left[kind+'_target_true'],right[kind+'_target_new'],right[kind+'_target_true'],strict=True):
            assert (n['case_id'],n['prompt_index'],n['prompt']) == (t['case_id'],t['prompt_index'],t['prompt']), 'NEW_TRUE_IDENTITY'
            m, mm = t['nll']-n['nll'], tt['nll']-nn['nll']
            assert math.isfinite(m) and math.isfinite(mm), 'NONFINITE_MARGIN'
            error = abs(m-mm);maxm = max(maxm,error)
            row = dict(kind=kind,case_id=n['case_id'],prompt_index=n['prompt_index'],left=m,right=mm,
                       absolute_error=error,limit=MARGIN_LIMIT,excess=max(0.,error-MARGIN_LIMIT))
            margins.append(row)
            if error>MARGIN_LIMIT:warnings.append(dict(code='MARGIN_FIDELITY',**row))
            if (m>0)!=(mm>0):
                f = dict(**row,within_original_boundary=abs(m)<=MARGIN_LIMIT and abs(mm)<=MARGIN_LIMIT)
                flips.append(f);warnings.append(dict(code='BOUNDARY_FLIP' if f['within_original_boundary'] else 'NONBOUNDARY_FLIP',**f))
    return dict(max_nll=maxn,max_margin=maxm,boundary_flips=flips,nll_rows=nr,margin_rows=margins,
                warnings=warnings,warning_count=len(warnings),diagnostic_status='WARNING' if warnings else 'NO_RECORDED_EXCEEDANCE',
                numerical_fidelity_policy=POLICY,numerical_certification=CERTIFICATION)


def historical_row(n,t,old,token):
    assert old['identity'] == token['pair_identity'], 'ORIGINAL_PAIR_IDENTITY'
    for value in (n['nll'],t['nll'],old['new_nll'],old['true_nll'],old['margin']):
        assert math.isfinite(value), 'NONFINITE_HISTORICAL_NLL'
    m=t['nll']-n['nll'];dn=abs(n['nll']-old['new_nll']);dt=abs(t['nll']-old['true_nll']);dm=abs(m-old['margin'])
    assert bool(old['success']) == (old['margin']>0), 'ORIGINAL_SUCCESS_INTEGRITY'
    flip=(m>0)!=old['success'];within=abs(m)<=MARGIN_LIMIT and abs(old['margin'])<=MARGIN_LIMIT
    warnings=[]
    if max(dn,dt)>NLL_LIMIT or dm>MARGIN_LIMIT:warnings.append('HISTORICAL_FIDELITY')
    if flip:warnings.append('ORIGINAL_BOUNDARY_FLIP' if within else 'ORIGINAL_NONBOUNDARY_FLIP')
    return dict(case_id=n['case_id'],panel=n['kind'],prompt_index=n['prompt_index'],new_error=dn,true_error=dt,
                margin_error=dm,nll_limit=NLL_LIMIT,margin_limit=MARGIN_LIMIT,nll_excess=max(0.,max(dn,dt)-NLL_LIMIT),
                margin_excess=max(0.,dm-MARGIN_LIMIT),boundary_flip=flip,within_original_boundary=within,
                actual_new=n['nll'],actual_true=t['nll'],original_new=old['new_nll'],original_true=old['true_nll'],
                original_margin=old['margin'],original_identity=old['identity'],warnings=warnings,warning_count=len(warnings))
