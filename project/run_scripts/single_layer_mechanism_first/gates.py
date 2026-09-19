"""Independent scalar/ID stage reducer. Never called by a candidate selector."""
import math
from .config import NUMERIC


def reduce_observation(observation):
    """Recompute published NLL and TF indicators from saved pair rows."""
    cases=None;result={};bycase={}
    for kind,multiple in (('RS',1),('PS',2),('NS',10)):
        group=observation['metrics'][kind];rows=group['rows'];seen=set();order=[]
        if len(rows)!=observation['requests']*multiple or group['denominator']!=len(rows):
            raise ValueError('OBSERVATION_DENOMINATOR')
        for row in rows:
            if row['identity'] in seen:raise ValueError('DUPLICATE_PAIR_ID')
            seen.add(row['identity']);order.append((row['case_id'],row['prompt_index']))
            if not all(math.isfinite(row[k]) for k in ('new_nll','true_nll')):raise ValueError('NONFINITE_NLL')
            success=row['true_nll']<row['new_nll'] if kind=='NS' else row['new_nll']<row['true_nll']
            if type(row['success']) is not bool or row['success']!=success:raise ValueError('PAIR_ARITHMETIC')
            bycase.setdefault(row['case_id'],{}).setdefault(kind,[]).append(row)
        current=[r['case_id'] for r in rows[::multiple]]
        if len(set(current))!=observation['requests'] or order!=[(c,i) for c in current for i in range(multiple)]:
            raise ValueError('CASE_PROMPT_ORDER')
        if cases is None:cases=current
        if cases!=current:raise ValueError('PANEL_CASE_ORDER')
        successes={r['identity'] for r in rows if r['success']}
        if group['numerator']!=len(successes):raise ValueError('AGGREGATE_ARITHMETIC')
        result[kind]=dict(numerator=len(successes),denominator=len(rows),success_ids=sorted(successes),
                          all_ids=[r['identity'] for r in rows])
    fields=('rewrite_strict','two_P_strict','R_two_P_strict','R_two_P_NLL_joint')
    strict={k:[] for k in fields}
    for c in cases:
        r=bycase[c]['RS'][0];p=bycase[c]['PS']
        flags=(r['new_strict'],all(x['new_strict'] for x in p),
               r['new_strict'] and all(x['new_strict'] for x in p),
               r['success'] and all(x['success'] for x in p))
        for key,flag in zip(fields,flags):
            if flag:strict[key].append(c)
    for key,ids in strict.items():
        if observation['strict'][key]!=len(ids):raise ValueError('STRICT_AGGREGATE')
    result['strict']=strict;result['case_ids']=cases
    return result


def compare(base, candidate):
    a=reduce_observation(base);b=reduce_observation(candidate)
    if a['case_ids']!=b['case_ids']:raise ValueError('COMPARISON_CASE_ORDER')
    result={}
    for kind in ('RS','PS','NS'):
        if a[kind]['all_ids']!=b[kind]['all_ids']:raise ValueError('COMPARISON_PROMPT_TARGET_ID')
        old=set(a[kind]['success_ids']);new=set(b[kind]['success_ids'])
        result[kind]=dict(lost=sorted(old-new),gained=sorted(new-old),
                         point_count_difference=len(new)-len(old),denominator=a[kind]['denominator'])
    result['strict']={k:dict(lost=sorted(set(a['strict'][k])-set(b['strict'][k])),
        gained=sorted(set(b['strict'][k])-set(a['strict'][k])),
        point_count_difference=len(b['strict'][k])-len(a['strict'][k])) for k in a['strict']}
    return result


def b1_gate(native, primary, selection):
    paired=compare(native,primary)
    required=('rewrite_strict','two_P_strict','R_two_P_strict','R_two_P_NLL_joint')
    checks=dict(nonzero_valid_correction=selection.get('accepted') is True and selection.get('actual_delta_norm',0)>0,
        current=selection.get('current_pass') is True,full512=selection.get('full512') is True,
        choice_or_margin=selection.get('repaired_choices',0)>0 or
            (selection.get('phi_reference_native',0)>0 and
             selection.get('phi_reference_gain',0)>=.05*selection['phi_reference_native'] and
             selection['phi_reference_gain']>selection.get('tau_risk',math.inf)),
        PS_point_nonloss=paired['PS']['point_count_difference']>=0,
        strict_joint_point_nonloss=all(paired['strict'][k]['point_count_difference']>=0 for k in required))
    return dict(name='B1_TO_S3',pass_=all(checks.values()),**{'pass':all(checks.values())},checks=checks,
        paired=paired,primary='DEC_MODES_CUM',observer_used_only_for_stage_expansion=True,
        point_nonloss_not_statistical_noninferiority=True)


def s3_gate(native, primary, selection, w0_native_retention, w0_primary_retention):
    paired=compare(native,primary)
    ids=lambda rows:{r['identity'] for r in rows if r['W0_success'] and r['selected_success']}
    before,after=ids(w0_native_retention['rows']),ids(w0_primary_retention['rows'])
    if [(r['identity'],r['W0_success']) for r in w0_native_retention['rows']]!=[
            (r['identity'],r['W0_success']) for r in w0_primary_retention['rows']]:raise ValueError('W0_RETENTION_IDS')
    checks=dict(valid_conditions=all(selection.get(k) is True for k in ('current_pass','history_pass','reference_pass','full512')),
        PS_point_nonloss=paired['PS']['point_count_difference']>=0,
        strict_joint_point_nonloss=all(paired['strict'][k]['point_count_difference']>=0
             for k in ('two_P_strict','R_two_P_strict','R_two_P_NLL_joint')),
        W0_N_grossloss_or_recovery=w0_primary_retention['lost']<w0_native_retention['lost'] or bool(after-before))
    return dict(name='S3_TO_S10',**{'pass':all(checks.values())},checks=checks,paired=paired,
        W0_correct_N_recovered_ids=sorted(after-before),primary='DEC_MODES_CUM',
        observer_used_only_for_stage_expansion=True)
