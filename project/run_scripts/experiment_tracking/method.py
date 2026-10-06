"""Bounded scalar contract; no tensor conversion, evaluator or network access."""
import math

COMPARISON_SCHEMA='price-first2k-scalar-v1'
PREFIXES=('current/pre','current/post','all_seen/post','W0_first2000')
NEIGHBOR_PREFIXES=('w0/current/N','w0/all_seen/N')
FIELDS=('count','success_count','success_pct','token_acc_pct','prompt_acc_pct',
        'strict_acc_pct','true_nll','new_nll','margin_true_minus_new')
METHOD_CONFIG={'model','model_family','writer','role','metric_schema'}
GROUPS=tuple(f'{p}/{k}' for p in PREFIXES for k in 'RPN')+NEIGHBOR_PREFIXES
METHOD_METRICS={f'{g}/{f}' for g in GROUPS for f in FIELDS}
METHOD_METRICS.update(p+'/success_harmonic_pct' for p in PREFIXES)
METHOD_METRICS.update(('edits','batch','candidate','pre_state_edits','post_state_edits','fit/global_candidate'))

def check(ok,code):
    if not ok:raise ValueError(code)

def harmonic(values,unit='percent'):
    """None means unavailable; any missing component wins over a measured zero."""
    check(len(values)==3 and unit in ('percent','fraction'),'HARMONIC_INPUT')
    if any(v is None for v in values):return None
    top=100 if unit=='percent' else 1
    check(all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=top for v in values),'HARMONIC_RANGE')
    if any(v==0 for v in values):return 0.
    return (3 if unit=='percent' else 300)/sum(1/v for v in values)

def close(a,b):return math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8)

def validate(values,scientific=False):
    groups=[g for g in GROUPS if any(k.startswith(g+'/') for k in values)]
    for key,value in values.items():
        if key in METHOD_METRICS:check(type(value) in (int,float),'METHOD_NUMERIC_NOT_BOOL')
        if key.endswith('_pct'):check(0<=value<=100,'PERCENT_RANGE')
        if key in ('edits','batch','candidate','pre_state_edits','post_state_edits','fit/global_candidate') or key.endswith(('/count','/success_count')):
            check(type(value) is int and value>=0,'INTEGER_COUNT_OR_AXIS')
        if any(key==g+'/'+f for g in GROUPS for f in ('true_nll','new_nll')):
            check(value>=0,'NLL_NONNEGATIVE')
    performance=bool(groups) or any(p+'/success_harmonic_pct' in values for p in PREFIXES)
    if performance:
        check('edits' in values,'EVAL_EDITS_REQUIRED')
    for g in groups:
        item={k:values[g+'/'+k] for k in FIELDS if g+'/'+k in values}
        if 'count' in item:
            check(item['count']>0,'EMPTY_POPULATION_OMIT')
            if 'success_count' in item:check(item['success_count']<=item['count'],'SUCCESS_COUNT_RANGE')
        if {'count','success_count','success_pct'}<=item.keys():
            check(close(item['success_pct'],100*item['success_count']/item['count']),'SUCCESS_ARITHMETIC')
        if {'true_nll','new_nll','margin_true_minus_new'}<=item.keys():
            check(close(item['margin_true_minus_new'],item['true_nll']-item['new_nll']),'MARGIN_SIGN')
    for p in PREFIXES:
        if p+'/success_harmonic_pct' in values:
            expected=harmonic([values.get(f'{p}/{k}/success_pct') for k in 'RPN'])
            check(expected is not None and close(values[p+'/success_harmonic_pct'],expected),'HARMONIC_MISSING_OR_MISMATCH')
    if any(k.startswith('current/pre/') for k in values):
        check('pre_state_edits' in values and 'post_state_edits' in values,'PRE_STATE_REQUIRED')
        check(values['pre_state_edits']<=values['post_state_edits']==values['edits'],'PRE_STATE_AXIS')
    if any(k.startswith(('current/post/','all_seen/post/')) for k in values):
        check(values.get('post_state_edits')==values['edits'],'POST_STATE_AXIS')
    if any(k.startswith('W0_first2000/') for k in values):
        check(values['edits']==0,'W0_STATE_ZERO')
        check(all(values.get(f'W0_first2000/{k}/count')==n for k,n in [('R',2000),('P',4000),('N',20000)]),'W0_EXACT_FIRST2000_COUNTS')
    if scientific:
        check(not any(k.startswith('eval/') for k in values),'LEGACY_EVAL_AMBIGUOUS_FOR_METHOD')
        if any(k.startswith(('fit/','optimizer/')) for k in values):
            check(all(k in values for k in ('fit/global_candidate','batch','candidate')),'FIT_AXIS_REQUIRED')
    return values

def define_axes(run):
    run.define_metric('edits')
    run.define_metric('fit/global_candidate')
    for prefix in (*PREFIXES,*NEIGHBOR_PREFIXES):
        run.define_metric(prefix+'/*',step_metric='edits',step_sync=False)
    run.define_metric('fit/*',step_metric='fit/global_candidate',step_sync=False)
    run.define_metric('optimizer/*',step_metric='fit/global_candidate',step_sync=False)

class AxisState:
    def __init__(self):self.edits=None;self.fit=None
    def check(self,values):
        for key,attr in [('edits','edits'),('fit/global_candidate','fit')]:
            if key in values:
                previous=getattr(self,attr)
                check(previous is None or values[key]>=previous,'AXIS_DECREASE:'+key)
    def accept(self,values):
        self.check(values)
        for key,attr in [('edits','edits'),('fit/global_candidate','fit')]:
            if key in values:setattr(self,attr,values[key])
