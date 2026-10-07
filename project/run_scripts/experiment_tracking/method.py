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
GENERATION_PREFIXES=(*PREFIXES,'w0/current','w0/all_seen')
GENERATION_REASONS=('missing_generation_prompts','missing_reference','zero_generated_vector',
                    'zero_reference_vector','nonfinite_score','length_cap_no_continuation')
GENERATION_COUNTS=('planned_count','fluency_count','consistency_count','generation_prompt_count',
                   'generated_token_count',*(f'missing_{r}_count' for r in GENERATION_REASONS))
GENERATION_METRICS={f'{p}/generation/{f}' for p in GENERATION_PREFIXES for f in GENERATION_COUNTS}
GENERATION_METRICS.update(f'{p}/{f}' for p in GENERATION_PREFIXES
                         for f in ('fluency/ngram_entropy','consistency/reference_score'))
METHOD_METRICS.update(GENERATION_METRICS)

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
    generation=[p for p in GENERATION_PREFIXES if any(k in GENERATION_METRICS and k.startswith(p+'/') for k in values)]
    for key,value in values.items():
        if key in METHOD_METRICS:check(type(value) in (int,float),'METHOD_NUMERIC_NOT_BOOL')
        if key.endswith('_pct'):check(0<=value<=100,'PERCENT_RANGE')
        if key in ('edits','batch','candidate','pre_state_edits','post_state_edits','fit/global_candidate') or key.endswith(('/count','/success_count')):
            check(type(value) is int and value>=0,'INTEGER_COUNT_OR_AXIS')
        if any(key==g+'/'+f for g in GROUPS for f in ('true_nll','new_nll')):
            check(value>=0,'NLL_NONNEGATIVE')
    performance=bool(groups) or bool(generation) or any(p+'/success_harmonic_pct' in values for p in PREFIXES)
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
    for p in generation:
        planned=values.get(p+'/generation/planned_count')
        check(type(planned) is int and planned>0,'GENERATION_PLANNED_COUNT_REQUIRED')
        for f in GENERATION_COUNTS:
            k=p+'/generation/'+f
            if k in values:check(type(values[k]) is int and values[k]>=0,'GENERATION_INTEGER_COUNT')
        for f,metric in [('fluency_count','fluency/ngram_entropy'),('consistency_count','consistency/reference_score')]:
            count=values.get(p+'/generation/'+f)
            check(type(count) is int and 0<=count<=planned,'GENERATION_VALID_COUNT_REQUIRED')
            mean=values.get(p+'/'+metric)
            check((count==0 and mean is None) or (count>0 and mean is not None),'GENERATION_MISSING_MEAN_COUNT')
            if mean is not None:
                # Native FP64 dot/norm cosine may exceed mathematical 1 by a
                # few representable endpoint ULPs. Preserve the raw scalar;
                # this is not a score clamp or scientific quality tolerance.
                upper=1+4*math.ulp(1.)
                check(mean>=0 and (metric!='consistency/reference_score' or mean<=upper),'GENERATION_RAW_UNIT_RANGE')
    if any(k.startswith('current/pre/') for k in values):
        check('pre_state_edits' in values and 'post_state_edits' in values,'PRE_STATE_REQUIRED')
        check(values['pre_state_edits']<=values['post_state_edits']==values['edits'],'PRE_STATE_AXIS')
    if any(k.startswith(('current/post/','all_seen/post/')) for k in values):
        check(values.get('post_state_edits')==values['edits'],'POST_STATE_AXIS')
    if any(k.startswith('W0_first2000/') for k in values):
        check(values['edits']==0,'W0_STATE_ZERO')
        if any(g.startswith('W0_first2000/') for g in groups):
            check(all(values.get(f'W0_first2000/{k}/count')==n for k,n in [('R',2000),('P',4000),('N',20000)]),'W0_EXACT_FIRST2000_COUNTS')
        if 'W0_first2000' in generation:
            check(values['W0_first2000/generation/planned_count']==2000,'W0_GENERATION_EXACT_FIRST2000')
    if scientific:
        check(not any(k.startswith('eval/') for k in values),'LEGACY_EVAL_AMBIGUOUS_FOR_METHOD')
        if any(k.startswith(('fit/','optimizer/')) for k in values):
            check(all(k in values for k in ('fit/global_candidate','batch','candidate')),'FIT_AXIS_REQUIRED')
    return values

def define_axes(run):
    run.define_metric('edits')
    run.define_metric('fit/global_candidate')
    for prefix in dict.fromkeys((*PREFIXES,*NEIGHBOR_PREFIXES,*GENERATION_PREFIXES)):
        run.define_metric(prefix+'/*',step_metric='edits',step_sync=False)
    run.define_metric('fit/*',step_metric='fit/global_candidate',step_sync=False)
    run.define_metric('optimizer/*',step_metric='fit/global_candidate',step_sync=False)
    run.define_metric('generation_progress/step')
    run.define_metric('generation_progress/*',step_metric='generation_progress/step',step_sync=False)

class AxisState:
    def __init__(self):self.edits=None;self.fit=None;self.generation=None
    def check(self,values):
        for key,attr in [('edits','edits'),('fit/global_candidate','fit'),('generation_progress/step','generation')]:
            if key in values:
                previous=getattr(self,attr)
                check(previous is None or values[key]>=previous,'AXIS_DECREASE:'+key)
    def accept(self,values):
        self.check(values)
        for key,attr in [('edits','edits'),('fit/global_candidate','fit'),('generation_progress/step','generation')]:
            if key in values:setattr(self,attr,values[key])
