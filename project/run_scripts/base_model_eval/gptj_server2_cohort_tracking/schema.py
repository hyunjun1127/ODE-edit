"""Authorized W0-reference-only schema. General edited-run validator is untouched."""
import math
from project.run_scripts.experiment_tracking import schema as base
from project.run_scripts.experiment_tracking.schema import (ENTITY, PROJECT, SDK_VERSION,
    load_env, identifier, endpoint, require, job_identity, run_name)
from project.run_scripts.experiment_tracking.method import FIELDS, harmonic

EXTRA = dict(reference_only=True, evaluation_model_state='W0',
    edits_axis_semantics='reference_cohort_progress', w0_reference_schema='w0-cohort-v1')
PREFIXES = ('W0_first2000', 'current/post', 'all_seen/post')
GROUPS = tuple(p+'/'+k for p in PREFIXES for k in 'RPN') + ('w0/current/N','w0/all_seen/N')
AXES = ('edits','reference_cohort_edits','actual_model_edits','actual_applied_edits',
        'pre_state_edits','post_state_edits')
ALLOWED = {g+'/'+f for g in GROUPS for f in FIELDS} | {p+'/success_harmonic_pct' for p in PREFIXES} | set(AXES)
PROGRESS = {'phase_id','step','edits','time/elapsed_seconds'}

def config(values):
    require(all(type(values.get(k)) is type(v) and values[k]==v for k,v in EXTRA.items()), 'W0_REFERENCE_CONFIG')
    clean=base.config({k:v for k,v in values.items() if k not in EXTRA})
    require(all(clean.get(k)==v for k,v in dict(server='server2',task_id='base-model-gptj-w0-cohort-curves',
        model='gptj',model_family='gptj',writer='none',role='scientific',arm='W0_BASE_MODEL').items()), 'W0_PROFILE_ONLY')
    return dict(clean, **EXTRA)

def bind_job_identity(values,environ=None):
    cfg=config(values)
    bound=base.bind_job_identity({k:v for k,v in cfg.items() if k not in EXTRA},environ)
    return config(dict(bound,**EXTRA))

def metrics(values,scientific=False):
    require(type(values) is dict and bool(values), 'SCALARS')
    require(all(type(v) in (int,float) and math.isfinite(v) for v in values.values()),'FINITE_BUILTIN')
    if 'reference_cohort_edits' not in values:
        require(set(values)<=PROGRESS and values.get('edits',0)==0,'PROGRESS_ONLY')
        return dict(values)
    require(set(values)<=ALLOWED and set(AXES)<=values.keys(),'W0_WHITELIST_AXIS')
    x=values['edits']
    require(type(x) is int and x in range(0,2001,100) and values['reference_cohort_edits']==x,'COHORT_AXIS')
    require(all(type(values[k]) is int and values[k]==0 for k in AXES[2:]),'ACTUAL_STATE_ZERO')
    expected={'W0_first2000':2000} if x==0 else {'current/post':100}
    if x in (500,1000,1500,2000):expected['all_seen/post']=x
    expected_groups={p+'/'+k:n*m for p,n in expected.items() for k,m in [('R',1),('P',2),('N',10)]}
    if x:
        expected_groups['w0/current/N']=1000
        if x%500==0:expected_groups['w0/all_seen/N']=x*10
    keys=set(AXES) | {g+'/'+f for g in expected_groups for f in FIELDS} | {p+'/success_harmonic_pct' for p in expected}
    require(set(values)==keys,'EXACT_COHORT_COVERAGE')
    for g,count in expected_groups.items():
        d={f:values[g+'/'+f] for f in FIELDS}
        require(type(d['count']) is int and d['count']==count and type(d['success_count']) is int and 0<=d['success_count']<=count,'COUNTS')
        require(all(0<=d[f]<=100 for f in FIELDS if f.endswith('_pct')),'PCT')
        require(d['true_nll']>=0 and d['new_nll']>=0,'NLL')
        require(math.isclose(d['success_pct'],100*d['success_count']/count,abs_tol=1e-8),'SUCCESS_ARITHMETIC')
        require(math.isclose(d['margin_true_minus_new'],d['true_nll']-d['new_nll'],abs_tol=1e-8),'MARGIN')
    for p in expected:
        require(math.isclose(values[p+'/success_harmonic_pct'],harmonic([values[p+'/'+k+'/success_pct'] for k in 'RPN']),abs_tol=1e-8),'HARMONIC')
    return dict(values)

class AxisState:
    def __init__(self):self.last=None
    def check(self,values):
        if 'reference_cohort_edits' in values:
            require(values['edits']==(0 if self.last is None else self.last+100),'ORDERED_CURVES')
        else:require(self.last is None,'NO_PROGRESS_AFTER_CURVES')
    def accept(self,values):
        self.check(values)
        if 'reference_cohort_edits' in values:self.last=values['edits']

def define_axes(run):
    run.define_metric('edits');run.define_metric('step')
    for p in (*PREFIXES,'w0/current/N','w0/all_seen/N'):
        run.define_metric(p+'/*',step_metric='edits',step_sync=False)
    for p in ('phase_id','time/elapsed_seconds'):run.define_metric(p,step_metric='step',step_sync=False)
