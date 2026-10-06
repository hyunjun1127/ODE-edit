"""Task caller of SH1's unmodified online scalar helper."""
import json
import math
from project.run_scripts.experiment_tracking import init
from project.run_scripts.experiment_tracking import schema
from project.run_scripts.jlz_realization.observe import reduce_rows
from .cap_common import require,rows_from,validate_rows
from .comparison_bridge import metric_row

SCHEMA='price-first2k-scalar-v1'
FIELDS=('count','success_count','success_pct','token_acc_pct','prompt_acc_pct',
    'strict_acc_pct','true_nll','new_nll','margin_true_minus_new')
PREFIXES=('current/pre','current/post','all_seen/post','W0_first2000')
REQUIRED_METRICS={f'{p}/{k}/{f}' for p in PREFIXES for k in ('R','P','N') for f in FIELDS}
REQUIRED_METRICS.update(p+'/success_harmonic_pct' for p in PREFIXES)
REQUIRED_METRICS.update(f'w0/{p}/N/{f}' for p in ('current','all_seen') for f in FIELDS)
REQUIRED_METRICS.update(('edits','batch','pre_state_edits','post_state_edits','fit/global_candidate'))
REQUIRED_CONFIG={'model','model_family','writer','role','metric_schema'}

def contract_ready():
    """Check actual accepted keys; the capability marker is not a submit gate."""
    missing_metrics=sorted(REQUIRED_METRICS-schema.METRICS)
    missing_config=sorted(REQUIRED_CONFIG-schema.CONFIG_KEYS)
    require(not missing_metrics and not missing_config,
        'TRACKING_SCHEMA_NOT_READY:'+json.dumps(dict(metrics=missing_metrics,config=missing_config)))

def model_scoped_arm(c,cell):
    """Stable saved-view routing using the existing helper's allowed arm field.

    Existing Llama/Qwen cell names are unchanged. This is logging metadata only;
    the scientific cell, output directory and frozen submissions are untouched.
    """
    model=c.get('model_profile')
    if model not in ('LLAMA','QWEN','GPTJ'):return cell
    known=next((m for m in ('LLAMA','QWEN','GPTJ') if cell.startswith(m+'_')),None)
    require(known is None or known==model,'TRACKING_MODEL_ARM_MISMATCH')
    return cell if known else model+'_'+cell

def start(c,lock,out,cell):
    import os
    contract_ready()
    model=c.get('model_profile')
    alias={'LLAMA':'llama3','QWEN':'qwen','GPTJ':'gptj'}.get(model)
    require(alias is not None,'TRACKING_MODEL_ALIAS')
    writer=c.get('writer',c['arm_profiles'][cell].get('writer','memit'))
    require(writer in ('memit','alphaedit'),'TRACKING_WRITER_IDENTITY')
    config={
        'server':'server4','task_id':c['task_id'],'arm':model_scoped_arm(c,cell),'attempt':c['run_instance']['attempt'],
        'source_sha':lock['source_commit'],'config_sha':lock['config_sha256'],
        'job_id':os.environ['SLURM_JOB_ID'],'model':alias,'model_family':model,
        'writer':writer,'role':'scientific','metric_schema':SCHEMA}
    parent=c['tracking'].get('parent_runs',{}).get(cell)
    if parent:config['parent_run_id']=parent
    # SH1's helper captures only allowlisted Slurm fields, binds actual job/name,
    # defines eval/fit axes, and creates tracking/identity.json exactly once.
    return init(env_file=c['tracking']['env_file'],spool=out/'tracking',config=config)

def candidate_metrics(payload,batch):
    """Native NLL is [owner][rewrite context]; equal context then owner mean.

    Consume already serialized CPU scalars only. No forward, tensor conversion,
    reduction in the scientific objective, or mutation of the raw candidate.
    """
    B=len(payload['F']);rows=payload['nll']
    require(B>0 and len(rows)==B,'TRACKING_NLL_OWNER_COUNT')
    require(all(isinstance(row,list) and row for row in rows),'TRACKING_NLL_CONTEXT_ROWS')
    require(all(type(x) in (int,float) and math.isfinite(x) for row in rows for x in row),'TRACKING_NLL_FINITE_SCALARS')
    require(all(len(payload[key])==B for key in ('KL','norm')),'TRACKING_OWNER_REDUCTIONS')
    return {'batch':batch,'candidate':payload['candidate'],'phase_id':1,
        'fit/global_candidate':(batch-1)*25+payload['candidate'],
        'fit/loss':payload['J_mean'],'fit/nll':sum(sum(row)/len(row) for row in rows)/B,
        'fit/kl':sum(payload['KL'])/B,'fit/norm':sum(payload['norm'])/B,
        'optimizer/calls':sum(payload['controller']['update_counts'])}

def safe_log(tracker,build_values,phase):
    """Caller conversion/transport failure cannot roll back a scientific commit.

    The authoritative event write is outside this guard. Startup auth remains
    a hard pre-model gate; only post-start logging errors are isolated here.
    """
    try:
        if tracker.log(build_values()) is False:raise RuntimeError('TRACKING_POINT_REJECTED')
        return True
    except Exception as error:
        count=getattr(tracker,'caller_dropped_points',0)+1
        tracker.caller_dropped_points=count
        receipt={'status':'LOGGING_DEGRADED_CALLER','phase':phase,'error_type':type(error).__name__,
            'caller_dropped_points':count,'scientific_state_unchanged':True}
        try:
            (tracker.spool/'caller-receipt.json').write_text(json.dumps(receipt)+'\n')
        except Exception:pass
        if count==1:print(json.dumps(receipt),flush=True)
        return False

class TrackedEvents:
    def __init__(self,events,tracker):self.events=events;self.tracker=tracker
    def __getattr__(self,name):return getattr(self.events,name)
    def emit(self,event,payload):
        self.events.emit(event,payload)
        if event=='candidate':
            safe_log(self.tracker,lambda:candidate_metrics(payload,self.events.batch),'candidate')

def batch_values(pre,current,all_seen,batch):
    """Map existing reductions; milestone current is still the incoming100."""
    values=dict(edits=batch*100,batch=batch,pre_state_edits=(batch-1)*100,post_state_edits=batch*100)
    values.update(metric_row('current/pre',pre,100))
    values.update(metric_row('current/post',current,100))
    if batch in (5,10,15,20):
        require(all_seen is not None,'MILESTONE_REQUIRED')
        values.update(metric_row('all_seen/post',all_seen,batch*100))
    else:require(all_seen is None,'UNMEASURED_ALL_SEEN_NOT_LOGGED')
    return values

def w0_rows(out,identities,ids,state,row_reader=None):
    rows=(row_reader or rows_from)(out/'W0',state)
    actual=validate_rows(rows,identities,ids,'W0')
    stored=json.loads((out/'W0/summary.json').read_text())
    require(actual==stored['summary'],'W0_TRACKING_RAW_SUMMARY')
    return rows,actual

def w0_subset(rows,ids,prefix):
    selected=set(ids)
    groups=reduce_rows([r for r in rows if r['case_id'] in selected])
    mapped=metric_row('_',groups,len(ids))
    return {prefix+k[len('_/N'):]:v for k,v in mapped.items() if k.startswith('_/N/')}

def log_w0(tracker,summary):
    return safe_log(tracker,lambda:dict(edits=0,batch=0,**metric_row('W0_first2000',summary,2000)),'W0_first2000')

def log_batch(tracker,receipt,w0,current_ids,seen_ids):
    def values():
        batch=receipt['batch']
        result=batch_values(receipt['pre'],receipt['post_current'],
            receipt['post'] if batch in (5,10,15,20) else None,batch)
        result.update(w0_subset(w0,current_ids,'w0/current/N'))
        if batch in (5,10,15,20):result.update(w0_subset(w0,seen_ids,'w0/all_seen/N'))
        return result
    return safe_log(tracker,values,'committed_batch_comparison')

def log_endpoint(tracker,metrics,batch,edits):
    ok=safe_log(tracker,lambda:endpoint_metrics(metrics,batch,edits),'endpoint')
    for i,kind in enumerate(('R','P','N')):
        def values():
            m=metrics[kind]
            return {'batch':batch,'edits':edits,'phase_id':3+i,'eval/TF_token_micro':m['token_micro'],
                'eval/TF_prompt_macro':m['prompt_macro'],'eval/TF_strict':m['strict_numerator']/m['strict_denominator'],
                'eval/true_nll':m['true_nll_mean'],'eval/new_nll':m['new_nll_mean']}
        ok=safe_log(tracker,values,'endpoint_'+kind) and ok
    return ok

def endpoint_metrics(metrics,batch,edits):
    rates=[metrics[k]['rate'] for k in ('R','P','N')]
    values={'batch':batch,'edits':edits,'phase_id':2,
        'eval/RS':rates[0],'eval/PS':rates[1],'eval/NS':rates[2],
        'eval/harmonic':0. if min(rates)==0 else 3/sum(1/r for r in rates)}
    for kind in ('R','P','N'):
        for field in ('numerator','denominator'):values['eval/'+kind+'_'+field]=metrics[kind][field]
    return values
