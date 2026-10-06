"""Task caller of SH1's unmodified online scalar helper."""
import json
import math
from project.run_scripts.experiment_tracking import init
from .cap_common import require

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
    config={
        'server':'server4','task_id':c['task_id'],'arm':model_scoped_arm(c,cell),'attempt':c['run_instance']['attempt'],
        'source_sha':lock['source_commit'],'config_sha':lock['config_sha256'],
        'job_id':os.environ['SLURM_JOB_ID']}
    parent=c['tracking'].get('parent_runs',{}).get(cell)
    if parent:config['parent_run_id']=parent
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
