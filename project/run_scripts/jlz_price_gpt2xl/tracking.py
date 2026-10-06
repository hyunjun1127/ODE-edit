"""Future-run comparison schema. SH1 helper is consumed, never patched here."""
import json
import os
from pathlib import Path
from project.run_scripts.experiment_tracking import init
from project.run_scripts.experiment_tracking import schema
from project.run_scripts.jlz_interference_l1.cap_tracking import safe_log,candidate_metrics
from project.run_scripts.jlz_interference_l1.comparison_bridge import metric_row
from project.run_scripts.jlz_realization.observe import reduce_rows
from .common import require,rows_from,validate_rows,write

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
    missing_metrics=sorted(REQUIRED_METRICS-schema.METRICS)
    missing_config=sorted(REQUIRED_CONFIG-schema.CONFIG_KEYS)
    require(not missing_metrics and not missing_config,
        'TRACKING_SCHEMA_NOT_READY:'+json.dumps(dict(metrics=missing_metrics,config=missing_config)))
    # Shared worker must bind eval axes and immutable online identity, not merely
    # accept keys. Explicit capability is supplied by its SH1 owner.
    require(getattr(schema,'COMPARISON_SCHEMA',None)==SCHEMA,'TRACKING_COMPARISON_CAPABILITY_NOT_READY')

def start(c,lock,out,cell):
    contract_ready()
    cfg=dict(server='server1',task_id=c['task_id'],arm='GPT2XL_'+cell,
        attempt=c['run_instance']['attempt'],source_sha=lock['source_commit'],
        config_sha=lock['config_sha256'],job_id=os.environ['SLURM_JOB_ID'],model='gpt2xl',
        model_family='gpt2',writer=c['writer'],role='scientific',metric_schema=SCHEMA)
    tracker=init(env_file=c['tracking']['env_file'],spool=out/'tracking',config=cfg)
    # Retain startup evidence separately; transport.result changes on each ACK.
    identity=dict(run_id=tracker.run_id,url=tracker.result['url'],job_identity=tracker.job_identity,
        source_sha=lock['source_commit'],config_sha=lock['config_sha256'],config=cfg,
        startup_remote_verified=True,scientific_complete=False)
    path=out/'tracking-identity.json';require(not path.exists(),'TRACKING_IDENTITY_CREATE_ONCE')
    write(path,identity)
    return tracker

def batch_values(pre,current,all_seen,batch):
    values=dict(edits=batch*100,batch=batch,pre_state_edits=(batch-1)*100,post_state_edits=batch*100)
    values.update(metric_row('current/pre',pre,100))
    values.update(metric_row('current/post',current,100))
    if batch in (5,10,15,20):
        require(all_seen is not None,'MILESTONE_REQUIRED')
        values.update(metric_row('all_seen/post',all_seen,batch*100))
    else:require(all_seen is None,'UNMEASURED_ALL_SEEN_NOT_LOGGED')
    return values

def w0_rows(out,identities,ids,state):
    rows=rows_from(out/'W0',state)
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
        b=receipt['batch']
        result=batch_values(receipt['pre'],receipt['post_current'],receipt['post'] if b in (5,10,15,20) else None,b)
        result.update(w0_subset(w0,current_ids,'w0/current/N'))
        if b in (5,10,15,20):result.update(w0_subset(w0,seen_ids,'w0/all_seen/N'))
        return result
    return safe_log(tracker,values,'committed_batch_comparison')

class TrackedEvents:
    def __init__(self,events,tracker):self.events=events;self.tracker=tracker
    def __getattr__(self,name):return getattr(self.events,name)
    def emit(self,event,payload):
        self.events.emit(event,payload)
        if event=='candidate':
            safe_log(self.tracker,lambda:dict(candidate_metrics(payload,self.events.batch),
                **{'fit/global_candidate':(self.events.batch-1)*20+payload['candidate']}),'candidate')
