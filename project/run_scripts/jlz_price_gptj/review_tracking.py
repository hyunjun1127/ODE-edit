"""One-shot CPU regression on immutable existing B4/B5/W0 raw, no model calls."""
import argparse,json
from pathlib import Path
from .common import member,verify,write,require,rows_from,validate_rows
from .tracking import batch_values,w0_subset,contract_ready,REQUIRED_METRICS,REQUIRED_CONFIG,SCHEMA
from project.run_scripts.jlz_interference_l1.comparison_bridge import observation,metric_row

PARENT=Path('/data/janghj/ODE-edit/local/jlz-price-cap-base-repair-2k/logging-repair-20261007')

def review(out):
    c=json.loads((PARENT/'config.json').read_text());mc=c['models']['LLAMA']
    identity=json.loads(verify(mc['observer_identity']).read_text())['rows']
    root=PARENT/'LLAMA_CAP075';checks=[];inputs=[]
    w0=rows_from(root/'W0');ids=[i for p in mc['packs'] for i in p['ids']]
    w0_summary=validate_rows(w0,identity,ids,'W0')
    require(w0_summary==json.loads((root/'W0/summary.json').read_text())['summary'],'W0_RAW_STORED')
    w0_payload=dict(edits=0,**metric_row('W0_first2000',w0_summary,2000))
    payloads=[w0_payload]
    for n in (4,5):
        folder=root/f'batch-{n:02d}';commit=json.loads((folder/'commit.json').read_text())
        current=mc['packs'][n-1]['ids'];seen=[i for p in mc['packs'][:n] for i in p['ids']]
        _,pre=observation(folder/'pre',commit['before'],identity,current,f'B{n}_PRE')
        postrows,post=observation(folder/'post',commit['after'],identity,seen if n==5 else current,f'W{n}')
        from project.run_scripts.jlz_realization.observe import reduce_rows
        birth=reduce_rows([r for r in postrows if r['case_id'] in set(current)])
        require(pre==commit['pre'] and post==commit['post'] and birth==commit['post_current'],'RAW_COMMIT_EQUAL')
        payload=batch_values(pre,birth,post if n==5 else None,n)
        payload.update(w0_subset(w0,current,'w0/current/N'))
        if n==5:payload.update(w0_subset(w0,seen,'w0/all_seen/N'))
        payloads.append(payload)
        for kind,m in [('R',1),('P',2),('N',10)]:
            for prefix in ('current/pre','current/post'):
                require(payload[f'{prefix}/{kind}/count']==100*m,'CURRENT_FIXED_DENOMINATOR')
            if n==5:require(payload[f'all_seen/post/{kind}/count']==500*m,'ALL_SEEN_DENOMINATOR')
        if n==4:require(not any(k.startswith('all_seen/') for k in payload),'NO_UNMEASURED_ALLSEEN')
        require(payload['edits']==n*100 and payload['pre_state_edits']==(n-1)*100,'PRE_STATE_AXIS')
        for k,v in payload.items():
            if k.endswith('_pct'):require(0<=v<=100,'PERCENT_UNIT')
        checks.append(dict(batch=n,keys=len(payload),current_denominators={k:payload[f'current/post/{k}/count'] for k in 'RPN'},
            all_seen_denominators={k:payload[f'all_seen/post/{k}/count'] for k in 'RPN'} if n==5 else None,
            current_harmonic=payload['current/post/success_harmonic_pct'],raw_stored_payload_equal=True))
        inputs.append(member(folder/'commit.json'))
        inputs.extend(member(p) for phase in ('pre','post') for p in sorted((folder/phase).glob('chunk-*.json')))
    from project.run_scripts.experiment_tracking import schema
    try:contract_ready();ready=True;block=None
    except RuntimeError as error:ready=False;block=str(error)
    integration='NOT_RUN_HELPER_CAPABILITY_MISSING'
    if ready:
        from project.run_scripts.experiment_tracking.test_tracking import SDK
        from project.run_scripts.experiment_tracking.worker import session
        class FakeSDK(SDK):
            def __init__(self):super().__init__();self.axes={}
            def define_metric(self,name,**kwargs):self.axes[name]=kwargs
        cfg=schema.bind_job_identity(dict(server='server4',task_id='jlz-price-gptj-2k',arm='GPTJ_MEMIT_CAP075',
            attempt='cpu-metadata-fixture',source_sha='a'*40,config_sha='b'*64,model='gptj',model_family='GPTJ',
            writer='memit',role='scientific',metric_schema=SCHEMA),{'SLURM_JOB_ID':'123','SLURM_STEP_ID':'batch'})
        request=dict(config=cfg,run_id='cpu-fixture-not-uploaded',spool='/tmp/fake-only-no-sdk-write',
            smoke=False,base_url='https://api.wandb.ai')
        sdk=FakeSDK();emitted=[]
        commands=[dict(op='log',values=schema.metrics(p),step=None) for p in payloads]
        commands.append(dict(op='finish',exit_code=0))
        session(sdk,request,commands,emitted.append)
        require(sdk.points==payloads,'WORKER_PAYLOAD_RAW_EQUAL')
        for prefix in ('current/pre','current/post','all_seen/post','W0_first2000','w0/current/N','w0/all_seen/N'):
            matches=[v for k,v in sdk.axes.items() if __import__('fnmatch').fnmatch(prefix+'/success_pct',k)]
            require(any(v.get('step_metric')=='edits' and v.get('step_sync') is False for v in matches),'WORKER_EVAL_AXIS:'+prefix)
        matches=[v for k,v in sdk.axes.items() if __import__('fnmatch').fnmatch('fit/loss',k)]
        require(any(v.get('step_metric')=='fit/global_candidate' and v.get('step_sync') is False for v in matches),'WORKER_FIT_AXIS')
        require(sdk.name.endswith('job123') and sdk.config['model']=='gptj','WORKER_MODEL_JOB_IDENTITY')
        integration='FAKE_SDK_PAYLOAD_AXES_IDENTITY_PASS'
    # Refuse unknown private fields under the real helper, never loosen it here.
    rejected=0
    for fn,value in [(schema.metrics,{'raw_prompt':'not-uploaded'}),(schema.metrics,{'unknown_raw':1}),
                     (schema.config,{'private_setting':'not-uploaded'})]:
        try:fn(value)
        except (RuntimeError,ValueError):rejected+=1
    require(rejected==3,'PRIVACY_REJECTION')
    result=dict(status='RAW_COMPARISON_REGRESSION_PASS' if ready else 'RAW_PASS_SHARED_HELPER_SCHEMA_BLOCKED',
        schema=SCHEMA,checks=checks,W0_first2000={k:w0_payload[f'W0_first2000/{k}/count'] for k in 'RPN'},
        helper_ready=ready,block=block,integration=integration,
        helper_sources=[member(p) for p in sorted(Path(schema.__file__).parent.glob('*.py'))],
        required_metrics=sorted(REQUIRED_METRICS),required_config=sorted(REQUIRED_CONFIG),
        privacy_rejections=rejected,inputs=inputs,original_raw_unchanged=True,new_forward=0,new_fit=0,
        new_WandB_run=False,independent_reviewer=False,review_level='owner CPU raw reducer')
    write(out,result)
    return {k:result[k] for k in ('status','checks','W0_first2000','helper_ready','new_forward')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(review(a.out)))
