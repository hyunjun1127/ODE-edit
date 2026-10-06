"""One bounded CPU integration of stored B4/B5/W0 and real logger assembly.

The SDK is a fixture: no online run, GPU, model load, or scientific computation.
"""
import argparse
import fnmatch
import json
import tempfile
from pathlib import Path
from project.run_scripts.jlz_interference_l1 import cap_tracking as cap
from project.run_scripts.jlz_interference_l1.cap_common import member,require,verify,write,rows_from,validate_rows
from project.run_scripts.jlz_interference_l1.comparison_bridge import observation,metric_row
from project.run_scripts.jlz_price_gptj import tracking as gptj
from project.run_scripts.jlz_price_gptj.review_tracking import review as review_gptj
from project.run_scripts.jlz_realization.observe import reduce_rows
from project.run_scripts.experiment_tracking import schema
from project.run_scripts.experiment_tracking.identity import create as create_identity
from project.run_scripts.experiment_tracking.method import harmonic
from project.run_scripts.experiment_tracking.test_method import MethodSDK
from project.run_scripts.experiment_tracking.worker import session

PARENT=Path('/data/janghj/ODE-edit/local/jlz-price-cap-base-repair-2k/logging-repair-20261007')

def first_candidate(path):
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['event']=='candidate':return row['payload']
    raise RuntimeError('STORED_CANDIDATE_REQUIRED')

def review(out):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    require(not out.exists(),'CPU_RECEIPT_CREATE_ONCE')
    cap.contract_ready();gptj.contract_ready()
    config=json.loads((PARENT/'config.json').read_text());mc=config['models']['LLAMA']
    identities=json.loads(verify(mc['observer_identity']).read_text())['rows']
    root=PARENT/'LLAMA_CAP075';ids=[i for pack in mc['packs'] for i in pack['ids']]
    w0=rows_from(root/'W0');summary=validate_rows(w0,identities,ids,'W0')
    require(summary==json.loads((root/'W0/summary.json').read_text())['summary'],'W0_RAW_STORED')
    payloads=[dict(edits=0,batch=0,**metric_row('W0_first2000',summary,2000))]
    checks=[];inputs=[member(PARENT/'config.json'),member(verify(mc['observer_identity'])),member(root/'W0/summary.json')]
    for batch in (4,5):
        folder=root/f'batch-{batch:02d}';commit=json.loads((folder/'commit.json').read_text())
        current=mc['packs'][batch-1]['ids'];seen=[i for pack in mc['packs'][:batch] for i in pack['ids']]
        _,pre=observation(folder/'pre',commit['before'],identities,current,f'B{batch}_PRE')
        postrows,post=observation(folder/'post',commit['after'],identities,seen if batch==5 else current,f'W{batch}')
        birth=reduce_rows([r for r in postrows if r['case_id'] in set(current)])
        require((pre,birth,post)==(commit['pre'],commit['post_current'],commit['post']),'RAW_COMMIT_EQUAL')
        mapped=cap.batch_values(pre,birth,post if batch==5 else None,batch)
        require(mapped==gptj.batch_values(pre,birth,post if batch==5 else None,batch),'LLAMA_GPTJ_SCALAR_MAPPING_EQUAL')
        require(cap.w0_subset(w0,current,'w0/current/N')==gptj.w0_subset(w0,current,'w0/current/N'),'W0_N_MAPPING_EQUAL')
        mapped.update(cap.w0_subset(w0,current,'w0/current/N'))
        if batch==5:mapped.update(cap.w0_subset(w0,seen,'w0/all_seen/N'))
        for kind,multiple in [('R',1),('P',2),('N',10)]:
            for prefix in ('current/pre','current/post'):
                require(mapped[f'{prefix}/{kind}/count']==100*multiple,'CURRENT100_DENOMINATOR')
            if batch==5:require(mapped[f'all_seen/post/{kind}/count']==500*multiple,'ALLSEEN500_DENOMINATOR')
        require(bool([k for k in mapped if k.startswith('all_seen/')])==(batch==5),'MEASURED_MILESTONES_ONLY')
        require(mapped['w0/current/N/count']==1000,'W0_CURRENT_N_COHORT')
        if batch==5:require(mapped['w0/all_seen/N/count']==5000,'W0_ALLSEEN_N_COHORT')
        fit=cap.candidate_metrics(first_candidate(folder/'events.jsonl'),batch)
        for payload in (fit,mapped):schema.metrics(payload,scientific=True)
        payloads.extend((fit,mapped))
        inputs.append(member(folder/'commit.json'))
        inputs.append(member(folder/'events.jsonl'))
        inputs.extend(member(p) for phase in ('pre','post') for p in sorted((folder/phase).glob('chunk-*.json')))
        checks.append(dict(batch=batch,current_RPN=[100,200,1000],all_seen_RPN=[500,1000,5000] if batch==5 else None,
            raw_summary_equal=True,cap_GPTJ_payload_equal=True,fit_global_candidate=fit['fit/global_candidate']))
    fixtures=[]
    source_sha=member(Path(cap.__file__))['sha256'];config_sha=member(PARENT/'config.json')['sha256']
    for model,family in [('llama3','LLAMA'),('gptj','GPTJ')]:
        for writer in ('memit','alphaedit'):
            cfg=schema.bind_job_identity(dict(server='server4',task_id='price-model-runs-tracking',
                arm=family+'_'+writer.upper()+'_CAP075',attempt='cpu-metadata-fixture',source_sha=source_sha,
                config_sha=config_sha,model=model,model_family=family,writer=writer,role='scientific',
                metric_schema=cap.SCHEMA),dict(SLURM_JOB_ID='123',SLURM_ARRAY_JOB_ID='120',SLURM_ARRAY_TASK_ID='0',SLURM_STEP_ID='-5'))
            request=dict(config=cfg,run_id='cpu-fixture-'+model+'-'+writer,spool='/tmp/fake-only-no-sdk-write',
                smoke=False,base_url='https://api.wandb.ai')
            sdk=MethodSDK();emitted=[]
            commands=[dict(op='log',values=p,step=None) for p in payloads]+[dict(op='finish',exit_code=0)]
            session(sdk,request,commands,emitted.append)
            require(sdk.points==payloads,'FAKE_WORKER_RAW_PAYLOAD_EQUAL')
            for prefix in (*cap.PREFIXES,'w0/current/N','w0/all_seen/N'):
                require(any(args==(prefix+'/*',) and kw==dict(step_metric='edits',step_sync=False)
                    for args,kw in sdk.definitions),'FAKE_EVAL_AXIS:'+prefix)
            require((('fit/*',),dict(step_metric='fit/global_candidate',step_sync=False)) in sdk.definitions,'FAKE_FIT_AXIS')
            require(sdk.name.endswith('job120_0') and sdk.config==cfg,'FAKE_JOB_CONFIG_IDENTITY')
            require(emitted[-1]['method_readback']['status']=='REMOTE_BOUNDED_ROWS_VERIFIED','FAKE_ONLY_BOUNDED_READBACK')
            require(all(r.get('delivery')=='SDK_ASYNC_NOT_REMOTE_ACK' for r in emitted if r['status']=='LOGGING_ACCEPTED'),'ASYNC_ACK_SEMANTICS')
            with tempfile.TemporaryDirectory(prefix='identity-fixture-',dir=out.parent) as temp:
                first=create_identity(temp,cfg,emitted[0],request['run_id']);original=(Path(temp)/'identity.json').read_bytes()
                try:create_identity(temp,cfg,emitted[0],request['run_id'])
                except FileExistsError:pass
                else:raise RuntimeError('IMMUTABLE_IDENTITY_NOT_ENFORCED')
                require((Path(temp)/'identity.json').read_bytes()==original and first['config']==cfg,'IMMUTABLE_IDENTITY_CHANGED')
            fixtures.append(dict(model=model,writer=writer,axes=True,job_array_index0=True,signed_step=True,
                identity_create_once=True,raw_payload_equal=True,readback='FAKE_SDK_ONLY_VERIFIED'))
    rejected=0
    for fn,value in [(schema.metrics,{'raw_prompt':'not-uploaded'}),(schema.metrics,{'unknown_raw':1}),
                     (schema.config,{'private_setting':'not-uploaded'})]:
        try:fn(value)
        except (RuntimeError,ValueError):rejected+=1
    require(rejected==3,'PRIVACY_REJECTION')
    require(harmonic([0,None,50]) is None and harmonic([0,40,50])==0,'HARMONIC_MISSING_ZERO')
    gptj_receipt=out.parent/'gptj-existing-raw-review.json'
    require(not gptj_receipt.exists(),'GPTJ_CPU_RECEIPT_CREATE_ONCE')
    gptj_result=review_gptj(gptj_receipt)
    require(gptj_result['status']=='RAW_COMPARISON_REGRESSION_PASS' and gptj_result['helper_ready'],'GPTJ_EXISTING_REVIEW_REQUIRED')
    source_paths=[Path(__file__),Path(cap.__file__),Path(gptj.__file__),
        Path('project/run_scripts/jlz_interference_l1/cap_run.py'),Path('project/run_scripts/jlz_price_alpha_writer/run.py'),
        Path('project/run_scripts/jlz_price_gptj/review_tracking.py'),Path('control/wandb-method-metric-schema.json')]
    source_paths.extend(sorted(Path(schema.__file__).parent.glob('*.py')))
    result=dict(status='CPU_PRODUCER_HELPER_RAW_INTEGRATION_PASS',schema=cap.SCHEMA,checks=checks,fixtures=fixtures,
        W0_RPN=[2000,4000,20000],N_desired='true',privacy_rejections=rejected,harmonic_missing_zero_checked=True,
        source=[member(p.resolve()) for p in source_paths],inputs=inputs,GPTJ_review=member(gptj_receipt),
        online_status='NOT_RUN_CPU_FAKE_SDK_ONLY',actual_remote_verification=False,new_forward=0,new_fit=0,new_GPU=0,
        new_Slurm=0,new_WandB_run=False,original_raw_unchanged=True,review_level='owner CPU raw reducer; no independent reviewer')
    write(out,result)
    return {k:result[k] for k in ('status','checks','W0_RPN','online_status','new_forward','new_fit')}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(review(args.out)))
