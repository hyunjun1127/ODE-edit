"""Two-stage held registration, then release after the unused server1 pair is cancelled."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'evaluation-source'))
from official.runners.server1 import submit as ctl
from official.runners.server1.common import member, read, verify
from official.experiments.prepare import digest, write_new

parser=argparse.ArgumentParser();parser.add_argument('action',choices=['hold','release']);args=parser.parse_args()
prep=read(ROOT/'preparation.json')
config=read(verify(prep['config']))
assert digest({k:v for k,v in config.items() if k!='identity_sha256'})==config['identity_sha256']
assert config['server']=='server4' and not config['covariance_repair'] and config['model_dtype']=='bfloat16'
for entry in config['source_members']+read(verify(prep['execution_lock']))['source_members']+prep['scripts']:
    verify(entry)

if args.action=='hold':
    assert not (ROOT/'held-registration.json').exists()
    for job in ('63561','63562'):
        old=ctl.metadata(ctl.command(['scontrol','show','job',job,'-o']))
        assert old['JobState']=='PENDING' and old['Reason']=='JobHeldUser'
        assert old['RunTime']=='00:00:00' and old['ReqNodeList']=='devbox'
        assert old['Command'].startswith('/mnt/raid5/janghj/ODE-edit/local/fe-qwen-cf-altdata-20261011/native-r1/')
    before=ctl.inventory(node='server4')
    assert before['allocated_gpus']<=1 and before['admitted_DAG_width']<=1, 'SERVER4_CAP2_SINGLE_FREE_LANE_REQUIRED'
    jobs={};submitted=[]
    try:
        for mode,script,cwd,mem in [('train','qwen25.sh',ROOT/'author',59392),
                                    ('evaluate','qwen25-eval.sh',ROOT/'evaluation-source',49152)]:
            name='s4-qwen25-cf-fe-next2k-'+mode
            argv=['sbatch','--hold','--parsable','--job-name='+name,'--partition=gpu','--nodelist=server4',
                  '--qos=lab_gpu_s4','--gres=gpu:1','--cpus-per-task=8',f'--mem={mem}M',
                  '--time=12:00:00','--export=NONE','--no-requeue','--kill-on-invalid-dep=yes',
                  '--chdir='+str(cwd),'--output='+str(ROOT/(mode+'-%j.out')),'--error='+str(ROOT/(mode+'-%j.err'))]
            parent=jobs['train']['job_id'] if mode=='evaluate' else None
            if parent:argv+=['--dependency=afterany:'+parent]
            argv+=[str(ROOT/script)]
            job=ctl.command(argv).split(';')[0];assert job.isdigit();submitted.append(job)
            d=ctl.metadata(ctl.command(['scontrol','show','job',job,'-o']))
            assert d['JobState']=='PENDING' and d['Reason']=='JobHeldUser'
            assert d['UserId'].split('(')[0]=='janghj' and d['ReqNodeList']=='server4'
            assert d['QOS']=='lab_gpu_s4' and d['Command']==str(ROOT/script) and d['WorkDir']==str(cwd)
            assert ctl.gpu_count(d['ReqTRES'])==1 and ctl.gpu_count(d.get('AllocTRES',''))==0
            assert ctl.requested_cpu_matches(d,8) and ctl.memory_MiB(d['MinMemoryNode'])==mem
            assert ctl.dependencies(d['Dependency'])==([('afterany',parent)] if parent else [])
            assert ctl.command(['scontrol','write','batch_script',job,'-']).strip()==(ROOT/script).read_text().strip()
            jobs[mode]=dict(job_id=job,name=name,argv=argv,held_detail=d,script=member(ROOT/script))
        held=ctl.inventory(node='server4')
        assert held['allocated_gpus']<=2 and held['admitted_DAG_width']<=2
        write_new(ROOT/'held-registration.json',dict(jobs=jobs,before=before,held=held,config=prep['config'],
                                                    cap=2,existing_server4_job_mutations=0))
        print(json.dumps({mode:{k:v for k,v in entry.items() if k in ('job_id','name')} for mode,entry in jobs.items()},indent=2),flush=True)
    except BaseException:
        for job in submitted:ctl.command(['scancel',job])
        write_new(ROOT/'registration-failure.json',dict(submitted=submitted,error=traceback.format_exc()))
        raise
else:
    assert not (ROOT/'submission.json').exists()
    held=read(ROOT/'held-registration.json');jobs=held['jobs']
    old_states={}
    for job in ('63561','63562'):
        d=ctl.metadata(ctl.command(['scontrol','show','job',job,'-o']))
        assert d['JobState']=='CANCELLED' and d['RunTime']=='00:00:00',d
        old_states[job]=d
    for mode,entry in jobs.items():
        verify(entry['script'])
        d=ctl.metadata(ctl.command(['scontrol','show','job',entry['job_id'],'-o']))
        assert d['JobState']=='PENDING' and d['Reason']=='JobHeldUser'
        for key in ('Command','WorkDir','JobName','ReqTRES','QOS','TimeLimit','Dependency'):
            assert d[key]==entry['held_detail'][key]
    for mode in ('evaluate','train'):
        ctl.command(['scontrol','release',jobs[mode]['job_id']])
    after=ctl.inventory(node='server4')
    assert after['allocated_gpus']<=2 and after['admitted_DAG_width']<=2
    for old in held['before']['jobs']:
        now=next((row for row in after['jobs'] if row['key']==old['key']),None)
        if now:
            for key in ('Command','WorkDir','JobName','ReqTRES','QOS','TimeLimit'):
                assert now['detail'][key]==old['detail'][key]
    receipt=dict(schema='fe-qwen-next2k-server4-submission-v1',jobs=jobs,after=after,cap=2,
                 config=prep['config'],preparation=member(ROOT/'preparation.json'),
                 superseded_server1_jobs=old_states,server1_GPU_execution=False,existing_server4_job_mutations=0,
                 GPU_results='NOT_OBSERVED',scientific_settings_equal=True,
                 checkpoint_history_batches=[0,5,10,15,20],destination='server4')
    write_new(ROOT/'submission.json',receipt)
    print(json.dumps(dict(jobs={k:v['job_id'] for k,v in jobs.items()},cap=2,server1_cancelled=['63561','63562'],
                         server4_states={r['key']:r['state'] for r in after['jobs']}),indent=2),flush=True)
