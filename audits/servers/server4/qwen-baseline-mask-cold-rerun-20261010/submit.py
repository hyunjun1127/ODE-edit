"""One deliberate held registration pass; never cancel or mutate protected jobs."""
import json
import hashlib
import subprocess
import shutil
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/data/janghj/ODE-edit/local/qwen-baseline-mask-cold-rerun-20261010/execution-r2')
OUT=Path(__file__).parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def call(*args):return subprocess.check_output(args,text=True,timeout=30).strip()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)

assert not (ROOT/'submission.json').exists(),'DUPLICATE_SUBMISSION_RECEIPT'
old=call('sacct','-X','-n','-P','-j','61776,61777','-o','JobIDRaw,User,JobName%80,State,ElapsedRaw,WorkDir%180')
assert all('|janghj|qwen-heldout-' in s and '|COMPLETED|' in s for s in old.splitlines())
queue=call('squeue','-u','janghj','-h','-o','%i|%j|%T|%b|%R|%E')
assert 'qwen-mask-heldout-' not in queue,'DUPLICATE_NAME'
front=call('scontrol','show','job','62037','--oneliner')
held=call('scontrol','show','job','62038','--oneliner')
assert 'UserId=janghj(1025)' in front and 'NodeList=server4' in front and 'JobState=RUNNING' in front
assert 'lane-llama-unitlr.sh' in front
assert 'JobState=PENDING' in held and 'Reason=JobHeldUser' in held and 'lane-qwen-unitlr.sh' in held
assert set(s.split('|')[0] for s in queue.splitlines() if '|server4|' in s)=={'62037'}
# Reserve both final payloads, Alpha atomic successor, 2GiB metadata, 32GiB free.
W=5*3584*18944*4;H=5*18944*18944*4
reserve=32*2**30 + 2*W + H + (W+H) + 2*2**30
space=shutil.disk_usage(ROOT)
assert space.free>=reserve,('RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE',space.free,reserve)
freeze=read(ROOT/'freeze.json')
for method in ('MEMIT','ALPHAEDIT'):
    c=read(ROOT/f'{method}.json');assert sha(ROOT/f'{method}.json')==freeze['configs'][method]
    assert c['source']==freeze['source'] and c['final_edits']==500
    for name,h in c['source_members'].items():assert sha(ROOT/'source'/name)==h
    assert sha(Path(c['stream']))==c['stream_sha256']
rows=[]
write(ROOT/'admission.json',dict(at=datetime.now(timezone.utc).isoformat(),old=old,queue=queue,
    protected_running=front,protected_held=held,cap=2,free_bytes=space.free,reserved_bytes=reserve,
    proof='One baseline serial lane starts after62037. Protected62038 stays HELD; even if user later releases it, baseline1+OURS1<=2. No ours rerun/input dependency.',
    old_cancelled=[],source=freeze['source'],launcher_sha256=sha(ROOT/'run.sh')))
dependency='afterany:62037'
for method in ('MEMIT','ALPHAEDIT'):
    name='qwen-mask-heldout-'+method.lower()
    argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s4','--nodelist=server4',
        '--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=59392M','--gres=gpu:rtx_pro_6000:1',
        '--time=12:00:00','--export=NONE','--no-requeue','--job-name='+name,
        '--dependency='+dependency,'--chdir='+str(ROOT/'source'),
        '--output='+str(ROOT/(method+'-%j.out')),'--error='+str(ROOT/(method+'-%j.err')),
        str(ROOT/'run.sh'),method]
    result=call(*argv);job=result.split(';')[0];assert job.isdigit()
    write(ROOT/(method+'-registered.json'),dict(job_id=job,argv=argv))
    record=call('scontrol','show','job',job,'--oneliner')
    for required in ('UserId=janghj(1025)','JobName='+name,'JobState=PENDING','Priority=0',
                     'Requeue=0','CPUs/Task=8','MinMemoryNode=58G','ReqNodeList=server4',
                     'Command='+str(ROOT/'run.sh'),'WorkDir='+str(ROOT/'source'),
                     'TresPerNode=gres/gpu:rtx_pro_6000:1',dependency):
        assert required in record,('HELD_INSPECTION_FAILED',job,required)
    c=read(ROOT/f'{method}.json')
    rows.append(dict(method=method,dataset='cf',cohort='heldout500',old_job_id=c['old_job_id'],old_state='COMPLETED_KEEP',
        job_id=job,job_name=name,dependency=dependency,source=freeze['source'],config_sha256=c['config_sha256'],
        config_file_sha256=sha(ROOT/f'{method}.json'),held_record=record,argv=argv,
        context_sha256=None,context_status='NOT_OBSERVED_ACTUAL_NATIVE_JOB_WILL_SEAL',
        checkpoint_path=str(ROOT/'runs'/method/'checkpoint/latest.json'),WandB_stage='NOT_STARTED'))
    dependency='afterany:'+job
write(ROOT/'held-inspection.json',rows)
for row in rows:call('scontrol','release',row['job_id'])
snapshot=call('squeue','-h','-j',','.join(r['job_id'] for r in rows),'--format=%i|%j|%T|%b|%E')
result=dict(nonce='USER-GH-QWEN-BASELINE-MASK-COLD-RERUN-20261010-R1',
    accepted_turn='01a1214b-7cc9-7f12-a299-73e8577d0d34',at=datetime.now(timezone.utc).isoformat(),
    rows=rows,snapshot=snapshot,status='REGISTERED_RELEASED',cap=2,cancellation_count=0,
    qualification='NOT_RUN_USER_DISABLED',protected_62038='UNCHANGED_HELD',
    collector='Runner writes own B5 final/terminal; no extra collector needed for registration',
    generation='DEFERRED_CHECKPOINT_EVALUATION',checkpoint='B5_500_KEEP_DEFERRED_CONSUMER',
    new_monitor=False,automatic_retry=False)
write(ROOT/'submission.json',result);write(OUT/'submission.json',result)
print(json.dumps(dict(status=result['status'],jobs=[r['job_id'] for r in rows],snapshot=snapshot)))
