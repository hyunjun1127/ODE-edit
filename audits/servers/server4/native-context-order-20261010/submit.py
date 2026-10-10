"""One-shot context-only registration with explicit model ordering."""
import hashlib,json,subprocess,shutil
from pathlib import Path
from datetime import datetime,timezone
R=Path('/data/janghj/ODE-edit/local/native-context-order-20261010/execution-r1')
OUT=Path(__file__).parent
def call(*a):return subprocess.check_output(a,text=True,timeout=30).strip()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,j):
 with p.open('x') as f:json.dump(j,f,indent=2)
assert not (R/'submission.json').exists()
q=call('squeue','-u','janghj','-h','-o','%i|%j|%T|%b|%R|%E')
assert 's4-native-context-' not in q
tail=call('scontrol','show','job','62064','--oneliner')
assert 'UserId=janghj(1025)' in tail and 'JobName=qwen-mask-heldout-alphaedit' in tail
assert 'afterany:62063' in tail and 'ReqNodeList=server4' in tail
head=call('scontrol','show','job','62063','--oneliner')
assert 'afterany:62037' in head and 'JobName=qwen-mask-heldout-memit' in head
protected=call('scontrol','show','job','62038','--oneliner')
assert 'JobState=PENDING' in protected and 'Reason=JobHeldUser' in protected
assert shutil.disk_usage(R).free>=4*2**30
freeze=read(R/'freeze.json');rows=[];dep='afterany:62064'
for family in ('qwen25','gptj','llama3'):
 c=read(R/f'{family}.json');assert sha(R/f'{family}.json')==freeze['configs'][family]
 for name,h in c['source_members'].items():assert sha(R/'source'/name)==h
 name='s4-native-context-'+family
 argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s4','--nodelist=server4',
   '--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=59392M','--gres=gpu:rtx_pro_6000:1',
   '--time=02:00:00','--export=NONE','--no-requeue','--job-name='+name,'--dependency='+dep,
   '--chdir='+str(R/'source'),'--output='+str(R/(family+'-%j.out')),
   '--error='+str(R/(family+'-%j.err')),str(R/'run.sh'),family]
 job=call(*argv).split(';')[0];assert job.isdigit()
 write(R/(family+'-registered.json'),dict(job_id=job,argv=argv))
 raw=call('scontrol','show','job',job,'--oneliner')
 for token in ('UserId=janghj(1025)','JobName='+name,'Priority=0','JobState=PENDING','Requeue=0',
   'CPUs/Task=8','MinMemoryNode=58G','ReqNodeList=server4','Command='+str(R/'run.sh'),
   'WorkDir='+str(R/'source'),'TresPerNode=gres/gpu:rtx_pro_6000:1',dep):assert token in raw,(job,token)
 rows.append(dict(family=family,job_id=job,job_name=name,dependency=dep,source=freeze['source'],
   config_sha256=c['config_sha256'],model_manifest_sha256=c['model_manifest_sha256'],model_path=c['model'],
   revision=c['revision'],seed=c['seed'],profile=c['profile'],context_path=c['out']+'/contexts.json',
   context_sha256=None,ready_path=c['out']+'/READY.json',status='REGISTERED_NOT_GENERATED',held_record=raw,argv=argv))
 dep='afterok:'+job
write(R/'held-inspection.json',rows)
for row in rows:call('scontrol','release',row['job_id'])
snapshot=call('squeue','-h','-j',','.join(r['job_id'] for r in rows),'-o','%i|%j|%T|%E')
j=dict(nonce='USER-GH-SH4-NATIVE-CONTEXT-ORDER-20261010-R1',accepted_turn='01a12156-29cb-7bc3-9bc1-75e0d132de43',
 at=datetime.now(timezone.utc).isoformat(),rows=rows,snapshot=snapshot,status='REGISTERED_RELEASED',
 queue_before=q,frontier_before=dict(job62063=head,job62064=tail,protected62038=protected),
 cap=2,resource_proof='Single context lane after baseline tail62064 covers62037->62063->62064; plus protected62038 at most2. No old job changed.',
 edits=0,fit=0,evaluation=0,qualification=0,old_job_mutations=0,new_monitor=False)
write(R/'submission.json',j);write(OUT/'submission.json',j)
print(json.dumps(dict(jobs=[r['job_id'] for r in rows],snapshot=snapshot)))
