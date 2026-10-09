"""One-shot source-cessation gated S3 sequential context-only submission."""
import subprocess,json,pathlib,hashlib,shutil,datetime
R=pathlib.Path('/data/janghj/ODE-edit/local/native-context-server3-20261010/execution-r1')
def call(*args):return subprocess.check_output(args,text=True,timeout=40).strip()
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,v):
 with pathlib.Path(p).open('x') as f:json.dump(v,f,indent=2)
assert not (R/'submission.json').exists()
cessation=read(R/'cessation-verified.json');assert cessation['verified'] and cessation['jobs']==['62090','62091','62092']
assert shutil.disk_usage(R).free>4*2**30
q=call('squeue','-u','janghj','-w','ubuntu','-h','-o','%i|%j|%T|%b|%E');assert not q,'ADMISSION_CHANGED_KEEP_UNSUBMITTED'
freeze=read(R/'freeze.json');rows=[];dep=None
for family in ('qwen25','gptj','llama3'):
 c=read(R/f'{family}.json');assert sha(R/f'{family}.json')==freeze['configs'][family]
 name='s3-native-context-'+family
 argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s3','--nodelist=ubuntu','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=59392M','--gres=gpu:h200:1','--time=02:00:00','--export=NONE','--no-requeue','--job-name='+name,'--chdir='+str(R/'source'),'--output='+str(R/(family+'-%j.out')),'--error='+str(R/(family+'-%j.err'))]
 if dep:argv+=['--dependency='+dep]
 argv += [str(R/'run.sh'),family]
 job=call(*argv).split(';')[0];assert job.isdigit();write(R/(family+'-registered.json'),dict(job_id=job,argv=argv))
 raw=call('scontrol','show','job',job,'--oneliner')
 for token in ['UserId=janghj(', 'JobName='+name,'Priority=0','JobState=PENDING','Requeue=0','CPUs/Task=8','MinMemoryNode=58G','ReqNodeList=ubuntu','QOS=lab_gpu_s3','TimeLimit=02:00:00','Command='+str(R/'run.sh'),'WorkDir='+str(R/'source'),'TresPerNode=gres/gpu:h200:1']:
  assert token in raw,(job,token)
 if dep:assert dep in raw
 rows.append(dict(family=family,job_id=job,job_name=name,dependency=dep,source=freeze['source'],config_file_sha256=sha(R/f'{family}.json'),config_sha256=c['config_sha256'],model_manifest_sha256=c['model_manifest_sha256'],ready_path=c['out']+'/READY.json',held_record=raw,argv=argv))
 dep='afterok:'+job
write(R/'held-inspection.json',rows)
active=call('squeue','-u','janghj','-w','ubuntu','-h','-o','%i');assert set(active.splitlines())=={r['job_id'] for r in rows},'ADMISSION_RACE_KEEP_HELD'
for row in rows:call('scontrol','release',row['job_id'])
snap=call('squeue','-h','-j',','.join(r['job_id'] for r in rows),'-o','%i|%j|%T|%E')
r=dict(nonce='USER-SH4-SH3-NATIVE-CONTEXT-MIGRATION-20261010-R1',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='REGISTERED_RELEASED',rows=rows,snapshot=snap,cap=1,task_cap=1,resource_proof='all three GPU jobs single afterok lane; own ubuntu queue empty at admission; physical/QoS stricter retained',launcher_sha256=sha(R/'run.sh'),runtime_python='/data/janghj/ODE-edit/local/runtime/price-s4-mirror-v1/venv/bin/python',runtime={'torch':'2.9.1+cu128','transformers':'4.57.1'},cessation_receipt_sha256=sha(R/'cessation-verified.json'),old_jobs_mutated=0,edits=0,fit=0,evaluation=0,WandB_science_runs=0)
write(R/'submission.json',r);write(pathlib.Path(__file__).parent/'submission.json',r);print(json.dumps({'jobs':[x['job_id'] for x in rows],'snapshot':snap}))
