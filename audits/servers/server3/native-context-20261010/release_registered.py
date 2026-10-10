"""Release existing held IDs only; fix pending node-filter inspection, no resubmit."""
import json,pathlib,subprocess,hashlib,datetime,re
R=pathlib.Path('/data/janghj/ODE-edit/local/native-context-server3-20261010/execution-r1')
def call(*a):return subprocess.check_output(a,text=True,timeout=30).strip()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
rows=json.loads((R/'held-inspection.json').read_text());ids={x['job_id'] for x in rows};assert ids=={'62101','62102','62103'}
queue=call('squeue','-u','janghj','-h','-o','%i');mine=[]
for job in queue.splitlines():
 raw=call('scontrol','show','job',job,'--oneliner')
 if 'ReqNodeList=ubuntu ' in raw or 'NodeList=ubuntu ' in raw:
  assert 'UserId=janghj(' in raw
  mine.append(job)
assert set(mine)==ids,('ADMISSION_RACE',mine)
for row in rows:
 raw=call('scontrol','show','job',row['job_id'],'--oneliner')
 for token in ['JobName='+row['job_name'],'UserId=janghj(','JobState=PENDING','Reason=JobHeldUser','Priority=0','ReqNodeList=ubuntu','Command='+str(R/'run.sh'),'Requeue=0','QOS=lab_gpu_s3','MinMemoryNode=58G','TresPerNode=gres/gpu:h200:1']:
  assert token in raw,token
 if row['dependency']:assert row['dependency'] in raw
for row in rows:call('scontrol','release',row['job_id'])
snapshot=call('squeue','-h','-j',','.join(sorted(ids)),'-o','%i|%j|%T|%E')
d=dict(nonce='USER-SH4-SH3-NATIVE-CONTEXT-MIGRATION-20261010-R1',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,status='REGISTERED_RELEASED',snapshot=snapshot,project_cap=1,task_cap=1,resource_proof='All own jobs exact owner plus ReqNodeList/NodeList accounting: these3 held only; afterok lane width1',launcher_sha256=sha(R/'run.sh'),cessation_sha256=sha(R/'sh4-cessation.json'),source='70b63f001ae200eed70d5278028b1cc7d33ce009',runtime={'python':'/data/janghj/ODE-edit/local/runtime/price-s4-mirror-v1/venv/bin/python','torch':'2.9.1+cu128','transformers':'4.57.1'},pre_release_diagnostic='Original -w ubuntu filter omitted unallocated PENDING; held preserved, corrected inspection, no duplicate submit',old_job_mutations=0,new_science_runs=0)
for p in [R/'submission.json',pathlib.Path(__file__).parent/'submission.json']:
 with p.open('x') as f:json.dump(d,f,indent=2)
print(snapshot)
