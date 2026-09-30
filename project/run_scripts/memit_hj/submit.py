"""Register the complete cap1 physical DAG held, inspect, release in reverse."""
import argparse,fnmatch,json,os,re,subprocess
from pathlib import Path
from project.run_scripts.memit_history_lifelong.io import save,file_sha

GROUPS=['P','A','B','C','D','CPU']
def dependency(group,jobs,external):
 if group=='P':return 'afterany:'+':'.join(external) if external else None
 if group=='A':return 'afterok:'+jobs['P']
 if group=='CPU':return 'afterany:'+':'.join(jobs[g] for g in 'PABCD')
 previous={'B':'A','C':'B','D':'C'}[group]
 return 'afterok:'+jobs['P']+',afterany:'+jobs[previous]
def run(argv):
 p=subprocess.run(argv,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(f'{argv[0]} exit={p.returncode}: {p.stderr}')
 return p.stdout
def fields(text):
 matches=list(re.finditer(r'(?:^|\s)([A-Za-z][A-Za-z0-9_:/]*)=',text));return {m.group(1):text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip() for i,m in enumerate(matches)}
def inspect(job,argv,script,lock,owner,dep,group):
 f=fields(run(['scontrol','show','job',job,'--oneliner']));gpu=group!='CPU'
 dep_ids=re.findall(r'\d+',dep or '')
 checks=dict(owner=f.get('UserId','').split('(')[0]==owner,job=f.get('JobId')==job,
  name=f.get('JobName')=='odeedit_memit_hj_'+group+'_s3',held=f.get('JobState')=='PENDING' and f.get('Priority')=='0',
  node=f.get('ReqNodeList')=='ubuntu',partition=f.get('Partition')=='gpu',cpu=f.get('NumCPUs')=='8' and f.get('CPUs/Task')=='8',
  gpu=('gres/gpu=1' in f.get('ReqTRES','') and f.get('TresPerNode')=='gres/gpu:1') if gpu else 'gres/gpu' not in f.get('ReqTRES',''),
  memory=f.get('MinMemoryNode') in (('119G','121856M') if gpu else ('32G','32768M')),
  requeue=f.get('Requeue')=='0',wall=f.get('TimeLimit')==('30-00:00:00' if gpu else '04:00:00'),
  command=f.get('Command')==str(script),argv=f.get('SubmitLine')==' '.join(argv),
  dependency=all(x in f.get('Dependency','') for x in dep_ids) if dep else f.get('Dependency')=='(null)',
  source=file_sha(script)==lock['launchers'][script.name])
 # Check every dependency TYPE too; afterany must not masquerade as afterok.
 for kind in ['afterok','afterany']:
  if dep and kind in dep:checks['dependency_'+kind]=kind in f.get('Dependency','')
 return dict(job_id=job,group=group,checks=checks,all_pass=all(checks.values()),fields=f)
def submit(lock_path):
 path=Path(lock_path).resolve();lock=json.loads(path.read_text());attempt=path.parent
 assert lock['resource']['cap']==1 and lock['resource']['memory_mib']==121856
 owner=run(['id','-un']).strip();assert owner=='janghj'
 # Create-once reservation survives a failed submission command. It prevents
 # automatic retry from making a duplicate, including a lost sbatch response.
 save(attempt/'submission-reservation.json',dict(nonce=lock['instruction'],source_commit=lock['source_commit'],lock_sha256=file_sha(path)))
 queue=run(['squeue','-a','-h','-u',owner,'-w','ubuntu','-o','%i|%j|%T|%b|%N'])
 external=[]
 for line in queue.splitlines():
  jid,name,state,gres,node=line.split('|');assert not name.startswith('odeedit_memit_hj_'),'EXISTING_SAME_TASK:'+jid
  if any(fnmatch.fnmatch(name,p) for p in ['odeedit_*','bfode_*','motivation_*','session01_*','project_*']):
   assert re.fullmatch(r'\d+(?:_\d+)?',jid);external.append(jid)
 save(attempt/'admission.json',dict(owner=owner,queue_resource_snapshot=queue,external_project_jobs=external,cap=1,maximum_DAG_concurrent_GPUs=1))
 jobs={};receipts=[];(attempt/'logs').mkdir(exist_ok=True)
 for group in GROUPS:
  script=Path(lock['source_root'])/'project/run_scripts/memit_hj'/('collect.sbatch' if group=='CPU' else 'run.sbatch')
  dep=dependency(group,jobs,external)
  argv=['sbatch','--parsable','--hold','--export=NONE','--kill-on-invalid-dep=yes','--job-name=odeedit_memit_hj_'+group+'_s3',
    '--output='+str(attempt/'logs/%j.out'),'--error='+str(attempt/'logs/%j.err')]
  if dep:argv+=['--dependency='+dep]
  argv += [str(script),lock['source_root'],str(path),str(attempt/'dag-submission.json') if group=='CPU' else group]
  result=run(argv).strip();job=result.split(';')[0];assert job.isdigit();jobs[group]=job
  receipt=dict(group=group,job_id=job,argv=argv,dependency=dep,source_commit=lock['source_commit'],lock_sha256=file_sha(path),response=result)
  save(attempt/f'submission-{group}.json',receipt)
  check=inspect(job,argv,script,lock,owner,dep,group);save(attempt/f'held-{group}.json',check)
  assert check['all_pass'],'HELD_INSPECTION_FAILED: '+str(check['checks'])
  receipts.append(receipt)
 mapping=dict(source_commit=lock['source_commit'],lock_sha256=file_sha(path),jobs=receipts,logical_mapping=lock['plan']['groups'],cap=1)
 save(attempt/'dag-submission.json',mapping)
 # Release consumers first so no runnable producer can finish before its
 # already-registered consumer is released. Dependencies retain cap1.
 for group in reversed(GROUPS):
  response=run(['scontrol','release',jobs[group]])
  save(attempt/f'release-{group}.json',dict(job_id=jobs[group],returncode=0,stdout=response,status='RELEASED',actual_initial='NOT_OBSERVED',actual_terminal='NOT_OBSERVED'))
 print(json.dumps(dict(status='ALL_SIX_REGISTERED_INSPECTED_RELEASED',jobs=jobs,actual_initial='NOT_OBSERVED')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--lock',required=True);submit(p.parse_args().lock)
