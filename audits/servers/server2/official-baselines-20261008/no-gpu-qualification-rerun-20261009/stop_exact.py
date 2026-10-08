"""One-shot exact USER-authorized old pipeline stop, all raw kept."""
import datetime, hashlib, json, subprocess
from pathlib import Path
ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1')
OUT=ROOT/'no-gpu-qualification-stop-r2'
SOURCES={'registration-cf-checkpoint-r1':'6d35c65de19ec378a30749683762eba87fee50e3',
 'registration-zsre-r1':'ccc1f5d683349b75d1780c49bcf1116b370b91a2'}
ORDER=[61672,61656,61671,61670,61669,61668,61667,61666,61655,61654,61653,61652,61651,61650]
ACTIVE={'PENDING','RUNNING','CONFIGURING','COMPLETING','SUSPENDED'}
def save(name,value):
 with (OUT/name).open('x') as f: json.dump(value,f,indent=2)
def command(argv): return subprocess.check_output(argv,text=True).strip()
def detail(j,role,root):
 if int(j)==61650:
  raw=command(['sacct','-n','-P','-X','-j',str(j),'--format=JobIDRaw,User,State,NodeList'])
  row=raw.splitlines()[0].split('|')
  assert row[:4]==['61650','janghj','COMPLETED','server2']
  terminal=json.loads((root/'FT/pipeline-terminal.json').read_text())
  assert terminal['status']=='EDIT_FACTUAL_CHECKPOINT_COMPLETE'
  return dict(raw=raw,JobId=str(j),JobState='COMPLETED',source_evidence='original exact submission/terminal')
 raw=command(['scontrol','show','job',str(j),'-o'])
 f=dict(t.split('=',1) for t in raw.split() if '=' in t)
 assert f['UserId']=='janghj(1025)' and f['ReqNodeList']=='server2'
 assert f['Command']==str(root/(role+'.sh')) and f['WorkDir']==str(root/'source')
 return {'raw':raw,**{k:f[k] for k in ('JobId','JobName','UserId','JobState','Command','WorkDir','Dependency')}}
def main():
 OUT.mkdir(exist_ok=False)
 targets={}
 for name,sha in SOURCES.items():
  root=ROOT/name;s=json.loads((root/'submission.json').read_text())
  assert s['source']['code_commit']==sha
  assert hashlib.sha256((root/'execution.lock.json').read_bytes()).hexdigest()==s['lock']['sha256']
  for role,j in s['jobs'].items():
   targets[int(j)]=dict(role=role,attempt=str(root),source=sha,before=detail(j,role,root))
 assert set(targets)==set(ORDER)
 save('targets-before.json',targets)
 actions=[]
 for j in ORDER:
  t=targets[j];d=detail(j,t['role'],Path(t['attempt']))
  if d['JobState']!='PENDING': continue
  subprocess.run(['scontrol','hold',str(j)],check=True)
  actions.append(dict(job=j,action='hold',before=d))
  save('held-'+str(j)+'.json',actions[-1])
 for j in ORDER:
  t=targets[j];d=detail(j,t['role'],Path(t['attempt']))
  action='cancel' if d['JobState'] in ACTIVE else 'KEEP_TERMINAL'
  if action=='cancel': subprocess.run(['scancel',str(j)],check=True)
  item=dict(job=j,role=t['role'],source=t['source'],action=action,before=d)
  actions.append(item);save('action-'+str(j)+'.json',item)
 after=command(['sacct','-n','-P','-j',','.join(map(str,ORDER)),
  '--format=JobIDRaw,JobName%100,User,State,ExitCode,NodeList,Elapsed,AllocTRES%100'])
 queue=command(['squeue','-h','-j',','.join(map(str,ORDER)),'-o','%i|%T|%r'])
 result=dict(instruction='USER-GH-SH1-SH2-OFFICIAL-NO-GPU-QUAL-RERUN-20261009-R1',
  timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),targets=targets,actions=actions,
  after_accounting=after,after_queue=queue,raw_source_CP_KEEP=True,new_submission=False)
 save('receipt.json',result)
 print(json.dumps(dict(cancelled=[a['job'] for a in actions if a['action']=='cancel'],
  kept=[a['job'] for a in actions if a['action']=='KEEP_TERMINAL'],after_queue=queue,accounting=after)))
if __name__=='__main__':main()
