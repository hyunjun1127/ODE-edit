"""Bounded exact source/health reconciliation and pending-only USER repair stop."""
import datetime, hashlib, json, subprocess
from pathlib import Path
BASE=Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1')
OLD=BASE/'registration-no-gpu-qual-r1'
OUT=BASE/'cf-display-reconcile-r1'
SOURCE='478464689231595eb63da368cc7c376e775eb559'
PENDING_CF=['CF_SPHERE','CF_MEMIT_FE','CF_ALPHAEDIT_BLUE','CF_ALPHAEDIT']
def read(p): return json.loads(p.read_text())
def cmd(argv): return subprocess.check_output(argv,text=True).strip()
def save(name,value):
 with (OUT/name).open('x') as stream: json.dump(value,stream,indent=2)
def detail(role,j):
 result=subprocess.run(['scontrol','show','job',str(j),'-o'],text=True,capture_output=True)
 if result.returncode:
  row=cmd(['sacct','-n','-P','-X','-j',str(j),'--format=JobIDRaw,User,State,NodeList']).splitlines()[0].split('|')
  assert row[0]==str(j) and row[1]=='janghj' and row[2].split()[0] in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY'}
  return dict(JobId=str(j),JobState=row[2],accounting=row,terminal=True)
 f=dict(t.split('=',1) for t in result.stdout.split() if '=' in t)
 assert f['UserId']=='janghj(1025)' and f['ReqNodeList']=='server2'
 assert f['Command']==str(OLD/(role+'.sh')) and f['WorkDir']==str(OLD/'source')
 return {k:f.get(k) for k in ('JobId','JobName','JobState','UserId','Command','WorkDir','Dependency','RunTime','NodeList')}
def main():
 OUT.mkdir(exist_ok=False)
 sub=read(OLD/'submission.json');jobs=sub['jobs']
 assert sub['source']['code_commit']==SOURCE
 assert hashlib.sha256((OLD/'execution.lock.json').read_bytes()).hexdigest()==sub['lock']['sha256']
 assert read(OLD/'manifest.json')['code_commit']==SOURCE
 states={role:detail(role,j) for role,j in jobs.items()}
 health={}
 for role in ['W0_CF','CF_MEMIT',*PENDING_CF]:
  root=OLD/role
  transport=read(root/'logging-transport.json') if (root/'logging-transport.json').is_file() else None
  receipt=read(root/'tracking/receipt.json') if (root/'tracking/receipt.json').is_file() else None
  commits=sorted((root/'commits').glob('batch-*.json'))
  accepted=root/'tracking/accepted-scalars.jsonl'
  health[role]=dict(state=states[role],commits=len(commits),
   latest_commit=None if not commits else dict(batch=read(commits[-1])['batch'],path=str(commits[-1])),
   logging=transport,tracking=None if receipt is None else {k:receipt.get(k) for k in ('status','dropped_points','run_id','startup_readback')},
   accepted_scalar_bytes=accepted.stat().st_size if accepted.exists() else 0,
   actual_factual_endpoints=[p.name for p in (root/'factual').glob('*.json')])
 assert health['W0_CF']['logging']['rejected_calls']==1
 assert health['W0_CF']['tracking']['dropped_points']==1
 assert health['CF_MEMIT']['commits']>=1 and health['CF_MEMIT']['tracking']['dropped_points']==0
 save('health-before.json',dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=SOURCE,states=states,health=health))
 actions=[];cancelled=[]
 # Resource-only hold protects the untouched zsRE successor while its CF edge is replaced.
 for role in ['collector','ZSRE_SPHERE',*PENDING_CF]:
  before=detail(role,jobs[role])
  if before['JobState']=='PENDING':
   cmd(['scontrol','hold',jobs[role]])
   actions.append(dict(role=role,job=jobs[role],action='hold',before=before,after=detail(role,jobs[role])))
 for role in ['collector',*PENDING_CF]:
  before=detail(role,jobs[role])
  if before['JobState']=='PENDING':
   cmd(['scancel',jobs[role]])
   cancelled.append(role);action='cancel_affected_pending'
  else: action='KEEP_NONPENDING_REQUIRES_SEPARATE_HEALTH_EVIDENCE'
  actions.append(dict(role=role,job=jobs[role],action=action,before=before))
  save('action-'+role+'.json',actions[-1])
 result=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=SOURCE,
  cancelled={r:jobs[r] for r in cancelled},actions=actions,
  healthy_keep={'CF_MEMIT':jobs['CF_MEMIT']},protected_zsre={r:j for r,j in jobs.items() if r.startswith('ZSRE_')},
  pending_control_hold='ZSRE_SPHERE' if any(a['role']=='ZSRE_SPHERE' and a['action']=='hold' for a in actions) else None,
  original_raw_CP_source_KEEP=True)
 save('receipt.json',result)
 print(json.dumps(result))
if __name__=='__main__':main()
