"""Exact pending-only resource edges; no science/source mutation."""
from pathlib import Path
from datetime import datetime,timezone
import copy,getpass
from official.runners.server1 import submit as s
from official.experiments.prepare import write_new,file_sha
LOCAL=Path('/mnt/raid5/janghj/ODE-edit/local/server1-two-lanes-20261011')
AUDIT=Path(__file__).parent
before=s.read(LOCAL/'before.json')
assert before['allocated_gpus']<=2
rows={r['key']:r for r in before['jobs']}
changes={'63217':[], '63296':['63217'], '63297':['63218']}
expected={'63217':['63154'],'63296':['63217','63218'],'63297':['63217','63218']}
roots=[Path('/mnt/raid5/janghj/ODE-edit/local')/v for v in ('cake-six-2k-server1-20261011/registration-r1','fe-original-w0-2k-20261011/qwen-server1-move-r1/registration-r1','fe-original-w0-2k-20261011/registration-r1')]
bindings={}
for root in roots:
 sub=s.read(root/'submission.json')
 for key,j in sub['jobs'].items():
  if j['job_id'] not in rows:continue
  s.verify(j['script']);s.verify(j['config'])
  lock=root/'execution-lock.json';lk=s.read(lock)
  for m in lk['source_members']:s.verify(m)
  d=rows[j['job_id']]['detail']
  assert d['UserId'].split('(')[0]==getpass.getuser() and d['Command']==j['script']['path'] and d['WorkDir']==str(root/'source') and d['JobName']==j['name']
  assert s.command(['scontrol','write','batch_script',j['job_id'],'-']).strip()==Path(j['script']['path']).read_text().strip()
  bindings[j['job_id']]=dict(source=sub['source'],script=j['script'],config=j['config'],lock=s.member(lock),argv=j['argv'])
assert set(bindings)==set(rows),'UNRESOLVED_OWN_GPU_SOURCE'
for jid in changes:
 assert rows[jid]['state']=='PENDING' and sorted(rows[jid]['parents'])==sorted(expected[jid])
 assert all(k=='afterany' for k,_ in s.dependencies(rows[jid]['detail']['Dependency'])),'INPUT_GATE_PROTECTED'
plan=copy.deepcopy(before['jobs'])
for r in plan:
 if r['key'] in changes:r['parents']=changes[r['key']]
assert s.graph_width(plan)==2
write_new(AUDIT/'before.json',before)
write_new(AUDIT/'plan.json',dict(changes=changes,bindings=bindings,after_width=2,input_afterok_removed=0))
held=[];updates={}
def current(jid):
 d=s.metadata(s.command(['scontrol','show','job',jid,'--oneliner']))
 assert d['UserId'].split('(')[0]==getpass.getuser() and d['Command']==rows[jid]['detail']['Command'] and d['WorkDir']==rows[jid]['detail']['WorkDir']
 return d
def pending(d):
 assert d['JobState']=='PENDING' and d['RunTime']=='00:00:00' and not s.gpu_count(d.get('AllocTRES','')) and d['NodeList'] in ('','(null)')
try:
 for jid in ['63296','63297','63217']:
  d=current(jid);pending(d)
  assert sorted(p for _,p in s.dependencies(d.get('Dependency','')))==sorted(expected[jid])
  s.command(['scontrol','hold',jid]);held.append(jid);pending(current(jid))
 live=s.inventory();assert live['allocated_gpus']<=2 and {r['key'] for r in live['jobs']}==set(rows),'ADMISSION_RACE_RECONCILE'
 for jid in ['63296','63297','63217']:
  d=current(jid);pending(d)
  dep='afterany:'+':'.join(changes[jid]) if changes[jid] else ''
  s.command(['scontrol','update','JobId='+jid,'Dependency='+dep])
  a=current(jid);pending(a)
  assert s.dependencies(a.get('Dependency',''))==[('afterany',p) for p in changes[jid]]
  updates[jid]=dict(before=d,after=a)
 checked=s.inventory();assert s.graph_width(checked['jobs'])==2
 write_new(AUDIT/'held-updates.json',dict(updates=updates,inventory=checked))
finally:
 for jid in held:s.command(['scontrol','release',jid])
after=s.inventory();after['observed_at']=datetime.now(timezone.utc).isoformat()
assert after['allocated_gpus']<=2 and after['admitted_DAG_width']==2
for jid,b in bindings.items():
 s.verify(b['script']);s.verify(b['config']);s.verify(b['lock'])
for r in after['jobs']:
 assert r['detail']['Reason']!='JobHeldUser','TEMPORARY_HOLD_REMAINS'
write_new(AUDIT/'after.json',after)
write_new(AUDIT/'receipt.json',dict(nonce='USER-GH-SERVER1-TWO-LANES-20261011-R1',changes=changes,
 lanes=[['63217','63296','63298','63300'],['63154','63218','63297','63299','63301']],before_allocated=before['allocated_gpus'],after_allocated=after['allocated_gpus'],
 width=2,cycle=False,temporary_holds_remaining=0,source_config_script_unchanged=True,source_closure_verified=True,
 new_jobs=0,cancelled_jobs=0,observed_at=after['observed_at']))
print({r['key']:(r['state'],r['detail']['Reason'],r['parents']) for r in after['jobs']})
