"""Read-only live identity and two-lane proof; no scheduler mutations."""
import subprocess,re,itertools,json
from pathlib import Path
from datetime import datetime,timezone
from official.runners.server1.common import read,member,verify
from official.experiments.prepare import write_new
O=Path(__file__).parent
ROOT=Path('/mnt/raid5/janghj/ODE-edit')
def cmd(*argv): return subprocess.check_output(argv,text=True).strip()
def job(j):
    raw=cmd('scontrol','show','job',j,'-o')
    return dict(re.findall(r'(\S+?)=(.*?)(?= \S+?=|$)',raw))
queue=cmd('squeue','-h','-u','janghj','-o','%i')
live=[job(j) for j in queue.splitlines()]
own=[j for j in live if j['ReqNodeList']=='devbox' or j.get('NodeList')=='devbox']
assert {j['JobId'] for j in own}=={'62529','62530','62583','62584'},'FRESH_QUEUE_CHANGED_REASSESS'
base=ROOT/'local/official-baselines/server1'
registrations=[base/'fe-author-hparams-2k-20261010/registration-r1/submission.json',base/'completed-table-flucon-cap3-20261010/registration-r1/submission.json']
bound={};sources=[]
for path in registrations:
    s=read(path);lock=read(verify(s['execution_lock']))
    for m in lock['source_members']:verify(m)
    sources.append(dict(submission=member(path),source=s['source'],lock=s['execution_lock'],verified_source_members=len(lock['source_members'])))
    for k,j in s['jobs'].items():
        if j['job_id'] not in {x['JobId'] for x in own}:continue
        verify(j['script'])
        if j.get('config'):verify(j['config'])
        current=next(x for x in own if x['JobId']==j['job_id'])
        initial=s['initial_snapshot'][k]
        for field in ('UserId','Command','WorkDir','JobName','ReqNodeList','ReqTRES','Requeue'):
            assert current[field]==initial[field],(j['job_id'],field)
        bound[j['job_id']]=dict(script=j['script'],config=j.get('config'),source=s['source'],original_dependencies=j['dependencies'])
assert len(bound)==4
for j in own:
    assert j['UserId'].startswith('janghj(') and j['Priority']!='0'
    if j['JobState']=='PENDING':assert j['RunTime']=='00:00:00' and j['AllocTRES']=='(null)' and j['Restarts']=='0'
gpu={j['JobId']:j for j in own if 'gres/gpu=' in j['ReqTRES']}
assert set(gpu)=={'62529','62530','62583'}
assert gpu['62530']['Dependency']=='afterany:62529(unfulfilled)'
assert next(j for j in own if j['JobId']=='62584')['Dependency']=='afterany:62583(unfulfilled)'
edges=[('62529','62530')]
antichains=[s for n in range(4) for s in itertools.combinations(gpu,n) if not any(a in s and b in s for a,b in edges)]
width=max(map(len,antichains));assert width==2
allocated=sum(int(re.search(r'gres/gpu=(\d+)',j['AllocTRES']).group(1)) for j in own if 'gres/gpu=' in j['AllocTRES'])
assert allocated==2
caps=[]
for p in (ROOT/'servers/local/gpu-caps.tsv',Path('servers/local/gpu-caps.tsv')):
    rows=[x.split('\t') for x in p.read_text().splitlines() if x.startswith('server1\t')]
    assert len(rows)==1 and rows[0][1:4]==['devbox','2','183296']
    caps.append(member(p))
write_new(O/'receipt.json',dict(nonce='USER-GH-S1-S2-CAP2-LANE-ADJUST-20261010-R1',server='server1',session='01a04939-f93a-7b50-bca0-65438eab2062',
    at=datetime.now(timezone.utc).isoformat(),authority='ae2f3eb4f7fb3700dd4df68cb3f042afa115d0bb',effective_cap=2,previous_local_cap=3,caps=caps,
    jobs=own,bindings=bound,sources=sources,allocated_GPUs=allocated,legacy_overcap=False,lanes=[['62529','62530'],['62583']],
    max_GPU_antichain_width=width,cycle=False,GPU0_collector='62584',pending_dependency_changes=[],temporary_holds=0,remaining_temporary_holds=0,
    scheduler_mutations=0,source_config_unchanged=True,other_server_memory_QOS_unchanged=True,new_jobs=0,cancellations=0,
    own_preparation_WTs_without_local_cap=['odeeditsh1-completed-flucon-cap3-20261010','odeeditsh1-fe-author-20261010','odeeditsh1-refresh-s2-flucon-20261010'],
    excluded_other_node_job_count=len(live)-len(own),independent_reviewer=False,broadcast='NO_BROADCAST_NOT_REQUIRED'))
print('PASS allocation2 cap2 graphwidth2 cycle0 mutation0 source/config hashes unchanged')
