"""Exact old FE stop with non-FE resource lane bypass preserved."""
import json,subprocess
from pathlib import Path
from official.runners.server1.submit import metadata,dependencies
from official.experiments.prepare import file_sha,write_new
B=Path('/mnt/raid5/janghj/ODE-edit/local');O=Path(__file__).parent
def call(*a):return subprocess.check_output(a,text=True,timeout=40)
def get(j):return metadata(call('scontrol','show','job','-o',j))
expected={}
for task in ('baseline-refresh-s2-flucon-20261010','fe-author-hparams-2k-20261010'):
 root=B/task/'registration-r1';s=json.loads((root/'submission.json').read_text())
 for r in s['jobs'].values():expected[str(r['job_id'])]=(r,root,s['source'])
targets=['62868','62872','62532'];edges={'62870':'62867','62874':'62871'}
def check(j):
 d=get(j);r,root,source=expected[j]
 assert d['UserId'].startswith('janghj(') and d['ReqNodeList']=='server2'
 assert d['Command']==r['script']['path'] and d['WorkDir']==str(root/'source') and d['JobName']==r['name']
 assert file_sha(r['script']['path'])==r['script']['sha256']
 assert file_sha(r['config']['path'])==r['config']['sha256']
 return d
before={j:check(j) for j in targets+list(edges)}
write_new(O/'stop-before.json',dict(jobs=before,source={j:expected[j][2] for j in before}))
ops=[]
for j in edges:
 d=check(j);assert d['JobState']=='PENDING' and d['RunTime']=='00:00:00';call('scontrol','hold',j);ops.append(['hold',j])
for j,p in edges.items():
 d=check(j);pairs=set(dependencies(d['Dependency']));pairs.add(('afterany',p))
 call('scontrol','update','JobId='+j,'Dependency='+','.join(a+':'+b for a,b in sorted(pairs)))
 assert set(dependencies(get(j)['Dependency']))==pairs
 ops.append(['resource_union',j,p])
for j in targets:
 d=check(j)
 if d['JobState'] in ('PENDING','RUNNING','CONFIGURING','COMPLETING'):
  call('scancel',j);ops.append(['scancel',j])
 else:ops.append(['terminal_keep',j,d['JobState']])
for j in edges:
 call('scontrol','release',j);ops.append(['release',j])
after={j:get(j) for j in before}
write_new(O/'stop-after.json',dict(jobs=after,operations=ops,non_FE_cancelled=0,
 mixed_collector='62877 KEEP; reads compact metadata, never restores CP',
 no_new_submit=True))
print(json.dumps({j:d['JobState'] for j,d in after.items()}))
