"""Read-only compact submission/startup and exact README-cell verification."""
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[3]
owner = root / 'audits/servers/server2/baseline-refresh-s2-flucon-20261010'
p = owner / 'submission.json'
assert hashlib.sha256(p.read_bytes()).hexdigest() == 'ae9da7a1b5d1bc283c090ef333b9077bbf036695128ff25bfd406ad5d7e3bfc5'
sub = json.loads(p.read_text()); startup = json.loads((owner / 'startup.json').read_text())
jobs = {j['job_id']: j for j in sub['jobs']}
assert set(jobs) == {str(n) for n in range(62864,62878)}
assert sub['effective_cap'] == sub['DAG_width'] == 3
assert not sub['existing_jobs_changed'] and sub['CP_transfers'] == sub['CP_deletions'] == 0
assert all(j['held_inspected'] and j['released'] and not j['GPU_complete'] for j in jobs.values())
assert all(j['source'] == sub['source'] for j in jobs.values())
assert jobs['62877']['GPU'] == 0
assert {x[1] for x in jobs['62877']['dependencies']} == {str(n) for n in range(62864,62877)}
parents = {'62864':None,'62865':None,'62866':'62864','62867':'62866','62868':'62867',
           '62869':'62865','62870':'62868','62871':'62869','62872':'62871','62873':'62870',
           '62874':'62872','62875':'62874','62876':'62875'}
for jid, parent in parents.items():
    j = jobs[jid]
    assert j['GPU'] == 1 and j['dataset'] == 'cf' and j['endpoint'] == 'W20_FIRST2000'
    assert j['dependencies'] == ([] if parent is None else [['afterany',parent]])
    assert j['checkpoint']['bytes'] > 0 and len(j['checkpoint']['sha256']) == 64
states = {x.split('|')[0]:x.split('|')[2] for x in startup['postrelease_bounded_snapshot'].splitlines()}
assert {j for j,s in states.items() if s == 'RUNNING'} == {'62864','62865'}
assert all(states[j] == 'PENDING' for j in jobs if j not in ('62864','62865'))
old = subprocess.check_output(['git','show','0e30ad58:README.md'],cwd=root,text=True)
new = (root / 'README.md').read_text()
names = {'FT':'FT','MEMIT':'MEMIT','AlphaEdit':'ALPHAEDIT','AlphaEdit-BLUE':'ALPHAEDIT_BLUE',
         'MEMIT-FE':'MEMIT_FE','AlphaEdit+SPHERE':'SPHERE'}
def table(text):
    out=[]; model=None
    for line in text.splitlines():
        if line.startswith('### '):
            model={'### Llama3-8B-Instruct':'llama3','### Qwen2.5-7B-Instruct':'qwen25','### GPT-J-6B':'gptj'}.get(line)
        if line.startswith('|'):
            out.append((model,[c.strip() for c in line.strip('|').split('|')]))
    return out
a,b=table(old),table(new);assert len(a)==len(b)
lookup={(j['model'],j['method']):j for j in jobs.values() if j['GPU']==1}
changed=[]
for (m,x),(n,y) in zip(a,b):
    assert m==n
    if x==y:continue
    assert len(x)==len(y)==10 and {i for i in range(10) if x[i]!=y[i]}=={5,6}
    j=lookup[(m,names[y[0]])]
    value=f"{'ING' if states[j['job_id']]=='RUNNING' else 'PENDING'}: {j['job_name']} ({j['job_id']})"
    assert y[5]==y[6]==value
    changed.append(j['job_id'])
assert set(changed)==set(parents) and len(changed)==13
print(json.dumps({'PASS':True,'generation_status_cells':26,'numeric_cells_changed':0,
                  'GPU_evals':13,'CPU_collectors':1,'cap':3,'GH_GPU_calls':0}))
