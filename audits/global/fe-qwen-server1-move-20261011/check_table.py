"""실제 owner 영수증과 Qwen FE 한 행만 검산; GPU/Slurm 호출 없음."""
import json
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[3]
old=subprocess.check_output(['git','show','7e778b86:README.md'],cwd=root,text=True)
new=(root/'README.md').read_text()
tables=lambda text:[s for s in text.splitlines() if s.startswith('|')]
a,b=tables(old),tables(new)
assert len(a)==len(b)
delta=[i for i in range(len(a)) if a[i]!=b[i]]
assert len(delta)==1
i=delta[0]
assert a[i]=='| FE (author repo, W0-fixed z) |  |  |  |  |  |  |  |  |  |'
assert [x.strip() for x in b[i].split('|')[2:-1]]==['PENDING: 63217']*4+['DEFERRED']*2+['PENDING: 63218']*3
s=json.loads((root/'audits/servers/server1/fe-qwen-server1-move-20261011/submission.json').read_text())
assert s['held_inspected'] and s['released']
assert {j['job_id'] for j in s['jobs']}=={'63217','63218'}
for j in s['jobs']:
    assert j['state']=='PENDING'
    assert {tuple(d) for d in j['dependencies']}=={('afterany',x) for x in ('63152','63154','63207')}
assert s['new_execution_width']==2 and s['legacy_allocated_GPU']==3
assert s['server2_duplicate_check']['submitted']==0
assert s['disk']['available_bytes']>=s['disk']['required_free_bytes']
assert s['source']=='2bac5732fc594e0eaf0d200772a56a640e02bc21'
assert 'PENDING: 63219' in new
print('PASS: Qwen FE9셀/실제2IDs·의존성/SH2중복0/비FE·PRICE·W0 포함 다른 표 불변')
