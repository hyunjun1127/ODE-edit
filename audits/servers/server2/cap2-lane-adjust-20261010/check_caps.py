import hashlib
import json
from pathlib import Path

roots=[Path('/mnt/raid5/janghj/ODE-edit')]+[
    Path('/mnt/raid5/janghj/.codex/worktrees')/('odeeditsh2-'+n) for n in (
        'completed-cap3-20261010','missing-flucon-20261010','fe-author-hparams-2k-20261010',
        'sphere-b9-resume-20261010','cap2-lane-adjust-20261010')]
rows=[]
for root in roots:
    p=root/'servers/local/gpu-caps.tsv'; b=p.read_bytes()
    own=[line.split('\t') for line in b.decode().splitlines() if line.startswith('server2\t')]
    assert own==[['server2','server2','2','60416','odeedit_*,odealloc_*']]
    rows.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),row=own[0]))
out=Path(__file__).with_name('local-caps.json')
assert not out.exists()
out.write_text(json.dumps(dict(effective_cap=2,files=rows,old_frozen_sources_modified=False),indent=2)+'\n')
print('local cap checks PASS:',len(rows))
