"""GPT-J W0 Flu/Con 두 셀만 바뀌었는지 CPU 검산한다."""
import subprocess
from pathlib import Path
import sys
import json

root = Path(__file__).resolve().parents[3]
before = subprocess.check_output(['git','show',sys.argv[1]+':README.md'],cwd=root,text=True)
after = (root/'README.md').read_text()

def rows(text):
    return [line for line in text.splitlines() if line.startswith('|')]

old,new=rows(before),rows(after)
assert len(old)==len(new)
changed=[i for i,(a,b) in enumerate(zip(old,new)) if a!=b]
assert len(changed)==1
i=changed[0]
assert old[i].startswith('| [W0 (편집 전)]') and '24.44 | 17.00 | 19.30 | 82.48' in old[i]
a,b=old[i].split('|'),new[i].split('|')
assert [j for j in range(len(a)) if a[j]!=b[j]]==[6,7]
assert b[6].strip()==b[7].strip()==sys.argv[2]
assert a[1:6]==b[1:6] and a[8:]==b[8:]
integration=json.loads((Path(__file__).parent/'integration.json').read_text())
submission=json.loads((root/integration['submission']).read_text())
for key in ('job_id','job_name','state','dependency','held_inspection','released','runtime_source','config_sha256'):
    assert integration[key]==submission[key], key
assert submission['cap_proof']['cap']==2 and submission['cap_proof']['existing_job_mutations']==0
assert submission['resources']['GPU']==1 and submission['noCP']
assert submission['duplicate_result']=='NOT_FOUND_IN_BOUNDED_SCOPE'
print('PASS: GPT-J W0 CF Flu/Con 2개 셀만 수정, 나머지 표 불변.')
