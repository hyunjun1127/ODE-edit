"""Compact evidence and exact-cell check; no model/scheduler/raw mutation."""
import hashlib
import json
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

root = Path(__file__).resolve().parents[3]
p = root / 'audits/servers/server2/baseline-refresh-s2-flucon-20261010/table-rows.json'
assert hashlib.sha256(p.read_bytes()).hexdigest() == 'ea0b0c6df054428565e0a489d8df88fd04f118442235a992a9d6fdefcfeb5549'
d = json.loads(p.read_text())
assert d['numeric_eligible'] == 23 and d['issues'] == []
rows = {r['job_id']: r for r in d['rows']}
for jid in ('62075', '62083', '62085'):
    r = rows[jid]
    assert r['numeric_eligible'] and r['requests'] == 2000 and r['commits'] == 20
    assert r['checkpoint_receipt']['final_W20'] and r['checkpoint_receipt']['batch'] == 20
    if r['dataset'] == 'zsre':
        a = r['zsre_audit']; q = a['query_proof']
        assert a['Loc'] == 'loc_ans_target_correctness_NOT_W0_agreement' and a['missing_requests'] == 0
        assert q['requests'] == 2000 and q['queries'] == 24858
        assert q['input_mismatches'] == q['target_mismatches'] == 0
        assert r['denominators'] == {'rewrite': 6691, 'paraphrase': 6691, 'neighborhood': 11476}
assert rows['62085']['metrics']['Efficacy'] == rows['62085']['metrics']['Generalization'] == 0
old = subprocess.check_output(['git', 'show', '417a12e7:README.md'], cwd=root, text=True)
new = (root / 'README.md').read_text()
def table(text):
    return [[s.strip() for s in line.strip('|').split('|')] for line in text.splitlines() if line.startswith('|')]
a, b = table(old), table(new)
assert len(a) == len(b)
changed = [(x, y, {i for i, (v,w) in enumerate(zip(x,y)) if v != w}) for x,y in zip(a,b) if x != y]
assert len(changed) == 3
def display(jid, field):
    return str(Decimal(str(rows[jid]['metrics'][field])).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
for before, after, fields in changed:
    if after[0] == 'AlphaEdit':
        assert fields == {1,2,3,4,7,8,9}
        assert after[1:5] == [display('62075', f) for f in ('Score','Efficacy','Generalization','Specificity')]
        assert after[7:10] == [display('62083', f) for f in ('Efficacy','Generalization','Specificity')]
    elif after[0] == 'MEMIT-FE':
        assert fields == {7,8,9}
        assert after[7:10] == [display('62085', f) for f in ('Efficacy','Generalization','Specificity')]
    else:
        assert after[:4] == ['MEMIT_FE_HISTORY (FE author hparams)','Qwen2.5','CF','server2']
        assert fields == {4} and after[4] == 'ING: s2-qwen25-cf-fe-author-history (62531)'
assert sum(len(f) for _,_,f in changed) == 11
print(json.dumps({'PASS':True, 'numeric_cells':10, 'status_cells':1, 'other_table_changes':0, 'GPU_calls':0}))
