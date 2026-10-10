"""Bounded compact-receipt/README verification; no model or scheduler access."""
import hashlib
import json
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = 'f74dcdae5938b6ebab4a8a5944b5644ee948e079'
owner = ROOT / 'audits/servers/server1/completed-table-flucon-cap3-20261010'
raw = (owner / 'table-rows.json').read_bytes()
assert hashlib.sha256(raw).hexdigest() == '9ff004351c8e6012d13284b9404a35d57c53cba2657f3deecb77f2aa0d0c3f2e'
rows = json.loads(raw)
sub = json.loads((owner / 'submission.json').read_text())
assert sub['effective_cap'] == sub['admitted_width'] == 3
assert sub['held_inspected'] and sub['released'] and sub['existing_jobs_unchanged']
assert not sub['new_generation_complete']
jobs = {j['job_id']: j for j in sub['jobs']}
assert set(jobs) == {'62581', '62582', '62583', '62584'}
assert all(j['initial_state'] == 'PENDING' for j in jobs.values())
assert jobs['62583']['dependency'] == [['afterany', '62581']]
assert jobs['62581']['dependency'] == jobs['62582']['dependency'] == []
assert jobs['62584']['GPU'] == 0
old = subprocess.check_output(['git', 'show', f'{BASE}:README.md'], cwd=ROOT, text=True)
new = (ROOT / 'README.md').read_text()
def tables(text):
    return [[c.strip() for c in line.strip('|').split('|')]
            for line in text.splitlines() if line.startswith('|')]
a, b = tables(old), tables(new)
assert len(a) == len(b)
changes = [(x, y, {i for i, (v, w) in enumerate(zip(x, y)) if v != w})
           for x, y in zip(a, b) if x != y]
assert len(changes) == 7
numeric = {g['job_id']: g for g in rows['generation'] if g['status'] == 'W20_GENERATION_CPU_RAW_VERIFIED'}
assert set(numeric) == {'62259', '62260', '62261'}
mapping = {'FT': '62259', 'AlphaEdit+SPHERE': '62260', 'MEMIT-FE': '62261'}
seen = set()
for x, y, indices in changes:
    if y[0] in mapping:
        assert indices == {5, 6}
        g = numeric[mapping[y[0]]]
        s = g['summary']
        assert s['planned_count'] == s['fluency_count'] == s['consistency_count'] == 2000
        assert s['generation_prompt_count'] == 20000
        assert not any(s['missing_reason_counts'].values())
        for field, index in [('Flu', 5), ('Con', 6)]:
            display = str((Decimal(str(g[field]['raw_value'])) * 100).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
            assert display == g[field]['paper_display_x100'] == y[index]
        seen.add(y[0])
    elif y[0] == 'MEMIT_FE_HISTORY':
        assert indices == {9, 10}
        jid = {'GPT-J': '62581', 'Llama3': '62582', 'Qwen2.5': '62583'}[y[1]]
        expected = f"PENDING: {jobs[jid]['job_name']} ({jid})"
        assert y[9] == y[10] == expected
    else:
        assert y[:4] == ['MEMIT_FE_HISTORY (FE author hparams)', 'Llama3', 'CF', 'server1']
        assert indices == {4} and y[4] == 'ING: official-s1-cf-llama3-memit-fe-author-history (62529)'
assert seen == set(mapping)
assert sum(len(indices) for _, _, indices in changes) == 13
print(json.dumps({'status': 'PASS', 'numeric_cells': 6, 'status_cells': 7,
                  'other_cells_changed': 0, 'new_GPU_jobs': 3, 'cap': 3, 'GPU_calls': 0}))
