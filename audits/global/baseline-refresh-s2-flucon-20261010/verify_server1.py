"""Compact owner evidence and exact README diff, with no GPU/scheduler calls."""
import hashlib
import json
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

root = Path(__file__).resolve().parents[3]
p = root / 'audits/servers/server1/baseline-refresh-s2-flucon-20261010/table-rows.json'
assert hashlib.sha256(p.read_bytes()).hexdigest() == 'aafa6d5e9d4cb85948a4d21c712bd0753635ec6adeedc44ad00123a620222024'
data = json.loads(p.read_text())
assert data['completed_rows'] == 12
history = {r['model']: r for r in data['rows'] if r['method'] == 'MEMIT_FE_HISTORY'}
old = subprocess.check_output(['git', 'show', 'd6733abb1afe9022584699b36410bd7e685b9d47:README.md'], cwd=root, text=True)
new = (root / 'README.md').read_text()
def table(text):
    return [[s.strip() for s in line.strip('|').split('|')] for line in text.splitlines() if line.startswith('|')]
a, b = table(old), table(new)
assert len(a) == len(b)
changed = [(x, y) for x, y in zip(a, b) if x != y]
assert len(changed) == 3
for before, after in changed:
    assert before[:9] == after[:9] and len(after) == 11 and after[0] == 'MEMIT_FE_HISTORY'
    model = {'GPT-J': 'gptj', 'Llama3': 'llama3', 'Qwen2.5': 'qwen25'}[after[1]]
    row = history[model]
    assert row['complete'] and row['commits'] == 20
    g = row['generation']
    if model == 'qwen25':
        assert g['state'] == 'RUNNING' and g['final_metrics'] is None
        assert after[9] == after[10] == f"ING: {g['job_name']} ({g['job_id']})"
        continue
    assert g['state'] == 'COMPLETED' and g['status'] == 'W20_GENERATION_CPU_RAW_VERIFIED'
    assert g['scoring'] == 'ALL_CASES_CPU_REFERENCE_BOUND_RECHECK'
    s = g['summary']
    assert s['planned_count'] == s['fluency_count'] == s['consistency_count'] == 2000
    assert s['generation_prompt_count'] == 20000 and not any(s['missing_reason_counts'].values())
    for field, index in [('Flu', 9), ('Con', 10)]:
        display = str((Decimal(str(g[field]['raw_value'])) * 100).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
        assert display == after[index] == g[field]['paper_display_x100']
print(json.dumps({'PASS': True, 'new_numeric_cells': 4, 'status_cells': 2, 'other_table_changes': 0, 'GPU_calls': 0}))
