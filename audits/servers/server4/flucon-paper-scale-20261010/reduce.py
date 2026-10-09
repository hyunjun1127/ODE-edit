"""Bounded read-only local eligibility audit; no model imports or remote writes."""
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def paper(raw):
    return None if raw is None else str((Decimal(str(raw))*100).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))
assert paper(None) is None and paper('1.23445') == '123.45'
prior_path = REPO/'audits/servers/server4/main-table-refresh-20261009/table-rows.json'
prior = load(prior_path)
manifest_path = prior_path.with_name('raw-manifest.json')
manifest = load(manifest_path)
counts, success = Counter(), Counter()
for m in manifest['members']:
    p=Path(m['path'])
    assert p.stat().st_size==m['bytes'] and sha(p)==m['sha256']
    for r in load(p)['rows']:
        assert r['endpoint']=='W20' and math.isfinite(r['true_nll']) and math.isfinite(r['new_nll'])
        k=r['kind']; counts[k]+=1
        success[k]+=int(r['true_nll']<r['new_nll'] if k=='N' else r['new_nll']<r['true_nll'])
assert counts=={'R':2000,'P':4000,'N':20000}
assert success=={'R':1994,'P':3711,'N':16441}
run=Path(manifest['members'][0]['path']).parents[2]
assert sha(run/'terminal.json')==manifest['terminal_sha256']
assert load(run/'terminal.json')['commits']==20
for i, expected in enumerate(manifest['commit_receipt_sha256'],1):
    assert sha(run/f'batch-{i:02d}/commit.json')==expected
# Filename metadata only, confined to the selected existing exception's run.
generation_candidates=[str(p.relative_to(run)) for p in run.rglob('*.json')
    if any(x in str(p.relative_to(run)).lower() for x in ('generation','fluency','consistency'))]
assert not generation_candidates, 'NEW_GENERATION_REQUIRES_REDUCTION'
ids=[r['job_id'] for r in prior['rows']]+[str(n) for n in range(60917,60923)]
accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
    '-o','JobIDRaw,User,JobName%100,State,ExitCode,ElapsedRaw,Start,End,NodeList'],text=True,timeout=30)
accounts={line.split('|')[0]:line.split('|') for line in accounting.splitlines()}
at=datetime.now(timezone.utc).isoformat()
rows=[]
for old in prior['rows']:
    a=accounts[old['job_id']]; assert a[1]=='janghj'
    row={k:old.get(k) for k in ('server','model','method','dataset','job_id','job_name','source_commit',
          'config_sha256','ordered_sample_sha256','cold_state_identity','rerun_attempt','final_raw_sha256','denominators')}
    row.update(observed_at=at,scheduler_state=a[3],raw_value_Flu=None,raw_unit_Flu='entropy_bits',
        raw_value_Con=None,raw_unit_Con='TF_IDF_cosine_0_to_1',paper_display_x100_Flu=None,
        paper_display_x100_Con=None,display_unit='raw_x100_not_accuracy_percent',
        generation_denominator=None,checkpoint_sha256=None,numeric_update=False)
    if old['job_id']=='60103':
        assert a[3]=='COMPLETED'
        row.update(generation_status='NOT_MEASURED',main_eligibility='EXISTING_EXPLICIT_FREE100_EXCEPTION',
            action='KEEP_FACTUAL_AND_MISSING_GENERATION',factual_counts=dict(counts),success_counts=dict(success))
    else:
        assert a[3].startswith(('CANCELLED','FAILED'))
        assert not Path(old['output_root']).exists()
        row.update(generation_status='NO_W20_LOCAL_OUTPUT',main_eligibility='NOT_ELIGIBLE_MIGRATED_TO_SERVER2',
            action='KEEP_SERVER2_OWNER_ROW')
    rows.append(row)
result=dict(nonce='USER-GH-FLUCON-PAPER-SCALE-TABLE-REFRESH-20261010-R1',server='server4',
    accepted_turn='01a121bf-d996-71d2-973c-3bb381fbe3ba',observed_at=at,rows=rows,
    new_eligible_W20=0,eligible_generation_rows=0,own_completed_zsre=0,own_selected_W0=0,
    selected_W0_provenance='Published W0 report: Llama SH1, GPTJ SH2, Qwen CF SH3/zsRE SH2; no SH4 replacement',
    prior_sha256=sha(prior_path),raw_manifest_sha256=sha(manifest_path),reducer_sha256=sha(Path(__file__)),
    policy_sha256=sha(REPO/'control/main-results-policy.json'),accounting=accounting,
    excluded='heldout500, OURS tuning/sweep, migrated replicas, input contexts and old historical runs not newly promoted',
    coverage='Exact known main-eligible exception and migrated12 + old cancelled Llama6; not filesystem-wide discovery',
    old_llama_states={str(i):accounts[str(i)][3] for i in range(60917,60923)},
    GPU=0,new_jobs=[],job_mutations=0,raw_or_WB_changes=0,README_changed=False,
    broadcast='NO_BROADCAST_NOT_REQUIRED: scalar metadata only; all raw remains local')
with (OUT/'table-rows.json').open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2)
with (OUT/'table-rows.csv').open('x',newline='') as f:
    fields=['model','method','dataset','job_id','scheduler_state','generation_status','raw_value_Flu','raw_value_Con',
        'paper_display_x100_Flu','paper_display_x100_Con','main_eligibility','action']
    w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps({'eligible_generation_rows':0,'new_eligible_W20':0,'raw_rows_verified':sum(counts.values()),'at':at}))
