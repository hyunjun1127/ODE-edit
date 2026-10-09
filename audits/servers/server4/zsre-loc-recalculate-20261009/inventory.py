"""Read-only metadata/accounting inventory. No metrics when no completed raw exists."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

OUT=Path(__file__).parent;REPO=OUT.parents[3]
LOCAL=Path('/data/janghj/ODE-edit/local/qwen-baselines-12-20261009/execution-noqual-r2')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def command(args):return subprocess.check_output(args,text=True,timeout=30).strip()
jobs=[r for r in read(LOCAL/'released.json')['jobs'] if r['dataset']=='zsre']
assert len(jobs)==12
accounting=command(['sacct','-X','-n','-P','-j',','.join(j['job_id'] for j in jobs),
                    '-o','JobIDRaw,User,JobName%70,State,ExitCode,ElapsedRaw,Start,NodeList'])
by_id={line.split('|')[0]:line.split('|') for line in accounting.splitlines()}
stream=read(LOCAL/'streams/zsre-stream.lock.json')
rows=[]
for j in jobs:
    a=by_id[j['job_id']]
    assert a[1]=='janghj' and a[2]==j['name'] and a[3].startswith('CANCELLED') and a[5]=='0'
    if j['kind']!='gpu':continue
    run=LOCAL/'runs'/j['logical_main_row']
    raw_files=list(run.rglob('*.json')) if run.exists() else []
    assert not raw_files,'RAW_PRESENT_REQUIRES_INDEPENDENT_REDUCTION'
    rows.append(dict(model='qwen25',method=j['method'],dataset='zsre',job_id=j['job_id'],job_name=j['name'],
        state=a[3],elapsed_seconds=int(a[5]),source=j['source'],config_sha256=j['config_sha256'],
        config_file_sha256=sha(LOCAL/'configs'/f"{j['logical_main_row']}.json"),
        ordered_cohort_sha256=stream['ordered_case_ids_sha256'],stream_sha256=stream['stream_sha256'],
        W20_complete=False,raw_status='NOT_CREATED_UNSTARTED_MIGRATION_CANCELLED',raw_sha256=None,
        before_W0_agreement=None,after_Loc=None,Efficacy=None,Generalization=None,
        evaluated_requests=None,locality_token_denominator=None,metric_reducer_sha256=None,
        eligible=False))
historical=command(['sacct','-X','-n','-P','-u','janghj','-N','server4','-S','2026-08-01','-E','2026-10-10',
                    '-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'])
hits=[line for line in historical.splitlines() if 'zsre' in line.lower()]
assert not hits,'ADDITIONAL_ZSRE_ACCOUNTING_REQUIRES_REVIEW'
tracked=subprocess.run(['git','grep','-l','-i','zsre','--','audits/servers/server4','experiment-reports/servers/server4'],
                       text=True,capture_output=True,check=False).stdout.splitlines()
result=dict(instruction_id='USER-GH-ZSRE-LOC-RECALCULATE-MAIN-20261009-R1',at=datetime.now(timezone.utc).isoformat(),
    server='server4',status='NO_COMPLETED_APPROVED_ZSRE_FOUND_POLICY_ACCEPTED',
    adopted_metric='100*mean_requests(mean_loc_ans_tokens(predicted==target))',
    aggregation='request macro, not token micro; stored Specificity_loc_ans only crosscheck',
    W0_agreement='auxiliary only, preserve separately',rows=rows,
    completed_baseline_count=0,completed_ours_count=0,metrics_recomputed=0,
    reducer_status='NOT_APPLICABLE_NO_COMPLETED_RAW',inventory_script_sha256=sha(Path(__file__)),
    source_publication=command(['git','rev-parse','HEAD']),
    coverage=dict(tracked_server4_zsre_mentions=tracked,
      historical_allocated_accounting_window='2026-08-01 through 2026-10-10; node server4; user janghj',
      historical_zsre_name_hits=hits,
      unallocated_accounting='exact registered Qwen main/archive IDs additionally queried; not inferred from node filter',
      historical_mentions='CAKE/BLUE upstream source inventory paths, not zsRE experiment completion',
      official_llama_preparation='local/official-baselines-20261008 preparation-v1/v2 only; no own published zsRE execution receipt found',
      limitations='Bounded registered metadata/accounting inventory; not a whole-filesystem search for unregistered experiments'),
    original_accounting=accounting,
    CF_changed=False,FLUCON_changed=False,job_mutations=0,raw_CP_frozen_keep=True,
    GPU_model_forward=0,online_history_rewrite=False,SH2_migration_unchanged=True,
    common_future_source='GH-owned; adopt READY only at future freeze, no existing job hotpatch',
    tokenization_equivalence_or_paper_reproduction_claim=False)
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'inventory.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
with (OUT/'rows.csv').open('x',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print(json.dumps(dict(status=result['status'],candidate_main_rows=len(rows),eligible=0,inventory_sha256=sha(OUT/'inventory.json'))))
