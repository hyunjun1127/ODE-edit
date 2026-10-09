"""One-shot metadata audit. No model, checkpoint or remote result loading."""
import csv, hashlib, json, subprocess
from datetime import datetime, timezone
from pathlib import Path
OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
LOCAL=Path('/data/janghj/ODE-edit/local')
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
priorpath=REPO/'audits/servers/server4/qwen-migration-20261009/result-eligibility.json'
prior=read(priorpath)['rows']
ids=[r['old_job_id'] for r in prior]+[str(i) for i in range(60917,60923)]
raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),'-o',
    'JobIDRaw,User,JobName%100,State,ExitCode,ElapsedRaw,Start,End,NodeList'],text=True,timeout=30)
acc={l.split('|')[0]:l.split('|') for l in raw.splitlines()}
at=datetime.now(timezone.utc).isoformat()
rows=[]
for old in prior:
    a=acc[old['old_job_id']]; assert a[1]=='janghj'
    assert a[3].startswith(('CANCELLED','FAILED'))
    root=LOCAL/'qwen-baselines-12-20261009'/('execution-cf-display-r1' if old['dataset']=='cf' else 'execution-noqual-r2')
    config=root/'configs'/(old['logical_main_row']+'.json')
    assert sha(config)==old['config_file_sha256']
    output=root/'runs'/old['logical_main_row']
    assert not output.exists(), 'New output requires independent raw audit'
    rows.append(dict(model=old['model'],method=old['method'],dataset=old['dataset'],job_id=a[0],
        job_name=a[2],scheduler_state=a[3],observed_at=at,source=old['source'],config_sha256=old['config_sha256'],
        config_file_sha256=sha(config),ordered_cohort_sha256=old['ordered_sample_sha256'],
        local_output=str(output),local_output_exists=False,W20=False,metrics=None,denominators=None,
        frozen_evaluator_audit='NOT_APPLICABLE_NO_COMPLETED_LOCAL_RAW',
        public_query_audit='NOT_APPLICABLE_NO_COMPLETED_LOCAL_RAW',
        request_macro_reduction='NOT_APPLICABLE_NO_COMPLETED_LOCAL_RAW',
        disposition='MIGRATED_TO_SH2_NO_DUPLICATE_COUNT',main_eligible=False))
for i,method in enumerate(['MEMIT','PRUNE','RECT','ALPHAEDIT','ALPHAEDIT_BLUE','CAKE'],60917):
    a=acc[str(i)]; assert a[1]=='janghj' and a[3].startswith('CANCELLED') and a[5]=='0' and a[6]=='None'
    rows.append(dict(model='llama3',method=method,dataset='cf',job_id=str(i),job_name=a[2],scheduler_state=a[3],
        observed_at=at,W20=False,metrics=None,denominators=None,main_eligible=False,
        disposition='CANCELLED_NEVER_STARTED',source_evidence='experiment-reports/servers/server4/llama3-baselines-fluency-consistency-2k/report-ko.md'))
result=dict(nonce='USER-GH-ALL-SH-BASELINE-COMPLETED-ZSRE-AUDIT-20261010-R1',
    accepted_turn='01a122d0-e21c-74f2-8c25-5ee01998b4de',server='server4',observed_at=at,
    authority='7dacd3fb',rows=rows,completed_eligible_baselines=0,completed_own_zsre=0,
    zsre_audit_status='NOT_APPLICABLE_NO_COMPLETED_LOCAL_ZSRE_BASELINE',
    semantics='Eff/Gen/Loc = 100*mean_requests(mean_target_token_correctness); Loc loc_ans, not W0 agreement; no token-micro substitution',
    public_query_GPU_parity='NOT_CLAIMED',numeric_changes=0,
    prior_inventory_sha256=sha(priorpath),reducer_sha256=sha(Path(__file__)),accounting=raw,
    coverage='Exact 12 migrated Qwen and 6 cancelled Llama baselines; existing known inventory, no whole filesystem completeness claim',
    exclusions='OURS/PRICE, heldout500, tuning, historical exceptions untouched; SH2 destination metrics/status not impersonated',
    GPU=0,new_jobs=[],job_mutations=0,checkpoint_loads=0,README_changed=False,
    broadcast='NO_BROADCAST_NOT_REQUIRED; existing raw/checkpoints KEEP')
with (OUT/'table-rows.json').open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2)
with (OUT/'table-rows.csv').open('x',newline='') as f:
    fields=['model','method','dataset','job_id','job_name','scheduler_state','W20','main_eligible','disposition']
    w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps({'rows':len(rows),'completed_eligible':0,'zsre_audit':result['zsre_audit_status'],'at':at}))
