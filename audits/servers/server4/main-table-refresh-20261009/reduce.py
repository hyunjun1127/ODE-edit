"""One bounded, read-only source inventory and existing FREE100 exception reducer."""
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).parent
REPO = OUT.parents[3]
LOCAL = Path('/data/janghj/ODE-edit/local')
NONCE = 'USER-GH-ALL-SH-COMPLETED-MAIN-TABLE-REFRESH-20261009-R1-SERVER4'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name, value):
    with (OUT/name).open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2)

prior = REPO/'audits/servers/server4/qwen-migration-20261009/result-eligibility.json'
oldrows = read(prior)['rows']
ids = [r['old_job_id'] for r in oldrows] + [str(n) for n in range(60917,60923)] + ['60103']
accounting = subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(ids),
    '-o','JobIDRaw,User,JobName%100,State,ExitCode,ElapsedRaw,Start,End,NodeList'],text=True,timeout=30)
accounts = {x.split('|')[0]:x.split('|') for x in accounting.splitlines()}
assert all(accounts[i][1]=='janghj' for i in ids)
at = datetime.now(timezone.utc).isoformat()
rows = []
for old in oldrows:
    root = LOCAL/'qwen-baselines-12-20261009'/('execution-cf-display-r1' if old['dataset']=='cf' else 'execution-noqual-r2')
    output = root/'runs'/old['logical_main_row']
    assert not output.exists(), 'NEW_OUTPUT_REQUIRES_RAW_REVIEW'
    cfg = root/'configs'/(old['logical_main_row']+'.json')
    assert sha(cfg)==old['config_file_sha256']
    a = accounts[old['old_job_id']]
    assert a[3].startswith('CANCELLED') or (old['old_job_id']=='61783' and a[3]=='FAILED')
    rows.append(dict(server='server4',model='qwen25',method=old['method'],dataset=old['dataset'],
        job_id=old['old_job_id'],job_name=a[2],observed_state=a[3],observed_at=at,
        source_commit=old['source'],config_sha256=old['config_sha256'],
        ordered_sample_sha256=old['ordered_sample_sha256'],rerun_attempt=root.name,
        cold_state_identity='NOT_STARTED_NO_MODEL_STATE',final_raw_sha256=None,denominators=None,
        metrics=None,main_numeric_eligible=False,action='KEEP_SERVER2_REPLACEMENT_ROW',
        reason='Migrated to server2; original local output absent, no W20; not a replacement status for server2',
        output_root=str(output),report_path='experiment-reports/servers/server4/main-table-refresh-20261009/report-ko.md'))

root = LOCAL/'jlz-price-cap-base-repair-2k/method-metrics-20261007'
run = root/'LLAMA_FREE100'
terminal = read(run/'terminal.json')
summary = read(run/'batch-20/post/summary.json')
assert terminal['job']=='60103' and terminal['commits']==20 and terminal['status']=='W20_COMPLETE'
assert summary['endpoint']=='W20' and summary['requests']==2000 and summary['row_count']==26000
assert read(run/'batch-20/commit.json')['after']==summary['state']
commits = [run/f'batch-{i:02d}/commit.json' for i in range(1,21)]
assert all(read(p)['batch']==i for i,p in enumerate(commits,1))
counts, success, cases = Counter(), Counter(), []
row_ids = set()
raw_manifest=[]
for p in sorted((run/'batch-20/post').glob('chunk-*.json')):
    raw_manifest.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    for r in read(p)['rows']:
        assert r['identity'] not in row_ids and r['endpoint']=='W20'
        row_ids.add(r['identity'])
        k=r['kind']; assert k in ('R','P','N')
        assert math.isfinite(r['true_nll']) and math.isfinite(r['new_nll'])
        counts[k]+=1
        success[k]+=int(r['true_nll']<r['new_nll'] if k=='N' else r['new_nll']<r['true_nll'])
        if k=='R': cases.append(r['case_id'])
assert counts=={'R':2000,'P':4000,'N':20000} and len(set(cases))==2000
dataset=read(LOCAL/'datasets/counterfact-fixed-10k-v1/counterfact.json')
assert cases==[r['case_id'] for r in dataset[:2000]]
for k in counts:
    assert success[k]==summary['summary'][k]['numerator']
    assert counts[k]==summary['summary'][k]['denominator']
rates={k:100*success[k]/counts[k] for k in counts}
score=3/sum(1/rates[k] for k in ('R','P','N'))
metrics=dict(Score=score,Eff=rates['R'],Gen=rates['P'],Loc=rates['N'],Flu=None,Con=None)
assert sha(run/'batch-20/post/summary.json')=='5acf7176ef23aef08a5d3708d600bb16621b0823a0fc0cb08f1bf83f1ad9a331'
assert success=={'R':1994,'P':3711,'N':16441}
a=accounts['60103']; assert a[3]=='COMPLETED'
rows.append(dict(server='server4',model='llama3',method='PRICE_FREE100_MEMIT',dataset='cf',
    job_id='60103',job_name=a[2],observed_state='W20_COMPLETE_HISTORICAL_EXCEPTION',observed_at=at,
    source_commit=terminal['source'],config_sha256=sha(root/'config.json'),config_hash_kind='file_sha256',
    ordered_sample_sha256=hashlib.sha256(json.dumps(cases,separators=(',',':')).encode()).hexdigest(),
    ordered_sample_hash_encoding='JSON ordered case IDs, compact separators',
    rerun_attempt='method-metrics-20261007/LLAMA_FREE100',cold_state_identity=sha(run/'initial.json'),
    final_raw_sha256=sha(run/'batch-20/post/summary.json'),denominators=dict(counts),success_counts=dict(success),
    metrics=metrics,old_metrics=metrics,delta=dict(Score=0,Eff=0,Gen=0,Loc=0),
    main_numeric_eligible=True,action='KEEP_EXISTING_EXPLICIT_FREE100_EXCEPTION_UNCHANGED',
    metric_definition='Legacy strict NLL prompt-pair preference; not relabeled official request-macro',
    generation_status='NOT_MEASURED',zsre_status='NOT_MEASURED',
    exception_evidence='README existing § and experiment-reports/servers/server1/official-baselines-20261008/llama-free100-table-20261009.md',
    report_path='experiment-reports/servers/server4/main-table-refresh-20261009/report-ko.md'))
historical_path=REPO/'experiment-reports/global/2026-10-08-ours-baseline-results/inventory.csv'
with historical_path.open() as f:
    historical=[r for r in csv.DictReader(f) if r.get('server')=='server4']
put('raw-manifest.json',dict(members=raw_manifest,terminal_sha256=sha(run/'terminal.json'),
    commit_receipt_sha256=[sha(p) for p in commits],submission_sha256=sha(root/'submission.json')))
put('table-rows.json',dict(nonce=NONCE,accepted_turn='01a120fb-33b8-7c12-a301-020a9c507251',accepted_turn_status='LOCAL_OWN_SESSION_TURN_CONTEXT_VERIFIED',
    at=at,preparation_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    policy_sha256=sha(REPO/'control/main-results-policy.json'),rows=rows,
    fresh_completed_eligible=0,existing_exception_revalidated=1,numeric_changes=0,
    own_completed_zsre=0,accounting=accounting,prior_inventory_sha256=sha(prior),
    old_llama_accounting={str(i):accounts[str(i)] for i in range(60917,60923)},
    historical_inventory_sha256=sha(historical_path),historical_own_rows=len(historical),
    excluded='Other historical/PRICE tuning/Q3/heldout/GPT2/W0/qualification/collector remain outside main eligibility; no promotion or replica inventory',
    coverage='Known current own official12, cancelled old Llama6, existing eligible FREE100 raw, published historical inventory; no whole-filesystem completeness claim',
    reducer_sha256=sha(Path(__file__)),GPU=0,new_jobs=[],job_mutations=0,README_changed=False,
    broadcast='NO_BROADCAST_NOT_REQUIRED: compact source/receipts only; raw remains local'))
with (OUT/'table-rows.csv').open('x',newline='') as f:
    fields=['server','model','method','dataset','job_id','job_name','observed_state','observed_at','source_commit','config_sha256','ordered_sample_sha256','main_numeric_eligible','action']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print(json.dumps(dict(fresh_completed=0,exception_verified=1,changes=0,counts=dict(counts),success=dict(success),metrics=metrics)))
