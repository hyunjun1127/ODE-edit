"""One user-authorized exact migration cessation; no submit or retry."""
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

BASE=Path('/data/janghj/ODE-edit/local/qwen-baselines-12-20261009')
CF=BASE/'execution-cf-display-r1'; ZS=BASE/'execution-noqual-r2'
LOCAL=Path('/data/janghj/ODE-edit/local/qwen-migration-20261009')
OUT=Path(__file__).parent
def read(p):return json.loads(p.read_text())
def command(args):return subprocess.check_output(args,text=True,timeout=30).strip()
def store(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2)
def fields(job):
    text=command(['scontrol','show','job',job,'-o'])
    return text,dict(x.split('=',1) for x in text.split() if '=' in x)
jobs=[{**j,'root':str(CF)} for j in read(CF/'released.json')['jobs']]
jobs += [{**j,'root':str(ZS)} for j in read(ZS/'released.json')['jobs'] if j['dataset']=='zsre']
assert [j['job_id'] for j in jobs]==list(map(str,range(61783,61795)))+list(map(str,range(61755,61767)))
assert not (LOCAL/'cessation-started.json').exists(), 'EXPLICIT_RECONCILIATION_REQUIRED'
def validate_pending(j):
    raw,f=fields(j['job_id']); root=Path(j['root'])
    assert f['JobName']==j['name'] and f['UserId'].startswith('janghj(')
    assert f['ReqNodeList']=='server4' and f['WorkDir']==str(root)
    assert f['Command']==str(root/'scripts'/f"{j['logical_main_row']}-{j['kind']}.sh")
    assert f['JobState']=='PENDING' and f['RunTime']=='00:00:00' and f['StartTime']=='Unknown'
    assert f['AllocTRES']=='(null)' and f['Restarts']=='0'
    assert read(root/'source-lock.json')['code_commit']==j['source']
    dep='afterok:61794' if j['job_id']=='61755' else j['dependency']
    assert dep in f['Dependency']
    return raw
accounting=command(['sacct','-X','-n','-P','-j',','.join(j['job_id'] for j in jobs),
                    '-o','JobIDRaw,User,JobName%60,State,ExitCode,ElapsedRaw,Start,End,NodeList'])
accounts={line.split('|')[0]:line.split('|') for line in accounting.splitlines()}
assert accounts['61783'][1:6]==['janghj','qwen-cf-ft','FAILED','1:0','1']
error=(CF/'logs/qwen25-cf-ft-gpu-61783.err').read_text()
assert 'RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE' in error
for root in (CF,ZS):
    assert not list((root/'runs').rglob('*.json')), 'MAIN_PROGRESS_REQUIRES_RECONCILIATION'
    assert not list((root/'shared-w0').rglob('*.json')), 'W0_OBSERVED_REQUIRES_RECONCILIATION'
active=jobs[1:]
before=[dict(job_id=j['job_id'],raw=validate_pending(j)) for j in active]
store(LOCAL/'cessation-started.json',dict(at=datetime.now(timezone.utc).isoformat(),accounting=accounting,jobs=jobs,before=before))
for j in reversed(active):
    raw=validate_pending(j)
    command(['scontrol','hold',j['job_id']])
    after=validate_pending(j);assert 'Reason=JobHeldUser ' in after
    store(LOCAL/'holds'/f"{j['job_id']}.json",dict(before=raw,after=after))
cancelled=[]
for j in reversed(active):
    raw=validate_pending(j);assert 'Reason=JobHeldUser ' in raw
    command(['scancel',j['job_id']])
    after,f=fields(j['job_id']);assert f['JobState']=='CANCELLED'
    store(LOCAL/'cancellations'/f"{j['job_id']}.json",dict(before=raw,after=after))
    cancelled.append(j['job_id'])
post=command(['sacct','-X','-n','-P','-j',','.join(j['job_id'] for j in jobs),'-o','JobIDRaw,User,JobName%60,State,ExitCode,ElapsedRaw'])
store(LOCAL/'accounting-after.json',dict(raw=post))
rows=[]
for j in jobs:
    rows.append({k:j[k] for k in ('job_id','name','dataset','method','kind','source','config_sha256','logical_main_row','root')})
    rows[-1].update(state='FAILED_KEEP' if j['job_id']=='61783' else 'USER_CANCELLED_FOR_SERVER2_MIGRATION',
                    main_edit_started=False,completed_edits=0)
store(OUT/'cessation.json',dict(nonce='USER-GH-S4-S2-QWEN-BASELINES-MIGRATION-RESULTS-20261009-R1',
      at=datetime.now(timezone.utc).isoformat(),server='server4',status='CEASED_READY_FOR_SOURCE_HANDOFF_NOT_DESTINATION_SUBMITTED',
      failure=dict(job_id='61783',exit_code='1:0',elapsed_seconds=1,error='RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE',cancelled=False),
      cancelled_pending_ids=cancelled,rows=rows,successful_terminal_main_ids=[],
      W0_observed=False,W20_observed=False,other_job_mutations=0,server4_new_submissions=0,
      raw_source_checkpoint_keep=True,local_full_receipts=str(LOCAL),automatic_retry=False))
print(json.dumps(dict(cancelled_count=len(cancelled),kept_failed='61783',status='CESSATION_CONFIRMED')))
