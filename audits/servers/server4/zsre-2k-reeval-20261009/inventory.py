"""Refresh own completed-zsRE checkpoint eligibility without reading weights."""
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

OUT=Path(__file__).parent
REPO=OUT.parents[3]
OLD=Path('/data/janghj/ODE-edit/local/qwen-baselines-12-20261009/execution-noqual-r2')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cmd(args):return subprocess.check_output(args,text=True,timeout=30).strip()
prior=REPO/'audits/servers/server4/zsre-loc-recalculate-20261009/inventory.json'
release=read(OLD/'released.json')
jobs=[j for j in release['jobs'] if j['dataset']=='zsre' and j['kind']=='gpu']
assert len(jobs)==6
raw=cmd(['sacct','-X','-n','-P','-j',','.join(j['job_id'] for j in jobs),
         '-o','JobIDRaw,User,JobName%70,State,ExitCode,ElapsedRaw,Start,NodeList'])
accounts={s.split('|')[0]:s.split('|') for s in raw.splitlines()}
rows=[]
for j in jobs:
    state=accounts[j['job_id']]
    assert state[1]=='janghj' and state[2]==j['name']
    assert state[3].startswith('CANCELLED') and state[5]=='0'
    root=OLD/'runs'/j['logical_main_row'];cp=root/'checkpoint/latest.json'
    assert not root.exists(), 'NEW_LOCAL_OUTPUT_REQUIRES_CHECKPOINT_REVIEW'
    rows.append(dict(model='qwen25',method=j['method'],original_job_id=j['job_id'],
        state=state[3],elapsed_seconds=0,source=j['source'],config_sha256=j['config_sha256'],
        output_root=str(root),output_exists=False,checkpoint_pointer=str(cp),checkpoint_exists=False,
        checkpoint_sha256=None,W20_complete=False,eligibility='NOT_APPLICABLE_UNSTARTED_CANCELLED',eval_job_id=None))
historical=cmd(['sacct','-X','-n','-P','-u','janghj','-N','server4','-S','2026-08-01','-E','2026-10-10',
                '-o','JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList'])
zsre=[x for x in historical.splitlines() if 'zsre' in x.lower()]
assert not zsre,'NEW_ALLOCATED_ZSRE_REQUIRES_REVIEW'
result=dict(nonce='USER-GH-ZSRE-SAVED-WEIGHTS-2K-REEVAL-20261009-R1',server='server4',
    session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd',accepted_turn='01a11fce-ae91-7bc1-a254-c347c6ee37fa',
    at=datetime.now(timezone.utc).isoformat(),status='NOT_APPLICABLE_NO_OWN_COMPLETED_ZSRE_CHECKPOINT',
    authority='a81e4daf',preparation_HEAD=cmd(['git','rev-parse','HEAD']),
    prior_inventory=dict(path=str(prior.relative_to(REPO)),sha256=sha(prior)),
    original_submission_sha256=sha(OLD/'released.json'),
    original_stream_lock_sha256=sha(OLD/'streams/zsre-stream.lock.json'),
    rows=rows,accounting=raw,allocated_zsre_name_hits=zsre,
    coverage='Own registered baseline/ours metadata from prior inventory plus fresh exact accounting and original local output existence; no unregistered whole-filesystem claim',
    own_completed_baseline=0,own_completed_ours=0,eligible_checkpoint_count=0,new_job_ids=[],
    new_GPU_evaluation=0,model_load=0,weight_restore=0,weights_hashed=0,duplicate_replica_evaluation=0,
    shared_evaluator='GH-owned API/SHA readiness not claimed; no applicable checkpoint, so not a submission blocker',
    native_query_parity='NOT_RUN_NOT_APPLICABLE_NO_CHECKPOINT',new_metrics={},WandB_new_run=False,
    existing_jobs_source_raw_CP='KEEP',SH2_migration='UNCHANGED',README_edit=False,
    recurring_monitor=False,automatic_retry=False,inventory_script_sha256=sha(Path(__file__)))
with (OUT/'inventory.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=result['status'],candidate_rows=6,eligible_CP=0,job_ids=[])))
