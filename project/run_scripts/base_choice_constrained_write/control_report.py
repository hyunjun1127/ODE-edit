"""Compact source/admission/cleanup publication and exact own-job accounting."""
import argparse
import json
from pathlib import Path
import subprocess
from .provenance import ROOT,create_json,create_bytes,sha,now

AUDIT='audits/servers/server4/bpcw512-20260918-v2'
def member(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))

def accounting(job,save):
    if job not in ('50291','50305'):raise ValueError('EXACT_OWN_TASK_JOB_ALLOWLIST')
    fields=['JobID','JobName','User','State','ExitCode','Start','End','ElapsedRaw','AllocTRES','NodeList']
    raw=subprocess.check_output(['sacct','-n','-X','-j',job,'-P','--format='+','.join(fields)],text=True)
    rows=[dict(zip(fields,line.split('|'))) for line in raw.splitlines() if line.strip()]
    if len(rows)!=1 or rows[0]['JobID']!=job or rows[0]['User']!='janghj' or rows[0]['JobName']!='odeedit_bpcw512_B1_s4':
        raise ValueError('ACCOUNTING_OWNER_MAPPING')
    row=rows[0]
    if row['State'].startswith(('RUNNING','PENDING','COMPLETING')):raise ValueError('NOT_TERMINAL')
    gpu=0
    for pair in row['AllocTRES'].split(','):
        if pair.startswith('gres/gpu='):gpu=int(pair.split('=')[1])
    result=dict(time=now(),job=job,parent_allocation_only=True,rows=rows,
        allocated_GPU_seconds=int(row['ElapsedRaw'])*gpu,allocated_GPU_hours=int(row['ElapsedRaw'])*gpu/3600,
        utilization='NOT_MEASURED',job_step_extern_double_count=False)
    create_json(save,result);print(json.dumps(result,indent=2))

def package(repo):
    audit=repo/AUDIT
    for name in ['checkpoint-inventory.json','removal-receipt.json','cancel-requests.json','jobs-terminal.json']:
        p=ROOT/'en-cleanup'/name;create_bytes(audit/'en-cleanup'/name,p.read_bytes())
    full=ROOT/'receipts/full-read-m0.json';create_bytes(audit/'full-read-m0.json',full.read_bytes())
    ref=ROOT/'reference-inputs-v2/manifest.json';r=json.loads(ref.read_text())
    create_json(audit/'reference-input-binding.json',dict(status=r['status'],train=r['train'],dev=r['dev'],
        inputs_SHA256=r['inputs_sha256'],policy=member(ROOT/'reference-inputs-v2/selection-policy.json'),
        local_manifest=member(ref),old_reference_identity=r['old_reference_identity'],
        no_Report_model_or_answer_access=r['no_Report_answer_or_model_access'],cpu_seconds=r['cpu_seconds']))
    create_bytes(audit/'reference-selection-policy.json',(ROOT/'reference-inputs-v2/selection-policy.json').read_bytes())
    for version in ['v1','r1']:
        p=ROOT/f'CPU-preflight-{version}/receipt.json';create_bytes(audit/f'CPU-{version}.json',p.read_bytes())
    repair=json.loads((ROOT/'repair-r1/receipt.json').read_text())
    create_json(audit/'repair-r1.json',{k:v for k,v in repair.items() if k!='capsules'} |
        dict(reused_capsule_count=len(repair['capsules']),capsule_manifest_local=member(ROOT/'repair-r1/receipt.json')))
    submissions=[]
    for attempt in ['attempt-v1','attempt-r1']:
        p=ROOT/'B1'/attempt;lock=json.loads((p/'execution.lock.json').read_text())
        submission=json.loads((p/'submission.json').read_text())
        row=dict(attempt=attempt,job=submission['job'],release_time=submission['release_time'],
            source_commit=lock['execution']['commit'],source_tree=lock['execution']['tree'],archive=lock['execution']['archive'],
            lock=member(p/'execution.lock.json'),held_inspection=member(p/'held-inspection.json'),
            resource_admission=member(p/'resource-admission.json'),resource=lock['resources'],storage=lock['storage'],
            B1_only=True,sequential_authorized=False,source_assets='execution.lock local transitive inventory')
        submissions.append(row)
    create_json(audit/'submission-lineage.json',dict(attempts=submissions,new_scope='B1 two endpoints only',new_native100_shared_once=True))
    print(json.dumps(dict(audit=str(audit),members=len(list(audit.rglob('*.json'))))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['accounting','package']);p.add_argument('--job');p.add_argument('--save',type=Path);p.add_argument('--repo',type=Path)
    a=p.parse_args()
    if a.action=='accounting':accounting(a.job,a.save)
    else:package(a.repo)
