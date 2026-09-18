"""CPU postrun closure. No model, evaluator, scheduler or mutation of raw."""
import argparse
import json
from pathlib import Path
import subprocess
from .provenance import create_json,sha,now

def audit(output,report,destination,repo):
    lock=json.loads((output.parent/'execution.lock.json').read_text())
    checked=[]
    for row in lock['execution']['members']+lock['external_members']:
        p=Path(row['path']);valid=p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
        if not valid:raise ValueError('EXECUTED_SOURCE_POSTRUN_CHANGED:'+str(p))
        checked.append(dict(path=str(p),sha256=row['sha256'],bytes=row['bytes']))
    manifest=json.loads((report/'manifest.json').read_text())
    for row in manifest['members']:
        p=Path(row['path'])
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('REPORT_MANIFEST_CHANGED')
    root=json.loads((report/'rooted-receipt.json').read_text())
    if root['manifest_SHA256']!=sha(report/'manifest.json') or root['report_SHA256']!=sha(report/'diagnostic-report-ko.md'):raise ValueError('ROOTED_BINDING')
    summary=json.loads((report/'summary.json').read_text());terminal=json.loads((output/'terminal.json').read_text())
    if terminal['max_batches']!=1 or terminal['sequential_authorized'] or terminal['automatic_resume']:raise ValueError('SEQUENTIAL_BOUNDARY')
    actual=summary['accounting']['rows'][0]
    if actual['State']!='COMPLETED' or actual['ExitCode']!='0:0':raise ValueError('TERMINAL_NOT_COMPLETE')
    changed=subprocess.check_output(['git','diff','--name-only','e666b37485d07332c662d7a4359aaf25904a9dbc','HEAD'],cwd=repo,text=True).splitlines()
    forbidden=[p for p in changed if p.endswith(('.pt','.bin','.safetensors','.out','.err')) or '/local/' in p]
    if forbidden:raise ValueError('RAW_OR_TENSOR_TRACKED:'+str(forbidden))
    result=dict(time=now(),status='PASS_CPU_POSTRUN_CLOSURE',executed_source_members=len(checked),
        unchanged_executed_source=True,source_manifest=checked,report_manifest_sha256=sha(report/'manifest.json'),
        report_sha256=sha(report/'diagnostic-report-ko.md'),rooted_receipt_sha256=sha(report/'rooted-receipt.json'),
        actual_job=summary['accounting'],source=summary['source'],analysis_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        max_batches=1,sequential_authorized=False,monitoring_active=False,automatic_resume=False,
        new_postrun_GPU=0,GPU_B2_continuation='NOT_RUN_NOT_AUTHORIZED',large_immutable_model_rehash='NOT_REPEATED; prior lock/stat and actual runtime W0/P binding',
        PNG_visual_inspection='SEPARATE_HUMAN_MODEL_VIEW_RECEIPT',HTML_renderer='NOT_AVAILABLE',raw_or_tensor_tracked=False,
        next='WAITING_USER_APPROVAL_FOR_SEQUENTIAL')
    create_json(destination,result);print(json.dumps({k:v for k,v in result.items() if k!='source_manifest'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['output','report','destination','repo']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();audit(a.output,a.report,a.destination,a.repo)
