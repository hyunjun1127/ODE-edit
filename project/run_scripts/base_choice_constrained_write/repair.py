"""CPU-only RCA and completed W0 capsule reuse seal, no runtime overwrite."""
import ast
import json
from pathlib import Path
import subprocess
from .provenance import ROOT,sha,now,create_json

def record():
    old=ROOT/'B1/attempt-v1';current=Path(__file__).parent
    def fn(path):
        tree=ast.parse(path.read_text())
        return ast.dump(next(x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name=='raw_greedy'),include_attributes=False)
    if fn(old/'source/project/run_scripts/base_choice_constrained_write/choice.py')!=fn(current/'choice.py'):
        raise ValueError('GENERATION_NUMERICAL_SOURCE_CHANGED')
    acct=subprocess.check_output(['sacct','-n','-X','-j','50291','-P','--format=JobID,JobName,User,State,ExitCode,Start,End,ElapsedRaw,AllocTRES,NodeList'],text=True)
    if 'CANCELLED' not in acct or 'RUNNING' in acct:raise ValueError('OLD_TASK_RELEASE_NOT_CONFIRMED')
    if (old/'output/native').exists():raise ValueError('NATIVE_BOUNDARY_REQUIRES_SPECIFIC_REUSE_AUDIT')
    inputs={r['source_row_id']:r for r in json.loads((ROOT/'reference-inputs-v2/inputs.json').read_text())}
    runtime=json.loads((old/'output/runtime-load.json').read_text());caps=[];incomplete=[]
    for p in sorted((old/'output/capsules').rglob('*.json')):
        try:r=json.loads(p.read_text())
        except json.JSONDecodeError:incomplete.append(str(p));continue
        row=inputs[r['source_row_id']]
        if any(r[k]!=v for k,v in row.items()) or r['W0']!=runtime['identity']['W0']:raise ValueError('W0_CAPSULE_REUSE_IDENTITY')
        if not 1<=len(r['y0'])<=16 or len(r['y0'])!=len(r['steps']):raise ValueError('INCOMPLETE_CAPSULE')
        caps.append(dict(source_row_id=r['source_row_id'],path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    result=dict(time=now(),old_job='50291',accounting=acct,allocated_GPU_seconds=267,
        status='USER_AUTHORIZED_NARROW_B1_TECHNICAL_REPAIR',
        failures=['near-positive Gram eigenspace misclassified numerical Farkas infeasibility',
                  'zero FD direction must not count as derivative PASS','actual production factor Gram comparison absent'],
        failure_detection='CPU/source independent red, before model correction/native',
        native_fit_completed=0,scientific_endpoints_completed=0,capsules=caps,incomplete_capsules=incomplete,
        generation_function_AST_equal=True,old_source_preserved=True,live_cancellation_rollback='NOT_VERIFIED',
        preserved='all source/log/capsules/receipts; no deletion',sequential_authorized=False)
    create_json(ROOT/'repair-r1/receipt.json',result);print(json.dumps(dict(reused_capsules=len(caps),incomplete=len(incomplete),GPU_seconds=267)))

if __name__=='__main__':record()
