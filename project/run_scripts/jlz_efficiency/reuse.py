"""Use sealed scalar evidence; reconstruct only lost RAM, never disk resume."""
import json
from pathlib import Path
import torch
from .budget import BudgetAccountant
from .solver import solve
from .measurement import write
from project.run_scripts.jlz_sequential.state import tensor_sha

def load(root,name):return json.loads((Path(root)/'output'/name).read_text())

def reconstruct(model,tok,records,contexts,history,cache,budget,out,root,entry,make_oracle):
    panel=entry(model,tok,records,contexts,history,cache,out,'small4')
    oldentry=load(root,'small4-entry.json');fresh=panel['entry_receipt']
    for key in ('case_ids','input_identity','teacher_sha','key_sha','active','entry_nll'):
        if fresh[key]!=oldentry[key]:raise RuntimeError('REUSE_ENTRY_IDENTITY_MISMATCH '+key)
    fixed={};shorts={};timings={}
    for file in sorted((Path(root)/'output').glob('fixed-*.json')):
        route=file.stem.removeprefix('fixed-');value=load(root,file.name)
        fixed[route]=value;timings[route]=value['oracle_records']
        candidate=Path(root)/'output'/('short-'+route+'.json')
        if candidate.exists():shorts[route]=load(root,candidate.name)
    oracle=make_oracle(model,panel,'REF_MB2');trace=[];account=BudgetAccountant(12)
    def call(x):
        budget.charge('short','REF_MB2');value=oracle(x)
        trace.append(dict(x_sha=tensor_sha(x),gradient_sha=tensor_sha(value[1]),smooth=float(value[0])))
        return value
    result=solve(call,torch.zeros(len(records)*5,model.config.hidden_size,device='cuda'),panel['c'],panel['rho'],panel['mask'],cap=12,tol=1e-4,account=account)
    x=result.pop('x');gradient=result.pop('gradient');payload=result.pop('final_payload')
    result['point_trace']=trace;result['accountant']=account.report()
    expected=shorts['REF_MB2']['solver']
    matching=all(result[k]==expected[k] for k in ('point_trace','status','calls','accepted_steps','backtracks','branches','final_recomputed'))
    write(out/'short-REF_MB2.json',dict(solver=result,comparison={'status':'PASS' if matching else 'RECONSTRUCTION_MISMATCH'},
        oracle_records=oracle.records,budget=budget.report(),purpose='LOST_REFERENCE_RETURN_R_RECONSTRUCTION_NOT_NEW_QUALIFICATION',reused_from=str(root)))
    if not matching:raise RuntimeError('REUSE_REFERENCE_RETURN_RECONSTRUCTION_MISMATCH; no repeat-to-PASS')
    returned={k:payload[k] for k in ('nll','kl')}
    write(out/'reuse-receipt.json',dict(source_attempt=str(root),mode='NEW_COLD_W0_H0; NOT_EXACT_CRASH_RESUME',
        reused_fixed={r:v['status'] for r,v in fixed.items()},reused_short={r:v['comparison']['status'] for r,v in shorts.items()},
        reused_native={n:load(root,n).get('status','RECORDED_ORIGINAL') for n in ('native-original.json','native-cached.json','native-batched.json')},
        native_new_requests=0,qualification_reruns=0,RAM_reconstruction_oracles=result['calls'],reference_return_trace_exact=True,
        old_allocated_GPU_seconds=401,new_cost_excludes_old=True,old_timing_GPU=load(root,'runtime.json')['GPU_UUID'],
        timing_reuse='Old paired small timing for selection only; new paired B100 measurements all on new job GPU'))
    del payload,gradient,oracle
    return panel,fixed,shorts,x.detach(),returned,timings
