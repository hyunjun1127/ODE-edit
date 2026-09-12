"""Explicit-recall B-BF4 gate/teacher reuse. No new FD policy or controller."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from ..contracts import sha,save,digest,tensor_sha
from ..evaluation import materialized
from ..observations import pack
from .analyze_partial import RUN,COMMON,SOURCE,PRIOR_NATIVE,verify_execution_source

PRIMITIVES=('functional.py','linear_solve.py','elastic_qp.py','observations.py','evaluation.py',
    'track_a/native_geometry.py','track_b/protocol.py','track_b/native_adapter.py','track_b/runtime_gate.py')

def prepare(output):
    repo=Path(__file__).resolve().parents[4]
    execution=json.loads((RUN/'execution.lock.json').read_text())
    verify_execution_source(execution,repo)
    root=repo/'project/run_scripts/multilayer_joint_compensation'
    members=[]
    for relative in PRIMITIVES:
        path=root/relative;rel=str(path.relative_to(repo))
        original=subprocess.check_output(['git','show',SOURCE+':'+rel],cwd=repo)
        expected=hashlib.sha256(original).hexdigest()
        if sha(path)!=expected:raise ValueError('GATE_REUSE_PRIMITIVE_CHANGED:'+rel)
        members.append(dict(path=str(path),sha256=expected))
    terminal=json.loads((RUN/'output/terminal.json').read_text())
    if terminal['source_head']!=SOURCE or terminal['status']!='B_ENDPOINT_FINITE_RECORDED' or not terminal['W0_selected_restored']:
        raise ValueError('FOLLOWUP_REFERENCE_TERMINAL')
    teacher=RUN.parent/'Middle-B-OS-attempt-v1/output/WN-current-teacher.pt'
    inputs=[RUN/'execution.lock.json',RUN/'output/run.lock.json',RUN/'output/INITIAL_VALID.json',
        RUN/'output/fd-refinement.json',RUN/'output/terminal.json',RUN/'output/endpoint.pt',
        RUN/'output/endpoint-full.json',RUN/'output/nodes/node-00.json',
        RUN/'output/WN-current-teacher-reuse.json',teacher]
    old=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in inputs]
    fd=json.loads((RUN/'output/fd-refinement.json').read_text())
    if fd['status']!='RAW_ADJACENT_FD_PASS':raise ValueError('NO_PAST_RAW_FD_GATE')
    references=[dict(state=s,path=str(PRIOR_NATIVE/n),bytes=(PRIOR_NATIVE/n).stat().st_size,sha256=sha(PRIOR_NATIVE/n))
        for s,n in [('We','ENTRY-full.json'),('N4','N-full.json')]]
    save(output,dict(status='EXPLICIT_USER_RECALL_FOLLOWUP_REUSE',old_source=SOURCE,
        old_source_tree=execution['source_tree'],reference_run=str(RUN),teacher=str(teacher),common=str(COMMON),
        old_members=old,unchanged_primitives=members,reference_members=references,allowed_arm='B-BF4',steps=4,h=.25,
        prior_endpoint_evaluation_repeated=False,missing_observation_supplement=True,
        reference_FD_GGN_reuse=True,FD_tolerance_change=0,controller_change=0,
        initial_gate='SAME_WN_REFERENCE_PLUS_ACTUAL_NONZERO_FUNCTIONAL_PHYSICAL_RESTORE',
        expected_wall_hours_conservative=32,wall_estimate_not_budget=True,
        authority='ODEEDIT-GH-SH1-SH2-MULTILAYER-AB-USER-RECALL-20260912-R1',
        monitoring='INITIAL_GATE_ONLY_THEN_PAUSE',scientific_promotion=False))
    print(json.dumps(dict(path=str(output),sha256=sha(output))),flush=True)

def load_verified(ref):
    if sha(ref['path'])!=ref['sha256']:raise ValueError('FOLLOWUP_LOCK_CHANGED')
    lock=json.loads(Path(ref['path']).read_text())
    if lock['status']!='EXPLICIT_USER_RECALL_FOLLOWUP_REUSE':raise ValueError('FOLLOWUP_LOCK_STATUS')
    for m in lock['old_members']+lock['unchanged_primitives']+lock['reference_members']:
        path=Path(m['path'])
        if sha(path)!=m['sha256'] or ('bytes' in m and path.stat().st_size!=m['bytes']):raise ValueError('FOLLOWUP_MEMBER_CHANGED:'+str(path))
    return lock

def initial_reuse(lock,problem,view,packed,tok,model,ledger,current_reference):
    """Reuse exact prior derivative checks, verify this process/application now."""
    old=Path(lock['reference_run'])/'output'
    run=json.loads((old/'run.lock.json').read_text())
    valid=json.loads((old/'INITIAL_VALID.json').read_text())
    node=json.loads((old/'nodes/node-00.json').read_text())
    import transformers
    if model.training or any(p.requires_grad or p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('FOLLOWUP_MODEL_BOUNDARY')
    if run['torch']!=torch.__version__ or run['transformers']!=transformers.__version__ or run['TF32']!=False:
        raise ValueError('FOLLOWUP_RUNTIME_IDENTITY')
    if problem.support!=(8,) or problem.fixture_identity['common_ready_sha']!=valid['common_ready_sha']:
        raise ValueError('FOLLOWUP_SUPPORT_COMMON')
    if [tensor_sha(w) for w in problem.wn]!=valid['evidence']['FD_identity']['WN']:
        raise ValueError('FOLLOWUP_EXACT_WN')
    if current_reference['mean_nll']!=node['before'][2]['mean_nll'] or current_reference['value']!=node['before'][2]['value']:
        raise ValueError('FOLLOWUP_WN_REFERENCE_ACTUAL_FORWARD_MISMATCH')
    saved=torch.load(old/'endpoint.pt',weights_only=True,map_location='cpu',mmap=True)
    probe=(saved['weights'][1],)
    terminal=json.loads((old/'terminal.json').read_text())
    if tensor_sha(saved['weights'][0])!=problem.fixture_identity['WN'][view.full.names[0]] or tensor_sha(probe[0])!=terminal['endpoint_sha'][view.full.names[1]]:
        raise ValueError('FOLLOWUP_PROBE_IDENTITY')
    first_id=packed['Current'][0]['ordinal']
    rows=[dict(r) for r in packed['Current'] if r['ordinal']==first_id]
    if digest([r['identity'] for r in rows])!=valid['evidence']['row_identity']:raise ValueError('FOLLOWUP_PROBE_PACKING')
    first=next(iter(pack(view,rows,tok.pad_token_id,2)))
    with torch.no_grad(),ledger.time('followup_actual_application_gate'):
        expected=first.logits(probe)
        physical=view.full_weights(tuple(w.to(first.input_ids.device) for w in probe))
        with materialized(model,view.full.names,physical,ledger,purpose='followup_nonzero_probe'):
            logits=model(input_ids=first.input_ids,attention_mask=first.attention_mask,use_cache=False).logits
            actual=logits[first.positions[0],first.positions[1]]
            if not torch.isfinite(actual).all() or not torch.equal(expected,actual):raise ValueError('FOLLOWUP_PHYSICAL_FORWARD_MISMATCH')
        view.full.assert_live(bytes_check=True)
    return dict(status='REUSED_DERIVATIVE_GATE_AND_ACTUAL_FOLLOWUP_APPLICATION_VALID',
        prior_initial_valid_sha=sha(old/'INITIAL_VALID.json'),prior_FD_sha=sha(old/'fd-refinement.json'),
        FD_GGN_repeated=0,FD_GGN_source_unchanged=True,WN_reference_exact=True,
        actual_nonzero_candidate_sha=terminal['endpoint_sha'],fixed_L4=True,
        functional_physical_max_abs=0,live_pointer_version_selected_bytes_restored=True,
        new_direct_materialized_forward=1,probe_carried_to_trajectory=False,
        BF4_solver_validation='FOCUSED_CPU_FIXED_REFERENCE_AND_JOINT_NONNEGATIVE_ELASTIC',
        full_BF4_endpoint_complete=False,semantic_performance_gate=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output)
