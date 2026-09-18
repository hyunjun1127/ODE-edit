"""Two full-bank local linearizations, then at most one current guard."""
import time
from pathlib import Path
import torch
from project.run_scripts.single_layer_edit_preserving_correction import geometry
from .checks import current_guard
from project.run_scripts.single_layer_edit_preserving_correction.common import tensor_sha
from project.run_scripts.single_layer_edit_preserving_correction.sequential_state import atomic_tensor
from .provenance import create_json
from . import factors,qp

def expose(scan,previous):
    rows={r['pair_id']:dict(r) for r in previous}
    for doc in scan['documents']:
        worst=doc['worst'];pid=f'{doc["source_row_id"]}|p{worst["position"]}|c{worst["competitor"]}'
        rows.setdefault(pid,dict(pair_id=pid,index=doc['index'],source_row_id=doc['source_row_id'],
            position=worst['position'],target=worst['target'],competitor=worst['competitor'],kappa=worst['kappa']))
    return sorted(rows.values(),key=lambda r:r['pair_id'])

def run(rt,WN,ref,space,allowed,cur,rows,K,anchor,tau,out):
    out=Path(out);start=time.monotonic();timing={};rounds=[];pairs=[];weight=WN;gradient_count=0
    t=time.monotonic();scan=ref.scan(WN,tau);timing['native_reference_scan']=time.monotonic()-t
    create_json(out/'native-reference.json',scan)
    selected=WN;ideal=torch.zeros_like(WN,dtype=torch.float64);status='NATIVE_ALREADY_FEASIBLE' if scan['all_choices'] else None
    selected_scan=scan;guard_result=None
    if status is None and space.status in ('RANK_UNRESOLVED','REPAIR_SPACE_EMPTY'):status=space.status
    for round_index in range(2):
        if status is not None:break
        directory=out/f'round{round_index+1}';pairs=expose(scan,pairs)
        if len(pairs)>(512 if round_index==0 else 1024):raise ValueError('EXPOSED_PAIR_CAP')
        gradient_count+=len(pairs)
        if gradient_count>1536:raise ValueError('PAIR_GRADIENT_CAP')
        t=time.monotonic();files,b=factors.build(ref,space,weight,WN,pairs,scan,directory/'factors')
        factor_seconds=time.monotonic()-t
        t=time.monotonic();G=factors.gram(files,ref.device);gram_seconds=time.monotonic()-t
        atomic_tensor(directory/'qp-problem.pt',dict(G=torch.from_numpy(G),b=torch.from_numpy(b),ids=[r['pair_id'] for r in pairs]))
        t=time.monotonic()
        try:
            solution=qp.solve(G,b,[r['pair_id'] for r in pairs],order='gss')
        except qp.QPInfeasible as exc:
            create_json(directory/'qp-local-infeasible.json',exc.receipt)
            rounds.append(dict(round=round_index+1,rows=len(pairs),pair_gradients=gradient_count,
                factor_seconds=factor_seconds,gram_seconds=gram_seconds,qp_seconds=time.monotonic()-t,
                scan_seconds=0.,status=exc.code,candidate_scanned=False))
            status='LOCAL_LINEAR_QP_INFEASIBLE_NATIVE_FALLBACK'
            break
        qp_seconds=time.monotonic()-t
        create_json(directory/'qp-solution.json',solution)
        t=time.monotonic();proposal=factors.reconstruct(files,solution['alpha'],WN.shape,ref.device)
        reconstruction_seconds=time.monotonic()-t
        candidate=(WN.double()+proposal).float();actual=candidate.double()-WN.double()
        # Cheap physical algebraic checks precede full-vocabulary choice scans.
        t=time.monotonic();algebra=geometry.invariant_diagnostics(proposal,actual,WN,K,allowed)
        create_json(directory/'algebra.json',algebra)
        if not algebra['ideal_pass']:raise RuntimeError('NOMINAL_NULLSPACE_IMPLEMENTATION_ERROR')
        if not(algebra['actual_response_pass'] and algebra['actual_leakage_pass']):
            rounds.append(dict(round=round_index+1,rows=len(pairs),pair_gradients=gradient_count,
                factor_seconds=factor_seconds,gram_seconds=gram_seconds,qp_seconds=qp_seconds,
                reconstruction_seconds=reconstruction_seconds,scan_seconds=0.,candidate_scanned=False,
                status='FP32_RESPONSE_GUARD_NATIVE_FALLBACK'))
            status='FP32_RESPONSE_GUARD_NATIVE_FALLBACK';break
        atomic_tensor(directory/'candidate.pt',dict(weight=candidate,ideal=proposal,actual=actual,
            native_sha256=tensor_sha(WN),center_sha256=tensor_sha(weight)))
        tscan=time.monotonic();candidate_scan=ref.scan(candidate,tau,exposed=pairs)
        scan_seconds=time.monotonic()-tscan;create_json(directory/'reference.json',candidate_scan)
        receipt=dict(round=round_index+1,rows=len(pairs),pair_gradients=gradient_count,
            center_sha256=tensor_sha(weight),candidate_sha256=tensor_sha(candidate),all_choices=candidate_scan['all_choices'],
            token_flips=candidate_scan['token_flips'],factor_seconds=factor_seconds,gram_seconds=gram_seconds,
            qp_seconds=qp_seconds,reconstruction_seconds=reconstruction_seconds,scan_seconds=scan_seconds,
            ideal_norm=float(proposal.norm()),actual_norm=float(actual.norm()),candidate_scanned=True)
        rounds.append(receipt);create_json(directory/'round.json',receipt)
        if candidate_scan['all_choices']:
            tguard=time.monotonic();guard_result=current_guard(cur,rows,anchor,candidate,WN,proposal,K,allowed)
            create_json(directory/'current-invariant.json',guard_result)
            timing['current_guard']=time.monotonic()-tguard
            if guard_result['pass']:
                selected=candidate;ideal=proposal;selected_scan=candidate_scan;status='ACCEPTED_REFERENCE_AND_CURRENT'
            else:status='CURRENT_GUARD_NATIVE_FALLBACK'
            break
        weight=candidate;scan=candidate_scan
    if status is None:
        unresolved_tie=any(p['margin']==0 and not p['preserved'] for d in scan['documents'] for p in d['positions'])
        status='TIE_UNRESOLVED' if unresolved_tie else 'REFERENCE_BUDGET_NATIVE_FALLBACK'
    result=dict(status=status,rounds=rounds,timing=timing,seconds=time.monotonic()-start,
        pair_scalar_gradients=gradient_count,current_guard_count=int(guard_result is not None),
        candidate_reference_scans=sum(r['candidate_scanned'] for r in rounds),selected_sha256=tensor_sha(selected),native_sha256=tensor_sha(WN),
        selected_reference_all_choices=selected_scan['all_choices'],selected_reference_token_flips=selected_scan['token_flips'],
        native_fallback=bool(torch.equal(selected,WN) and status!='NATIVE_ALREADY_FEASIBLE'),
        native_already_feasible_skip=status=='NATIVE_ALREADY_FEASIBLE',past_guard='NOT_APPLICABLE_B1_EMPTY',
        correction_actual_norm=float((selected.double()-WN.double()).norm()),ideal_norm=float(ideal.norm()),
        global_nonlinear_optimality_claim=False,sequential_authorized=False)
    create_json(out/'selection.json',result)
    return selected,ideal,result,selected_scan
