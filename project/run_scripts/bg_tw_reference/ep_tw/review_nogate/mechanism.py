"""Saved-tensor CPU scalar/design audit. Never execute the editing policy."""
import argparse
import ast
import csv
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from review_nogate import ROOT,read,save,table,ref,sha

def norm(t):return sum(float(v.double().square().sum()) for v in t.reshape(-1).split(1<<20))**.5
def dot(a,b):return sum(float((x.double()*y.double()).sum()) for x,y in zip(a.reshape(-1).split(1<<20),b.reshape(-1).split(1<<20)))
def rawsha(t):
    import hashlib
    h=hashlib.sha256();v=t.contiguous().view(torch.uint8).reshape(-1).numpy()
    for i in range(0,v.size,8<<20):h.update(memoryview(v[i:i+(8<<20)]))
    return h.hexdigest()

def run(worktree,output):
    w,out=Path(worktree),Path(output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    lock=read(ROOT/'execution.lock.json');src=Path(lock['source_root']);raw=ROOT/'scientific-v1';rows=[];previous=None
    files=['project/run_scripts/bg_tw_reference/ep_tw/'+n for n in ('policy.py','model_adapter.py','runner.py','ledger.py','gate_skip.py')]
    files+=['project/run_scripts/bg_tw_reference/native_map.py','project/run_scripts/low_cost_write_donor_pilot/fitting.py']
    for file in files:assert sha(w/file)==sha(src/file),'ANALYSIS_VIEW_DIFFERS_FROM_EXECUTED_SOURCE'
    for b in range(1,11):
        p=raw/f'B{b:03d}';pol=read(p/'policy.json');diag=pol['correction'];proj=diag['projection'];cand=pol['selection']['candidate_receipts']
        r=torch.load(p/'route.pt',weights_only=True,map_location='cpu',mmap=True)
        n=torch.load(p/'native-targets-map.pt',weights_only=True,map_location='cpu',mmap=True)
        cp=torch.load(p/'checkpoint.pt',weights_only=True,map_location='cpu',mmap=True)
        W=next(iter(cp['weights'].values()));Vp=n['native_proposal'];gE=r['gE'];gD=r['gD'];C=r['C'];A=r['map_A']
        q=dot(gE,gD);e2=dot(gE,gE);coef=min(q,0.)/e2 if e2 else 0.
        # Reconstruction is CPU evidence, not an original saved direction.
        d=-gD if coef==0 else -gD+coef*gE
        z=torch.stack([x['target'] for x in n['target_observations']],1);anchor=n['anchors'];rad=n['radii']
        alpha_expected=min(lock['numerical_policy']['alpha_cap'],.25*diag['actual_native_delta_norm']/(diag['mapped_direction_norm']+lock['numerical_policy']['epsilon_num']))
        assert alpha_expected==diag['alpha']
        action=W-Vp;anorm=norm(action);selected=next(x for x in cand if x['id']==pol['selection']['selected_id'])
        mapped=C@A
        reconstructed=Vp.clone() if selected['id']=='RAW' else Vp+mapped*selected['beta']
        difference=reconstructed-W
        row=dict(batch=b,selected=selected['id'],ge_raw_SHA=rawsha(gE),gd_raw_SHA=rawsha(gD),C_raw_SHA=rawsha(C),
            ge_norm=norm(gE),gd_norm=norm(gD),gradient_cos=q/(norm(gE)*norm(gD)) if norm(gE)*norm(gD) else None,
            q_CPU=q,q_recorded=proj['q_ge_gd'],projection_active=q<0,coefficient_CPU=coef,coefficient_recorded=proj['coefficient'],
            ge_exact_zero=e2==0,ge_dot_d_CPU=dot(gE,d),ge_dot_d_recorded=proj['ge_dot_d'],gd_dot_d_CPU=dot(gD,d),gd_dot_d_recorded=proj['gd_dot_d'],
            KKT_stationarity_recorded=proj['kkt_stationarity_norm'],KKT_complementarity_recorded=proj['kkt_complementarity'],
            alpha=diag['alpha'],alpha_cap=lock['numerical_policy']['alpha_cap'],epsilon_num=lock['numerical_policy']['epsilon_num'],zeta=.25,
            alpha_cap_active=diag['alpha']==lock['numerical_policy']['alpha_cap'],native_action_norm=diag['actual_native_delta_norm'],
            mapped_direction_norm_recorded=diag['mapped_direction_norm'],C_norm_CPU=norm(C),C_norm_recorded=diag['residual_correction_norm'],
            CA_norm_CPU=norm(mapped),CA_norm_recorded=diag['mapped_correction_norm'],trust_limit=diag['native_trust_limit'],trust_retraction=diag['trust_retraction'],
            ball_clamped_requests=diag['ball_projected_requests'],native_ball_excess_recorded=diag['native_proposal_ball_max_excess'],
            ball_excess_after_recorded=diag['postprojection_ball_max_excess'],
            ball_excess_after_CPU=float(((z+C).double()-anchor.double()).norm(dim=0).sub(rad.double()).max()),
            ge_dot_C_CPU=dot(gE,C),ge_dot_C_recorded=diag['ge_dot_correction'],gd_dot_C_CPU=dot(gD,C),gd_dot_C_recorded=diag['gd_dot_correction'],
            actual_correction_norm=anorm,correction_native_ratio=anorm/diag['actual_native_delta_norm'],
            CPU_reconstruction_max_abs=float(difference.abs().max()),CPU_reconstruction_norm=norm(difference),
            CPU_reconstruction_unequal_elements=int((difference!=0).sum()),CPU_reconstruction_not_GPU_parity=True,
            independent_native_comparison_RHS_skipped=True,factorized_map_solve_source_count=1,native_target_count=100,native_fit_solve_count=1,
            preprojection_temporary_Z_tensor='NOT_STORED',postprojection_C_tensor='STORED',pretrust_C_tensor='NOT_STORED')
        if previous is not None:
            native=Vp-previous;total=W-previous;nn=norm(native);dp=dot(native,action)
            para=dp/nn if nn else 0.;orth=max(0.,anorm**2-para**2)**.5
            row.update(native_norm_CPU=nn,selected_increment_norm=norm(total),correction_native_cos=dp/(nn*anorm) if nn*anorm else None,
                correction_parallel_signed_norm=para,correction_orthogonal_norm=orth,
                correction_parallel_coefficient=dp/(nn**2) if nn else None,
                pure_native_scalar_shrink='FALSE_NONZERO_ORTHOGONAL' if orth>0 else 'RAW_IDENTITY',
                entry_weight_evidence='PREVIOUS_SELECTED_CHECKPOINT')
        else:row.update(entry_weight_evidence='W0_HASH_AND_RECORDED_NORM_ONLY_NO_NEW_BASE_LOAD',pure_native_scalar_shrink='GEOMETRY_NOT_RECOMPUTED_B1')
        rows.append(row);previous=W.clone()
        del cp,r,n,gE,gD,C,A,Vp,W,action,d,mapped,reconstructed,difference
        print('MECHANISM_CPU',b,flush=True)
    table(out/'per-batch-mechanism.csv',rows)
    conformance=[]
    def add(req,formula,file,func,evidence,level,limit):
        path=src/file;tree=ast.parse(path.read_text());nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==func]
        conformance.append(dict(requirement=req,formula=formula,executed_source=lock['source_head'],file=file,function=func,
            line=nodes[0].lineno if nodes else 'MODULE',file_sha256=sha(path),artifacts=evidence,level=level,limitation=limit))
    ep='project/run_scripts/bg_tw_reference/ep_tw/'
    add('Cold W0 L4 native own trajectory','100targets/nativeRHSsolve1 eachbatch; no warm state',ep+'runner.py','run','entry/native-fit/10commit/9links','STORED_EVIDENCE_CONSISTENT','Other base parameters not reloaded in CPU review')
    add('Native fit/history split','Native source fit removes only final history loop; finalizer separately1','project/run_scripts/low_cost_write_donor_pilot/fitting.py','split_native_source','native-fit source AST receipt and counts','SOURCE_CONFIRMED','Not a new original-native model parity test')
    add('Actual Vp anchor and stopgradient','C=0 returns actual raw.clone; backward gW A.T',ep+'model_adapter.py','_RawAnchoredWeight','RAW_Vp stored hash bridge; gE/gD route hashes','SOURCE_CONFIRMED','FD/direct neural derivative verification SKIPPED_USER_DIRECTED')
    add('Fixed map','A=solve(P(KK.T+M)+I,PK).T; q=1; no inverse/SPD assumption','project/run_scripts/bg_tw_reference/native_map.py','from_native_system','native-map A saved, P/context/history bindings','SOURCE_CONFIRMED','10 map solves source-derived, distinct from10 native RHS solves; G/P not resaved perbatch')
    add('Current loss mass','tokenmean/request then100request mean',ep+'model_adapter.py','_current','candidate-evaluation current rows/counts; independent E recompute','STORED_EVIDENCE_CONSISTENT','Native multicontext target loss is separate')
    add('Preservation loss mass','KL(p0||pV),full128256 vocabulary,logits128:256,128positions,64docs mean',ep+'model_adapter.py','_generic','fixed teacher/role/source IDs/candidate D64 rows','STORED_EVIDENCE_CONSISTENT','Fixed S64 is controller not holdout; teacher selfKL skipped')
    add('Independent sweeps','gE then gD at same C0 leaf; 7+64 backward microbatches perbatch',ep+'model_adapter.py','gradient_sweeps','current70/S64640 backwards plus route tensors','STORED_EVIDENCE_CONSISTENT','Counts/finite do not prove derivative correctness')
    add('Halfspace projection','d=-gD+min(q,0)/||gE||^2*gE; zero gE=>-gD',ep+'policy.py','project_direction','per-batch-mechanism q/coefficient/innerproducts/KKT','STORED_EVIDENCE_CONSISTENT','Projection active9/10; zero branch unexercised; CPU/GPU scalar roundoff recorded')
    add('Step ball trust ordering','alpha=min(cap,.25||Vp-We||/(||dA||+eps)); ball then C-only trust',ep+'policy.py','build_correction','policy.json +C/radius/anchor CPU algebra','STORED_EVIDENCE_CONSISTENT','Intermediate temporary/pretrust tensors NOT_STORED; postprojection first-order sign not guaranteed')
    add('Independent candidate materialization','RAW=Vp; corrected=Vp+beta*(C@A)',ep+'policy.py','materialize_candidates','40candidate hashes; selected checkpoint3-hash bridge; CPU reconstruction differences','STORED_EVIDENCE_CONSISTENT','CPU matmul re-materialization is not exact GPU replay')
    add('Finite E/strict screen and D selection','E<=Ep and Ap subset Ac; minD then RAW/norm/ID tie',ep+'policy.py','choose_candidate','40candidate-details and10 independent selector matches','STORED_EVIDENCE_CONSISTENT','No quality tolerance added; finite-step per-request NLL need not all decrease')
    add('Observer firewall','Only Current+S64 passed into selector; old/PN/Dev downstream observer',ep+'runner.py','run','source argument flow and saved controller_access=False','SOURCE_CONFIRMED','Dataflow audit not an independent GPU execution trace of each operation')
    add('History and atomic accepted ledger','inner0; selectedfinalizer1; accepted strict IDs only',ep+'ledger.py','commit_batch','10CP M/ledger and9next-entry links','STORED_EVIDENCE_CONSISTENT','Unaccepted override edge not exercised: all1000 accepted atwrite')
    add('State and save','W4/M4/context/RNG/ledger/next ordinal eachbatch',ep+'runner.py','run','10weights_only reloads/fullSHA/65references','STORED_EVIDENCE_CONSISTENT','Full model/GPUcontinuation NOT_TESTED')
    add('User-directed skip','Do not call technical callbacks or require repairpass',ep+'gate_skip.py','diagnostic','waiver/selfKL/episode receipts +actual INITIAL_EXECUTION marker','SKIPPED_USER_DIRECTED','numerical_validation=NOT_ESTABLISHED; not silent omission or PASS')
    table(out/'design-conformance.csv',conformance)
    save(out/'mechanism-summary.json',dict(projection_active=sum(x['projection_active'] for x in rows),
        alpha_cap_active=sum(x['alpha_cap_active'] for x in rows),ball_clamped_requests_total=sum(x['ball_clamped_requests'] for x in rows),
        trust_retractions=sum(x['trust_retraction']<1 for x in rows),positive_postprojection_ge_C=sum(x['ge_dot_C_recorded']>0 for x in rows),
        correction_native_ratio_min=min(x['correction_native_ratio'] for x in rows if x['correction_native_ratio']>0),
        correction_native_ratio_max=max(x['correction_native_ratio'] for x in rows),
        CPU_reconstruction_max_abs=max(x['CPU_reconstruction_max_abs'] for x in rows),
        CPU_reconstruction_max_norm=max(x['CPU_reconstruction_norm'] for x in rows),
        projection_q_CPU_max_abs_difference=max(abs(x['q_CPU']-x['q_recorded']) for x in rows),
        executed_source_files=[ref(src/p) for p in files],numerical_validation='NOT_ESTABLISHED'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.worktree,a.output)
