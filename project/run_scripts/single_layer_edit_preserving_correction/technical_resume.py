"""Narrow T continuation after a receipt-only source failure.

Reuse requires exact sealed successful prior stages, native actual weight and
gradient. No native re-fit, teacher regeneration, or erased failure cost.
"""
import json
import resource
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from .common import member,write,save_tensor,tensor_sha,digest
from .runtime import Runtime,invariant
from .binding import score_rows
from .technical import NUMERIC,TechnicalHold,fd,require
from .geometry import RightSpace,edit_null_space,projector_diagnostics,row_space,matrix_sha256

def run(lock):
    directory=Path(lock['output']);directory.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();stage='validate_prior';rt=None
    write(directory/'technical-contract.json',NUMERIC)
    try:
        prior=lock['resume_prior']
        for spec in prior['members']:
            if member(spec['path'])!=spec:raise ValueError('PRIOR_EVIDENCE_IDENTITY_DRIFT')
        original=Path(prior['output'])
        failure=json.loads((original/'failure.json').read_text())
        if not (failure['stage']=='full_token_nullspace' and 'multiple values' in failure['error'] and 'status' in failure['error']):
            raise ValueError('UNAPPROVED_RESUME_FAILURE_BOUNDARY')
        stages={}
        for name in ('native-repeat','teacher-fixed-binding','noop-repeat','direct-cached-gradient','all-token-forward-stationarity'):
            value=json.loads((original/(name+'.json')).read_text())
            if value['status']!='PASS':raise ValueError('PRIOR_STAGE_NOT_PASS:'+name)
            stages[name]=member(original/(name+'.json'))
        write(directory/'prior-stage-reuse.json',dict(stages=stages,failure_preserved=member(original/'failure.json'),
            source_applicability=prior['source_applicability'],new_native_fit=0,new_native_targets=0,new_teacher=0,
            new_physical_gradient_sweep=0,new_cached_gradient_sweep=0))
        stage='model_load';rt=Runtime(lock,directory)
        before=rt.byte_hash_nonselected()
        if before!=json.loads((original/'nonselected-before.json').read_text()):raise ValueError('NONSELECTED_MODEL_NOT_SAME')
        write(directory/'nonselected-before.json',before)
        native=torch.load(original/'native1/native-capsule.pt',weights_only=True,map_location='cpu',mmap=True)
        WN=native['weight'].clone()
        if tensor_sha(rt.W0)!=native['receipt']['entry_weight_sha256'] or tensor_sha(rt.M)!=native['receipt']['history_sha256']:
            raise ValueError('PRIOR_NATIVE_COLD_BINDING')
        rt.copy_weight(WN)
        stage='cache_binding';ref=rt.reference_oracle();cur,rows,K,meta=rt.protected_oracle(rt.records[:8])
        oldmeta=json.loads((original/'protected-provenance.json').read_text())
        if meta!=oldmeta:raise ValueError('PROTECTED_TOKEN_KEY_PROVENANCE_NOT_IDENTICAL')
        write(directory/'protected-provenance.json',meta)
        oldG=torch.load(original/'cached-gradient.pt',weights_only=True,map_location='cpu',mmap=True)
        if oldG['weight_sha']!=tensor_sha(WN):raise ValueError('GRADIENT_WEIGHT_IDENTITY')
        G=oldG['gradient'];Lcached=oldG['loss']
        q0=json.loads((original/'noop-repeat.json').read_text())['quality0']
        qnow=score_rows(cur,WN,rows)
        nllgap=max(abs(q0[k]['nll']-qnow[k]['nll']) for k in q0)
        nowKL,_,nowrows=ref.kl(WN)
        require(directory,'same-endpoint-recheck',nllgap<=1e-5 and abs(nowKL-Lcached)<=5e-7,
            dict(NLL_max=nllgap,KL_now=nowKL,KL_prior=Lcached,KL_rows=nowrows,prior_gradient=member(original/'cached-gradient.pt')))
        noise=json.loads((original/'noop-repeat.json').read_text())['observed_KL_noise']
        p=torch.load(original/'P-star-basis.pt',weights_only=True,map_location='cpu',mmap=True)
        if matrix_sha256(rt.P[0])!=p['P_raw_sha']:raise ValueError('P_STAR_NATIVE_P_IDENTITY')
        allowed=RightSpace(p['basis'].numpy(),np.empty((p['basis'].shape[1],0)),
            'RESOLVED' if p['basis'].shape[1] else 'REPAIR_SPACE_EMPTY',p['diagnostic'])
        stage='projector_receipt'
        space=edit_null_space(allowed,K);write(directory/'EN-F-space.json',space.receipt())
        if space.receipt()!=json.loads((original/'EN-F-space.json').read_text()):
            # Geometry timer is not a numerical identity. Compare actual factors
            # and spectrum only; no timer equality or byte-math claim invented.
            old=json.loads((original/'EN-F-space.json').read_text());new=space.receipt()
            for key in ('basis_sha256','blocked_sha256','dimension','status'):
                if new[key]!=old[key]:raise ValueError('RECOMPUTED_GEOMETRY_NOT_IDENTICAL:'+key)
        pj=projector_diagnostics(space)
        if space.status!='RANK_UNRESOLVED':require(directory,'projector',pj['status']=='PASS',pj)
        stage='A_map';A=rt.factor_A(native);ca=row_space(A);save_tensor(directory/'A.pt',A);write(directory/'CA-space.json',ca.receipt())
        stage='FD'
        generator=torch.Generator().manual_seed(20260918)
        directions={'CA':ca.project(G),'random':torch.randn(WN.shape,generator=generator,dtype=torch.float64)}
        if space.status!='RANK_UNRESOLVED':directions['EN-F']=space.project(G)
        else:write(directory/'EN-F-FD-not-run.json',dict(status='RANK_UNRESOLVED',derivative_pass=False))
        fd_results={}
        for name,v in directions.items():fd_results[name]=fd(ref,WN,G,v,float((WN.double()-rt.W0.double()).norm()),noise,directory/'FD',name)
        stage='actual_nullspace_response'
        if space.status=='RESOLVED':
            v=space.project(G);chi=float(v.square().sum())
            if chi and Lcached>1e-6:
                ideal=-(Lcached/chi)*v;weight=(WN.double()+ideal).float()
                save_tensor(directory/'invariant-probe.pt',dict(ideal=ideal,actual_weight=weight,WN_sha=tensor_sha(WN)))
                inv=invariant(cur,rows,q0,weight,WN,ideal,K,allowed);require(directory,'actual-invariant',inv['pass'],inv)
                station=[cur.key_stationarity(i,weight) for i in range(len(cur.caches))];rt.sync_oracles()
                require(directory,'perturbed-key-stationarity',all(s['byte_equal'] for s in station),dict(rows=station))
                parity=[cur.compare_logits(i,weight,weight) for i in range(len(cur.caches))];rt.sync_oracles()
                require(directory,'nonzero-physical-cached-parity',max(s['max_abs'] for s in parity)<=1e-4,dict(rows=parity))
            else:write(directory/'actual-invariant.json',dict(status='NO_DIRECTION_OR_LOSS_FLOOR',nonzero_path_NOT_TESTED=True))
        else:write(directory/'actual-invariant.json',dict(status=space.status,nonzero_path_NOT_TESTED=True))
        stage='restore';rt.guard();after=rt.byte_hash_nonselected()
        require(directory,'nonselected-exact-restore',before==after and torch.equal(rt.W.detach().cpu(),WN),
            dict(nonselected_before=digest(before),nonselected_after=digest(after),selected_expected=tensor_sha(WN),selected_actual=tensor_sha(rt.W)))
        cov=json.loads((original/'COV-resolution.json').read_text())
        write(directory/'READY.json',dict(status='T_READY',scope='cold8 actual T, saved stages reused after receipt-only repair; M not run',
            space_status=space.status,FD=fd_results,EN_COV_resolution=cov['resolution'],technical_contract=digest(NUMERIC),
            source=lock['execution'],lock_identity=lock['lock_identity'],teacher=rt.teacher.receipt,identity=rt.identity,
            timing=rt.timing,oracle_work=dict(reference=ref.work,protected=cur.work),wall_seconds=time.monotonic()-start,
            peak_gpu_allocated=torch.cuda.max_memory_allocated(),peak_gpu_reserved=torch.cuda.max_memory_reserved(),
            peak_host_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,prior_stage_reuse=prior,
            P_star_basis=member(original/'P-star-basis.pt'),M_new_submissions=0))
    except BaseException as exc:
        write(directory/'failure.json',dict(status='TECHNICAL_HOLD' if isinstance(exc,TechnicalHold) else 'TECHNICAL_FAILURE',stage=stage,
            error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start,
            timing={} if rt is None else rt.timing,oracle_work=[] if rt is None else [o.work for o in rt.oracles],thresholds_unchanged=True,main_started=False))
        raise
