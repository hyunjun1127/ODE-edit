"""One fixed B2 candidate, original/full-gradient comparison; no extra fit."""
import torch
from project.run_scripts.jlz_realized_subject.qualification import compare,full_masked
from project.run_scripts.jlz_realization.writer import rng_snapshot,rng_equal
from .builder import build,reverse
from .subject import evaluate
from .routes import annotate
from .optimizer import project
from .common import require,write

def check(a,entry,out):
    annotate(entry);B=entry['pack']['n_requests'];R={}
    # One zero owner and one boundary owner; both budgets checked by CPU fixtures.
    for i,l in enumerate(a.sites):
        u=torch.zeros(a.dims[l][0],B,device=a.device)
        if i<2:
            direction=torch.sin(torch.arange(a.dims[l][0],device=a.device,dtype=torch.float32)+.3)
            u[:,B-1]=direction/direction.double().norm().float()*.75
        R[l]=(u*entry['anchors'][l][None,:]).detach()
    projections={}
    for radius in (.75,1.5):
        blocks=[R[l][:,B-1]/entry['anchors'][l][B-1] for l in a.sites]
        _,projections[str(radius)]=project(blocks,radius)
    built=build(a,entry,R,0);rng=rng_snapshot();guard=a.guard()
    values=evaluate(a,entry,built,capture=True)
    adj=evaluate(a,entry,built,backward=True)
    require(rng_equal(rng) and a.guard()==guard,'QUALIFICATION_STATE_MUTATION')
    original,oldreceipt=reverse(a,entry,R,built,adj['adjoint'],cached=False,prune_first=False)
    optimized,receipt=reverse(a,entry,R,built,adj['adjoint'],cached=True,prune_first=True)
    comparisons=compare(optimized,original)
    with full_masked(a):native=evaluate(a,entry,built,backward=True)
    nativegrad,_=reverse(a,entry,R,built,native['adjoint'],cached=False,prune_first=False)
    nativecmp=compare(original,nativegrad)
    native_loss=bool(((native['F']-values['F']).abs()<=1e-5+1e-4*native['F'].abs()).all())
    require(native_loss and all(x['passed'] for x in nativecmp.values()),'ORIGINAL_NATIVE_FULLGRADIENT_PARITY')
    selected={i for i,g in enumerate(entry['groups']) if any(r['request']==0 for r in g['rows'])}
    probe=evaluate(a,entry,built,group_indices=selected)
    stateless=rng_equal(rng) and a.guard()==guard
    parity=torch.equal(adj['F'],values['F']) and probe['complete_owner'][0] and float(probe['F'][0])==float(values['F'][0])
    require(stateless,'NATIVE_STATE_MUTATION')
    optimized_pass=all(x['passed'] for x in comparisons.values())
    routes=dict(R1='original_contiguous_MB2_explicit_rows',R2=optimized_pass,R3=optimized_pass,R4a=bool(parity),R4b=False)
    record=dict(actual_FP32_projection_checks=projections,status='TECHNICAL_READY',routes=routes,B=B,fixed_candidates=1,extra_fits=0,optimizer_updates=0,
        full_native_loss=native_loss,native_full_gradient=nativecmp,original_vs_cached_pruned=comparisons,
        grad_vs_nograd=bool(torch.equal(adj['F'],values['F'])),complete_owner_probe=bool(parity),state_RNG_unchanged=True,
        reverse=receipt,original_reverse=oldreceipt,forward_evaluations=4,reverse_evaluations=3,
        fallback='original_same_method if optional optimized comparisons fail',
        CPU15_not_actual_model=True,terminal_capture_preserved=values['masked_bases'] is not None)
    write(out/'qualification.json',record);return record
