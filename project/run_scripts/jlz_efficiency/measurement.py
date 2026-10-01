"""Scalar-only evidence and qualification, never a tensor resume artifact."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import torch
from project.run_scripts.jlz_pilot.solver import prox_blocks
from project.run_scripts.jlz_sequential.state import tensor_sha

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def reject(v):
        if isinstance(v,torch.Tensor):raise TypeError('NO_TENSOR_PERSISTENCE')
        raise TypeError(type(v).__name__)
    payload=json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,default=reject)+'\n'
    temp=path.with_name(path.name+'.tmp')
    with temp.open('x') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def point_record(result,x,c,rho,mask,scale):
    f,g,p=result
    pg=x-prox_blocks(x-g,1.,c,rho,mask)
    record=dict(smooth=float(f),nll=p['nll'],kl=p['kl'],gradient_norm=float(g.double().norm()),
        gradient_block_norm=g.double().norm(dim=1).cpu().tolist(),gradient_sha=tensor_sha(g),point_sha=tensor_sha(x),
        weight_sha={str(l):tensor_sha(w) for l,w in p['weights'].items()},
        normalized_prox=float(pg.double().norm())/scale,feasible=bool((x.norm(dim=1)<=rho*(1+1e-6)).all()) and not bool((x[~mask]!=0).any()))
    return record,g.detach().cpu().clone()

def compare_points(ref,rg,actual,ag):
    diff=(ag-rg).double();den=rg.double().norm().clamp_min(1)
    blockdiff=diff.norm(dim=1);blockden=rg.double().norm(dim=1).clamp_min(1)
    errors=dict(smooth_abs=abs(ref['smooth']-actual['smooth']),
        per_request_abs=max(abs(a-b) for k in ('nll','kl') for a,b in zip(ref[k],actual[k],strict=True)),
        gradient_global=float(diff.norm()/den),gradient_block=float((blockdiff/blockden).max()),
        gradient_component=float(diff.abs().max()),prox_abs=abs(ref['normalized_prox']-actual['normalized_prox']))
    limits=dict(smooth_abs=1e-3,per_request_abs=1e-4,gradient_global=1e-5,gradient_block=1e-5,gradient_component=1e-5,prox_abs=5e-6)
    exact=ref['weight_sha']==actual['weight_sha'] and ref['point_sha']==actual['point_sha']
    ok=exact and actual['feasible'] and all(errors[k]<=v for k,v in limits.items())
    return dict(status='PASS' if ok else 'UNQUALIFIED',errors=errors,limits=limits,same_materialization=exact,
        diff_norm=float(diff.norm()),reference_norm=float(rg.double().norm()),block_diff_norm=blockdiff.tolist(),
        block_reference_norm=rg.double().norm(dim=1).tolist(),bitwise_gradient=ref['gradient_sha']==actual['gradient_sha'])

def timing(reference,candidate):
    a=list(reference);b=list(candidate)
    result=dict(reference=dict(min=min(a),median=statistics.median(a),max=max(a)),candidate=dict(min=min(b),median=statistics.median(b),max=max(b)))
    result['status']='TIMING_SEPARATED' if statistics.median(b)<statistics.median(a) and max(b)<min(a) else 'SPEED_UNRESOLVED'
    result['median_ratio_reference_over_candidate']=statistics.median(a)/statistics.median(b)
    return result

def timed(fn,*args,**kwargs):
    torch.cuda.synchronize();start=time.monotonic();v=fn(*args,**kwargs);torch.cuda.synchronize()
    return v,time.monotonic()-start
