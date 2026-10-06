"""Alpha no-grad fresh upper BUILD; same materialized FP32 payload and whole-B Q."""
import time
import torch
import torch.nn.functional as F
from project.run_scripts.jlz_realized_subject.geometry import mean_keys
from .geometry import ridge
from project.run_scripts.jlz_native_writer_aware.physical import move,materialize,linear
from project.run_scripts.jlz_native_writer_aware.common import require

def positions(group,device):
    rows=group['rows']
    return (torch.arange(len(rows),device=device),
            torch.tensor([r['lookup'] for r in rows],device=device),
            torch.tensor([r['global_row'] for r in rows],device=device))

@torch.no_grad()
def build(a,entry,R,candidate,expose_mean_M=False):
    started=time.monotonic();groups=entry['groups']
    rows=[r for g in groups for r in g['rows']]
    require([r['global_row'] for r in rows]==list(range(len(rows))),'ROW_ORDER')
    rw=[i for i,r in enumerate(rows) if r['kind']=='rewrite'];rwrows=[rows[i] for i in rw]
    boundary={a.first:[dict(key=g['cache']['key'],residual=g['cache']['residual']) for g in groups]}
    weights={};P={};K={};raw={};v={};prebase={};metadata={};mean_M={}
    for index,l in enumerate(a.sites):
        subjects=[];bases=[]
        for g,b in zip(groups,boundary[l]):
            ix,pos,_=positions(g,'cpu');key=b['key'][ix,pos].to(a.device)
            subjects.append(key)
            bases.append(a.compose(l,key,b['residual'][ix,pos].to(a.device),entry['entry_weights'][l]))
        key=torch.cat(subjects);k=mean_keys(key[rw].T,rwrows,entry['pack']).double()
        if candidate==0:
            require(bool((R[l]==0).all()),'GPTJ_C0_REQUEST_ZERO')
            ref=entry['mean_keys'][l].to(k.device).double()
            require(bool(((k-ref).abs()<=2e-5+2e-4*ref.abs()).all()),'GPTJ_C0_NATIVE_UPPER_KEY_PARITY')
        if l==a.first and 'alpha_operator' in entry['first_geometry']:
            geo=entry['first_geometry']['alpha_operator']
            require(torch.equal(geo['K'],k),'FIRST_KEY_CACHE_IDENTITY')
        else:
            geo=ridge(a,k,entry['factors'][l])
            if l==a.first:entry['first_geometry']['alpha_operator']=geo
        p=geo['P'];w=materialize(entry['entry_weights'][l],R[l],p)
        require(bool(torch.isfinite(w).all()),'NONFINITE_WEIGHT')
        weights[l]=w;P[l]=p;K[l]=k;raw[l]=key.cpu()
        v[l]=(a.local_linear(l,key,w)-a.local_linear(l,key,entry['entry_weights'][l])).cpu()
        prebase[l]=torch.cat(bases).cpu();metadata[l]=geo['metadata']
        if expose_mean_M:mean_M[l]=geo['M']
        if index+1<len(a.sites):
            nxt=a.sites[index+1];boundary[nxt]=[]
            for g,b in zip(groups,boundary[l]):
                cache=move(b,a.device);kw=move(g['cache']['kwargs'],a.device)
                nk,nr=a.stage(l,nxt,cache['key'],cache['residual'],R[l],p,w,kw)
                boundary[nxt].append(dict(key=nk.cpu(),residual=nr.cpu()))
                del cache,nk,nr,kw
    result=dict(boundary=boundary,weights=weights,P=P,K=K,raw=raw,v=v,prebase=prebase,
                metadata=metadata,candidate=candidate,rows=rows,entry_id=id(entry),cache_versions={l:(id(entry['factors'][l]['LU' if a.profile['writer']=='alphaedit' else 'A']),entry['factors'][l]['LU' if a.profile['writer']=='alphaedit' else 'A']._version,K[l]._version,P[l]._version) for l in a.sites},seconds=time.monotonic()-started)
    if expose_mean_M:result['mean_M']=mean_M
    return result
