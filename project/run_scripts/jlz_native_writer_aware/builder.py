"""Whole-B barrier, RAM-only CPU boundaries, one reverse solve VJP per layer."""
import time
import torch
import torch.nn.functional as F
from project.run_scripts.jlz_realized_subject.geometry import ridge,mean_keys
from .physical import move,materialize,linear
from .common import require

def positions(group,device):
    rows=group['rows']
    return (torch.arange(len(rows),device=device),
            torch.tensor([r['lookup'] for r in rows],device=device),
            torch.tensor([r['global_row'] for r in rows],device=device))

@torch.no_grad()
def build(a,entry,R,candidate):
    started=time.monotonic();groups=entry['groups']
    rows=[r for g in groups for r in g['rows']]
    require([r['global_row'] for r in rows]==list(range(len(rows))),'ROW_ORDER')
    rw=[i for i,r in enumerate(rows) if r['kind']=='rewrite'];rwrows=[rows[i] for i in rw]
    boundary={a.first:[dict(key=g['cache']['key'],residual=g['cache']['residual']) for g in groups]}
    weights={};P={};K={};raw={};v={};prebase={};metadata={}
    for index,l in enumerate(a.sites):
        subjects=[];bases=[]
        for g,b in zip(groups,boundary[l]):
            ix,pos,_=positions(g,'cpu');key=b['key'][ix,pos].to(a.device)
            subjects.append(key)
            bases.append(b['residual'][ix,pos].to(a.device)+F.linear(key,entry['entry_weights'][l]))
        key=torch.cat(subjects);k=mean_keys(key[rw].T,rwrows,entry['pack']).double()
        if l==a.first and 'v14_ridge' in entry['first_geometry']:
            geo=entry['first_geometry']['v14_ridge']
            require(torch.equal(geo['K'],k),'FIRST_KEY_CACHE_IDENTITY')
        else:
            geo=ridge(k,entry['factors'][l])
            if l==a.first:entry['first_geometry']['v14_ridge']=geo
        p=geo['P'];w=materialize(entry['entry_weights'][l],R[l],p)
        require(bool(torch.isfinite(w).all()),'NONFINITE_WEIGHT')
        weights[l]=w;P[l]=p;K[l]=k;raw[l]=key.cpu()
        v[l]=(F.linear(key,w)-F.linear(key,entry['entry_weights'][l])).cpu()
        prebase[l]=torch.cat(bases).cpu();metadata[l]=geo['metadata']
        if index+1<len(a.sites):
            nxt=a.sites[index+1];boundary[nxt]=[]
            for g,b in zip(groups,boundary[l]):
                cache=move(b,a.device);kw=move(g['cache']['kwargs'],a.device)
                nk,nr=a.stage(l,nxt,cache['key'],cache['residual'],R[l],p,w,kw)
                boundary[nxt].append(dict(key=nk.cpu(),residual=nr.cpu()))
                del cache,nk,nr,kw
    return dict(boundary=boundary,weights=weights,P=P,K=K,raw=raw,v=v,prebase=prebase,
                metadata=metadata,candidate=candidate,rows=rows,seconds=time.monotonic()-started)

def reverse(a,entry,R,built,v_adjoint,route='direct',stop_solve=False):
    """Gather ALL row adjoints before whole-B solve/mean VJP, then lower stage."""
    start=time.monotonic();incoming=None;result={};ledger=[]
    rows=built['rows'];rw=[i for i,r in enumerate(rows) if r['kind']=='rewrite'];rwrows=[rows[i] for i in rw]
    for index in reversed(range(len(a.sites))):
        l=a.sites[index];r=R[l].detach().requires_grad_(True);p=built['P'][l].detach().requires_grad_(True)
        gr=torch.zeros_like(r);gp=torch.zeros_like(p);boundary_grad=[]
        # This weight graph has no reuse across row-group backwards.
        for gi,(g,b) in enumerate(zip(entry['groups'],built['boundary'][l])):
            k=b['key'].to(a.device).detach().requires_grad_(True)
            res=b['residual'].to(a.device).detach().requires_grad_(True)
            ix,pos,rid=positions(g,a.device);kw=move(g['cache']['kwargs'],a.device)
            W=materialize(entry['entry_weights'][l],r,p) if route=='dense' else built['weights'][l]
            action=linear(k[ix,pos],r,p,W,route)-F.linear(k[ix,pos],entry['entry_weights'][l])
            outputs=[action];seeds=[v_adjoint[l][rid.cpu()].to(a.device)]
            if incoming is not None:
                nk,nr=a.stage(l,a.sites[index+1],k,res,r,p,W,kw,route)
                outputs.extend((nk,nr));seeds.extend(x.to(a.device) for x in incoming[gi])
            gradients=torch.autograd.grad(outputs,(k,res,r,p),seeds,allow_unused=True)
            gk,gx,dr,dp=gradients
            boundary_grad.append([gk.detach().cpu() if gk is not None else torch.zeros_like(b['key']),
                                  gx.detach().cpu() if gx is not None else torch.zeros_like(b['residual'])])
            if dr is not None:gr.add_(dr)
            if dp is not None:gp.add_(dp)
            del action,outputs,seeds,gradients,k,res,W,kw
        # Exactly one candidate-wide solve VJP. First K has no lower dependency,
        # but computing its adjoint makes the same audit invariant universal.
        Kleaf=built['K'][l].detach().requires_grad_(True)
        solved=ridge(Kleaf,entry['factors'][l])
        gK=torch.autograd.grad(solved['P'],Kleaf,gp)[0]
        if stop_solve:gK=torch.zeros_like(gK) # qualification negative control only
        raw=built['raw'][l].to(a.device).T.detach().requires_grad_(True)
        mean=mean_keys(raw[:,rw],rwrows,entry['pack']).double()
        grow=torch.autograd.grad(mean,raw,gK)[0].T.cpu()
        for g,dst in zip(entry['groups'],boundary_grad):
            ix,pos,rid=positions(g,'cpu');dst[0][ix,pos]+=grow[rid]
        result[l]=gr;incoming=boundary_grad
        ledger.append(dict(layer=l,groups=len(entry['groups']),solve_VJP=1,
                           P_adjoint_norm=float(gp.norm()),K_solve_adjoint_norm=float(gK.norm()),
                           direct_R_norm=float(gr.norm()),all_columns=R[l].shape[1]))
    return result,dict(seconds=time.monotonic()-start,layers=ledger,bridge_count=1,
                       boundary_storage='RAM_CPU',route=route,stop_solve_negative_control=stop_solve)
