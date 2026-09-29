"""Five-layer all-token low-rank AL, fixed budgets, physical full guard only at round end."""
import contextlib
import time
from .common import *

def energy(v):return .5*sum(float(x.detach().double().square().sum()) for x in v)

def projected(v,g,step):
    import torch
    norm=sum(x.double().square().sum() for x in g).sqrt().clamp_min(1e-12)
    out=[x.detach()-step*y/norm.to(y.dtype) for x,y in zip(v,g)]
    radius=sum(x.double().square().sum() for x in out).sqrt()
    if float(radius)>.01:out=[x*(.01/radius).to(x.dtype) for x in out]
    return out

def al_value(c,dual,beta):
    return sum((max(0.,dual.get(k,0.)+beta*x)**2-dual.get(k,0.)**2)/(2*beta) for k,x in c.items())

def next_dual(dual,c,beta):
    ans=dict(dual)
    for k,x in c.items():ans[k]=max(0.,dual.get(k,0.)+beta*x)
    return ans

def bounds_for(arm,base_anchor,anchors,base_entry,history_entry):
    require(arm in ('JOINT_STEP','JOINT_CUM'),'JOINT_ARM')
    return dict(base=(base_entry if arm=='JOINT_STEP' else base_anchor)+.05,
        history={h:(history_entry[h] if arm=='JOINT_STEP' else anchors[h])+.10 for h in anchors})

def warm_dual(dual):return {k:v for k,v in dual.items() if k=='base' or k.startswith('history:')}

def working_set(history,residuals):
    old=[h for h in history if h.startswith('old:')]
    new=sorted([h for h in history if h not in old],key=lambda h:(-residuals['history:'+h],h))[:32]
    return old+new

def basis(P,M,K):
    import torch
    p=P.to(K.device,dtype=torch.float64);m=M.to(K.device,dtype=torch.float64);k=K.double()
    system=p@(k@k.T+m);system.diagonal().add_(10.)
    rhs=p@k; a=torch.linalg.solve(system,rhs)
    residual=float((system@a-rhs).norm()/rhs.norm().clamp_min(1e-30));del system,p,m,rhs
    u,s,_=torch.linalg.svd(a,full_matrices=False)
    keep=s>s.max()*1e-6; q=u[:,keep].float()
    require(bool(torch.isfinite(q).all()) and q.shape[1]<=K.shape[1],'BASIS')
    err=float((a-q.double()@(q.double().T@a)).norm()/a.norm().clamp_min(1e-30))
    return q,dict(rank=q.shape[1],singular_values=s.cpu().tolist(),discarded=int((~keep).sum()),raw_solve_relative_residual=residual,span_residual=err,dtype='FP64 raw-P solve/thin SVD; FP32 Q')


class JointHook:
    def __init__(self,rt,q,v,norms):self.rt=rt;self.q=q;self.v=v;self.norms=norms;self.handles=[]
    def __enter__(self):
        import torch.nn.functional as F
        for i,l in enumerate(LAYERS):
            def hook(m,args,out,i=i):
                # args[0] is LIVE downstream input, every token; no detach or entry-key substitution.
                return out+F.linear(F.linear(args[0],self.q[i].T),self.v[i]*self.norms[i])
            self.handles.append(self.rt.model.model.layers[l].mlp.down_proj.register_forward_hook(hook))
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()


class Objective:
    def __init__(self,scorer,current,native,base,teacher,history,bounds):
        self.scorer=scorer;self.current=current;self.native=native;self.base=base;self.teacher=teacher;self.history=history;self.bounds=bounds

    def components(self,history_ids):
        out={}
        for cid in self.native:
            out[f'edit:{cid}']=(1.,[(r,1/6,'nll') for r in self.native[cid]])
            r=self.current[cid];r=[x for x in r if x['kind']=='R'];new=next(x for x in r if x['label']=='new');true=next(x for x in r if x['label']=='true')
            out[f'preference:{cid}']=(0.,[(new,1.,'nll'),(true,-1.,'nll')])
        out['base']=(self.bounds['base'],[(r,1/len(self.base),'kl') for r in self.base])
        for h in history_ids:out['history:'+h]=(self.bounds['history'][h],[(self.history[h],1.,'nll')])
        return out

    def scalar(self,row,kind,category):
        import torch
        lp,_=self.scorer.forward(row,category)
        if kind=='kl':
            p0=self.teacher[row['row_id']].to(lp.device);require(p0.shape==lp.shape,'TEACHER_SHAPE')
            return (p0.exp()*(p0-lp)).sum(-1).mean()
        y=torch.tensor(row['target_ids'],device=lp.device)
        return -lp.gather(1,y[:,None])[:,0].mean()

    def evaluate(self,history_ids,category):
        import torch
        ans={};rows={}
        with torch.no_grad():
            for name,(bound,terms) in self.components(history_ids).items():
                values=[]
                for row,weight,kind in terms:
                    key=(row['row_id'],kind)
                    if key not in rows:rows[key]=float(self.scalar(row,kind,category))
                    values.append(weight*rows[key])
                ans[name]=sum(values)-bound
        require(all(__import__('math').isfinite(x) for x in ans.values()),'OBJECTIVE_FINITE')
        return ans,[dict(row_id=k[0],kind=k[1],value=v) for k,v in rows.items()]

    def gradient(self,history_ids,c,dual,beta,v):
        import torch
        for x in v:x.grad=None
        edit=None
        parts=self.components(history_ids)
        for group in ('edit','protection'):
            coefficients={}
            for name,(_,terms) in parts.items():
                if (name.startswith(('edit:','preference:')))!=(group=='edit'):continue
                coeff=max(0.,dual.get(name,0.)+beta*c[name])
                for row,w,kind in terms:
                    key=(row['row_id'],kind)
                    if key not in coefficients:coefficients[key]=[row,0.,kind]
                    coefficients[key][1]+=coeff*w
            for row,coefficient,kind in coefficients.values():
                # Zero multiplier contributes exact zero; every constraint was evaluated first.
                if coefficient:
                    loss=self.scalar(row,kind,'joint_gradient_'+group)*coefficient
                    start=time.monotonic();loss.backward()
                    if loss.is_cuda:torch.cuda.synchronize(loss.device)
                    self.scorer.backward_seconds+=time.monotonic()-start
                    self.scorer.backwards+=1;self.scorer.backward_tokens+=len(row['input_ids'])
            if group=='edit':edit=[torch.zeros_like(x) if x.grad is None else x.grad.detach().clone() for x in v]
        protection=[(torch.zeros_like(x) if x.grad is None else x.grad.detach())-a for x,a in zip(v,edit)]
        grad=[(torch.zeros_like(x) if x.grad is None else x.grad.detach().clone())+x.detach() for x in v]
        require(all(bool(torch.isfinite(x).all()) for x in grad),'GRADIENT_FINITE')
        diag=[]
        for l,a,b in zip(LAYERS,edit,protection):
            an=float(a.double().norm());bn=float(b.double().norm());dot=float((a.double()*b.double()).sum())
            diag.append(dict(layer=l,edit_grad_norm=an,protection_grad_norm=bn,dot=dot,cosine=dot/(an*bn) if an*bn else None))
        return grad,diag


def solve(rt,scorer,requests,current,native,base,teacher,history,anchors,base_anchor,dual,arm,output,batch):
    import torch
    require(len(requests)==BATCH_SIZE,'JOINT_USER_BS1')
    start=time.monotonic();output=Path(output);entry=rt.snapshot();entry_hash=rt.hashes();mh=tensor_sha(rt.M);rng=rt.rng_get()
    norms=[float(rt.w0[l].double().norm()) for l in LAYERS];q=[];geom=[]
    geometry_start=time.monotonic()
    for l in LAYERS:
        k=rt.keys(requests,l);require(k.shape[1]==BATCH_SIZE,'JOINT_KEY_CARDINALITY')
        qi,gi=basis(rt.P[l-4],rt.M[l-4],k);gi.update(layer=l,key_sha=tensor_sha(k),basis_sha=tensor_sha(qi));q.append(qi);geom.append(gi)
    geometry_seconds=time.monotonic()-geometry_start
    dev=rt.weights[4].device
    v=[torch.zeros((rt.weights[l].shape[0],x.shape[1]),device=dev,dtype=torch.float32,requires_grad=True) for l,x in zip(LAYERS,q)]
    old_ids=[h for h in history if h.startswith('old:')]; all_ids=list(history)
    bounds=dict(base=base_anchor+.05,history={h:anchors[h]+.10 for h in history})
    obj=Objective(scorer,current,native,base,teacher,history,bounds)
    # Entry observation is needed for STEP bounds and working-set slack; not a candidate full guard.
    entry_c,entry_rows=obj.evaluate(all_ids,'joint_entry_risk')
    observed_base=entry_c['base']+bounds['base']
    observed_history={h:entry_c['history:'+h]+bounds['history'][h] for h in history}
    bounds=bounds_for(arm,base_anchor,anchors,observed_base,observed_history);obj.bounds=bounds
    entry_c['base']=observed_base-bounds['base']
    for h in history:entry_c['history:'+h]=observed_history[h]-bounds['history'][h]
    dual=warm_dual(dual)
    best=None;best_e=None;full_c=entry_c;rounds=[];proposals=trials=0
    save(output/'entry-objective.json',dict(bounds=bounds,residuals=entry_c,rows=entry_rows,geometry=geom,weight=entry_hash,history=mh))
    if max(entry_c.values(),default=0.)<=1e-5:
        # Already physically feasible zero write: the entry full scan is the sole acceptance guard.
        hist=rt.append(requests)
        return dict(accepted=True,outcome='ZERO_WRITE_ALREADY_FEASIBLE',selected=dict(round=0,energy=0.,weight=entry_hash,residuals=entry_c,rows=entry_rows),
            rounds=[],proposals=0,trials=0,full_guard_passes=1,dual=dual,bounds=bounds,geometry=geom,history=hist,
            weight_before=entry_hash,weight_after=rt.hashes(),history_before=mh,history_after=tensor_sha(rt.M),geometry_seconds=geometry_seconds,
            entry_risk_scans=1,seconds=time.monotonic()-start),entry
    for round_index,beta in enumerate((1,2,4,8,16),1):
        working=working_set(history,full_c);rr=dict(round=round_index,beta=beta,working_history=working,proposals=[])
        for step in range(8):
            proposals+=1
            with JointHook(rt,q,v,norms):
                c,raw=obj.evaluate(working,'joint_merit');grad,gd=obj.gradient(working,c,dual,beta,v)
            merit=energy(v)+al_value(c,dual,beta);accepted=False;pr=dict(proposal=proposals,merit=merit,residuals=c,gradient=gd,trials=[])
            for bt in range(6):
                trials+=1;candidate=projected(v,grad,1e-3*.5**bt)
                disp=sum(float((g.double()*(y.double()-x.detach().double())).sum()) for x,y,g in zip(v,candidate,grad))
                with JointHook(rt,q,candidate,norms):cc,raw_trial=obj.evaluate(working,'joint_backtrack')
                trial_e=energy(candidate);trial_merit=trial_e+al_value(cc,dual,beta);ok=trial_merit<=merit+1e-4*disp
                pr['trials'].append(dict(backtrack=bt,step=1e-3*.5**bt,energy=trial_e,merit=trial_merit,displacement_dot_gradient=disp,armijo=ok,residuals=cc))
                if ok:
                    v=[x.detach().requires_grad_(True) for x in candidate];accepted=True;break
            pr['stop']='ARMIJO_ACCEPTED' if accepted else 'ROUND_LINE_SEARCH_EXHAUSTED';rr['proposals'].append(pr)
            if not accepted:break
        with JointHook(rt,q,v,norms):hook_c,hook_rows=obj.evaluate(working,'materialization_hook_panel')
        with torch.no_grad():
            for i,l in enumerate(LAYERS):
                rt.weights[l].copy_(entry[l].to(dev)+(v[i]*norms[i])@q[i].T)
        require(all(bool(torch.isfinite(w).all()) for w in rt.weights.values()),'MATERIALIZATION_FINITE')
        # Only this call scans all history for this round. Selection uses PHYSICAL endpoint values.
        full_c,physical_rows=obj.evaluate(all_ids,'full_guard_physical')
        finite_e=energy(v);feasible=max(full_c.values(),default=0.)<=1e-5 and 2*finite_e<=1e-4+1e-5
        rr.update(full_residuals=full_c,full_rows=physical_rows,hook_rows=hook_rows,hook_residuals=hook_c,
             energy=finite_e,feasible=feasible,weight=rt.hashes(),materialization_max_constraint_difference=max(abs(hook_c[k]-full_c[k]) for k in hook_c))
        hook_by={(r['row_id'],r['kind']):r['value'] for r in hook_rows};phys_by={(r['row_id'],r['kind']):r['value'] for r in physical_rows}
        err=max(abs(x-phys_by[k]) for k,x in hook_by.items())
        margin_err=max((abs(hook_c[k]-full_c[k]) for k in hook_c if k.startswith('preference:')),default=0.)
        rr.update(materialization_max_row_difference=err,materialization_max_margin_difference=margin_err,
            parity_nll_limit=rt.config['settings']['materialization_parity_nll_abs'],parity_margin_limit=rt.config['settings']['materialization_parity_margin_abs'])
        save(output/f'round-{round_index}.json',rr)
        # Record all raw comparisons before any numerical parity verdict.
        require(err<=rt.config['settings']['materialization_parity_nll_abs'],'HOOK_MATERIALIZED_NUMERICAL_PARITY')
        require(margin_err<=rt.config['settings']['materialization_parity_margin_abs'],'HOOK_MATERIALIZED_MARGIN_PARITY')
        if feasible and (best is None or finite_e<best_e):
            best=rt.snapshot();best_e=finite_e;best_receipt=dict(round=round_index,residuals=full_c,rows=physical_rows,weight=rr['weight'],energy=finite_e)
        dual=next_dual(dual,full_c,beta);rounds.append(dict(round=round_index,energy=finite_e,feasible=feasible,full_history_count=len(all_ids)))
        rt.restore(entry)
    require(proposals<=40 and trials<=240 and len(rounds)<=5,'SOLVER_BUDGET')
    require(rt.hashes()==entry_hash and tensor_sha(rt.M)==mh,'PRECOMMIT_ROLLBACK')
    accepted=best is not None
    hist=[]
    if accepted:
        rt.restore(best);require(rt.hashes()==best_receipt['weight'],'SELECTED_PHYSICAL_STATE')
        hist=rt.append(requests)
    else:
        rt.restore(entry);rt.rng_set(rng)
    rt.check_fixed()
    receipt=dict(accepted=accepted,outcome='ACCEPTED' if accepted else 'ATOMIC_REJECT_NO_FINITE_FEASIBLE_CANDIDATE',
         selected=best_receipt if accepted else None,rounds=rounds,proposals=proposals,trials=trials,full_guard_passes=len(rounds),
         dual={k:v for k,v in dual.items() if k=='base' or k.startswith('history:')},bounds=bounds,geometry=geom,
         history=hist,weight_before=entry_hash,weight_after=rt.hashes(),history_before=mh,history_after=tensor_sha(rt.M),seconds=time.monotonic()-start)
    receipt.update(geometry_seconds=geometry_seconds,entry_risk_scans=1)
    return receipt,entry
