"""Synchronous EfficiencyAdam with terminal-only accepted materialized payload."""
import json,time
import torch
from project.run_scripts.jlz_shared_budget.optimizer import EfficiencyAdam,norm_gradient
from .builder import build,reverse
from .subject import evaluate
from .common import require,write,tensor_sha
from .telemetry import candidate_summary

def stop(J,index,cap=25):
    return bool((J<.05).all()) or index==cap-1

def fit(a,entry,out):
    out.mkdir(exist_ok=False);B=entry['pack']['n_requests'];sites=a.sites
    u={l:torch.zeros(a.dims[l][0],B,device=a.device) for l in sites}
    star=entry['anchors'][a.profile['anchor_layer']]
    optimizers=[EfficiencyAdam([u[l][:,r] for l in sites],star[r]) for r in range(B)]
    updates=0;start=time.monotonic()
    for index in range(25):
        R={l:(entry['anchors'][l][None,:]*u[l]).detach() for l in sites}
        built=build(a,entry,R,index)
        obs=evaluate(a,entry,built,capture=True)
        norms=torch.stack([x.norm(dim=0) for x in u.values()])
        penalty=(.5/star.cpu().double())*norms.sum(0).cpu().double()
        J=obs['F']+penalty
        require(bool(torch.isfinite(J).all()),'NONFINITE_J')
        terminal=stop(J,index)
        event=dict(candidate=index,ordinal=index+1,B=B,J_mean=float(J.mean()),J=J.tolist(),
            nll_mean=float(obs['nll'].mean()),KL_mean=float(obs['kl'].mean()),norm_mean=float(penalty.mean()),
            requested_budget=norms.double().sum(0).cpu().tolist(),terminal=terminal,updates_before=updates,
            builder_seconds=built['seconds'],masked_seconds=obs['seconds'],native_forward_replay=False,
            gradient=None,gradient_measured=False,component_gradient_norms=None,solve=built['metadata'],
            realization=candidate_summary(a,entry,R,built))
        if terminal:
            event['gradient_status']='NO_BACKWARD_TERMINAL'
            write(out/f'candidate-{index:02d}.json',event)
            write(out/'fit.json',dict(candidates=index+1,updates=updates,request_evaluations=B*(index+1),
                request_update_participations=B*updates,terminal_candidate=index,all_J_under_threshold=bool((J<.05).all()),
                early_stop='ALL_CURRENT_REQUESTS_ONLY',seconds=time.monotonic()-start,
                weights={str(l):tensor_sha(w) for l,w in built['weights'].items()},
                R={str(l):tensor_sha(r) for l,r in R.items()},checkpoint_saved=False))
            return dict(u=u,R=R,built=built,observed=obs,candidates=index+1,updates=updates)
        # Common stop must be decided BEFORE any backward. Recompute SAME
        # candidate native loss only for backward; count it as explicit replay.
        replay=evaluate(a,entry,built,backward=True)
        require(torch.equal(replay['F'],obs['F']),'SAME_CANDIDATE_NATIVE_REPLAY')
        gradR,reverse_receipt=reverse(a,entry,R,built,replay['adjoint'])
        gradients={l:gradR[l]*entry['anchors'][l][None,:] for l in sites}
        native_norm={str(l):float(g.norm()) for l,g in gradients.items()};norm_norm={}
        for l in sites:
            analytic=torch.stack([norm_gradient(u[l][:,r],.5/star[r]) for r in range(B)],1)
            norm_norm[str(l)]=float(analytic.norm());gradients[l].add_(analytic)
        require(all(bool(torch.isfinite(g).all()) for g in gradients.values()),'NONFINITE_SUM_GRADIENT')
        projection=[];optstart=time.monotonic()
        for r,opt in enumerate(optimizers):
            blocks,receipt=opt.step([u[l][:,r] for l in sites],[gradients[l][:,r] for l in sites])
            for l,value in zip(sites,blocks):u[l][:,r].copy_(value)
            projection.append(dict(request=r,**receipt))
        updates+=1
        event.update(gradient_status='SUM_F_PLUS_ANALYTIC_NORM_ONCE',gradient_measured=True,
            component_gradient_norms=dict(F=native_norm,norm=norm_norm,total={str(l):float(g.norm()) for l,g in gradients.items()}),
            native_forward_replay=True,replay_seconds=replay['seconds'],replay_groups=replay['backward_groups'],
            reverse=reverse_receipt,projection=projection,optimizer_seconds=time.monotonic()-optstart,
            updates_after=updates)
        write(out/f'candidate-{index:02d}.json',event)
        print(json.dumps(dict(event='V14_CANDIDATE',candidate=index,Jmean=float(J.mean()),updates=updates)),flush=True)
        del built,obs,replay,gradR,gradients
    raise AssertionError('UNREACHABLE_NO_TERMINAL')
