"""Synchronous EfficiencyAdam with terminal-only accepted materialized payload."""
import json,time
import torch
from .optimizer import EfficiencyAdam,norm_gradient
from .routes import annotate
from project.run_scripts.jlz_realization.writer import rng_snapshot,rng_equal
from .builder import build,reverse
from .subject import evaluate
from .common import require,write,tensor_sha
from .telemetry import candidate_summary

def stop(J,index,cap=25):
    return bool((J<.05).all()) or index==cap-1

def fit(a,entry,out,routes=None):
    annotate(entry);routes=routes or dict(R2=True,R3=True,R4a=False)
    out.mkdir(exist_ok=False);B=entry['pack']['n_requests'];sites=a.sites
    u={l:torch.zeros(a.dims[l][0],B,device=a.device) for l in sites}
    star=entry['anchors'][a.profile['anchor_layer']]
    optimizers=[EfficiencyAdam([u[l][:,r] for l in sites],star[r]) for r in range(B)]
    updates=0;start=time.monotonic()
    for index in range(25):
        calls_before=dict(getattr(a,'physical_calls',{}))
        R={l:(entry['anchors'][l][None,:]*u[l]).detach() for l in sites}
        built=build(a,entry,R,index)
        norms=torch.stack([x.norm(dim=0) for x in u.values()])
        penalty=(.5/star.cpu().double())*norms.sum(0).cpu().double()
        probe=None;replay=None;witness=False
        if routes.get('R4a') and index<24:
            owner=int(penalty.argmax())
            groups={i for i,g in enumerate(entry['groups']) if any(r['request']==owner for r in g['rows'])}
            rng=rng_snapshot();probe=evaluate(a,entry,built,group_indices=groups)
            require(rng_equal(rng),'PROBE_RNG_MUTATION')
            require(probe['complete_owner'][owner],'PARTIAL_LOSS_IS_NOT_OWNER_WITNESS')
            value=float(probe['F'][owner]+penalty[owner])
            guard=1e-5+1e-4*(float(probe['nll'][owner].abs().mean())+.0625*abs(float(probe['kl'][owner])))
            witness=value>=.05 and value-.05>guard
            probe_receipt=dict(owner=owner,complete_owner=True,J=value,guard=guard,witness=witness,
                groups=probe['forward_groups'],seconds=probe['seconds'],valid_tokens=probe['valid_tokens'],padded_tokens=probe['padded_tokens'])
        else:probe_receipt=dict(witness=False,reason='terminal_value_only' if index==24 else 'R4a_unqualified_original_two_pass')
        if witness:
            replay=evaluate(a,entry,built,backward=True)
            obs=replay
        else:obs=evaluate(a,entry,built,capture=index==24)
        J=obs['F']+penalty
        require(bool(torch.isfinite(J).all()),'NONFINITE_J')
        terminal=stop(J,index)
        require(not witness or not terminal,'WITNESS_PARITY_VIOLATION')
        if terminal and obs['masked_bases'] is None:
            captured=evaluate(a,entry,built,capture=True)
            require(torch.equal(captured['F'],obs['F']),'TERMINAL_CAPTURE_PARITY')
            obs=captured
        event=dict(routes=routes,probe=probe_receipt,physical=dict(subject_groups=obs['forward_groups'],valid_tokens=obs['valid_tokens'],padded_tokens=obs['padded_tokens']),candidate=index,ordinal=index+1,B=B,J_mean=float(J.mean()),J=J.tolist(),
            nll_mean=float(obs['nll'].mean()),KL_mean=float(obs['kl'].mean()),norm_mean=float(penalty.mean()),
            requested_budget=norms.double().sum(0).cpu().tolist(),terminal=terminal,updates_before=updates,
            builder_seconds=built['seconds'],masked_seconds=obs['seconds'],native_forward_replay=False,
            gradient=None,gradient_measured=False,component_gradient_norms=None,solve=built['metadata'],
            realization=candidate_summary(a,entry,R,built))
        if terminal:
            event['gradient_status']='NO_BACKWARD_TERMINAL'
            event['checkpoint_function_work']={k:v-calls_before.get(k,0) for k,v in getattr(a,'physical_calls',{}).items()}
            event['physical']['backward_groups']=0
            write(out/f'candidate-{index:02d}.json',event)
            write(out/'fit.json',dict(candidates=index+1,updates=updates,request_evaluations=B*(index+1),
                request_update_participations=B*updates,terminal_candidate=index,all_J_under_threshold=bool((J<.05).all()),
                early_stop='ALL_CURRENT_REQUESTS_ONLY',seconds=time.monotonic()-start,
                routes=routes,shared_radius=1.5,local_radius=.75,weights={str(l):tensor_sha(w) for l,w in built['weights'].items()},
                R={str(l):tensor_sha(r) for l,r in R.items()},checkpoint_saved=False))
            return dict(u=u,R=R,built=built,observed=obs,candidates=index+1,updates=updates)
        # Common stop must be decided BEFORE any backward. Recompute SAME
        # candidate native loss only for backward; count it as explicit replay.
        if replay is None:
            replay=evaluate(a,entry,built,backward=True)
            require(torch.equal(replay['F'],obs['F']),'SAME_CANDIDATE_NATIVE_REPLAY')
        gradR,reverse_receipt=reverse(a,entry,R,built,replay['adjoint'],cached=routes.get('R2',False),prune_first=routes.get('R3',False))
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
            native_forward_replay=not witness,replay_seconds=0. if witness else replay['seconds'],replay_groups=replay['backward_groups'],
            reverse=reverse_receipt,projection=projection,optimizer_seconds=time.monotonic()-optstart,
            updates_after=updates)
        event['physical'].update(backward_groups=replay['backward_groups'],replay_valid_tokens=0 if witness else replay['valid_tokens'],replay_padded_tokens=0 if witness else replay['padded_tokens'])
        event['checkpoint_function_work']={k:v-calls_before.get(k,0) for k,v in getattr(a,'physical_calls',{}).items()}
        write(out/f'candidate-{index:02d}.json',event)
        print(json.dumps(dict(event='V14_CANDIDATE',candidate=index,Jmean=float(J.mean()),updates=updates)),flush=True)
        del built,obs,replay,gradR,gradients
    raise AssertionError('UNREACHABLE_NO_TERMINAL')
