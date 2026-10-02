"""Fixed-budget global Adam, all requests/all sites; no quality selection."""
import time
import torch
from .allocation import policy
from .common import require, write, tensor_sha
from .oracle import Oracle

def fit(adapter,spec,geometry,eta,out,candidates=25,microbatch=4,route='strict_prefix',qualification=False):
    a=adapter
    D={l:torch.zeros(shape[0],spec['n_requests'],device=a.device,requires_grad=True) for l,shape in a.dims.items()}
    opt=torch.optim.Adam(list(D.values()),lr=a.profile['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0)
    oracle=Oracle(a,spec,microbatch,route)
    traces=[];start=time.monotonic()
    for it in range(candidates):
        opt.zero_grad(set_to_none=True)
        back=it<candidates-1 or candidates!=25
        task=oracle.evaluate(D,backward=back,initialize=it==0)
        if it==0:
            write(out/'entry-capture.json',dict(input_identity=spec['identity'],
                teacher_sha256=tensor_sha(oracle.teacher),own_entry_weight_versions=oracle.versions,
                anchor_sha256={l:tensor_sha(x) for l,x in oracle.anchors.items()},
                key_sha256={l:tensor_sha(x) for l,x in oracle.keys.items()},
                candidate1_differentiable=back,entry_fused=True,history_appends=0))
        if it==0 and eta:
            for l in a.sites:geometry.solve(l,oracle.keys[l],entry=True,reference_check=qualification)
        anchor2={l:x.square().sum(0) for l,x in oracle.anchors.items()}
        norm=sum((a.profile['norm_factor']*d.norm(dim=0)/anchor2[l]).sum() for l,d in D.items())
        taskgrad={l:d.grad.detach().clone() for l,d in D.items()} if back else {}
        if back:norm.backward()
        v=torch.zeros((),device=a.device,dtype=torch.float64);components={};gradstats={}
        for l,d in D.items():
            if eta:
                value,g,parts=policy(d.detach(),geometry.entry[l],anchor2[l].mean())
                v+=value
                components[l]={k:float(x) for k,x in parts.items()}
                if back:
                    t=taskgrad[l].double()
                    gradstats[l]=dict(task_norm=float(t.norm()),policy_norm=float(g.norm()),
                        ratio=float(g.norm()/t.norm().clamp_min(1e-30)),
                        cosine=float((t*g).sum()/(t.norm()*g.norm()).clamp_min(1e-30)))
                    d.grad.add_(g.to(d.dtype),alpha=eta)
            if back:require(d.grad is not None and torch.isfinite(d.grad).all(),'NONFINITE_OR_MISSING_DELTA_GRAD')
        row=dict(candidate=it+1,nll=task['nll'],kl=task['kl'],native_norm=float(norm.detach()),
                 normalized_V=float(v),eta=eta,V_components=components,gradient=gradstats,
                 task_seconds=task['seconds'],calls=dict(oracle.calls),logical_batch=spec['n_requests'],
                 backward=back,Adam_update=it<candidates-1)
        write(out/f'candidate-{it+1:02d}.json',row);traces.append(row)
        if qualification and it in (0,1):
            # Bounded zero and one nonzero-candidate reference; no optimizer/write.
            saved={l:d.grad.detach().clone() for l,d in D.items()}
            opt.zero_grad(set_to_none=True)
            ref=oracle.evaluate(D,backward=True,route='full_reference',full_head=True)
            diffs={l:float((D[l].grad-taskgrad[l]).norm()/taskgrad[l].norm().clamp_min(1e-30)) for l in D}
            write(out/f'qualification-{it:02d}.json',dict(nll_maxabs=float((ref['nll']-task['nll']).abs().max()),
                  kl_maxabs=float((ref['kl']-task['kl']).abs().max()),all_layer_gradient_relative=diffs,
                  separate_probe=True,quality_gate=False))
            # On unqualified shape rounding use same-method full reference, once.
            if max(diffs.values())>0.002 or float((ref['nll']-task['nll']).abs().max())>5e-5:
                oracle.route='full_reference';oracle.caches.clear()
            for l,d in D.items():
                # If route falls back, use this already measured reference task
                # gradient plus the unchanged native norm/policy contribution.
                d.grad = saved[l] + d.grad - taskgrad[l] if oracle.route=='full_reference' else saved[l]
        if it<candidates-1:
            opt.step()
            with torch.no_grad():
                for l,d in D.items():
                    radius=a.profile['clamp_factor']*anchor2[l].sqrt()
                    d.mul_(torch.minimum(torch.ones_like(radius),radius/d.norm(dim=0).clamp_min(1e-30)))
                    require(torch.isfinite(d).all(),'NONFINITE_UPDATED_DELTA')
        print({'event':'candidate','candidate':it+1,'eta':eta,'nll':float(task['nll'].sum()),'route':oracle.route},flush=True)
    result={l:d.detach().clone() for l,d in D.items()}
    receipt=dict(candidates=candidates,Adam_updates=candidates-1,calls=oracle.calls,
                 seconds=time.monotonic()-start,route=oracle.route,fit_microbatch=microbatch,
                 logical_batch=spec['n_requests'],norms={l:d.norm(dim=0) for l,d in result.items()},
                 checkpoint_saved=False)
    write(out/'fit.json',receipt)
    return result,oracle,receipt
