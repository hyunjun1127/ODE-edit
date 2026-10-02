"""V9 q-Adam, physical D leaf and a single scaled total-adjoint bridge."""
import math
import random
import time
import torch
from .common import require,write,digest,tensor_sha
from .subject import native
from .causal_builder import build
from .allocation import loss as allocation_loss
from .physical_aux import actual
from .telemetry import measure

def partitions(n,seed,stream,batch):
    rng=random.Random(int(digest([seed,stream,batch,'current']),16));order=list(range(n));rng.shuffle(order)
    return [order[i::4] for i in range(4)]

def scales(anchors,dims):
    return {l:(a.double()/math.sqrt(dims[l][0]*len(dims))).float()[None,:] for l,a in anchors.items()}

@torch.no_grad()
def clamp(q,s,anchors):
    result={}
    for l,v in q.items():
        D=s[l]*v;n=D.norm(dim=0);radius=.75*anchors[l]
        factor=torch.minimum(torch.ones_like(n),radius/n.clamp_min(torch.finfo(n.dtype).tiny))
        v.mul_(factor);require(bool(torch.isfinite(v).all()),'NONFINITE_Q')
        result[str(l)]=dict(clipped=int((factor<1).sum()),radial_scale=factor.tolist(),pre_rho=(n/anchors[l]).tolist(),
                            post_rho=((s[l]*v).norm(dim=0)/anchors[l]).tolist())
    return result

def fit(a,entry,arm,out,batch,seed,stream,terminal_callback=None):
    B=entry['pack']['n_requests'];s=scales(entry['anchors'],a.dims)
    q={l:torch.zeros((a.dims[l][0],B),device=a.device,requires_grad=True) for l in a.sites}
    optimizer=torch.optim.Adam(list(q.values()),lr=.1,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
    current=partitions(B,seed,stream,batch);write(out/'partitions.json',dict(current=current,actual_B=B,replay=False))
    count=0;payload=None;artifacts=[]
    for k in range(1,26):
        started=time.monotonic();backward=k<25;I=current[k//5-1] if k in (5,10,15,20) else []
        optimizer.zero_grad(set_to_none=True)
        D={l:(s[l]*v).detach().requires_grad_(backward) for l,v in q.items()}
        # Terminal captures every RW virtual hidden in this native pass, no extra teacher forward.
        n=native(a,entry,D,backward,range(B) if k==25 else I)
        with torch.set_grad_enabled(backward):
            built=build(a,entry,D,k);policy,energy=allocation_loss(D,built['geometry'],entry['anchors'],arm)
        layer,stored=measure(D,built,entry,out/'diagnostics',k);artifacts.extend(stored)
        aux=actual(a,entry,built,D,n['teachers'],I,backward) if I else None
        if backward:
            outputs=[policy];adjoints=[policy.new_tensor(B)]
            if aux:
                for l,p in built['P'].items():
                    if p.requires_grad:outputs.append(p);adjoints.append(aux['grad_P'][l])
            torch.autograd.backward(outputs,adjoints)
            for l,d in D.items():
                if aux:d.grad.add_(aux['grad_D'][l])
                require(d.grad is not None and bool(torch.isfinite(d.grad).all()),'NONFINITE_TOTAL_D_GRADIENT')
                q[l].grad=s[l]*d.grad # exactly once after ALL adjoints.
        gradient={str(l):dict(D=float(d.grad.double().norm()),q=float(q[l].grad.double().norm())) for l,d in D.items()} if backward else None
        if k==25:
            result=actual(a,entry,built,D,n['teachers'],range(B),False,terminal=True);payload=result['payload'];comparison={}
            for l,key in payload['keys'].items():
                ref=built['geometry'][l]['K'].detach().cpu().float();err=(key-ref).abs();limit=1e-4+1e-5*ref.abs()
                comparison[str(l)]=dict(max_absolute=float(err.max()),excess=float((err-limit).max()),builder_sha=tensor_sha(ref),actual_sha=tensor_sha(key))
            write(out/'terminal-key-comparison.json',comparison)
            require(all(v['excess']<=0 for v in comparison.values()),'TERMINAL_KAPPA_MISMATCH')
            write(out/'terminal-actual.json',dict(context_nll=payload['context_nll'],native_kl=payload['native_kl'],
                virtual_context_nll=n['nll'],virtual_kl=n['kl'],decomposition=result['decomposition']))
            if terminal_callback:terminal_callback(D,built,payload,n['teachers'])
        else:result=None
        receipt=dict(candidate=k,completed_updates_before=count,gradient_measured=backward,gradient_norm=gradient,
            native_nll_mean=float(n['nll'].mean()),native_kl_mean=float(n['kl'].mean()),native_norm_mean=n['norm_sum']/B,
            policy_mean=float(policy.detach()),lambda_allocation=.1,energy=energy,layer=layer,lr=.1,warmup=0,
            native_seconds=n['seconds'],builder_seconds=built['seconds'],native_rows=n['rows'],native_prediction_tokens=n['prediction_tokens'],
            upper_solve_count=len(a.sites)-1,first_solve_this_candidate=k==1,
            solve={str(l):g['metadata'] for l,g in built['geometry'].items()},
            auxiliary=None if aux is None else dict(loss_sum=aux['loss_sum'],stats=aux['stats'],seconds=aux['seconds'],current=I,decomposition=aux['decomposition']),
            key_source='actual_lower_writer_native_FP32_mean',K_candidate=k,P_candidate=k,all_current_columns=True,
            grad_P_enabled=backward,grad_K_solve_enabled=backward,builder_delta_hook=False,replay=False,
            seconds=time.monotonic()-started,terminal_actual_seconds=None if result is None else result['seconds'])
        if backward:
            optimizer.step();count+=1
            receipt['post_update_clamp']=clamp(q,s,entry['anchors'])
            if k==1:
                rho2=torch.stack([((s[l]*v).double().norm(dim=0)/entry['anchors'][l].double()).square() for l,v in q.items()]).sum(0)
                receipt['first_step_joint_rho2']=rho2.tolist();receipt['first_step_limit']=.01+1e-8+1e-5*.01
                write(out/'first-step.json',receipt)
                require(bool((rho2<=receipt['first_step_limit']).all()),'FIRST_STEP_Q_BRIDGE_BOUND')
        receipt['Adam_updates_after']=count;write(out/f'candidate-{k:02d}.json',receipt)
        print(dict(event='candidate',arm=arm,batch=batch,candidate=k,updates=count,seconds=receipt['seconds']),flush=True)
        del built,policy,n,aux,result,D
    write(out/'diagnostic-manifest.json',artifacts)
    require(count==24,'ADAM_COUNT')
    return payload,dict(candidates=25,Adam_updates=count,arm=arm,actual_B=B,terminal_gradient_measured=False,checkpoints_saved=False)
