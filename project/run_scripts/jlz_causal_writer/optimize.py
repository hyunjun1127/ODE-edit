"""25 candidates, 24 constant-LR absolute Adam steps, native post-clamp."""
import random
import time
import torch
from .common import require, write, digest, tensor_sha
from .subject import native
from .causal_builder import build
from .allocation import loss as allocation_loss
from .physical_aux import actual

def partitions(n, seed, stream, batch, purpose):
    rng=random.Random(int(digest([seed,stream,batch,purpose]),16));order=list(range(n));rng.shuffle(order)
    return [order[i::4] for i in range(4)]

def fit(a,entry,residents,references,arm,out,batch,seed,stream,candidates=25,updates=24):
    require((candidates,updates) in ((25,24),(5,4)),'FIXED_CANDIDATE_BUDGET')
    B=entry['pack']['n_requests'];D={l:torch.zeros((a.dims[l][0],B),device=a.device,requires_grad=True) for l in a.sites}
    optimizer=torch.optim.Adam(list(D.values()),lr=.1,betas=(.9,.999),eps=1e-8,weight_decay=0)
    current=partitions(B,seed,stream,batch,'current');past=partitions(len(residents),seed,stream,batch,'past')
    write(out/'partitions.json',dict(current=current,past=past,reference_count=len(residents),actual_B=B))
    count=0;payload=None;entry_P=None
    for k in range(1,candidates+1):
        started=time.monotonic();backward=(k<=updates or candidates==5)
        pulse=k in (5,10,15,20);slot=k//5-1 if pulse else None
        I=current[slot] if pulse else [];J=past[slot] if pulse else []
        optimizer.zero_grad(set_to_none=True)
        n=native(a,entry,D,backward,I)
        with torch.set_grad_enabled(backward):
            built=build(a,entry,D,k)
            policy,layer=allocation_loss(D,built['geometry'],entry['anchors'],arm)
        if entry_P is None:entry_P={l:p.detach().clone() for l,p in built['P'].items()}
        with torch.no_grad():
            for l in a.sites:
                key=built['geometry'][l]['K'];p=built['P'][l];w=built['weights'][l]
                layer[str(l)] = layer.pop(l) if l in layer else layer[str(l)]
                layer[str(l)].update(delta_norm=float(D[l].norm()),
                    actual_write_norm=float((w-entry['entry_weights'][l]).norm()),
                    key_entry_drift=float((key-entry['entry_keys'][l].to(key)).norm()),
                    P_zero_candidate_drift=float((p-entry_P[l]).norm()))
        aux=None
        if pulse:
            aux=actual(a,entry,built,D,n['teachers'],I,residents,references,J,backward)
        if backward:
            outputs=[policy];adjoints=[torch.ones_like(policy)*B]
            if aux:
                for l,p in built['P'].items():
                    if p.requires_grad:outputs.append(p);adjoints.append(aux['grad_P'][l])
            torch.autograd.backward(outputs,adjoints)
            for l,d in D.items():
                if aux:d.grad.add_(aux['grad_D'][l])
                require(d.grad is not None and bool(torch.isfinite(d.grad).all()),'NONFINITE_TOTAL_GRADIENT')
        gradient={str(l):float(d.grad.double().norm()) for l,d in D.items()} if backward else None
        if k==25:
            result=actual(a,entry,built,D,{},range(B),residents,[],[],False,terminal=True)
            payload=result['payload'];comparison={}
            for l,key in payload['keys'].items():
                ref=built['geometry'][l]['K'].detach().cpu().float()
                err=(key-ref).abs();limit=1e-4+1e-5*ref.abs()
                comparison[str(l)]=dict(max_absolute=float(err.max()),excess=float((err-limit).max()),
                    builder_key_sha=tensor_sha(ref),terminal_key_sha=tensor_sha(key))
            write(out/'terminal-key-comparison.json',comparison)
            require(all(v['excess']<=0 for v in comparison.values()),'BUILDER_TERMINAL_KEY_MISMATCH')
        else:result=None
        receipt=dict(candidate=k,completed_updates_before=count,gradient_measured=backward,gradient_norm=gradient,
            native_nll_mean=float(n['nll'].mean()),native_kl_mean=float(n['kl'].mean()),native_norm_mean=n['norm_sum']/B,
            policy_mean=float(policy.detach()),layer=layer,lr=.1,warmup=0,
            native_seconds=n['seconds'],builder_seconds=built['seconds'],
            native_rows=n['rows'],native_prediction_tokens=n['prediction_tokens'],
            builder_RW_rows=len(entry['pack']['key_request']),
            upper_solve_count=len(a.sites)-1,first_solve_this_candidate=k==1,
            peak_GPU_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            solve={str(l):g['metadata'] for l,g in built['geometry'].items()},
            auxiliary=None if aux is None else dict(loss_sum=aux['loss_sum'],stats=aux['stats'],seconds=aux['seconds'],current=I,past=J),
            key_source='actual_lower_writer',K_candidate=k,P_candidate=k,all_current_columns=True,
            grad_P_enabled=True,grad_K_solve_enabled=True,direct_K_policy_gradient=True,builder_delta_hook=False,
            seconds=time.monotonic()-started,terminal_actual_seconds=None if result is None else result['seconds'])
        if k<=updates:
            optimizer.step();count+=1;clamp={}
            with torch.no_grad():
                for l,d in D.items():
                    norms=d.norm(dim=0);radius=.75*entry['anchors'][l]
                    scales=torch.minimum(torch.ones_like(norms),radius/norms.clamp_min(torch.finfo(norms.dtype).tiny))
                    d.mul_(scales)
                    require(bool(torch.isfinite(d).all()),'NONFINITE_ADAM')
                    clamp[str(l)]=dict(clipped=int((scales<1).sum()),max_norm=float(d.norm(dim=0).max()))
            receipt['post_update_clamp']=clamp
        receipt['Adam_updates_after']=count
        write(out/f'candidate-{k:02d}.json',receipt)
        print(dict(event='candidate',arm=arm,batch=batch,candidate=k,updates=count,seconds=receipt['seconds']),flush=True)
        del built,policy,n,aux,result
    require(count==updates,'ADAM_COUNT')
    return payload,dict(candidates=candidates,Adam_updates=count,arm=arm,actual_B=B,
                        terminal_gradient_measured=False if candidates==25 else True,
                        checkpoints_saved=False)
