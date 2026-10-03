"""Fixed 25 candidates / 24 q-Adam updates; no clamp, pulse or early exit."""
import math
import time
import torch
from .causal_builder import build
from .subject import evaluate
from .terminal import observe
from .telemetry import candidate,gradient_measure
from .common import require,write


def scales(anchors,dims):
    return {l:(a.double()/math.sqrt(dims[l][0]*len(dims))).float()[None,:] for l,a in anchors.items()}


def fit(a,entry,arm,out,batch,observer=None):
    B=entry['pack']['n_requests'];scale=scales(entry['anchors'],a.dims)
    q={l:torch.zeros((a.dims[l][0],B),device=a.device,requires_grad=True) for l in a.sites}
    optimizer=torch.optim.Adam(list(q.values()),lr=.1,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
    updates=0;payload=None
    for k in range(1,26):
        start=time.monotonic();optimizer.zero_grad(set_to_none=True)
        R={l:(scale[l]*v).detach().requires_grad_(True) for l,v in q.items()}
        built=build(a,entry,R,k)
        if observer is not None:
            observer('before_evaluate',a,entry,k,R,q,scale,optimizer,built,None)
        result=evaluate(a,entry,R,built,arm,components=k in (2,9,25))
        grad=gradient_measure(R,scale,result)
        layer=candidate(a,entry,R,q,built)
        if k==25:
            payload,terminal=observe(a,entry,R,built)
            write(out/'terminal-actual.json',terminal)
        for l,v in q.items():
            v.grad=B*scale[l]*result['gradient'][l] # full mean -> SUM once; q scale once.
            require(bool(torch.isfinite(v.grad).all()),'NONFINITE_Q_GRADIENT')
        if observer is not None:
            observer('evaluated',a,entry,k,R,q,scale,optimizer,built,result)
        if k<25:optimizer.step();updates+=1
        if observer is not None:
            observer('after_update',a,entry,k,R,q,scale,optimizer,built,result)
        write(out/f'candidate-{k:02d}.json',dict(candidate=k,completed_updates_before=k-1,Adam_updates_after=updates,
            losses=result['losses'],total_mean=result['total_mean'],energy=result['energy'],gradient=grad,layer=layer,
            native_seconds=result['seconds'],builder_seconds=built['seconds'],seconds=time.monotonic()-start,
            prediction_tokens=result['prediction_tokens'],native_rows=result['masked_rows'],
            backward_calls=result['masked_backward_calls'],optimizer_builder_bridges=result['builder_optimizer_bridges'],
            component_builder_backwards=result['component_builder_backwards'],
            solve={str(l):g['metadata'] for l,g in built['geometry'].items()},first_solve_this_candidate=k==1,
            upper_solve_count=len(a.sites)-1,terminal_backward=k==25,terminal_update=False,
            coefficients=dict(norm=.5,allocation=.1,kl=.0625),clamp=False,pulse=False,E_penalty=False,
            replay=False,R_or_P_saved=False,checkpoint_saved=False,actual_B=B))
        print(dict(event='candidate',arm=arm,batch=batch,candidate=k,updates=updates,seconds=time.monotonic()-start),flush=True)
        del built,result,R
    require(updates==24,'UPDATE_BUDGET')
    return payload,dict(candidates=25,Adam_updates=24,terminal_gradient_measured=True,actual_B=B,
                        optimizer_SUM_bridge_once=True,checkpoint_saved=False)
