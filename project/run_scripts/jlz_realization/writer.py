"""RAM-only W/H/RNG transaction; native CPU FP32 history, no replay."""
import random
import numpy as np
import torch
from .common import require,state,tensor_sha

def rng_snapshot():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

def rng_restore(rng):
    random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2])
    if rng[3]:torch.cuda.set_rng_state_all(rng[3])

def rng_equal(rng):
    now=rng_snapshot()
    return now[0]==rng[0] and now[1][0]==rng[1][0] and np.array_equal(now[1][1],rng[1][1]) and now[1][2:]==rng[1][2:] and torch.equal(now[2],rng[2]) and len(now[3])==len(rng[3]) and all(torch.equal(x,y) for x,y in zip(now[3],rng[3]))

class Transaction:
    def __init__(self,a,history):self.a,self.history=a,history;self.done=False;self.rollback_verified=False
    def __enter__(self):
        self.before=state(self.a,self.history);self.guard=self.a.guard();self.rng=rng_snapshot()
        self.W={l:w.detach().cpu().clone() for l,w in self.a.weights.items()}
        self.H={l:h.clone() for l,h in self.history.items()};return self
    def finish(self):
        require(self.a.guard()==self.guard,'NONSELECTED_PARAMETER_MUTATION');self.done=True
    def __exit__(self,kind,value,tb):
        if not self.done:
            with torch.no_grad():
                for l,w in self.a.weights.items():w.copy_(self.W[l])
                for l,h in self.history.items():h.copy_(self.H[l])
            rng_restore(self.rng)
            require(state(self.a,self.history)==self.before and rng_equal(self.rng) and self.a.guard()==self.guard,'ROLLBACK_MISMATCH')
            self.rollback_verified=True
        self.W.clear();self.H.clear()

@torch.no_grad()
def commit(a,history,entry,payload):
    require(set(payload['weights'])==set(a.weights),'WHOLE_LAYER_COMMIT')
    keys={}
    for l,w in a.weights.items():
        require(bool(torch.isfinite(payload['weights'][l]).all()),'NONFINITE_ACCEPTED_WEIGHT')
        w.copy_(payload['weights'][l]);require(torch.equal(w,payload['weights'][l]),'COMMIT_NOT_EXACT')
        k=payload['keys'][l].to(device='cpu',dtype=torch.float32).contiguous()
        h=history[l]
        require(h.dtype==torch.float32 and h.device.type=='cpu' and k.shape==(h.shape[0],entry['pack']['n_requests']) and bool(torch.isfinite(k).all()),'NATIVE_HISTORY_SCHEMA')
        keys[str(l)]=tensor_sha(k)
        h.add_(k@k.T)  # One native CPU FP32 Gram; no mirror/division/dedup.
        require(bool(torch.isfinite(h).all()),'NONFINITE_HISTORY')
    return dict(after=state(a,history),history_appends=len(history),history_key_hash=keys,
        history_accumulation='native_CPU_FP32_kappa_Gram_exactly_once_no_mirror',accepted_weight_copy_exact=True,
        replay=False,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
