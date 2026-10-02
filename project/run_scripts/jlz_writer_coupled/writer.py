"""RAM transaction: exact accepted weights, postcommit keys, native memory."""
import random
import numpy as np
import torch
from .common import require,state,digest,tensor_sha
from .inputs import fact_identity,version

class Transaction:
    def __init__(self,a,history,memory):
        self.a,self.history,self.memory=a,history,memory
        self.done=False;self.rollback_verified=False
    def __enter__(self):
        self.before=state(self.a,self.history);self.mem_before=self.memory.summary()
        self.W={l:w.detach().cpu().clone() for l,w in self.a.weights.items()}
        self.H={l:h.clone() for l,h in self.history.items()};self.M=self.memory.snapshot()
        self.rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
        self.guard=self.a.guard();return self
    def finish(self):
        require(self.a.guard()==self.guard,'NONSELECTED_PARAMETER_MUTATION');self.done=True
    def __exit__(self,kind,value,tb):
        if not self.done:
            with torch.no_grad():
                for l,w in self.a.weights.items():w.copy_(self.W[l])
                for l,h in self.history.items():h.copy_(self.H[l])
            self.memory.restore(self.M)
            random.setstate(self.rng[0]);np.random.set_state(self.rng[1]);torch.set_rng_state(self.rng[2])
            if self.rng[3]:torch.cuda.set_rng_state_all(self.rng[3])
            require(state(self.a,self.history)==self.before and self.memory.summary()==self.mem_before,'ROLLBACK_MISMATCH')
            self.rollback_verified=True
        self.W.clear();self.H.clear();self.M=None

@torch.no_grad()
def commit(a,history,memory,entry,payload,records):
    require(set(payload['weights'])==set(a.weights),'WHOLE_LAYER_COMMIT')
    for l,w in a.weights.items():
        require(torch.isfinite(payload['weights'][l]).all(),'NONFINITE_ACCEPTED_WEIGHT')
        w.copy_(payload['weights'][l])
        require(torch.equal(w,payload['weights'][l]),'ACCEPTED_WEIGHT_COPY_NOT_EXACT')
    alpha=torch.tensor(entry['pack']['key_context_weights'],device=a.device,dtype=torch.float32)
    require(entry['pack']['entry_key_prefix_exact'],'COMMITTED_KEY_IDENTITY')
    # These are current keys captured inside the exact accepted all-token
    # physical forward, not entry keys, and include every native occurrence.
    key_hash={}
    for l,h in history.items():
        k=payload['keys'][l].to(a.device)
        require(k.shape==(h.shape[0],len(alpha)) and torch.isfinite(k).all(),'COMMIT_KEY_SCHEMA')
        key_hash[str(l)]=tensor_sha(k)
        weighted=k*alpha.sqrt()
        # One symmetric Gram operand. Explicitly mirror the lower triangle
        # so roundoff cannot turn a symmetric history into a nonsymmetric
        # next-entry SPD system. No jitter, refresh, or changed alpha.
        for start in range(0,h.shape[0],256):
            stop=min(start+256,h.shape[0])
            delta=(weighted[start:stop]@weighted[:stop].T).cpu()
            for local,row in enumerate(range(start,stop)):
                h[row,:row+1].add_(delta[local,:row+1])
                h[:row,row].copy_(h[row,:row])
        require(torch.isfinite(h).all(),'NONFINITE_HISTORY')
    events=[]
    for i,r in enumerate(records):
        inp=entry['kl_inputs'][i]
        events.append(dict(fact_id=fact_identity(r),version=version(r),
            record={'case_id':r['case_id'],'requested_rewrite':r['requested_rewrite']},
            context_nll=payload['context_nll'][i].tolist(),kl_identity=digest(inp),
            kl_input=inp,teacher=entry['teachers'][i]))
    admission=memory.admit(events)
    return dict(after=state(a,history),history_appends=len(history),history_key_hash=key_hash,
        history_key_source='exact_accepted_physical_forward_causal_prefix_identity',
        history_accumulation='FP32 sqrt-alpha Gram; lower-triangle mirror, exactly one append',
        memory=memory.summary(),admission=admission,accepted_weight_copy_exact=True,
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
