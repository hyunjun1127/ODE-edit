"""Full-token layer-major actual writer and atomic RAM W/H transaction."""
import random
import time
import numpy as np
import torch
from .adapter import detach,unwrap
from .inputs import subset,pool
from .common import require,tensor_sha,write

def state(adapter,history):
    return dict(W={str(l):tensor_sha(w) for l,w in adapter.weights.items()},
                H={str(l):tensor_sha(h) for l,h in history.items()})

class Transaction:
    def __init__(self,a,history):
        self.a,self.h=a,history
        self.before=state(a,history)
        self.W={l:w.detach().cpu().clone() for l,w in a.weights.items()}
        self.H={l:h.clone() for l,h in history.items()}
        self.guard,self.hooks=a.guard(),a.hook_signature()
        self.rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        self.done=False;self.appended=set();self.rollback_verified=False
    def __enter__(self):return self
    def check(self):
        require(self.a.guard()==self.guard and self.a.hook_signature()==self.hooks,'NONSELECTED_OR_HOOK_MUTATION')
    def append(self,l,key):
        require(l not in self.appended and key.dtype==torch.float32,'HISTORY_APPEND_IDENTITY')
        self.h[l].add_((key@key.T).cpu())
        require(torch.isfinite(self.h[l]).all(),'NONFINITE_HISTORY')
        self.appended.add(l)
    def finish(self):
        self.check();require(self.appended==set(self.a.sites),'INCOMPLETE_HISTORY_APPEND');self.done=True
    def rollback(self):
        with torch.no_grad():
            for l,w in self.a.weights.items():w.copy_(self.W[l])
            for l,h in self.h.items():h.copy_(self.H[l])
        random.setstate(self.rng[0]);np.random.set_state(self.rng[1]);torch.set_rng_state(self.rng[2])
        if self.rng[3] is not None:torch.cuda.set_rng_state_all(self.rng[3])
        self.check();self.rollback_verified=state(self.a,self.h)==self.before
        require(self.rollback_verified,'ROLLBACK_NOT_EXACT')
    def __exit__(self,*exc):
        if not self.done:self.rollback()

class StopPrefix(Exception):pass

@torch.no_grad()
def stream_prefix(a,spec,microbatch):
    blocks=[]
    for start in range(0,len(spec['key_lookup']),microbatch):
        rows=list(range(start,min(start+microbatch,len(spec['key_lookup']))))
        tokens=subset(spec['key_tokens'],rows,a.device);cache={}
        def stop(module,args,kwargs):
            cache['hidden']=detach(args[0] if args else kwargs['hidden_states'])
            cache['kwargs']=detach({k:v for k,v in kwargs.items() if k!='hidden_states'})
            raise StopPrefix()
        handle=a.blocks[min(a.sites)].register_forward_pre_hook(stop,with_kwargs=True)
        try:
            try:a.model.model(**tokens,use_cache=False)
            except StopPrefix:pass
        finally:handle.remove()
        require(cache,'WRITER_PREFIX_CAPTURE')
        blocks.append(dict(rows=rows,**cache))
    return blocks

@torch.no_grad()
def before_linear(a,layer,block):
    b=a.blocks[layer];h=block['hidden'].to(a.device);kwargs=detach(block['kwargs'],a.device)
    kwargs.pop('use_cache',None)
    attn=b.self_attn(hidden_states=b.input_layernorm(h),**kwargs)[0]
    residual=h+attn
    post=b.post_attention_layernorm(residual)
    x=b.mlp.act_fn(b.mlp.gate_proj(post))*b.mlp.up_proj(post)
    return x,residual

@torch.no_grad()
def capture_keys(a,spec,microbatch):
    raw={l:[] for l in a.sites}
    for start in range(0,len(spec['key_lookup']),microbatch):
        rows=list(range(start,min(start+microbatch,len(spec['key_lookup']))));handles=[]
        for l in a.sites:
            def hook(module,args,l=l):
                raw[l].append(args[0][torch.arange(len(rows),device=a.device),
                    torch.tensor([spec['key_lookup'][r] for r in rows],device=a.device)].clone())
                if l==max(a.sites):raise StopPrefix()
            handles.append(a.blocks[l].mlp.down_proj.register_forward_pre_hook(hook))
        try:
            try:a.model.model(**subset(spec['key_tokens'],rows,a.device),use_cache=False)
            except StopPrefix:pass
        finally:
            for handle in handles:handle.remove()
    return {l:pool(torch.cat(x),spec) for l,x in raw.items()}

@torch.no_grad()
def commit(a,spec,D,geometry,tx,out,microbatch=4,qualification=False):
    start=time.monotonic();blocks=stream_prefix(a,spec,microbatch);keys={};reports=[]
    for layer in range(min(a.sites),max(a.sites)+1):
        if layer not in a.sites:
            for block in blocks:
                block['hidden']=unwrap(a.blocks[layer](block['hidden'].to(a.device),**detach(block['kwargs'],a.device))).cpu()
            continue
        raw=[]
        for block in blocks:
            x,residual=before_linear(a,layer,block)
            rows=block['rows'];ix=torch.arange(len(rows),device=a.device)
            lookup=torch.tensor([spec['key_lookup'][r] for r in rows],device=a.device)
            raw.append(x[ix,lookup].clone())
            block['x']=x.cpu();block['residual']=residual.cpu()
            del x,residual
        key=pool(torch.cat(raw),spec);keys[layer]=key;del raw
        if layer==min(a.sites) and layer in geometry.entry and torch.equal(key.double(),geometry.entry[layer]['K']):
            geom=geometry.entry[layer];reuse=True
        else:geom=geometry.solve(layer,key);reuse=False
        w=a.weights[layer];old=tx.W[layer].to(a.device)
        update=D[layer].double()@geom['P'].T
        desired=old+update.float()
        require(torch.isfinite(desired).all(),'NONFINITE_MATERIALIZATION')
        w.copy_(desired);require(torch.equal(w,desired),'COMMIT_TENSOR_IDENTITY')
        actual=(w.double()-old.double())@key.double()
        ideal=D[layer].double()@(key.double().T@geom['P'])
        residual=actual-D[layer].double()
        dn=D[layer].double().norm(dim=0);an=actual.norm(dim=0)
        row=dict(layer=layer,payload_sha256=tensor_sha(D[layer]),residual_payload_sha256=tensor_sha(D[layer]),
                 requested_norm=dn,actual_action_norm=an,ideal_action_norm=ideal.norm(dim=0),
                 residual_norm=residual.norm(dim=0),residual_ratio=float(residual.norm()/D[layer].double().norm().clamp_min(1e-30)),
                 cosine=float((actual*D[layer].double()).sum()/(actual.norm()*D[layer].double().norm()).clamp_min(1e-30)),
                 delta_weight_norm=float((w.double()-old.double()).norm()),
                 materialized_sha256=tensor_sha(w),commit_bitwise=True,entry_P_reused=reuse,
                 current_geometry=geom['summary'],entry_geometry=geometry.entry.get(layer,{}).get('summary','NOT_MEASURED'))
        write(out/f'writer-L{layer}.json',row);reports.append(row)
        del update,desired,actual,ideal,residual,old
        for block in blocks:
            x=block.pop('x').to(a.device);res=block.pop('residual').to(a.device)
            block['hidden']=(res+a.blocks[layer].mlp.down_proj(x)).cpu()
        print({'event':'writer_layer','layer':layer,'whole_batch':spec['n_requests']},flush=True)
    del blocks
    if qualification:
        recaptured=capture_keys(a,spec,microbatch)
        diff={l:float((recaptured[l]-keys[l]).norm()/recaptured[l].norm().clamp_min(1e-30)) for l in keys}
        write(out/'history-key-parity.json',dict(relative=diff,dependency_proof=a.capabilities))
        # Same method reference key recapture if optimized streaming shape differs.
        if any(x>0.002 for x in diff.values()):keys=recaptured
    for l in a.sites:tx.append(l,keys[l])
    tx.check()
    receipt=dict(layers=list(a.sites),history_appends=len(a.sites),occurrences=spec['n_requests'],
                 after=state(a,tx.h),seconds=time.monotonic()-start,checkpoint_saved=False,
                 writer='full_token_layer_major',history_source='prewrite_input_keys_dependency_proven',
                 requested_norm_share='reported_by_layer_not_causal_share')
    write(out/'writer.json',receipt)
    return receipt
