"""One actual modified linear, exact accepted FP32 weights, direct-D VJP."""
from contextlib import contextmanager
import torch
from official.ours.config import require_config
import torch.nn.functional as F
from official.ours.common import require

class PhysicalLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,D,P,W):
        ctx.save_for_backward(x,P,W)
        ctx.d_dtype=D.dtype
        return F.linear(x,W)
    @staticmethod
    def backward(ctx,g):
        x,p,w=ctx.saved_tensors
        xf=x.reshape(-1,x.shape[-1]);gf=g.reshape(-1,g.shape[-1])
        dx=torch.matmul(g,w) if ctx.needs_input_grad[0] else None
        dD=(gf.double().T@(xf.double()@p)).to(ctx.d_dtype) if ctx.needs_input_grad[1] else None
        return dx,dD,None,None

def materialize(entry,D,P):
    return entry+(D.double()@P.T).to(entry.dtype)

class LlamaAdapter:
    def __init__(self,model,profile):
        profile=require_config(profile)
        self.model,self.profile=model,profile
        require(model.config.model_type=='llama','UNSUPPORTED_MODEL_ADAPTER')
        self.blocks=model.model.layers;self.sites=tuple(profile['eligible_layers'])
        require(self.sites and self.sites==tuple(sorted(set(self.sites))) and min(self.sites)>=0 and max(self.sites)<len(self.blocks),'ELIGIBLE_ORDER')
        self.first=min(self.sites)
        require(all(self.blocks[l].mlp.down_proj.bias is None for l in self.sites),'BIAS_ADAPTER_NOT_QUALIFIED')
        self.weights={l:self.blocks[l].mlp.down_proj.weight for l in self.sites}
        ptrs={w.data_ptr() for w in self.weights.values()}
        require(len(ptrs)==len(self.sites),'WEIGHT_ALIAS')
        require(sum(p.data_ptr() in ptrs for _,p in model.named_parameters(remove_duplicate=False))==len(self.sites),'PARAMETER_ALIAS')
        require(all(w.dtype==torch.float32 for w in self.weights.values()),'FP32_REQUIRED')
        self.device=next(model.parameters()).device
        self.nll_layer=profile['nll_layer'];self.final_layer=len(self.blocks)-1
        require(max(self.sites)<=self.nll_layer<len(self.blocks),'NLL_READOUT_RANGE')
        self.dims={l:tuple(w.shape) for l,w in self.weights.items()}
        model.eval();model.requires_grad_(False)

    def guard(self):
        selected={id(w) for w in self.weights.values()}
        return {n:(p.data_ptr(),p._version,tuple(p.shape),str(p.dtype)) for n,p in self.model.named_parameters() if id(p) not in selected}

    def hook_signature(self):
        return {n:(tuple(m._forward_hooks),tuple(m._forward_pre_hooks),'forward' in m.__dict__) for n,m in self.model.named_modules()}

    def head(self,x):
        return self.model.lm_head(self.model.model.norm(x)).float()

    @contextmanager
    def install(self,D,P,weights,route='direct',capture=None):
        old={}
        for l in self.sites:
            module=self.blocks[l].mlp.down_proj
            old[l]=module.__dict__.get('forward')
            def forward(x,l=l):
                if capture is not None:capture(l,x)
                if route=='dense':return F.linear(x,weights[l])
                return PhysicalLinear.apply(x,D[l],P[l],weights[l])
            module.forward=forward
        try:yield
        finally:
            for l,value in old.items():
                module=self.blocks[l].mlp.down_proj
                if value is None:del module.__dict__['forward']
                else:module.forward=value

    def full(self,tokens):
        saved={};handles=[]
        for l in set((self.nll_layer,self.final_layer)):
            handles.append(self.blocks[l].register_forward_hook(lambda m,a,o,l=l:saved.update({l:o})))
        try:
            self.model.model(**tokens,use_cache=False)
            return saved[self.nll_layer],saved[self.final_layer]
        finally:
            for h in handles:h.remove()

    @torch.no_grad()
    def prefix(self,tokens):
        """Cache exactly before first modified down_proj, no upper-layer KV."""
        from transformers.masking_utils import create_causal_mask
        m=self.model.model;x=m.embed_tokens(tokens['input_ids'])
        pos=torch.arange(x.shape[1],device=x.device);ids=pos.unsqueeze(0)
        mask=create_causal_mask(config=m.config,input_embeds=x,attention_mask=tokens['attention_mask'],
            cache_position=pos,past_key_values=None,position_ids=ids)
        rope=m.rotary_emb(x,ids)
        kw=dict(attention_mask=mask,position_ids=ids,cache_position=pos,position_embeddings=rope,use_cache=False)
        for l in range(self.first):x=self.blocks[l](x,**kw)
        block=self.blocks[self.first]
        attn,_=block.self_attn(hidden_states=block.input_layernorm(x),**kw)
        residual=x+attn
        y=block.post_attention_layernorm(residual)
        key=block.mlp.act_fn(block.mlp.gate_proj(y))*block.mlp.up_proj(y)
        return dict(key=key.detach().cpu(),residual=residual.detach().cpu(),
            kwargs={k:(tuple(t.detach().cpu() for t in v) if isinstance(v,tuple) else v.detach().cpu() if isinstance(v,torch.Tensor) else v) for k,v in kw.items()})

    def cached(self,cache):
        kw={k:(tuple(t.to(self.device) for t in v) if isinstance(v,tuple) else v.to(self.device) if isinstance(v,torch.Tensor) else v) for k,v in cache['kwargs'].items()}
        x=cache['residual'].to(self.device)+self.blocks[self.first].mlp.down_proj(cache['key'].to(self.device))
        nll=x if self.nll_layer==self.first else None
        for l in range(self.first+1,len(self.blocks)):
            x=self.blocks[l](x,**kw)
            if l==self.nll_layer:nll=x
        return nll,x
