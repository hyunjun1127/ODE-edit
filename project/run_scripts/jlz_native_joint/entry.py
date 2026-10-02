"""Entry-scoped clean L4 and strict subject-prefix KV. No persistent cache."""
import torch
from .adapter import unwrap, detach
from .inputs import subset
from .common import require

class ReadOnlyPrefix:
    def __init__(self, kv):
        self.kv = kv

    def update(self, key, value, layer, cache_kwargs=None):
        k, v = self.kv[layer]
        return torch.cat((k.to(key.device), key), -2), torch.cat((v.to(value.device), value), -2)

class PrefixCapture:
    def __init__(self, adapter, rows, spec):
        self.adapter, self.rows, self.spec = adapter, rows, spec
        self.starts = [spec['lookup'][r] for r in rows]
        self.first = min(adapter.sites)
        self.handles, self.kv = [], {}
        self.hidden = None

    def __enter__(self):
        def first(module, args, output):
            self.hidden = unwrap(output).detach().cpu().clone()
        self.handles.append(self.adapter.blocks[self.first].register_forward_hook(first))
        for layer in range(self.first+1, len(self.adapter.blocks)):
            def attn(module, args, kwargs, layer=layer):
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                hidden = kwargs.get('hidden_states', args[0] if args else None)
                with torch.no_grad():
                    shape = (*hidden.shape[:-1], -1, module.head_dim)
                    key = module.k_proj(hidden).view(shape).transpose(1,2)
                    value = module.v_proj(hidden).view(shape).transpose(1,2)
                    cos, sin = kwargs['position_embeddings']
                    # apply_rotary is identical for K, Q result is discarded.
                    _, key = apply_rotary_pos_emb(key, key, cos, sin)
                    width = max(self.starts)
                    k = torch.zeros((*key.shape[:2], width, key.shape[-1]), dtype=key.dtype, device=key.device)
                    v = torch.zeros_like(k)
                    for i, n in enumerate(self.starts):
                        k[i, :, :n] = key[i, :, :n]
                        v[i, :, :n] = value[i, :, :n]
                    self.kv[layer] = (k.cpu(), v.cpu())
            self.handles.append(self.adapter.blocks[layer].self_attn.register_forward_pre_hook(attn, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()

    def finish(self, tokens, versions):
        require(self.hidden is not None and len(self.kv) == len(self.adapter.blocks)-self.first-1, 'PREFIX_CAPTURE_INCOMPLETE')
        self.lengths = tokens['attention_mask'].sum(1).cpu().tolist()
        self.versions = versions
        self.identity = self.spec['identity']
        return self

    def replay(self, deltas):
        a = self.adapter
        require(a.versions() == self.versions and self.identity == self.spec['identity'], 'STALE_ENTRY_CACHE')
        b, d = len(self.rows), self.hidden.shape[-1]
        tails = [n-s for n,s in zip(self.lengths, self.starts)]
        width, prefix = max(tails), max(self.starts)
        hidden = torch.zeros((b,width,d), device=a.device, dtype=torch.float32)
        for i,(s,n) in enumerate(zip(self.starts, self.lengths)):
            hidden[i,:n-s] = self.hidden[i,s:n].to(a.device)
        req = torch.tensor([self.spec['row_request'][r] for r in self.rows], device=a.device)
        ix = torch.arange(b,device=a.device)
        hidden = hidden.clone()
        hidden[ix,0] = hidden[ix,0] + deltas[self.first][:,req].T
        pos = torch.tensor(self.starts,device=a.device)[:,None] + torch.arange(width,device=a.device)[None,:]
        keypos = torch.cat((torch.arange(prefix,device=a.device).expand(b,-1),pos),1)
        validprefix = torch.arange(prefix,device=a.device)[None,:] < torch.tensor(self.starts,device=a.device)[:,None]
        validtail = torch.arange(width,device=a.device)[None,:] < torch.tensor(tails,device=a.device)[:,None]
        validkey = torch.cat((validprefix,validtail),1)
        allowed = validkey[:,None,:] & (keypos[:,None,:] <= pos[:,:,None])
        mask = torch.zeros((b,1,width,prefix+width),device=a.device,dtype=hidden.dtype)
        mask.masked_fill_(~allowed[:,None],torch.finfo(hidden.dtype).min)
        rotary = a.model.model.rotary_emb(hidden,pos)
        cache = ReadOnlyPrefix(self.kv)
        nll = hidden if a.nll_layer == self.first else None
        for layer in range(self.first+1,len(a.blocks)):
            hidden = unwrap(a.blocks[layer](hidden,attention_mask=mask,position_ids=pos,
                           position_embeddings=rotary,past_key_values=cache,use_cache=False))
            if layer in deltas:
                hidden = hidden.clone()
                hidden[ix,0] = hidden[ix,0] + deltas[layer][:,req].T
            if layer == a.nll_layer:
                nll = hidden
        final = a.model.model.norm(hidden)
        return (final,final,True) if a.nll_layer == a.final_layer else (nll,final,False)
