"""Qualified Llama layout; pure layer functions survive checkpoint recompute."""
import torch
from torch.utils.checkpoint import checkpoint
from official.ours.core.jlz_writer_coupled.physical import LlamaAdapter as BaseAdapter
from .physical_linear import linear

def move(value, device):
    if isinstance(value, torch.Tensor): return value.to(device)
    if isinstance(value, tuple): return tuple(move(x, device) for x in value)
    if isinstance(value, dict): return {k: move(x, device) for k, x in value.items()}
    return value

class LlamaAdapter(BaseAdapter):
    def __init__(self, model, profile):
        super().__init__(model, profile)
        self.checkpoint_enabled = True

    @staticmethod
    def unwrap(value):
        return value[0] if isinstance(value, tuple) else value

    def full(self, tokens):
        saved = {}; handles = []
        for l in set((self.nll_layer, self.final_layer)):
            handles.append(self.blocks[l].register_forward_hook(
                lambda m, args, out, l=l: saved.update({l: self.unwrap(out)})))
        try:
            self.model.model(**tokens, use_cache=False)
            return saved[self.nll_layer], saved[self.final_layer]
        finally:
            for h in handles: h.remove()

    def masked(self, group, increments, capture=False):
        """Add the SAME actual-context v to each full block subject, once."""
        cache = move(group['cache'], self.device); kw = cache['kwargs']
        ix = torch.arange(len(group['rows']), device=self.device)
        pos = torch.tensor([r['lookup'] for r in group['rows']], device=self.device)
        rid = torch.tensor([r['global_row'] for r in group['rows']], device=self.device)
        def inject(h, v):
            result = h.clone(); result[ix, pos] = result[ix, pos] + v[rid]
            return result
        key = cache['key']; pre = cache['residual'] + self.blocks[self.first].mlp.down_proj(key)
        keys = {self.first: key[ix, pos]} if capture else {}
        bases = {self.first: pre[ix, pos]} if capture else {}
        x = inject(pre, increments[self.first])
        nll = x if self.nll_layer == self.first else None
        for l in range(self.first+1, len(self.blocks)):
            if l in increments:
                def step(h, v, l=l):
                    k, residual = self.pre_projection(l, h, kw)
                    base = residual + self.blocks[l].mlp.down_proj(k)
                    return inject(base, v), k[ix, pos], base[ix, pos]
                x, k, base = self.recompute(step, x, increments[l])
                if capture: keys[l], bases[l] = k, base
            else:
                def step(h, l=l): return self.unwrap(self.blocks[l](h, **kw))
                x = self.recompute(step, x)
            if l == self.nll_layer: nll = x
        return nll, x, keys, bases

    def recompute(self, fn, *args):
        if torch.is_grad_enabled() and self.checkpoint_enabled and any(x.requires_grad for x in args if isinstance(x, torch.Tensor)):
            return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=True)
        return fn(*args)

    def pre_projection(self, layer, x, kw):
        block = self.blocks[layer]
        attn = block.self_attn(hidden_states=block.input_layernorm(x), **kw)[0]
        residual = x + attn
        normalized = block.post_attention_layernorm(residual)
        key = block.mlp.act_fn(block.mlp.gate_proj(normalized)) * block.mlp.up_proj(normalized)
        return key, residual

    def stage(self, layer, next_layer, key, residual, D, P, W, kw, route='direct'):
        def step(k, r, d, p, w):
            x = r + linear(k, d, p, w, route)
            for j in range(layer+1, next_layer):
                x = self.unwrap(self.blocks[j](x, **kw))
            return self.pre_projection(next_layer, x, kw)
        return self.recompute(step, key, residual, D, P, W)

    def actual(self, group, D, P, weights, route='direct', capture=False):
        cache = move(group['cache'], self.device); kw = cache['kwargs']
        rows = group['rows']; ix = torch.arange(len(rows), device=self.device)
        pos = torch.tensor([r['lookup'] for r in rows], device=self.device)
        key, residual = cache['key'], cache['residual']
        keys = {self.first: key[ix, pos]} if capture else {}
        x = residual + linear(key, D[self.first], P[self.first], weights[self.first], route)
        subjects = {self.first: x[ix, pos]} if capture else {}
        nll = x if self.nll_layer == self.first else None
        for l in range(self.first+1, len(self.blocks)):
            if l in D:
                def step(h, d, p, w, l=l):
                    k, r = self.pre_projection(l, h, kw)
                    y = r + linear(k, d, p, w, route)
                    return y, k[ix, pos], y[ix, pos]
                x, k, h = self.recompute(step, x, D[l], P[l], weights[l])
                if capture: keys[l], subjects[l] = k, h
            else:
                def step(h, l=l): return self.unwrap(self.blocks[l](h, **kw))
                x = self.recompute(step, x)
            if l == self.nll_layer: nll = x
        return nll, x, subjects, keys
