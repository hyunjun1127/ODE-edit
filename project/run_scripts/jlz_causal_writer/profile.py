"""Qualified Llama layout; pure layer functions survive checkpoint recompute."""
import torch
from torch.utils.checkpoint import checkpoint
from project.run_scripts.jlz_writer_coupled.physical import LlamaAdapter as BaseAdapter
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

    def recompute(self, fn, *args):
        if torch.is_grad_enabled() and self.checkpoint_enabled and any(x.requires_grad for x in args if isinstance(x, torch.Tensor)):
            return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=True)
        return fn(*args)

    def pre_projection(self, layer, x, kw):
        block = self.blocks[layer]
        attn, _ = block.self_attn(hidden_states=block.input_layernorm(x), **kw)
        residual = x + attn
        normalized = block.post_attention_layernorm(residual)
        key = block.mlp.act_fn(block.mlp.gate_proj(normalized)) * block.mlp.up_proj(normalized)
        return key, residual

    def stage(self, layer, next_layer, key, residual, D, P, W, kw, route='direct'):
        def step(k, r, d, p, w):
            x = r + linear(k, d, p, w, route)
            for j in range(layer+1, next_layer):
                x = self.blocks[j](x, **kw)
            return self.pre_projection(next_layer, x, kw)
        return self.recompute(step, key, residual, D, P, W)

    def native(self, group, D, capture=False):
        cache = move(group['cache'], self.device); kw = cache['kwargs']
        rows = group['rows']; ix = torch.arange(len(rows), device=self.device)
        lookup = torch.tensor([r['lookup'] for r in rows], device=self.device)
        owners = torch.tensor([r['request'] for r in rows], device=self.device)
        def inject(x, d):
            y = x.clone(); y[ix, lookup] = y[ix, lookup] + d[:, owners].T
            return y
        x = cache['residual'] + self.blocks[self.first].mlp.down_proj(cache['key'])
        x = inject(x, D[self.first])
        subjects = {self.first: x[ix, lookup]} if capture else {}
        nll = x if self.nll_layer == self.first else None
        for l in range(self.first+1, len(self.blocks)):
            if l in D:
                def step(h, d, l=l): return inject(self.blocks[l](h, **kw), d)
                x = self.recompute(step, x, D[l])
                if capture: subjects[l] = x[ix, lookup]
            else:
                def step(h, l=l): return self.blocks[l](h, **kw)
                x = self.recompute(step, x)
            if l == self.nll_layer: nll = x
        return nll, x, subjects

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
                def step(h, l=l): return self.blocks[l](h, **kw)
                x = self.recompute(step, x)
            if l == self.nll_layer: nll = x
        return nll, x, subjects, keys
