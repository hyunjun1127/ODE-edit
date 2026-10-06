"""GPT-J parallel attention/MLP path; no Llama structural aliases or weight copies.

The residual cache retains input and attention separately to preserve native
FP32 `(attention + fc_out(key)) + input` order, including fc_out bias.
Actual GPU confirmation is integrated in the real trajectory, not a toy pass.
"""
import torch
import torch.nn.functional as F
from project.run_scripts.jlz_native_writer_aware.physical import Adapter as Parent
from project.run_scripts.jlz_realized_subject.profile import move
from project.run_scripts.jlz_native_writer_aware.common import require


class Adapter(Parent):
    def __init__(self, model, profile):
        c = model.config
        require(c.model_type == 'gptj', 'GPTJ_MODEL_TYPE')
        require((c.n_embd, c.n_layer, c.n_inner or 4*c.n_embd, c.vocab_size)
                == (4096, 28, 16384, 50400), 'GPTJ_ARCHITECTURE')
        require(c._attn_implementation == 'eager', 'GPTJ_EAGER')
        self.model, self.profile = model, profile
        self.blocks = model.transformer.h
        self.sites = tuple(profile['eligible_layers'])
        require(self.sites == (3, 4, 5, 6, 7, 8), 'GPTJ_LAYER_BINDING')
        self.first = min(self.sites)
        self.nll_layer = self.final_layer = 27
        require(profile['nll_layer'] == 27 and profile['anchor_layer'] == 8, 'GPTJ_READOUT')
        self.weights = {l: self.projection(l).weight for l in self.sites}
        ptrs = {w.data_ptr() for w in self.weights.values()}
        require(len(ptrs) == len(self.sites) and sum(p.data_ptr() in ptrs for _, p in
            model.named_parameters(remove_duplicate=False)) == len(self.sites), 'GPTJ_ALIAS')
        require(all(w.dtype == torch.float32 and tuple(w.shape) == (4096, 16384)
                    for w in self.weights.values()), 'GPTJ_FP32_WEIGHT_LAYOUT')
        self.device = next(model.parameters()).device
        self.dims = {l: tuple(w.shape) for l, w in self.weights.items()}
        self.checkpoint_enabled = True
        model.eval()
        model.requires_grad_(False)

    def projection(self, layer):
        return self.blocks[layer].mlp.fc_out

    def attention(self, layer):
        return self.blocks[layer].attn

    def entry_residual(self, x, attention):
        return torch.cat((x, attention), dim=-1)

    def local_linear(self, layer, key, weight):
        return F.linear(key, weight, self.projection(layer).bias)

    def compose(self, layer, key, residual, weight):
        x, attention = residual.split(self.model.config.n_embd, dim=-1)
        return (attention + self.local_linear(layer, key, weight)) + x

    def pre_projection(self, layer, x, kw):
        block = self.blocks[layer]
        normalized = block.ln_1(x)
        attention = block.attn(hidden_states=normalized, **kw)[0]
        key = block.mlp.act(block.mlp.fc_in(normalized))
        return key, self.entry_residual(x, attention)

    def stage(self, layer, next_layer, key, residual, R, P, W, kw, route='direct'):
        require(not torch.is_grad_enabled(), 'GPTJ_PRODUCTION_BUILDER_NO_REVERSE')
        x = self.compose(layer, key, residual, W)
        for j in range(layer + 1, next_layer):
            x = self.unwrap(self.blocks[j](x, **kw))
        return self.pre_projection(next_layer, x, kw)

    def head(self, x):
        return self.model.lm_head(self.model.transformer.ln_f(x)).float()

    def observer_hidden(self, **tokens):
        return self.model.transformer(**tokens, use_cache=False).last_hidden_state

    def full(self, tokens):
        saved = {}
        handle = self.blocks[27].register_forward_hook(
            lambda module, args, out: saved.update({27: self.unwrap(out)}))
        try:
            self.model.transformer(**tokens, use_cache=False)
            return saved[27], saved[27]
        finally:
            handle.remove()

    def masked(self, group, increments, capture=False):
        cache = move(group['cache'], self.device)
        kw = cache['kwargs']
        ix = torch.arange(len(group['rows']), device=self.device)
        pos = torch.tensor([r['lookup'] for r in group['rows']], device=self.device)
        rid = torch.tensor([r['global_row'] for r in group['rows']], device=self.device)

        def inject(base, v):
            out = base.clone()
            out[ix, pos] = out[ix, pos] + v[rid]
            return out

        key = cache['key']
        base = self.compose(self.first, key, cache['residual'], self.weights[self.first])
        keys = {self.first: key[ix, pos]} if capture else {}
        bases = {self.first: base[ix, pos]} if capture else {}
        x = inject(base, increments[self.first])
        for l in range(self.first + 1, len(self.blocks)):
            if l in increments:
                def step(h, v, l=l):
                    k, residual = self.pre_projection(l, h, kw)
                    before = self.compose(l, k, residual, self.weights[l])
                    return inject(before, v), k[ix, pos], before[ix, pos]
                x, k, base = self.recompute(step, x, increments[l])
                if capture:
                    keys[l], bases[l] = k, base
            else:
                def step(h, l=l):
                    return self.unwrap(self.blocks[l](h, **kw))
                x = self.recompute(step, x)
        if group.get('native_c0_pending'):
            require(all(bool((v==0).all()) for v in increments.values()),'GPTJ_FIRST_SUBJECT_C0_ZERO')
            errors=[]
            for j,row in enumerate(group['rows']):
                pos=(torch.nonzero(row['target']!=-100).flatten().to(self.device) if row['kind']=='rewrite'
                     else torch.tensor([row['lookup']],device=self.device))
                reference=group['native_c0_selected'][j].to(self.device)
                error=(x[j,pos].detach()-reference).abs();limit=2e-5+2e-4*reference.abs()
                require(bool((error<=limit).all()),'GPTJ_C0_NATIVE_MASKED_HIDDEN_PARITY')
                errors.append(float(error.max()))
            group['native_c0_pending']=False
            group['native_c0_error_max']=max(errors)
            del group['native_c0_selected']
        return x, x, keys, bases
