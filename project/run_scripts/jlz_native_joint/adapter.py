"""Explicit model/native profile. Unimplemented architectures fail closed."""
from contextlib import contextmanager
import torch
from .common import require

def unwrap(output):
    return output[0] if isinstance(output, (tuple, list)) else output

def replace(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    if isinstance(output, list):
        return [hidden] + output[1:]
    return hidden

def detach(value, device='cpu'):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {k: detach(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(detach(v, device) for v in value)
    return value

class LlamaAdapter:
    """Only causal Llama with distinct down_proj parameters is qualified here.

    Generic core obtains actual ordered sites and heterogeneous dimensions via
    this interface; other native mappings must supply their own adapter.
    """
    def __init__(self, model, profile):
        self.model, self.profile = model, profile
        require(model.config.model_type == 'llama', 'UNSUPPORTED_MODEL_ADAPTER')
        self.sites = tuple(profile['eligible_layers'])
        require(self.sites and self.sites == tuple(sorted(set(self.sites))), 'ELIGIBLE_ORDER')
        self.blocks = model.model.layers
        self.weights = {l: self.blocks[l].mlp.down_proj.weight for l in self.sites}
        self.dims = {l: tuple(w.shape) for l, w in self.weights.items()}
        ptrs = [w.data_ptr() for w in self.weights.values()]
        require(len(set(ptrs)) == len(ptrs), 'SHARED_PARAMETER_NEEDS_ALIAS_RULE')
        require(all(w.dtype == torch.float32 for w in self.weights.values()), 'MODEL_DTYPE')
        selected = set(ptrs)
        aliases = [n for n, p in model.named_parameters(remove_duplicate=False) if p.data_ptr() in selected]
        require(len(aliases) == len(self.sites), 'WRITE_PARAMETER_ALIAS')
        self.device = next(model.parameters()).device
        self.nll_layer = profile['nll_layer']
        require(self.nll_layer >= max(self.sites), 'NLL_READOUT_PRECEDES_EDIT')
        self.final_layer = len(self.blocks) - 1
        self.capabilities = dict(causal_prefix=True, tokenwise_mlp=True,
                                 layer_streaming=True, prewrite_history_reuse=True,
                                 alias_free=True, native_block_to_down_additive=True)

    def guard(self):
        selected = {id(w) for w in self.weights.values()}
        return {n: (p.data_ptr(), p._version, tuple(p.shape), str(p.dtype))
                for n, p in self.model.named_parameters() if id(p) not in selected}

    def versions(self):
        return tuple((l, w.data_ptr(), w._version) for l, w in self.weights.items())

    def hook_signature(self):
        return {n: (tuple(m._forward_hooks), tuple(m._forward_pre_hooks))
                for n, m in self.model.named_modules()}

    def head(self, hidden, normalized=False):
        return self.model.lm_head(hidden if normalized else self.model.model.norm(hidden))

    @contextmanager
    def inject(self, deltas, rows, spec, capture=None):
        handles = []
        for layer in self.sites:
            def hook(module, args, output, layer=layer):
                hidden = unwrap(output)
                if capture is not None:
                    capture(layer, hidden)
                if deltas is None:
                    return output
                ix = torch.arange(len(rows), device=hidden.device)
                cols = torch.tensor([spec['lookup'][r] for r in rows], device=hidden.device)
                req = torch.tensor([spec['row_request'][r] for r in rows], device=hidden.device)
                changed = hidden.clone()
                # Always differentiable, including delta==0.
                changed[ix, cols] = hidden[ix, cols] + deltas[layer][:, req].T
                return replace(output, changed)
            handles.append(self.blocks[layer].register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def full_hidden(self, tokens, deltas, rows, spec, capture=None):
        intermediate = {}
        handle = None
        if self.nll_layer != self.final_layer:
            handle = self.blocks[self.nll_layer].register_forward_hook(
                lambda m, a, out: intermediate.update(nll=unwrap(out)))
        try:
            with self.inject(deltas, rows, spec, capture):
                final = self.model.model(**tokens, use_cache=False).last_hidden_state
            nll = final if self.nll_layer == self.final_layer else intermediate['nll']
            return nll, final, self.nll_layer == self.final_layer
        finally:
            if handle is not None:
                handle.remove()
