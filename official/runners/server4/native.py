"""State adapter only; all editing mathematics stays in official.baselines."""
import hashlib
import torch
from official.baselines import registry
from official.experiments import checkpoint

METHODS = ('ALPHAEDIT', 'ALPHAEDIT_BLUE', 'SPHERE')


def rng_hash():
    from official.experiments.prepare import digest
    import numpy as np
    def encode(value):
        if isinstance(value, torch.Tensor):return tensor_hash(value)
        if isinstance(value, np.ndarray):return hashlib.sha256(value.tobytes()).hexdigest()
        if isinstance(value, dict):return {k:encode(v) for k,v in value.items()}
        if isinstance(value, (tuple,list)):return [encode(v) for v in value]
        return value
    return digest(encode(checkpoint.rng_snapshot()))


def tensor_hash(tensor):
    data = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(data).cast('B')).hexdigest()


class NativeState:
    def __init__(self, model, tokenizer, method, assets):
        if method not in METHODS:
            raise ValueError('SERVER4_METHOD_SCOPE')
        self.model, self.tokenizer, self.method = model, tokenizer, method
        self.module, _, self.apply_native = registry.implementation(method, 'llama3')
        overrides = {} if method == 'ALPHAEDIT_BLUE' else {
            'P_loc': assets['projector']['path'], 'stats_dir': assets['stats_dir'], 'device': 0}
        self.hp = registry.hparams(method, 'llama3', overrides=overrides)
        self.weights = {self.hp.rewrite_module_tmp.format(layer)+'.weight':
                        model.get_parameter(self.hp.rewrite_module_tmp.format(layer)+'.weight')
                        for layer in self.hp.layers}
        mapping = assets['projector']['physical_layers']
        if len(set(mapping)) != len(mapping) or not set(self.hp.layers) <= set(mapping):
            raise ValueError('PROJECTOR_PHYSICAL_LAYER_MAPPING')
        # mmap avoids copying a five-layer file just to select BLUE's L4/L8.
        packed = torch.load(assets['projector']['path'], map_location='cpu',
                            weights_only=True, mmap=True)
        if not isinstance(packed, torch.Tensor) or packed.dtype != torch.float32:
            raise ValueError('PROJECTOR_FP32_TENSOR_REQUIRED')
        if tuple(packed.shape) != (len(mapping), 14336, 14336):
            raise ValueError('PROJECTOR_SHAPE')
        self.P = torch.stack([packed[mapping.index(layer)] for layer in self.hp.layers])
        if not torch.isfinite(self.P).all():
            raise ValueError('PROJECTOR_NONFINITE')
        self.cache = torch.zeros_like(self.P)
        self.module.CONTEXT_TEMPLATES_CACHE = None
        self.module.COV_CACHE.clear()
        self._bind()

    def _bind(self):
        if self.method != 'ALPHAEDIT_BLUE':
            self.module.P, self.module.P_loaded = self.P, True
            self.module.cache_c, self.module.cache_c_new = self.cache, True

    def contexts(self):
        return self.module.get_context_templates(self.model, self.tokenizer)

    def edit(self, records):
        options = registry.call_options(self.method, 'llama3')
        if self.method == 'ALPHAEDIT_BLUE':
            options.update(P=self.P, cache_c=self.cache)
        result = self.apply_native(self.model, self.tokenizer,
            registry.requests(records, self.method, 'llama3'), self.hp, **options)
        if result[0] is not self.model:
            raise ValueError('NATIVE_MODEL_IDENTITY')
        self.cache = result[1] if self.method == 'ALPHAEDIT_BLUE' else self.module.cache_c
        if not torch.isfinite(self.cache).all() or any(
                not torch.isfinite(w).all() for w in self.weights.values()):
            raise ValueError('NATIVE_NONFINITE_STATE')

    def signature(self):
        return dict(weights={k: tensor_hash(v) for k,v in self.weights.items()},
                    cache_c=tensor_hash(self.cache), contexts=self.contexts())

    def save(self, folder, batch, cursor, identity):
        return checkpoint.save(folder, batch=batch, weights=self.weights,
            cache_c={str(layer): self.cache[i] for i,layer in enumerate(self.hp.layers)},
            contexts=self.contexts(), evaluation_cursor=cursor, identity=identity,
            method=self.method, evaluation_complete=True)

    def restore(self, folder, identity):
        payload = checkpoint.load(folder, identity)
        if payload['method'] != self.method or set(payload['weights']) != set(self.weights):
            raise ValueError('RESTORE_NATIVE_METHOD_WEIGHTS')
        if set(payload['cache_c']) != set(map(str, self.hp.layers)):
            raise ValueError('RESTORE_NATIVE_HISTORY_LAYERS')
        with torch.no_grad():
            for name, value in payload['weights'].items():
                if value.shape != self.weights[name].shape:
                    raise ValueError('RESTORE_WEIGHT_SHAPE')
                self.weights[name].copy_(value)
        self.cache = torch.stack([payload['cache_c'][str(l)] for l in self.hp.layers])
        self.module.CONTEXT_TEMPLATES_CACHE = payload['contexts']
        self._bind()
        checkpoint.rng_restore(payload['rng'])
        return payload['batch'], payload['evaluation_cursor']
