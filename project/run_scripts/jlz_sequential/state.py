"""RAM-only atomic weight/history/observer transaction and compact identities."""
import copy
import hashlib
import json
import random
import numpy as np
import torch

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def tensor_sha(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(value.shape), str(value.dtype))).encode())
    h.update(memoryview(value.numpy()).cast('B'))
    return h.hexdigest()

def weights(model):
    return {l: model.model.layers[l].mlp.down_proj.weight for l in (4,5,6,7,8)}

def state_hash(model, history):
    return {'W': {str(l):tensor_sha(w) for l,w in weights(model).items()},
            'H': {str(l):tensor_sha(h) for l,h in history.items()}}

def parameter_guard(model):
    selected = {id(w) for w in weights(model).values()}
    return {n:(p.data_ptr(), p._version, str(p.dtype), tuple(p.shape))
            for n,p in model.named_parameters() if id(p) not in selected}

def hooks(model):
    return {n:(tuple(m._forward_hooks), tuple(m._forward_pre_hooks)) for n,m in model.named_modules()}

class Transaction:
    def __init__(self, model, history, contexts, ledger):
        self.model, self.history, self.contexts, self.ledger = model, history, contexts, ledger
        self.before = state_hash(model, history)
        self.W = {l:w.detach().cpu().clone() for l,w in weights(model).items()}
        self.H = {l:h.clone() for l,h in history.items()}
        self.context_snapshot, self.ledger_snapshot = copy.deepcopy(contexts), copy.deepcopy(ledger)
        self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                    torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        self.guard, self.hooks = parameter_guard(model), hooks(model)
        self.appended = set()
        self.done = False
        self.restored = False

    def __enter__(self):
        return self

    def append(self, layer, key):
        if layer in self.appended:
            raise RuntimeError('DUPLICATE_HISTORY_APPEND')
        if key.dtype != torch.float32 or not bool(torch.isfinite(key).all()):
            raise RuntimeError('INVALID_POST_KEY')
        with torch.no_grad():
            self.history[layer].add_((key @ key.T).cpu())
        if not bool(torch.isfinite(self.history[layer]).all()):
            raise FloatingPointError('NONFINITE_HISTORY')
        self.appended.add(layer)

    def check(self):
        if parameter_guard(self.model) != self.guard or hooks(self.model) != self.hooks:
            raise RuntimeError('NONSELECTED_OR_HOOK_MUTATION')
        if self.contexts != self.context_snapshot:
            raise RuntimeError('CONTEXT_MUTATION')

    def finish(self, row, publish):
        self.check()
        if self.appended != set(self.H):
            raise RuntimeError('INCOMPLETE_HISTORY_APPEND')
        self.ledger.append(row)
        publish(self.ledger)  # failure here is also rollback, not durable success
        self.done = True

    def rollback(self):
        with torch.no_grad():
            for l,w in weights(self.model).items(): w.copy_(self.W[l])
            for l,h in self.history.items(): h.copy_(self.H[l])
        self.contexts[:] = copy.deepcopy(self.context_snapshot)
        self.ledger[:] = copy.deepcopy(self.ledger_snapshot)
        random.setstate(self.rng[0]); np.random.set_state(self.rng[1]); torch.set_rng_state(self.rng[2])
        if self.rng[3] is not None: torch.cuda.set_rng_state_all(self.rng[3])
        self.check()
        self.restored = state_hash(self.model, self.history) == self.before
        if not self.restored: raise RuntimeError('ROLLBACK_HASH_FAILED')

    def __exit__(self, typ, value, tb):
        if not self.done: self.rollback()
        return False
