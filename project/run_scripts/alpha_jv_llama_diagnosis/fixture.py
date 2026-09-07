"""Bind W/M before Family construction, and attach the complete fixed-z authority."""
from dataclasses import dataclass
from copy import deepcopy
from contextlib import contextmanager
import random
import numpy as np
import torch
from project.run_scripts.ordered_response_barrier_ode.adapters import FixedZArtifact
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256, tensor_sha256
from project.run_scripts.ordered_response_barrier_ode.preflight import canonical_hash
from .contracts import BindingBoundary


@dataclass(frozen=True)
class EntrySnapshot:
    weights: dict
    alpha_cache: torch.Tensor
    cache_c_new: bool
    W_sha256: str
    M_sha256: str

    @classmethod
    def capture(cls, parameters, module):
        weights = {n:p.detach().cpu().clone() for n,p in parameters.items()}
        cache = module.cache_c.detach().cpu().clone()
        return cls(weights, cache, bool(module.cache_c_new), tensor_set_sha256(weights), tensor_sha256(cache))

    def assert_sealed(self):
        if (tensor_set_sha256(self.weights) != self.W_sha256
                or tensor_sha256(self.alpha_cache) != self.M_sha256):
            raise BindingBoundary('ENTRY_SNAPSHOT_MUTATED')

    def restore(self, model, module):
        self.assert_sealed()
        pointers = {n:model.get_parameter(n).data_ptr() for n in self.weights}
        with torch.no_grad():
            for n,v in self.weights.items():
                p=model.get_parameter(n); p.copy_(v.to(p.device))
            if module.cache_c.shape != self.alpha_cache.shape:
                raise BindingBoundary('ENTRY_HISTORY_SHAPE')
            module.cache_c.copy_(self.alpha_cache.to(module.cache_c.device))
            module.cache_c_new=self.cache_c_new
        actual={n:model.get_parameter(n) for n in self.weights}
        if (tensor_set_sha256(actual)!=self.W_sha256 or tensor_sha256(module.cache_c)!=self.M_sha256
                or pointers!={n:p.data_ptr() for n,p in actual.items()}):
            raise BindingBoundary('ENTRY_RESTORE_IDENTITY')

    def make_family(self, factory, *, model, module, **kwargs):
        self.restore(model,module)  # must precede FamilyRuntime.__init__ snapshot
        family=factory(model=model,module=module,**kwargs)
        family.bind_existing_method_state()  # never cold prepare a warm fixture
        if family.w0_sha256!=self.W_sha256:
            raise BindingBoundary('FAMILY_STALE_ENTRY')
        return family


@dataclass(frozen=True)
class FixedTargetBundle:
    artifact: FixedZArtifact
    semantic_inventory: object
    contexts_sha256: str
    W_sha256: str
    M_sha256: str
    cache_c_new: bool

    @classmethod
    def capture(cls, family):
        z=family.fixed_z
        if z is None or family.semantic_inventory is None:
            raise BindingBoundary('COMPLETE_FIXED_Z_AUTHORITY_MISSING')
        copy=FixedZArtifact(z.values,z.identity_sha256,z.request_order_sha256,z.target_context_identity_sha256)
        return cls(copy,deepcopy(family.semantic_inventory),canonical_hash(family.contexts),
                   family.w0_sha256,tensor_sha256(family.module.cache_c),bool(family.module.cache_c_new))

    def attach(self, family):
        z=self.artifact
        if (family.fixed_z is not None or family.w0_sha256!=self.W_sha256
                or tensor_set_sha256(family.parameters)!=self.W_sha256
                or tensor_sha256(family.module.cache_c)!=self.M_sha256
                or bool(family.module.cache_c_new)!=self.cache_c_new
                or canonical_hash(family.contexts)!=self.contexts_sha256
                or family.request_order_sha256!=z.request_order_sha256
                or self.semantic_inventory.identity_sha256!=z.target_context_identity_sha256):
            raise BindingBoundary('FIXED_Z_ENTRY_CONTEXT_BINDING')
        # Recompute the cheap tokenizer semantic inventory; never recompute z/model.
        from project.run_scripts.ordered_response_barrier_ode.semantic import build_compute_z_semantic_inventory
        current=build_compute_z_semantic_inventory(tokenizer=family.tokenizer,requests=family.requests,
            context_templates=family.contexts,request_order_sha256=family.request_order_sha256)
        if current.identity_sha256!=z.target_context_identity_sha256:
            raise BindingBoundary('FIXED_Z_SEMANTIC_INVENTORY')
        family.fixed_z=FixedZArtifact(z.values,z.identity_sha256,z.request_order_sha256,z.target_context_identity_sha256)
        family.semantic_inventory=deepcopy(self.semantic_inventory)


class RNGSnapshot:
    def __init__(self):
        self.python=random.getstate(); self.numpy=np.random.get_state(); self.torch=torch.get_rng_state().clone()
        self.cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None

    def matches(self):
        n=np.random.get_state()
        return (self.python==random.getstate() and self.numpy[0]==n[0]
                and np.array_equal(self.numpy[1],n[1]) and self.numpy[2:]==n[2:]
                and torch.equal(self.torch,torch.get_rng_state())
                and (self.cuda is None or all(torch.equal(a,b) for a,b in
                     zip(self.cuda,torch.cuda.get_rng_state_all(),strict=True))))

    def restore(self):
        random.setstate(self.python);np.random.set_state(self.numpy);torch.set_rng_state(self.torch)
        if self.cuda is not None:torch.cuda.set_rng_state_all(self.cuda)


@contextmanager
def observation_callback_guard(family, overlay, *, tensors=()):
    """Callbacks are output sinks, not extensions of the controller.

    Check RNG, live W, method state, overlay, fixed target and supplied read-only
    tensors around the complete callback. Failure is never silently repaired
    into a valid endpoint; the enclosing trajectory restores its entry.
    """
    rng=RNGSnapshot(); state=overlay.state_version
    M=tensor_sha256(family.module.cache_c); flag=bool(family.module.cache_c_new)
    params={n:(p.data_ptr(),p._version) for n,p in family.parameters.items()}
    protected=tuple(tensors)+(family.fixed_z.values,)+tuple(
        t for delta in overlay.deltas for t in (delta.left,delta.right))
    before=tuple(tensor_sha256(t) for t in protected)
    delta_ids=tuple(id(d) for d in overlay.deltas)
    try:
        yield
        if (not rng.matches() or state!=overlay.state_version
                or params!={n:(p.data_ptr(),p._version) for n,p in family.parameters.items()}
                or M!=tensor_sha256(family.module.cache_c) or flag!=bool(family.module.cache_c_new)
                or delta_ids!=tuple(id(d) for d in overlay.deltas)
                or before!=tuple(tensor_sha256(t) for t in protected)):
            raise BindingBoundary('OBSERVATION_CALLBACK_INTERVENTION')
        # A prefix callback is inside suspend(): the overlay refreshes its
        # version baseline only on suspend exit. Check callback-local versions
        # above and entry bytes here; the outer observer checks overlay after.
        if tensor_set_sha256(family.parameters)!=family.w0_sha256:
            raise BindingBoundary('OBSERVATION_CALLBACK_WEIGHT_INTERVENTION')
    finally:
        rng.restore()


def observe_prefix(family, overlay, shadow, *, arm, on_endpoint=None):
    """Temporary evaluation only; refuse mutation rather than continue from it."""
    rng=RNGSnapshot(); state=overlay.state_version; deltas=overlay.deltas
    M=tensor_sha256(family.module.cache_c); flag=bool(family.module.cache_c_new)
    capture=family._capture_persistent_endpoint
    captured=tuple(id(getattr(family,k,None)) for k in (
        '_captured_endpoint_weights','_captured_endpoint_method_state','_captured_endpoint_sha256'))
    try:
        with overlay.suspend(authoritative=False):
            result=family.finalize(arm=arm,terminal_state_version=state,shadow_weights=shadow,
                                   deltas=deltas,derived_observation_only=True)
            if on_endpoint is not None:
                with observation_callback_guard(family,overlay,tensors=tuple(shadow.values())):
                    on_endpoint(result,shadow)
        if (result['history_append_count']!=0 or result['physical_write_count']!=0
                or family._capture_persistent_endpoint!=capture or overlay.state_version!=state
                or captured!=tuple(id(getattr(family,k,None)) for k in (
                    '_captured_endpoint_weights','_captured_endpoint_method_state','_captured_endpoint_sha256'))
                or tensor_sha256(family.module.cache_c)!=M or bool(family.module.cache_c_new)!=flag
                or not rng.matches()):
            raise BindingBoundary('PREFIX_OBSERVER_INTERVENTION')
        overlay.assert_w0_unchanged(full_bytes=True)
        return dict(result,controller_influence_count=0,persistent_endpoint_capture_count=0,
                    W_M_RNG_overlay_restore_pass=True)
    finally:
        # Also restore on exception; never hide a failed intervention check.
        rng.restore()
