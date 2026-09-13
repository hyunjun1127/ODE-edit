"""E0 singleton binding and reversible state fixtures; no model loading.

Warm selected checkpoints do not contain full W0: callers must separately bind
the pinned base-model bytes. CUDA RNG is restored exactly or rejected, never
reseeded. Parameter versions cannot be decremented by native copy_; their
changes are reported, not misrepresented as version-exact restoration.
"""
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import random

import numpy as np
import torch


class FixtureBoundary(RuntimeError):
    def __init__(self, code, **receipt):
        self.code, self.receipt = code, dict(code=code, **receipt)
        super().__init__(code)


def tensor_sha(value):
    """Historical integrity.tensor_sha byte convention, including dtype/shape."""
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((str(value.dtype), list(value.shape))).encode())
    raw = value.reshape(-1).view(torch.uint8).numpy()
    for start in range(0, raw.size, 8 << 20):
        h.update(memoryview(raw[start:start + (8 << 20)]))
    return h.hexdigest()


@dataclass(frozen=True)
class SingletonSpec:
    layer: int
    rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj"

    def __post_init__(self):
        if self.layer not in range(4, 9):
            raise FixtureBoundary("UNSUPPORTED_SINGLETON_LAYER", layer=self.layer)

    @property
    def weight_name(self):
        return self.rewrite_module_tmp.format(self.layer) + ".weight"

    @property
    def source_index(self):
        return self.layer - 4


def select_projector(full_stack, spec):
    if (full_stack.ndim != 3 or full_stack.shape[0] != 5 or
            full_stack.shape[1] != full_stack.shape[2] or
            full_stack.dtype != torch.float32 or not torch.isfinite(full_stack).all()):
        raise FixtureBoundary("PROJECTOR_STACK_SCHEMA")
    selected = full_stack[spec.source_index:spec.source_index + 1].clone()
    return selected, dict(physical_layer=spec.layer, source_stack_index=spec.source_index,
                         local_index=0, shape=list(selected.shape),
                         sha256=tensor_sha(selected),
                         slice_sha256=tensor_sha(selected[0]))


def capture_rng():
    n = np.random.get_state()
    return dict(python=random.getstate(),
                numpy=[n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])],
                torch=torch.get_rng_state().tolist(),
                cuda=[v.tolist() for v in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else [])


def _tuple_tree(value):
    return tuple(_tuple_tree(x) for x in value) if isinstance(value, (list, tuple)) else value


def restore_rng(saved):
    if not isinstance(saved, dict) or set(saved) != {"python", "numpy", "torch", "cuda"}:
        raise FixtureBoundary("RNG_STATE_UNVERIFIED")
    # Validate CUDA availability/count before changing any RNG stream.
    live_cuda_count = torch.cuda.device_count() if torch.cuda.is_initialized() else 0
    if len(saved["cuda"]) != live_cuda_count:
        raise FixtureBoundary("CUDA_RNG_DEVICE_MAPPING_UNVERIFIED",
                              saved_devices=len(saved["cuda"]))
    prior = capture_rng()
    try:
        random.setstate(_tuple_tree(saved["python"]))
        n = saved["numpy"]
        np.random.set_state((n[0], np.asarray(n[1], dtype="uint32"), n[2], n[3], n[4]))
        torch.set_rng_state(torch.tensor(saved["torch"], dtype=torch.uint8))
        if saved["cuda"]:
            torch.cuda.set_rng_state_all([torch.tensor(v, dtype=torch.uint8) for v in saved["cuda"]])
    except Exception as exc:
        random.setstate(prior["python"])
        n = prior["numpy"]
        np.random.set_state((n[0], np.asarray(n[1], dtype="uint32"), n[2], n[3], n[4]))
        torch.set_rng_state(torch.tensor(prior["torch"], dtype=torch.uint8))
        if prior["cuda"]:
            torch.cuda.set_rng_state_all([torch.tensor(v, dtype=torch.uint8) for v in prior["cuda"]])
        raise FixtureBoundary("RNG_RESTORE_FAILED", original_exception=repr(exc)) from exc


def restore_checkpoint(model, module, history, checkpoint, spec, *,
                       expected_model_revision, expected_seen_ids):
    """Install a warm fixture. Enclose in FixtureTransaction for failure rollback."""
    required = {"weights", "cache_c", "metadata"}
    if not required.issubset(checkpoint):
        raise FixtureBoundary("CHECKPOINT_SCHEMA")
    metadata = checkpoint["metadata"]
    for key in ("batch", "seen_ids", "base_model_revision", "contexts", "rng", "state", "covariance"):
        if key not in metadata:
            raise FixtureBoundary("CHECKPOINT_STATE_UNVERIFIED", missing=key)
    if (metadata["base_model_revision"] != expected_model_revision or
            metadata["seen_ids"] != list(expected_seen_ids) or
            metadata["batch"] * 100 != len(expected_seen_ids)):
        raise FixtureBoundary("CHECKPOINT_ENTRY_IDENTITY")
    if metadata["contexts"] is None:
        raise FixtureBoundary("WARM_CONTEXTS_UNVERIFIED")
    if metadata["covariance"]:
        # Singleton BLUE does not consume COV_CACHE, but silently dropping a
        # nonempty original module cache would not be an exact state restore.
        raise FixtureBoundary("CHECKPOINT_COVARIANCE_RESTORE_UNVERIFIED")
    if set(checkpoint["weights"]) != {spec.weight_name}:
        raise FixtureBoundary("CHECKPOINT_PHYSICAL_LAYER_MAPPING")
    weights = dict(model.named_parameters())
    weight = weights[spec.weight_name]
    saved = checkpoint["weights"][spec.weight_name]
    cache = checkpoint["cache_c"]
    if (saved.shape != weight.shape or saved.dtype != weight.dtype or
            cache.shape != history.shape or cache.dtype != history.dtype or
            history.ndim != 3 or history.shape != (1, weight.shape[1], weight.shape[1]) or
            not torch.isfinite(saved).all() or not torch.isfinite(cache).all()):
        raise FixtureBoundary("CHECKPOINT_TENSOR_SCHEMA")
    state = metadata["state"]
    if (state["weights"][spec.weight_name] != tensor_sha(saved) or
            state["cache"] != tensor_sha(cache)):
        raise FixtureBoundary("CHECKPOINT_TENSOR_HASH")
    restore_rng(metadata["rng"])
    with torch.no_grad():
        weight.copy_(saved.to(weight.device))
        history.copy_(cache.to(history.device))
    module.CONTEXT_TEMPLATES_CACHE = deepcopy(metadata["contexts"])
    module.COV_CACHE = {}
    return dict(status="WARM_SELECTED_STATE_RESTORED", next_batch_index=metadata["batch"] + 1,
                physical_layer=spec.layer, weight_sha256=tensor_sha(weight),
                history_sha256=tensor_sha(history), rng_restored_exact=True,
                full_W0_authority="CALLER_PINNED_MODEL_REQUIRED", C0_restored=False)


class FixtureTransaction(AbstractContextManager):
    """Full live parameter/buffer CPU backup, globals/RNG/hooks exact rollback.

    This intentionally costs one host model copy. It does not make GPU clones.
    Native selected writes increment versions; rollback reports this explicitly.
    Existing hook registries and parameter objects are reinstated on all exits.
    """
    def __init__(self, model, module, history):
        self.model, self.module, self.history = model, module, history
        self.receipt = {}

    def __enter__(self):
        self.rng = capture_rng()
        self.registries, self.tensors = [], {}
        self.tensor_views, self.requires_grad = {}, {}
        for mod in self.model.modules():
            for name in ("_parameters", "_buffers", "_forward_hooks", "_forward_pre_hooks",
                         "_backward_hooks", "_forward_hooks_with_kwargs",
                         "_forward_pre_hooks_with_kwargs", "_forward_hooks_always_called"):
                if not hasattr(mod, name):
                    continue
                registry = getattr(mod, name)
                self.registries.append((mod, name, registry, registry.copy()))
                if name in ("_parameters", "_buffers"):
                    for tensor in registry.values():
                        if tensor is not None and id(tensor) not in self.tensors:
                            self.tensors[id(tensor)] = (tensor, tensor.detach().cpu().clone(),
                                                        tensor.data_ptr(), tensor._version)
                            self.tensor_views[id(tensor)] = tensor.detach()
                            self.requires_grad[id(tensor)] = tensor.requires_grad
        self.cache = (self.history.detach().cpu().clone(), self.history.data_ptr())
        self.context_object = getattr(self.module, "CONTEXT_TEMPLATES_CACHE", None)
        self.context_value = deepcopy(self.context_object)
        self.cov_object = getattr(self.module, "COV_CACHE", {})
        self.cov_values = {k: (v, v.detach().clone()) for k, v in self.cov_object.items()}
        self.training = [(mod, mod.training) for mod in self.model.modules()]
        self.backend_flags = (torch.backends.cuda.matmul.allow_tf32,
                              torch.backends.cudnn.allow_tf32,
                              torch.backends.cudnn.benchmark,
                              torch.backends.cudnn.deterministic)
        self.receipt["host_backup_bytes"] = sum(v.numel() * v.element_size() for _, v, _, _ in self.tensors.values())
        return self

    def __exit__(self, exc_type, exc, tb):
        with torch.no_grad():
            for mod, name, original, contents in self.registries:
                original.clear(); original.update(contents); setattr(mod, name, original)
            for tensor, saved, _, _ in self.tensors.values():
                original_view = self.tensor_views[id(tensor)]
                if (tensor.data_ptr() != original_view.data_ptr() or tensor.shape != original_view.shape or
                        tensor.dtype != original_view.dtype or tensor.device != original_view.device):
                    tensor.data = original_view
                if tensor_sha(tensor) != tensor_sha(saved):
                    tensor.copy_(saved.to(tensor.device))
                tensor.requires_grad_(self.requires_grad[id(tensor)])
            self.history.copy_(self.cache[0].to(self.history.device))
            self.cov_object.clear()
            for key, (original, saved) in self.cov_values.items():
                original.copy_(saved); self.cov_object[key] = original
        self.module.COV_CACHE = self.cov_object
        if isinstance(self.context_object, list):
            self.context_object[:] = self.context_value
        self.module.CONTEXT_TEMPLATES_CACHE = self.context_object
        for mod, training in self.training:
            mod.training = training
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = self.backend_flags
        restore_rng(self.rng)
        exact = all(t.data_ptr() == ptr and tensor_sha(t) == tensor_sha(saved)
                    for t, saved, ptr, _ in self.tensors.values())
        exact &= self.history.data_ptr() == self.cache[1] and tensor_sha(self.history) == tensor_sha(self.cache[0])
        self.receipt.update(status="RESTORED" if exact else "RESTORE_FAILED",
                            pointer_bytes_exact=bool(exact),
                            version_restore="NOT_CLAIMED_NATIVE_COPY_INCREMENTS",
                            changed_version_count=sum(t._version != version for t, _, _, version in self.tensors.values()),
                            original_exception=None if exc is None else repr(exc), rng_restored_exact=True)
        if not exact:
            raise FixtureBoundary("FIXTURE_ROLLBACK_FAILED", **self.receipt) from exc
        return False
