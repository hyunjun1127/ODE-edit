"""Invoke the unchanged BLUE dense native entrypoint with observation-only hooks."""
from contextlib import nullcontext
from copy import deepcopy
import time

import torch

from .fixtures import FixtureBoundary, tensor_sha


def run_native_batch(model, tok, module, hparams, history, projector, requests, spec,
                     *, observer=None, on_stage=None):
    """Return actual native endpoint before caller-controlled fixture rollback.

    observer(module,hparams,weights,history,requests,model) is an optional context
    manager factory. It must only wrap/pass through the native source; target
    cache substitution is not implicit here. Whole sequence retains original
    dense RHS solve order and original final history append. The caller owns
    source SHA verification, input order seals, and branch rollback on failure.
    """
    if (hparams.layers != [spec.layer] or hparams.L2 != 1 or not hparams.blue or
            hparams.rewrite_module_tmp != spec.rewrite_module_tmp):
        raise FixtureBoundary("NATIVE_SINGLETON_HPARAMS")
    if tok.padding_side != "right":
        raise FixtureBoundary("NATIVE_TOKENIZER_PADDING")
    parameters = dict(model.named_parameters())
    weight = parameters[spec.weight_name]
    if (any(v.dtype != torch.float32 for v in parameters.values()) or
            history.dtype != torch.float32 or projector.dtype != torch.float32 or
            history.shape != (1, weight.shape[1], weight.shape[1]) or
            projector.shape != history.shape):
        raise FixtureBoundary("NATIVE_SINGLETON_TENSOR_SCHEMA")
    if not all(torch.isfinite(v).all() for v in (weight, history, projector)):
        raise FixtureBoundary("NATIVE_NONFINITE_ENTRY")
    # Full nonselected byte evidence; snapshots/rollback live in FixtureTransaction.
    guards = {k: (v, v.data_ptr(), v._version, tensor_sha(v))
              for k, v in parameters.items() if k != spec.weight_name}
    selected_pointer, history_pointer = weight.data_ptr(), history.data_ptr()
    before = weight.detach().cpu().clone()
    entry_history_sha = tensor_sha(history)
    p_sha = tensor_sha(projector)
    rows = deepcopy(requests)
    request_copy = deepcopy(rows)
    if on_stage:
        on_stage("NATIVE_ENTRY_BOUND", dict(layer=spec.layer, requests=len(rows),
                                         history_sha256=entry_history_sha))
    started = time.monotonic()
    context = observer(module, hparams, {spec.weight_name: weight}, history, rows, model) if observer else nullcontext({})
    with context as counters:
        if rows:
            returned, cache = module.apply_AlphaEdit_to_model(
                model, tok, rows, hparams, cache_template=None, cache_c=history, P=projector)
            if returned is not model or cache is not history:
                raise FixtureBoundary("NATIVE_RETURN_IDENTITY")
    after_parameters = dict(model.named_parameters())
    drift = [k for k, (obj, ptr, version, sha) in guards.items()
             if k not in after_parameters or after_parameters[k] is not obj or
             obj.data_ptr() != ptr or obj._version != version or tensor_sha(obj) != sha]
    if drift or set(after_parameters) != set(parameters):
        raise FixtureBoundary("NONSELECTED_PARAMETER_MUTATION", names=drift)
    if (after_parameters[spec.weight_name] is not weight or weight.data_ptr() != selected_pointer or
            history.data_ptr() != history_pointer or tensor_sha(projector) != p_sha):
        raise FixtureBoundary("NATIVE_POINTER_OR_PROJECTOR_MUTATION")
    if rows != request_copy:
        raise FixtureBoundary("NATIVE_INPUT_MUTATION")
    if not torch.isfinite(weight).all() or not torch.isfinite(history).all():
        raise FixtureBoundary("NATIVE_NONFINITE_ENDPOINT")
    endpoint = weight.detach().cpu().clone()
    delta = endpoint.double() - before.double()
    receipt = dict(status="NATIVE_BATCH_EXECUTED" if rows else "EMPTY_BATCH_NOOP",
                   layer=spec.layer, request_count=len(rows),
                   endpoint_sha256=tensor_sha(endpoint), history_sha256=tensor_sha(history),
                   entry_history_sha256=entry_history_sha, projector_sha256=p_sha,
                   actual_delta_norm=float(delta.norm()), actual_delta_squared_norm=float(delta.square().sum()),
                   selected_pointer_exact=True, nonselected_bytes_versions_exact=True,
                   dense_native_entrypoint_called=int(bool(rows)),
                   history_append_validation="OBSERVER_REQUIRED" if rows else "NOOP",
                   seconds=time.monotonic() - started,
                   source_fidelity="CALLER_SOURCE_SHA_AND_OBSERVER_BINDING_REQUIRED")
    if on_stage:
        on_stage("NATIVE_ENDPOINT_CAPTURED", receipt)
    return dict(weight=endpoint, history=history.detach().cpu().clone(), receipt=receipt,
                observer=counters, contexts=deepcopy(module.CONTEXT_TEMPLATES_CACHE))
