"""Hooks around pinned stock EasyEdit Official entrypoints."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch

from .contracts import Method, ScientificBoundary


@dataclass
class OfficialEndpoint:
    deltas: dict[str, torch.Tensor]
    originals: dict[str, torch.Tensor]
    z: torch.Tensor
    edit_key: torch.Tensor
    direct_z_compute_count: int
    direct_z_recompute_count: int
    last_weight_name: str
    projector: torch.Tensor | None
    tokenizer_calls: list[dict[str, Any]]


def _weight_deltas(method: Method, module: Any, raw: dict[str, Any], model: Any) -> dict[str, torch.Tensor]:
    answer = {}
    for name, value in raw.items():
        if method is Method.ALPHAEDIT:
            update = value
        else:
            key, val = value
            update = key.to(next(model.parameters()).device) @ val.to(next(model.parameters()).device).T
        weight = dict(model.named_parameters())[name]
        answer[name] = module.upd_matrix_match_shape(update, weight.shape).detach().to("cpu", torch.float32)
    return answer


def run_official_once(
    *,
    method: Method,
    model: Any,
    tokenizer: Any,
    request: dict[str, Any],
    hparams: Any,
) -> OfficialEndpoint:
    if method is Method.ALPHAEDIT:
        module = importlib.import_module("easyeditor.models.alphaedit.AlphaEdit_main")
        apply_name, execute_name = "apply_AlphaEdit_to_model", "execute_AlphaEdit"
        module.cache_c_new = False
    else:
        module = importlib.import_module("easyeditor.models.memit.memit_main")
        apply_name, execute_name = "apply_memit_to_model", "execute_memit"
    original_compute_z = module.compute_z
    original_compute_ks = module.compute_ks
    original_execute = getattr(module, execute_name)
    captured: dict[str, Any] = {"z_calls": 0, "last_key": None, "raw_deltas": None}
    last_layer = int(hparams.layers[-1])

    def compute_z_hook(*args: Any, **kwargs: Any) -> torch.Tensor:
        captured["z_calls"] += 1
        if captured["z_calls"] != 1:
            raise ScientificBoundary("direct-z recomputation detected")
        value = original_compute_z(*args, **kwargs)
        captured["z"] = value.detach().clone()
        return value

    def compute_ks_hook(*args: Any, **kwargs: Any) -> torch.Tensor:
        value = original_compute_ks(*args, **kwargs)
        layer = kwargs.get("layer", args[4] if len(args) > 4 else None)
        if int(layer) == last_layer:
            captured["last_key"] = value.detach().clone()
        return value

    def execute_hook(*args: Any, **kwargs: Any) -> Any:
        value = original_execute(*args, **kwargs)
        captured["raw_deltas"] = value
        return value

    module.compute_z = compute_z_hook
    module.compute_ks = compute_ks_hook
    setattr(module, execute_name, execute_hook)
    try:
        apply_fn = getattr(module, apply_name)
        edited, originals = apply_fn(
            model,
            tokenizer,
            [request],
            hparams,
            copy=False,
            return_orig_weights=True,
            reset_cache=True,
        )
        if edited is not model:
            raise ScientificBoundary("Official entrypoint replaced model unexpectedly")
    finally:
        module.compute_z = original_compute_z
        module.compute_ks = original_compute_ks
        setattr(module, execute_name, original_execute)
    if captured["z_calls"] != 1 or captured["last_key"] is None or captured["raw_deltas"] is None:
        raise ScientificBoundary("Official capture incomplete")
    deltas = _weight_deltas(method, module, captured["raw_deltas"], model)
    last_name = f"{hparams.rewrite_module_tmp.format(last_layer)}.weight"
    projector = None
    if method is Method.ALPHAEDIT:
        projector = module.P[-1].detach().to("cpu", torch.float32)
    calls = list(getattr(tokenizer, "call_identities", []))
    return OfficialEndpoint(
        deltas=deltas,
        originals=originals,
        z=captured["z"],
        edit_key=captured["last_key"].reshape(-1).float(),
        direct_z_compute_count=1,
        direct_z_recompute_count=0,
        last_weight_name=last_name,
        projector=projector,
        tokenizer_calls=calls,
    )


def restore_originals(model: Any, originals: dict[str, torch.Tensor]) -> bool:
    from easyeditor.util.device import copy_to_param

    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, original in originals.items():
            copy_to_param(params[name], original)
    return all(torch.equal(params[name], original.to(params[name].device, params[name].dtype)) for name, original in originals.items())


def set_official_endpoint(model: Any, endpoint: OfficialEndpoint) -> None:
    from easyeditor.util.device import copy_to_param

    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, original in endpoint.originals.items():
            delta = endpoint.deltas[name].to(original.device, original.dtype)
            copy_to_param(params[name], original + delta)
