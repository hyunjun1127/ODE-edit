"""Pinned stock MEMIT target capture shared immutably by every experiment arm."""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from project.run_scripts.fixed_z_nonuniqueness.official import official_model_name_binding

from .contracts import ScientificBoundary
from .hashing import canonical_hash, tensor_sha256


@dataclass(slots=True)
class FrozenTarget:
    targets: torch.Tensor
    target_hashes: tuple[str, ...]
    target_root: str
    official_deltas: dict[str, torch.Tensor]
    originals: dict[str, torch.Tensor]
    official_endpoint_root: str
    direct_z_calls: int
    direct_z_calls_by_case: dict[str, int]
    tokenizer_calls: list[dict[str, Any]]


def _match_update(module: Any, raw: dict[str, Any], model: Any) -> dict[str, torch.Tensor]:
    answer: dict[str, torch.Tensor] = {}
    device = next(model.parameters()).device
    for name, value in raw.items():
        key, residual = value
        update = key.to(device) @ residual.to(device).T
        weight = dict(model.named_parameters())[name]
        answer[name] = module.upd_matrix_match_shape(update, weight.shape).detach().to("cpu", torch.float32)
    return answer


def _weight_root(model: Any, names: tuple[str, ...]) -> str:
    parameters = dict(model.named_parameters())
    return canonical_hash([
        {"name": name, "shape": list(parameters[name].shape), "sha256": tensor_sha256(parameters[name])}
        for name in names
    ])


def capture_official_memit_once(
    *, model: Any, tokenizer: Any, requests: list[dict[str, Any]], hparams: Any
) -> FrozenTarget:
    """Execute Official MEMIT once and capture one direct-z call per edit.

    The resulting target and Official endpoint are shared by all arms.  This
    routine is the only runtime authority allowed to call stock ``compute_z``.
    """

    module = importlib.import_module("easyeditor.models.memit.memit_main")
    original_compute_z = module.compute_z
    original_execute = module.execute_memit
    captured: dict[str, Any] = {"targets": [], "calls": {}, "raw": None}

    def compute_z_hook(*args: Any, **kwargs: Any) -> torch.Tensor:
        request = args[2] if len(args) > 2 else kwargs["request"]
        case_id = str(request["case_id"])
        captured["calls"][case_id] = captured["calls"].get(case_id, 0) + 1
        if captured["calls"][case_id] != 1:
            raise ScientificBoundary(f"stock direct-z recomputed for case {case_id}")
        value = original_compute_z(*args, **kwargs)
        captured["targets"].append((case_id, value.detach().clone()))
        return value

    def execute_hook(*args: Any, **kwargs: Any) -> Any:
        value = original_execute(*args, **kwargs)
        captured["raw"] = value
        return value

    module.compute_z = compute_z_hook
    module.execute_memit = execute_hook
    try:
        with official_model_name_binding(model, str(hparams.model_name)):
            edited, originals = module.apply_memit_to_model(
                model, tokenizer, requests, hparams,
                copy=False, return_orig_weights=True, reset_cache=True,
            )
        if edited is not model:
            raise ScientificBoundary("Official MEMIT unexpectedly replaced model")
        expected = [str(request["case_id"]) for request in requests]
        observed = [row[0] for row in captured["targets"]]
        if observed != expected or any(captured["calls"].get(case_id) != 1 for case_id in expected):
            raise ScientificBoundary("direct-z call order/count differs from sealed request order")
        if captured["raw"] is None:
            raise ScientificBoundary("Official MEMIT raw writer factors were not captured")
        names = tuple(originals)
        endpoint_root = _weight_root(model, names)
        targets = torch.stack([row[1] for row in captured["targets"]], dim=0).to(torch.float32)
        hashes = tuple(tensor_sha256(value) for value in targets)
        return FrozenTarget(
            targets=targets,
            target_hashes=hashes,
            target_root=canonical_hash(list(hashes)),
            official_deltas=_match_update(module, captured["raw"], model),
            originals=originals,
            official_endpoint_root=endpoint_root,
            direct_z_calls=len(captured["targets"]),
            direct_z_calls_by_case=dict(captured["calls"]),
            tokenizer_calls=list(getattr(tokenizer, "call_identities", [])),
        )
    finally:
        module.compute_z = original_compute_z
        module.execute_memit = original_execute


def restore_w0(model: Any, frozen: FrozenTarget) -> None:
    from easyeditor.util.device import copy_to_param

    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, original in frozen.originals.items():
            copy_to_param(parameters[name], original)
    if not all(
        torch.equal(parameters[name], original.to(parameters[name].device, parameters[name].dtype))
        for name, original in frozen.originals.items()
    ):
        raise ScientificBoundary("Official MEMIT W0 restoration failed")


def materialize_official(model: Any, frozen: FrozenTarget) -> None:
    from easyeditor.util.device import copy_to_param

    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, original in frozen.originals.items():
            copy_to_param(
                parameters[name],
                original + frozen.official_deltas[name].to(original.device, original.dtype),
            )
    if _weight_root(model, tuple(frozen.originals)) != frozen.official_endpoint_root:
        raise ScientificBoundary("Official MEMIT bypass parity failed")

