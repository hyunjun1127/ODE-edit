"""Stock AlphaEdit fixed-z capture for a shared atomic cohort.

MEMIT continues to use :mod:`fzcb_edit.target`; this module adds only the
equation-identical AlphaEdit capture needed by the production pilot.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch

from project.run_scripts.fixed_z_nonuniqueness.official import official_model_name_binding

from .contracts import ScientificBoundary
from .hashing import canonical_hash, tensor_sha256
from .target import FrozenTarget


def _weight_root(model: Any, names: tuple[str, ...]) -> str:
    parameters = dict(model.named_parameters())
    return canonical_hash([
        {"name": name, "shape": list(parameters[name].shape), "sha256": tensor_sha256(parameters[name])}
        for name in names
    ])


def capture_official_alphaedit_once(
    *, model: Any, tokenizer: Any, requests: list[dict[str, Any]], hparams: Any
) -> FrozenTarget:
    """Execute stock AlphaEdit once and freeze one target per request.

    The caller owns the create-once cold-cache snapshot and restores it after
    observing the Official endpoint.  No target cache file is used.
    """

    module = importlib.import_module("easyeditor.models.alphaedit.AlphaEdit_main")
    original_compute_z = module.compute_z
    original_execute = module.execute_AlphaEdit
    captured: dict[str, Any] = {"targets": [], "calls": {}, "raw": None}

    def compute_z_hook(*args: Any, **kwargs: Any) -> torch.Tensor:
        request = args[2] if len(args) > 2 else kwargs["request"]
        case_id = str(request["case_id"])
        captured["calls"][case_id] = captured["calls"].get(case_id, 0) + 1
        if captured["calls"][case_id] != 1:
            raise ScientificBoundary(f"stock AlphaEdit direct-z recomputed for case {case_id}")
        value = original_compute_z(*args, **kwargs)
        captured["targets"].append((case_id, value.detach().clone()))
        return value

    def execute_hook(*args: Any, **kwargs: Any) -> Any:
        value = original_execute(*args, **kwargs)
        captured["raw"] = value
        return value

    module.compute_z = compute_z_hook
    module.execute_AlphaEdit = execute_hook
    try:
        with official_model_name_binding(model, str(hparams.model_name)):
            edited, originals = module.apply_AlphaEdit_to_model(
                model,
                tokenizer,
                requests,
                hparams,
                copy=False,
                return_orig_weights=True,
                reset_cache=False,
                cache_template=None,
            )
        if edited is not model:
            raise ScientificBoundary("Official AlphaEdit unexpectedly replaced model")
        expected = [str(request["case_id"]) for request in requests]
        observed = [row[0] for row in captured["targets"]]
        if observed != expected or any(captured["calls"].get(case_id) != 1 for case_id in expected):
            raise ScientificBoundary("AlphaEdit direct-z call order/count differs from sealed order")
        if captured["raw"] is None:
            raise ScientificBoundary("Official AlphaEdit raw writer deltas were not captured")
        names = tuple(originals)
        parameters = dict(model.named_parameters())
        deltas: dict[str, torch.Tensor] = {}
        for name, update in captured["raw"].items():
            matched = module.upd_matrix_match_shape(update.to(parameters[name]), parameters[name].shape)
            deltas[name] = matched.detach().to("cpu", torch.float32)
        endpoint_root = _weight_root(model, names)
        targets = torch.stack([row[1] for row in captured["targets"]], dim=0).to(torch.float32)
        hashes = tuple(tensor_sha256(value) for value in targets)
        return FrozenTarget(
            targets=targets,
            target_hashes=hashes,
            target_root=canonical_hash(list(hashes)),
            official_deltas=deltas,
            originals=originals,
            official_endpoint_root=endpoint_root,
            direct_z_calls=len(captured["targets"]),
            direct_z_calls_by_case=dict(captured["calls"]),
            tokenizer_calls=list(getattr(tokenizer, "call_identities", [])),
        )
    finally:
        module.compute_z = original_compute_z
        module.execute_AlphaEdit = original_execute
