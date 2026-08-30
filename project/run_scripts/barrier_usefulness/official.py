"""Pinned Official hook with exact native compute-z context capture."""

from __future__ import annotations

import copy
import importlib
from dataclasses import dataclass
from typing import Any

from project.run_scripts.fixed_z_nonuniqueness.official import (
    OfficialEndpoint,
    restore_originals,
    run_official_once as _run_generic_official_once,
    set_official_endpoint,
)

from .contracts import Method, ScientificBoundary


@dataclass
class CapturedOfficial:
    endpoint: OfficialEndpoint
    compute_z_context_templates: list[list[str]]
    normalized_request: dict[str, Any]


def _normalized_request(request: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(request)
    if not row["target_new"].startswith(" "):
        row["target_new"] = " " + row["target_new"]
    if "{}" not in row["prompt"]:
        if row["subject"] not in row["prompt"]:
            raise ScientificBoundary("subject absent from Official prompt")
        row["prompt"] = row["prompt"].replace(row["subject"], "{}")
    return row


def run_official_once(*, method: Method, model: Any, tokenizer: Any, request: dict[str, Any], hparams: Any) -> CapturedOfficial:
    module_name = "easyeditor.models.alphaedit.AlphaEdit_main" if method is Method.ALPHAEDIT else "easyeditor.models.memit.memit_main"
    module = importlib.import_module(module_name)
    original = module.compute_z
    captured: list[list[list[str]]] = []

    def context_hook(*args: Any, **kwargs: Any) -> Any:
        templates = kwargs.get("context_templates", args[5] if len(args) > 5 else None)
        if templates is None:
            raise ScientificBoundary("Official compute_z context templates absent")
        captured.append(copy.deepcopy(templates))
        return original(*args, **kwargs)

    module.compute_z = context_hook
    try:
        endpoint = _run_generic_official_once(
            method=method, model=model, tokenizer=tokenizer, request=request, hparams=hparams
        )
    finally:
        module.compute_z = original
    if len(captured) != 1:
        raise ScientificBoundary(f"native context capture count differs: {len(captured)}")
    return CapturedOfficial(
        endpoint=endpoint,
        compute_z_context_templates=captured[0],
        normalized_request=_normalized_request(request),
    )


__all__ = ["CapturedOfficial", "restore_originals", "run_official_once", "set_official_endpoint"]
