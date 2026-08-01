"""Pinned AlphaEdit solver configuration for ODE-side mechanism probes.

This module never imports AlphaEdit or calls its mutation entrypoint.  It
verifies the exact EasyEdit reference sources and YAML, validates the small
solver contract used by the isolated-first-edit adapter, and proves that the
already-verified MEMIT bridge's key-extractor body is identical to the pinned
AlphaEdit key extractor after removing type-only annotations/imports.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import (
    ContractError,
    ProvenanceManifest,
    canonical_json,
    preflight_pinned_files,
    sha256_bytes,
)
from .manifests import (
    ALPHAEDIT_HPARAM_BY_MODEL,
    ALPHAEDIT_REFERENCE_SOURCE_PATHS,
    FIXED_FILE_IDENTITIES,
    fixed_model_spec,
)


_MEMIT_COMPUTE_KS = "easyeditor/models/memit/compute_ks.py"
_ALPHA_COMPUTE_KS = "easyeditor/models/alphaedit/compute_ks.py"
_EXPECTED_KEYS = frozenset(
    {
        "alg_name",
        "model_name",
        "stats_dir",
        "P_loc",
        "device",
        "layers",
        "clamp_norm_factor",
        "layer_selection",
        "fact_token",
        "v_num_grad_steps",
        "v_lr",
        "v_loss_layer",
        "v_weight_decay",
        "kl_factor",
        "mom2_adjustment",
        "mom2_update_weight",
        "rewrite_module_tmp",
        "layer_module_tmp",
        "mlp_module_tmp",
        "attn_module_tmp",
        "ln_f_module",
        "lm_head_module",
        "mom2_dataset",
        "mom2_n_samples",
        "mom2_dtype",
        "nullspace_threshold",
        "L2",
    }
)


@dataclass(frozen=True, slots=True)
class AlphaEditSolverConfig:
    """The exact AlphaEdit fields that influence this ODE-side solve."""

    model_alias: str
    model_name: str
    layers: tuple[int, ...]
    fact_token: str
    rewrite_module_tmp: str
    layer_module_tmp: str
    l2: float
    yaml_relative_path: str
    yaml_sha256: str
    reference_manifest_id: str
    key_extractor_equivalent: bool

    @property
    def config_id(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict()).encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_alias": self.model_alias,
            "model_name": self.model_name,
            "layers": list(self.layers),
            "fact_token": self.fact_token,
            "rewrite_module_tmp": self.rewrite_module_tmp,
            "layer_module_tmp": self.layer_module_tmp,
            "l2": self.l2,
            "yaml_relative_path": self.yaml_relative_path,
            "yaml_sha256": self.yaml_sha256,
            "reference_manifest_id": self.reference_manifest_id,
            "key_extractor_equivalent": self.key_extractor_equivalent,
            "cache_policy": "isolated-zero-per-case-no-accumulation",
            "projector_policy": "pinned-mmap-read-only-no-recompute",
        }


def _normalized_function_body(path: Path, function_name: str) -> str:
    try:
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ContractError("pinned AlphaEdit reference source is unreadable") from exc
    matches = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise ContractError(f"reference source has no unique {function_name} function")
    function = matches[0]
    function.decorator_list = []
    function.returns = None
    for argument in (
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ):
        argument.annotation = None
    if function.args.vararg is not None:
        function.args.vararg.annotation = None
    if function.args.kwarg is not None:
        function.args.kwarg.annotation = None
    return ast.dump(function, include_attributes=False)


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ContractError("pinned AlphaEdit YAML is unreadable") from exc
    if type(loaded) is not dict or set(loaded) != _EXPECTED_KEYS:
        raise ContractError("AlphaEdit YAML key set differs from the pinned contract")
    if any(not isinstance(key, str) for key in loaded):
        raise ContractError("AlphaEdit YAML contains a non-string key")
    return loaded


def load_alphaedit_solver_config(
    easyedit_root: str | Path,
    model_alias: str,
    *,
    memit_hparams: Any | None = None,
) -> tuple[AlphaEditSolverConfig, ProvenanceManifest]:
    """Preflight and load one fixed AlphaEdit reference without importing it."""

    root = Path(easyedit_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    try:
        yaml_relative = ALPHAEDIT_HPARAM_BY_MODEL[model_alias]
    except KeyError as exc:
        raise ContractError("model has no pinned AlphaEdit solver YAML") from exc
    relative_paths = (
        *ALPHAEDIT_REFERENCE_SOURCE_PATHS,
        yaml_relative,
        _MEMIT_COMPUTE_KS,
    )
    expected = {}
    for relative in relative_paths:
        identity = FIXED_FILE_IDENTITIES.get(relative)
        if identity is None:
            raise ContractError(f"AlphaEdit reference identity is missing: {relative}")
        expected[relative] = identity
    manifest = preflight_pinned_files(
        expected,
        label=f"AlphaEdit read-only solver reference for {model_alias}",
        base_dir=root,
    )
    mapping = _load_yaml_mapping(root / yaml_relative)
    try:
        layers = tuple(int(layer) for layer in mapping["layers"])
        l2 = float(mapping["L2"])
    except (TypeError, ValueError) as exc:
        raise ContractError("AlphaEdit layer/L2 configuration is invalid") from exc
    if (
        mapping["alg_name"] != "AlphaEdit"
        or mapping["model_name"] != spec.repository_id
        or layers != tuple(spec.layers)
        or mapping["layer_selection"] != "all"
        or mapping["fact_token"] != "subject_last"
        or mapping["rewrite_module_tmp"] != "model.layers.{}.mlp.down_proj"
        or mapping["layer_module_tmp"] != "model.layers.{}"
        or not math.isfinite(l2)
        or l2 <= 0.0
    ):
        raise ContractError("AlphaEdit solver YAML differs from the fixed model contract")
    if memit_hparams is not None:
        comparable = {
            "layers": tuple(int(layer) for layer in memit_hparams.layers),
            "fact_token": str(memit_hparams.fact_token),
            "rewrite_module_tmp": str(memit_hparams.rewrite_module_tmp),
            "layer_module_tmp": str(memit_hparams.layer_module_tmp),
        }
        expected_common = {
            "layers": layers,
            "fact_token": str(mapping["fact_token"]),
            "rewrite_module_tmp": str(mapping["rewrite_module_tmp"]),
            "layer_module_tmp": str(mapping["layer_module_tmp"]),
        }
        if comparable != expected_common:
            raise ContractError(
                "MEMIT target anchor and AlphaEdit solver disagree on shared semantics"
            )
    equivalent = _normalized_function_body(
        root / _MEMIT_COMPUTE_KS,
        "compute_ks",
    ) == _normalized_function_body(root / _ALPHA_COMPUTE_KS, "compute_ks")
    if not equivalent:
        raise ContractError(
            "verified MEMIT bridge key extractor differs from AlphaEdit compute_ks"
        )
    config = AlphaEditSolverConfig(
        model_alias=model_alias,
        model_name=str(mapping["model_name"]),
        layers=layers,
        fact_token=str(mapping["fact_token"]),
        rewrite_module_tmp=str(mapping["rewrite_module_tmp"]),
        layer_module_tmp=str(mapping["layer_module_tmp"]),
        l2=l2,
        yaml_relative_path=yaml_relative,
        yaml_sha256=expected[yaml_relative].sha256,
        reference_manifest_id=manifest.manifest_id,
        key_extractor_equivalent=True,
    )
    manifest.assert_current()
    return config, manifest
