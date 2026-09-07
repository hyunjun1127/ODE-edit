"""Production one-GPU binding for the Server1 ORBODE array cells.

The science loop lives in :mod:`integrator`.  This module binds that loop to
the pinned, read-only EasyEdit checkout and the sealed CounterFact stream.  It
does not import EasyEdit until :func:`run_cell`, after the launcher has passed
the source/artifact gate.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import random
import re
import stat
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, cast

import torch

from .adapters import CallbackMethodAdapter, FixedZArtifact, LayerBuild
from .artifacts import reduce_round_payload, validate_round_publication
from .contracts import (
    ArmId,
    NumericalMethodBoundary,
    ORBODERuntimeLock,
    ScientificBoundary,
    TechnicalBoundary,
    assert_full_fp32,
    canonical_arm_configs,
)
from .fp32_overlay import GroupedFP32Overlay, OverlayDelta, tensor_set_sha256, tensor_sha256
from .integrator import ArmExecutionResult, OrderedResponseIntegrator
from .preflight import (
    CELL_SPECS,
    MODEL_BINDINGS,
    ORDER_ROOT,
    STREAM_ROOT,
    STREAM_SEAL_RELATIVE,
    canonical_hash,
    wave_rounds,
)
from .semantic import (
    SemanticInventory,
    build_compute_z_semantic_inventory,
    normalize_official_requests,
    observe_semantic_predicate,
)
from .terminal_jvp import TerminalResponseObserver, seal_eager_attention


PRIMARY_ARM_ORDER = ("O", "QCL", "NQFIX", "ORBFH", "JAC")
ALL_ARM_ORDER = (*PRIMARY_ARM_ORDER, "ORBHit")
RESULT_SCHEMA = "orbode.server1.cell-result.v1"
TERMINAL_SCHEMA = "orbode.round0.cell-terminal.v1"


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(torch.device("cuda:0"))


def _writer_device_residual_division_replay(
    full_left: torch.Tensor,
    *,
    divisor: int,
    device: torch.device,
) -> torch.Tensor:
    """Replay production R/n arithmetic before detached CPU factor storage."""

    if (
        full_left.ndim != 2
        or full_left.dtype is not torch.float32
        or full_left.requires_grad
        or isinstance(divisor, bool)
        or not isinstance(divisor, int)
        or divisor <= 0
        or not bool(torch.isfinite(full_left).all())
    ):
        raise TechnicalBoundary("residual scaling replay contract differs")
    return (
        full_left.to(device=device, dtype=torch.float32)
        .div(float(divisor))
        .detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
    )


def _model_forward_count(model: torch.nn.Module) -> int:
    value = getattr(model, "_orbode_model_forward_invocation_count", None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TechnicalBoundary("model forward invocation counter is unavailable")
    return value


def _classify_failure_status(exc: BaseException) -> tuple[str, bool]:
    text = str(exc).lower()
    words = set(re.findall(r"(?<![a-z])[+\-]?(?:nan|inf(?:inity)?)(?![a-z])", text))
    nonfinite = "non-finite" in text or "nonfinite" in text or bool(words)
    if nonfinite:
        return "TECHNICAL_INVALID_NONFINITE_ORACLE_EXCLUSION_INCOMPLETE", True
    if isinstance(exc, NumericalMethodBoundary):
        return "NUMERICAL_METHOD_BOUNDARY", False
    if isinstance(exc, ScientificBoundary):
        return "SCIENTIFIC_BOUNDARY", False
    return "TECHNICAL_INVALID", False


def _install_model_forward_counter(model: torch.nn.Module) -> Any:
    if hasattr(model, "_orbode_model_forward_invocation_count"):
        raise TechnicalBoundary("model forward invocation counter was already installed")
    setattr(model, "_orbode_model_forward_invocation_count", 0)

    def count(module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
        current = getattr(module, "_orbode_model_forward_invocation_count", None)
        if isinstance(current, bool) or not isinstance(current, int) or current < 0:
            raise TechnicalBoundary("model forward invocation counter drifted")
        setattr(module, "_orbode_model_forward_invocation_count", current + 1)

    return model.register_forward_pre_hook(count)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _create_once_json(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise TechnicalBoundary(f"create-once JSON exists: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise TechnicalBoundary(f"JSON parent is not a regular directory: {path.parent}")
    raw = (_canonical_json(dict(value)) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or stat.S_IMODE(observed.st_mode) != 0o600:
        raise TechnicalBoundary("create-once JSON type/mode differs")
    return hashlib.sha256(raw).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _bootstrap_easyedit(root: Path) -> dict[str, Any]:
    expected = root.resolve(strict=True)
    conflicts = sorted(name for name in sys.modules if name == "easyeditor" or name.startswith("easyeditor."))
    if conflicts:
        raise TechnicalBoundary(f"EasyEdit was imported before source binding: {conflicts[:4]}")
    sys.path.insert(0, str(expected))
    importlib.invalidate_caches()
    module = importlib.import_module("easyeditor")
    observed = Path(module.__file__).resolve(strict=True)
    try:
        observed.relative_to(expected)
    except ValueError as exc:
        raise TechnicalBoundary(f"EasyEdit import escaped pinned source: {observed}") from exc
    return {
        "root": str(expected),
        "head": _git(expected, "rev-parse", "HEAD"),
        "tree": _git(expected, "rev-parse", "HEAD^{tree}"),
        "tracked_clean": not bool(_git(expected, "status", "--porcelain", "--untracked-files=no")),
    }


def _cell(cell_id: int) -> Any:
    if isinstance(cell_id, bool) or not isinstance(cell_id, int) or not 0 <= cell_id < len(CELL_SPECS):
        raise TechnicalBoundary("array cell id differs")
    value = CELL_SPECS[cell_id]
    if value.cell_id != cell_id:
        raise TechnicalBoundary("array cell mapping drift")
    return value


def _model_snapshot(hf_hub_cache: Path, alias: str) -> Path:
    spec = MODEL_BINDINGS[alias]
    path = hf_hub_cache / str(spec["hf_repo"]) / "snapshots" / str(spec["revision"])
    return path.resolve(strict=True)


def _load_hparams(repo_root: Path, family: str, alias: str) -> tuple[Any, Path]:
    relative = Path(str(MODEL_BINDINGS[alias]["hparams"][family][0]))
    path = (repo_root / relative).resolve(strict=True)
    if family == "AlphaEdit":
        cls = importlib.import_module("easyeditor.models.alphaedit.AlphaEdit_hparams").AlphaEditHyperParams
    else:
        cls = importlib.import_module("easyeditor.models.memit.memit_hparams").MEMITHyperParams
    return cls.from_hparams(str(path)), path


def _method_module(family: str) -> Any:
    name = "easyeditor.models.alphaedit.AlphaEdit_main" if family == "AlphaEdit" else "easyeditor.models.memit.memit_main"
    return importlib.import_module(name)


def _stock_repr_tools(method_module: Any) -> Any:
    """Bind repr-tools from the exact stock ``compute_z`` implementation.

    The family main modules re-export ``compute_z`` but do not themselves
    export its ``repr_tools`` import.  Following the function binding keeps
    this graph-producing observer tied to the same pinned EasyEdit source as
    the fixed-z authority without modifying that source.
    """

    compute_z = getattr(method_module, "compute_z", None)
    namespace = getattr(compute_z, "__globals__", None)
    repr_tools = namespace.get("repr_tools") if isinstance(namespace, dict) else None
    if not callable(getattr(repr_tools, "get_reprs_at_word_tokens", None)):
        raise TechnicalBoundary("stock compute_z repr_tools binding is unavailable")
    return repr_tools


def _selected(model: torch.nn.Module, hparams: Any) -> dict[str, torch.nn.Parameter]:
    parameters = dict(model.named_parameters())
    values: dict[str, torch.nn.Parameter] = {}
    for layer in (4, 5, 6, 7, 8):
        name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        if name not in parameters or parameters[name].ndim != 2:
            raise TechnicalBoundary(f"selected weight is unavailable: {name}")
        values[name] = parameters[name]
    if len({int(value.data_ptr()) for value in values.values()}) != len(values):
        raise TechnicalBoundary("selected weights alias storage")
    return values


def _clone_selected(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in values.items()}


def _delta_frobenius_geometry(
    values: Mapping[str, torch.Tensor], entry: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    if set(values) != set(entry):
        raise TechnicalBoundary("delta geometry inventory differs")
    by_weight: dict[str, float] = {}
    total_squared = 0.0
    for name in sorted(values):
        delta = values[name].detach().to(device="cpu", dtype=torch.float64) - entry[name].detach().to(
            device="cpu", dtype=torch.float64
        )
        squared = float(torch.sum(delta.square()).item())
        if not math.isfinite(squared) or squared < 0.0:
            raise NumericalMethodBoundary("terminal delta Frobenius geometry is non-finite")
        by_weight[name] = squared
        total_squared += squared
    return {
        "terminal_net_frobenius": math.sqrt(total_squared),
        "terminal_net_frobenius_squared": total_squared,
        "terminal_net_frobenius_squared_by_weight": by_weight,
    }


def _tensor_content_state_identity(
    prefix: str,
    values: Mapping[Any, torch.Tensor],
    *,
    include_version: bool = True,
) -> str:
    payload = []
    for key, value in sorted(values.items(), key=lambda item: repr(item[0])):
        row = {
            "key": repr(key),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "data_ptr": int(value.data_ptr()),
            "content_sha256": tensor_sha256(value),
        }
        if include_version:
            row["version"] = int(value._version)
        payload.append(row)
    return canonical_hash({"schema": prefix, "tensors": payload})


def _restore_selected(values: Mapping[str, torch.Tensor], snapshot: Mapping[str, torch.Tensor]) -> None:
    if set(values) != set(snapshot):
        raise TechnicalBoundary("W0 restore inventory differs")
    with torch.no_grad():
        for name, parameter in values.items():
            parameter.copy_(snapshot[name].to(device=parameter.device, dtype=parameter.dtype))
    if tensor_set_sha256(values) != tensor_set_sha256(snapshot):
        raise TechnicalBoundary("W0 byte restore failed")


def _request_order_identity(requests: Sequence[Mapping[str, Any]]) -> str:
    values = [str(item["request_sha256"]) for item in requests]
    return canonical_hash(values)


def _raw_records(dataset: Path, case_ids: set[int]) -> dict[int, dict[str, Any]]:
    value = json.loads(dataset.read_text(encoding="utf-8"))
    result = {int(row["case_id"]): row for row in value if int(row["case_id"]) in case_ids}
    if set(result) != case_ids:
        raise TechnicalBoundary("CounterFact endpoint record load is incomplete")
    return result


def _official_requests(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    return normalize_official_requests(rows)


def _parameter_inventory(model: torch.nn.Module) -> dict[str, Any]:
    counts: dict[str, int] = {}
    tensors: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0) + int(parameter.numel())
        tensors[key] = tensors.get(key, 0) + 1
    return {
        "parameter_elements_by_dtype": counts,
        "parameter_tensors_by_dtype": tensors,
        "quantized": bool(getattr(model, "is_quantized", False)),
    }


@contextlib.contextmanager
def _model_name(model: torch.nn.Module, value: str) -> Iterator[None]:
    previous = model.config._name_or_path
    model.config._name_or_path = value
    try:
        yield
    finally:
        model.config._name_or_path = previous


def _tensor_state_identity(prefix: str, values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256((prefix + "\0").encode("utf-8"))
    for key in sorted(values):
        value = values[key]
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_sha256(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _canonical_ns_accounting(
    round_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entry_prompt_denominator = 0
    primary_prompt_denominator = 0
    primary_evaluation_count = 0
    for round_payload in round_payloads:
        request_count = int(round_payload["request_count"])
        expected = request_count * 10
        entry = round_payload.get("entry_evaluation")
        endpoints = round_payload.get("primary_endpoints")
        if not isinstance(entry, Mapping) or not isinstance(endpoints, list):
            raise TechnicalBoundary("canonical NS publication inventory differs")
        entry_ns = entry.get("locality", {}).get("canonical_ns")
        if (
            entry.get("schema") != "orbode.raw-free-evaluation.v2"
            or not isinstance(entry_ns, Mapping)
            or entry_ns.get("predicate") != "target_true_nll < target_new_nll"
            or entry_ns.get("prompt_denominator") != expected
        ):
            raise TechnicalBoundary("entry canonical NS denominator differs")
        entry_prompt_denominator += expected
        for endpoint in endpoints:
            if not isinstance(endpoint, Mapping):
                raise TechnicalBoundary("canonical NS endpoint is not an object")
            evaluation = endpoint.get("evaluation")
            ns = evaluation.get("locality", {}).get("canonical_ns") if isinstance(evaluation, Mapping) else None
            if (
                not isinstance(evaluation, Mapping)
                or evaluation.get("schema") != "orbode.raw-free-evaluation.v2"
                or not isinstance(ns, Mapping)
                or ns.get("predicate") != "target_true_nll < target_new_nll"
                or ns.get("prompt_denominator") != expected
            ):
                raise TechnicalBoundary("endpoint canonical NS denominator differs")
            primary_prompt_denominator += expected
            primary_evaluation_count += 1
    return {
        "schema": "orbode.canonical-ns-accounting.v1",
        "predicate": "target_true_nll < target_new_nll",
        "prompts_per_request": 10,
        "entry_prompt_denominator": entry_prompt_denominator,
        "primary_endpoint_prompt_denominator": primary_prompt_denominator,
        "primary_evaluation_count": primary_evaluation_count,
        "all_primary_endpoint_denominators_exact": True,
        "token_prediction_preservation_is_canonical_ns": False,
    }


def _terminal_receipt(
    *,
    cell_id: int,
    wave: str,
    round_indices: Sequence[int],
    model_alias: str,
    writer_family: str,
    run_token: str,
    request_count: int,
    primary_endpoint_count: int,
    entry_already_hit_count: int,
    fixed_z_compute_count: int,
    source_head: str,
    source_tree: str,
    result_sha256: str,
    round_publication_identities: Sequence[str],
    round_publication_file_sha256: Sequence[str],
    canonical_ns_accounting: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact receipt consumed by the round0→remaining gate."""

    terminal: dict[str, Any] = {
        "schema": TERMINAL_SCHEMA,
        "status": "TERMINAL_VALID",
        "cell_id": cell_id,
        "wave": wave,
        "rounds": list(round_indices),
        "model_alias": model_alias,
        "writer_family": writer_family,
        "run_token": run_token,
        "request_count": request_count,
        "primary_arms": list(PRIMARY_ARM_ORDER),
        "primary_endpoint_count": primary_endpoint_count,
        "scientific_attempted_count": primary_endpoint_count,
        "terminal_valid_count": primary_endpoint_count,
        "entry_already_hit_count": entry_already_hit_count,
        "full_fp32": True,
        "full_fp32_claim_scope": "MODEL_STORAGE_AND_ORBODE_DYNAMIC_PATH",
        "official_native_memit_ephemeral_fp64_solve_exception": writer_family == "MEMIT",
        "all_algorithm_solves_full_fp32": writer_family != "MEMIT",
        "unqualified_full_fp32_claim": writer_family != "MEMIT",
        "fixed_z_compute_count": fixed_z_compute_count,
        "fixed_z_recompute_count": 0,
        "model_reload_wave_count": 1,
        "w0_restore_pass": True,
        "cache_restore_pass": True,
        "transaction_pass": True,
        "jvp_gate_pass": True,
        "overlay_gate_pass": True,
        "attention_backend_pass": True,
        "source_head": source_head,
        "source_tree": source_tree,
        "stream_root": STREAM_ROOT,
        "order_root": ORDER_ROOT,
        "technical_failure_count": 0,
        "scientific_failure_count": 0,
        "imputation_count": 0,
        "result_sha256": result_sha256,
        "round_publication_count": len(round_publication_identities),
        "round_publication_identity_root": canonical_hash(
            list(round_publication_identities)
        ),
        "round_publication_file_sha256_root": canonical_hash(
            list(round_publication_file_sha256)
        ),
        "literal_prompt_target_token_prediction_publication_count": 0,
        "canonical_ns_accounting": (
            dict(canonical_ns_accounting)
            if canonical_ns_accounting is not None
            else {"status": "NOT_RECORDED_LEGACY_RECEIPT"}
        ),
        "scientific_promotion": False,
    }
    terminal["receipt_identity_sha256"] = canonical_hash(terminal)
    return terminal


class FamilyRuntime:
    """One model/family/round binding with exact W0 and cold method state."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        family: str,
        hparams: Any,
        module: Any,
        requests: Sequence[Mapping[str, Any]],
        endpoint_records: Sequence[Mapping[str, Any]],
        request_order_sha256: str,
        contexts: Sequence[Sequence[str]],
        capture_persistent_endpoint: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.family = family
        self.hparams = hparams
        self.module = module
        self._repr_tools = _stock_repr_tools(module)
        self.requests = tuple(dict(item) for item in requests)
        self.endpoint_records = tuple(dict(item) for item in endpoint_records)
        self.request_order_sha256 = request_order_sha256
        self.contexts = tuple(tuple(str(item) for item in group) for group in contexts)
        self.device = next(model.parameters()).device
        self.parameters = _selected(model, hparams)
        self.w0 = _clone_selected(self.parameters)
        self.w0_sha256 = tensor_set_sha256(self.w0)
        self.pointer_identity = {name: int(value.data_ptr()) for name, value in self.parameters.items()}
        self.semantic_inventory: SemanticInventory | None = None
        self.fixed_z: FixedZArtifact | None = None
        self.pre_evaluation: Mapping[str, Any] | None = None
        self.endpoint_evaluation_count = 0
        self._alpha_cache_entry_is_zero = False
        self._alpha_cache_entry: torch.Tensor | None = None
        self._cov_versions: dict[tuple[Any, ...], int] = {}
        self._prepared_method_state_identity: str | None = None
        self._capture_persistent_endpoint = bool(capture_persistent_endpoint)
        self._captured_endpoint_weights: dict[str, torch.Tensor] | None = None
        self._captured_endpoint_method_state: torch.Tensor | None = None
        self._captured_endpoint_sha256: str | None = None

    def prepare_method_state(self) -> None:
        if self._prepared_method_state_identity is not None:
            raise TechnicalBoundary("method state preparation is repeated")
        if self.family == "AlphaEdit":
            projector = torch.load(self.hparams.P_loc, map_location="cpu", weights_only=False)
            if projector.dtype is not torch.float32 or tuple(projector.shape[:1]) != (5,) or not bool(torch.isfinite(projector).all()):
                raise TechnicalBoundary("AlphaEdit projector differs from finite FP32 layer inventory")
            self.module.P = projector.contiguous()
            self.module.P_loaded = True
            width = int(next(iter(self.parameters.values())).shape[1])
            self.module.cache_c = torch.zeros((5, width, width), dtype=torch.float32)
            self.module.cache_c_new = True
            self._alpha_cache_entry_is_zero = True
            self._alpha_cache_entry = self.module.cache_c.detach().clone()
        else:
            # Populate stock COV_CACHE from sealed files before the arm-entry
            # identity is captured.  It is read-only throughout every arm.
            with _model_name(self.model, str(self.hparams.model_name)):
                for layer in (4, 5, 6, 7, 8):
                    covariance = self.module.get_cov(
                        self.model,
                        self.tokenizer,
                        self.hparams.rewrite_module_tmp.format(layer),
                        self.hparams.mom2_dataset,
                        self.hparams.mom2_n_samples,
                        self.hparams.mom2_dtype,
                        force_recompute=False,
                        hparams=self.hparams,
                    )
                    if covariance.dtype is not torch.float32 or not bool(torch.isfinite(covariance).all()):
                        raise TechnicalBoundary("MEMIT covariance left FULL_FP32")
                    del covariance
            self._cov_versions = {key: int(value._version) for key, value in self.module.COV_CACHE.items()}
        self._prepared_method_state_identity = self.method_state_identity()

    def bind_existing_method_state(self) -> None:
        """Bind an already prepared entry state without rebuilding science assets.

        Sequential runners use this at every B100 entry.  It captures the
        current AlphaEdit history bytes (which need not be cold) or the static
        MEMIT covariance versions, while leaving P/C and every Official
        equation untouched.
        """

        if self._prepared_method_state_identity is not None:
            raise TechnicalBoundary("existing method state binding is repeated")
        if self.family == "AlphaEdit":
            cache = getattr(self.module, "cache_c", None)
            if (
                not bool(getattr(self.module, "P_loaded", False))
                or not isinstance(cache, torch.Tensor)
                or cache.dtype is not torch.float32
                or cache.ndim != 3
                or tuple(cache.shape[:1]) != (5,)
                or not bool(torch.isfinite(cache).all())
            ):
                raise TechnicalBoundary("AlphaEdit sequential entry state is not prepared")
            self._alpha_cache_entry = cache.detach().clone()
            self._alpha_cache_entry_is_zero = bool(
                getattr(self.module, "cache_c_new", False)
            )
        else:
            if not isinstance(getattr(self.module, "COV_CACHE", None), dict) or not self.module.COV_CACHE:
                raise TechnicalBoundary("MEMIT sequential covariance state is not prepared")
            self._cov_versions = {
                key: int(value._version) for key, value in self.module.COV_CACHE.items()
            }
        self._prepared_method_state_identity = self.method_state_identity()

    def reset_entry(self) -> None:
        _restore_selected(self.parameters, self.w0)
        if {name: int(value.data_ptr()) for name, value in self.parameters.items()} != self.pointer_identity:
            raise TechnicalBoundary("selected parameter pointer changed")
        if self.family == "AlphaEdit":
            if self._alpha_cache_entry is None:
                raise TechnicalBoundary("AlphaEdit entry cache snapshot is absent")
            self.module.cache_c.copy_(self._alpha_cache_entry)
            self.module.cache_c_new = self._alpha_cache_entry_is_zero
        else:
            if {key: int(value._version) for key, value in self.module.COV_CACHE.items()} != self._cov_versions:
                raise TechnicalBoundary("MEMIT static COV_CACHE mutated")
        if (
            self._prepared_method_state_identity is None
            or self.method_state_identity() != self._prepared_method_state_identity
        ):
            raise TechnicalBoundary("prepared Alpha cache/MEMIT COV content identity drifted")

    def method_state_identity(self) -> str:
        if self.family == "AlphaEdit":
            cache = self.module.cache_c
            if cache.dtype is not torch.float32 or not bool(torch.isfinite(cache).all()):
                raise TechnicalBoundary("AlphaEdit cache differs from finite FP32")
            return _tensor_content_state_identity(
                "orbode.alpha-cache-content.v1",
                {"cache_c": cache},
                include_version=False,
            )
        return _tensor_content_state_identity(
            "orbode.memit-cov-content.v1", self.module.COV_CACHE
        )

    def compute_fixed_z(self) -> FixedZArtifact:
        if self.fixed_z is not None:
            raise TechnicalBoundary("fixed native z recomputation is forbidden")
        values: list[torch.Tensor] = []
        with _model_name(self.model, str(self.hparams.model_name)), torch.enable_grad():
            for request in self.requests:
                value = self.module.compute_z(
                    self.model,
                    self.tokenizer,
                    dict(request),
                    self.hparams,
                    8,
                    [list(group) for group in self.contexts],
                )
                value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
                if value.ndim != 1 or not bool(torch.isfinite(value).all()):
                    raise TechnicalBoundary("stock native z is not a finite FP32 vector")
                values.append(value)
        stacked = torch.stack(values, dim=1).contiguous()
        semantic = build_compute_z_semantic_inventory(
            tokenizer=self.tokenizer,
            requests=self.requests,
            context_templates=self.contexts,
            request_order_sha256=self.request_order_sha256,
        )
        self.semantic_inventory = semantic
        self.fixed_z = FixedZArtifact(
            values=stacked,
            identity_sha256=tensor_sha256(stacked),
            request_order_sha256=self.request_order_sha256,
            target_context_identity_sha256=semantic.identity_sha256,
        )
        return self.fixed_z

    def terminal_graph(self) -> torch.Tensor:
        # AlphaEdit's stock wrapper always asks for both representations and
        # detaches them, while MEMIT's sibling accepts the private ``track``
        # extension.  The response JVP needs the undetached output for both
        # families, so bind the shared stock repr-tools primitive directly
        # instead of relying on that family-specific wrapper signature.
        fact_token_strategy = str(self.hparams.fact_token)
        if not fact_token_strategy.startswith("subject_"):
            raise TechnicalBoundary("terminal graph requires a subject-token strategy")
        get_outputs = getattr(self._repr_tools, "get_reprs_at_word_tokens", None)
        if not callable(get_outputs):
            raise TechnicalBoundary("stock family module does not expose repr_tools")
        value = get_outputs(
            model=self.model,
            tok=self.tokenizer,
            layer=8,
            context_templates=[str(item["prompt"]) for item in self.requests],
            words=[str(item["subject"]) for item in self.requests],
            module_template=self.hparams.layer_module_tmp,
            track="out",
            subtoken=fact_token_strategy.removeprefix("subject_"),
        )
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise TechnicalBoundary("stock terminal representation has invalid geometry")
        return value.T

    def terminal(self) -> torch.Tensor:
        value = self.terminal_graph().detach().to(device="cpu", dtype=torch.float32).contiguous()
        if self.fixed_z is not None and value.shape != self.fixed_z.values.shape:
            raise TechnicalBoundary("current terminal/fixed-z shape differs")
        return value

    def observe_semantic(self) -> Any:
        if self.semantic_inventory is None:
            raise TechnicalBoundary("semantic inventory is not sealed")
        return observe_semantic_predicate(self.model, self.semantic_inventory, microbatch_size=64)

    def build_layer(
        self,
        *,
        layer: int,
        current_terminal: torch.Tensor,
        fixed_z: FixedZArtifact,
        residual_denominator: int,
        state_version: int,
    ) -> LayerBuild:
        keys = self.module.compute_ks(
            self.model,
            self.tokenizer,
            list(self.requests),
            self.hparams,
            layer,
            [list(group) for group in self.contexts],
        ).T.detach().to(device=self.device, dtype=torch.float32).contiguous()
        residual = (fixed_z.values - current_terminal).to(device=self.device, dtype=torch.float32)
        if keys.shape[1] % residual.shape[1] != 0:
            raise TechnicalBoundary("writer key/request multiplicity differs")
        residual = residual.repeat_interleave(keys.shape[1] // residual.shape[1], dim=1)
        residual = residual / float(residual_denominator)
        position = (4, 5, 6, 7, 8).index(layer)
        with _model_name(self.model, str(self.hparams.model_name)):
            if self.family == "MEMIT":
                covariance = self.module.get_cov(
                    self.model,
                    self.tokenizer,
                    self.hparams.rewrite_module_tmp.format(layer),
                    self.hparams.mom2_dataset,
                    self.hparams.mom2_n_samples,
                    self.hparams.mom2_dtype,
                    force_recompute=False,
                    hparams=self.hparams,
                ).to(device=self.device, dtype=torch.float32)
                matrix = float(self.hparams.mom2_update_weight) * covariance + keys @ keys.T
                right = torch.linalg.solve(matrix, keys)
                solver_name = "stock-memit-equation/fp32/factorized"
            else:
                projector = self.module.P[position].to(device=self.device, dtype=torch.float32)
                cache = self.module.cache_c[position].to(device=self.device, dtype=torch.float32)
                matrix = projector @ (keys @ keys.T + cache)
                matrix = matrix + float(self.hparams.L2) * torch.eye(
                    keys.shape[0], dtype=torch.float32, device=self.device
                )
                rhs = projector @ keys
                right = torch.linalg.solve(matrix, rhs)
                solver_name = "stock-alphaedit-P-inside-equation/fp32/factorized"
        solve_residual = matrix @ right - (keys if self.family == "MEMIT" else rhs)
        denominator = (
            torch.linalg.matrix_norm(matrix) * torch.linalg.matrix_norm(right)
            + torch.linalg.matrix_norm(keys if self.family == "MEMIT" else rhs)
        ).clamp_min(torch.finfo(torch.float32).tiny)
        backward_error = float((torch.linalg.matrix_norm(solve_residual) / denominator).item())
        if not math.isfinite(backward_error) or not bool(torch.isfinite(right).all()):
            raise TechnicalBoundary("official-form FP32 solve is non-finite")
        name = f"{self.hparams.rewrite_module_tmp.format(layer)}.weight"
        weight_shape = tuple(self.parameters[name].shape)
        if (residual.shape[0], right.shape[0]) == weight_shape:
            left, oriented_right = residual, right
        elif (right.shape[0], residual.shape[0]) == weight_shape:
            left, oriented_right = right, residual
        else:
            raise TechnicalBoundary("canonical Official factor orientation differs from weight")
        result = LayerBuild(
            layer=layer,
            weight_name=name,
            left=left.detach().to(device="cpu", dtype=torch.float32),
            right=oriented_right.detach().to(device="cpu", dtype=torch.float32),
            built_state_version=state_version,
            residual_denominator=residual_denominator,
            residual_sha256=tensor_sha256(residual),
            keys_sha256=tensor_sha256(keys),
            solver_identity=canonical_hash({
                "family": self.family,
                "layer": layer,
                "name": solver_name,
                "state_version": state_version,
            }),
            solve_backward_error=backward_error,
        )
        if getattr(self, "cumulative_observer", None) is not None:
            self.cumulative_observer.on_build(result, keys)
        return result

    def evaluate_endpoint(self) -> Mapping[str, Any]:
        from project.run_scripts.ordered_response_barrier_ode.counterfact_locality_evaluator import (
            evaluate_counterfact_with_canonical_ns,
        )

        self.endpoint_evaluation_count += 1
        return evaluate_counterfact_with_canonical_ns(
            self.model,
            self.tokenizer,
            self.endpoint_records,
            device=self.device,
            microbatch_size=16,
        )

    def _apply_shadow(self, shadow: Mapping[str, torch.Tensor]) -> None:
        if set(shadow) != set(self.parameters):
            raise TechnicalBoundary("terminal shadow inventory differs")
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                parameter.copy_(shadow[name].to(device=parameter.device, dtype=torch.float32))

    def _capture_endpoint_for_persistence(self) -> None:
        if not self._capture_persistent_endpoint:
            return
        self._captured_endpoint_weights = _clone_selected(self.parameters)
        self._captured_endpoint_sha256 = tensor_set_sha256(
            self._captured_endpoint_weights
        )
        self._captured_endpoint_method_state = (
            self.module.cache_c.detach().clone()
            if self.family == "AlphaEdit"
            else None
        )

    def commit_captured_endpoint(self, *, expected_sha256: str) -> Mapping[str, Any]:
        """Promote the already observed terminal bytes to the next B100 entry.

        The integrator's guarded terminal callback always restores its entry
        state before hooks are reinstalled.  This method performs the distinct
        sequential harness commit from the exact captured bytes, with no
        writer, key, target, evaluator, or model-forward recomputation.
        """

        if (
            not self._capture_persistent_endpoint
            or self._captured_endpoint_weights is None
            or self._captured_endpoint_sha256 is None
            or self._captured_endpoint_sha256 != expected_sha256
        ):
            raise TechnicalBoundary("captured sequential endpoint identity differs")
        entry_weight_sha256 = tensor_set_sha256(self.parameters)
        entry_method_state_sha256 = self.method_state_identity()
        self._apply_shadow(self._captured_endpoint_weights)
        if self.family == "AlphaEdit":
            if self._captured_endpoint_method_state is None:
                raise TechnicalBoundary("captured AlphaEdit endpoint cache is absent")
            self.module.cache_c.copy_(self._captured_endpoint_method_state)
            # A committed sequential endpoint is initialized AlphaEdit
            # history.  Stock interprets ``False`` as a request to replace it
            # with a fresh zero tensor on the next Official call, severing the
            # B100 cache pointer/content chain.
            self.module.cache_c_new = True
        observed = tensor_set_sha256(self.parameters)
        if observed != expected_sha256:
            raise TechnicalBoundary("sequential endpoint commit bytes differ")
        method_state_after = self.method_state_identity()
        result = {
            "entry_weight_sha256": entry_weight_sha256,
            "committed_weight_sha256": observed,
            "entry_method_state_sha256": entry_method_state_sha256,
            "committed_method_state_sha256": method_state_after,
            "writer_recompute_count": 0,
            "fixed_z_recompute_count": 0,
            "model_forward_count": 0,
            "evaluator_count": 0,
            "persistent_copy_count": 1,
        }
        self._captured_endpoint_weights = None
        self._captured_endpoint_method_state = None
        self._captured_endpoint_sha256 = None
        return result

    def finalize(
        self,
        *,
        arm: str,
        terminal_state_version: int,
        shadow_weights: Mapping[str, torch.Tensor],
        deltas: Sequence[OverlayDelta],
        derived_observation_only: bool,
    ) -> Mapping[str, Any]:
        del terminal_state_version, deltas
        endpoint_sha = tensor_set_sha256(shadow_weights)
        try:
            self._apply_shadow(shadow_weights)
            if tensor_set_sha256(self.parameters) != endpoint_sha:
                raise TechnicalBoundary("physical terminal/shadow bytes differ")
            terminal = self.terminal()
            semantic = self.observe_semantic()
            evaluation = self.evaluate_endpoint()
            history_append_count = 0
            delta_geometry = _delta_frobenius_geometry(self.parameters, self.w0)
            if self.family == "AlphaEdit" and not derived_observation_only:
                for position, layer in enumerate((4, 5, 6, 7, 8)):
                    keys = self.module.compute_ks(
                        self.model,
                        self.tokenizer,
                        list(self.requests),
                        self.hparams,
                        layer,
                        [list(group) for group in self.contexts],
                    ).T.detach().to(device="cpu", dtype=torch.float32)
                    self.module.cache_c[position].add_(keys @ keys.T)
                history_append_count = 1
            if not derived_observation_only:
                self._capture_endpoint_for_persistence()
            return {
                "status": "TERMINAL_VALID",
                "arm": arm,
                "evaluation": evaluation,
                "terminal_activation_sha256": tensor_sha256(terminal),
                "semantic_observation": asdict(semantic),
                "selected_weight_endpoint_sha256": endpoint_sha,
                "physical_write_count": 0 if derived_observation_only else 1,
                "temporary_observation_materialization_count": int(derived_observation_only),
                "history_append_count": history_append_count,
                "terminal_history_key_capture_count": (
                    5 if self.family == "AlphaEdit" and not derived_observation_only else 0
                ),
                "inner_history_append_count": 0,
                "inner_cache_mutation_count": 0,
                "w0_restore_deferred_to_transaction": True,
                **delta_geometry,
            }
        finally:
            _restore_selected(self.parameters, self.w0)
            if self.family == "AlphaEdit":
                if self._alpha_cache_entry is None:
                    raise TechnicalBoundary("AlphaEdit entry cache snapshot is absent")
                self.module.cache_c.copy_(self._alpha_cache_entry)
                self.module.cache_c_new = self._alpha_cache_entry_is_zero

    def evaluate_current(self, *, arm: str, status: str, state_version: int) -> Mapping[str, Any]:
        if state_version != 0 or tensor_set_sha256(self.parameters) != self.w0_sha256:
            raise TechnicalBoundary("no-op endpoint is not exact W0")
        terminal = self.terminal()
        semantic = self.observe_semantic()
        if arm != ArmId.ORDERED_RESPONSE_FIRST_HIT.value:
            self._capture_endpoint_for_persistence()
        return {
            "status": status,
            "arm": arm,
            "evaluation": self.evaluate_endpoint(),
            "terminal_activation_sha256": tensor_sha256(terminal),
            "semantic_observation": asdict(semantic),
            "selected_weight_endpoint_sha256": self.w0_sha256,
            "physical_write_count": 0,
            "temporary_observation_materialization_count": 0,
            "history_append_count": 0,
            "terminal_history_key_capture_count": 0,
            "inner_history_append_count": 0,
            "inner_cache_mutation_count": 0,
            "w0_pointer_bytes_restore_pass": True,
            "terminal_net_frobenius": 0.0,
            "terminal_net_frobenius_squared": 0.0,
            "terminal_net_frobenius_squared_by_weight": {
                name: 0.0 for name in sorted(self.parameters)
            },
        }

    def run_official(self, *, fixed_z: FixedZArtifact) -> Mapping[str, Any]:
        """Run the literal pinned Official entrypoint with the shared fixed-z.

        The stock MEMIT implementation uses an ephemeral FP64 linear solve;
        model/storage/final update remain FP32 and the fact is explicitly
        recorded rather than hidden.  Dynamic ORBODE solves are FP32.
        """

        call_index = 0
        original_compute_z = self.module.compute_z

        def fixed_compute_z(
            current_model: Any,
            current_tokenizer: Any,
            request: Mapping[str, Any],
            current_hparams: Any,
            layer: int,
            context_templates: Sequence[Sequence[str]],
        ) -> torch.Tensor:
            nonlocal call_index
            if (
                current_model is not self.model
                or current_tokenizer is not self.tokenizer
                or current_hparams is not self.hparams
                or int(layer) != 8
                or tuple(tuple(str(item) for item in group) for group in context_templates) != self.contexts
                or call_index >= fixed_z.request_count
            ):
                raise TechnicalBoundary("Official fixed-z callback binding differs")
            expected = self.requests[call_index]
            normalized = normalize_official_requests([request])[0]
            if any(str(normalized[key]) != str(expected[key]) for key in ("case_id", "prompt", "subject", "target_new")):
                raise TechnicalBoundary("Official fixed-z request order/content differs")
            value = fixed_z.values[:, call_index].to(device=self.device, dtype=torch.float32)
            call_index += 1
            return value

        self.module.compute_z = fixed_compute_z
        try:
            cumulative = getattr(self, "cumulative_observer", None)
            if cumulative is not None:
                cumulative.install_official(self)
            apply = self.module.apply_AlphaEdit_to_model if self.family == "AlphaEdit" else self.module.apply_memit_to_model
            with _model_name(self.model, str(self.hparams.model_name)):
                edited, _ = apply(
                    self.model,
                    self.tokenizer,
                    list(self.requests),
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    cache_template=None,
                    reset_cache=False,
                )
            if edited is not self.model or call_index != fixed_z.request_count:
                raise TechnicalBoundary("Official endpoint/fixed-z consumption differs")
            if cumulative is not None:
                cumulative.finish_official(self)
            if any(not bool(torch.isfinite(value).all()) for value in self.parameters.values()):
                raise TechnicalBoundary("Official endpoint is non-finite before parity exclusion")
            endpoint_sha = tensor_set_sha256(self.parameters)
            delta_geometry = _delta_frobenius_geometry(self.parameters, self.w0)
            terminal = self.terminal()
            semantic = self.observe_semantic()
            evaluation = self.evaluate_endpoint()
            history_count = 1 if self.family == "AlphaEdit" else 0
            self._capture_endpoint_for_persistence()
            return {
                "status": "TERMINAL_VALID",
                "arm": ArmId.OFFICIAL.value,
                "evaluation": evaluation,
                "terminal_activation_sha256": tensor_sha256(terminal),
                "semantic_observation": asdict(semantic),
                "selected_weight_endpoint_sha256": endpoint_sha,
                "physical_write_count": 1,
                "history_append_count": history_count,
                "terminal_history_key_capture_count": 5 if self.family == "AlphaEdit" else 0,
                "inner_history_append_count": 0,
                "inner_cache_mutation_count": 0,
                "fixed_z_callback_count": call_index,
                "official_entrypoint": f"{self.module.__name__}.{apply.__name__}",
                "model_and_storage_dtype": "torch.float32",
                "official_native_ephemeral_solve_dtype": (
                    "torch.float64" if self.family == "MEMIT" else "torch.float32"
                ),
                "numeric_storage_cast_count": 0,
                **delta_geometry,
            }
        finally:
            self.module.compute_z = original_compute_z
            if getattr(self, "cumulative_observer", None) is not None:
                self.cumulative_observer.restore_official(self)
            _restore_selected(self.parameters, self.w0)
            if self.family == "AlphaEdit":
                if self._alpha_cache_entry is None:
                    raise TechnicalBoundary("AlphaEdit entry cache snapshot is absent")
                self.module.cache_c.copy_(self._alpha_cache_entry)
                self.module.cache_c_new = self._alpha_cache_entry_is_zero

    def adapter(self, *, compute: bool) -> CallbackMethodAdapter:
        value = CallbackMethodAdapter(
            family=self.family,
            layers=(4, 5, 6, 7, 8),
            weight_names_by_layer={layer: f"{self.hparams.rewrite_module_tmp.format(layer)}.weight" for layer in (4, 5, 6, 7, 8)},
            compute_fixed_z=self.compute_fixed_z,
            capture_terminal=self.terminal,
            build_official_form_layer=self.build_layer,
            run_official_endpoint=self.run_official,
            finalize_terminal_state=self.finalize,
            evaluate_current_endpoint=self.evaluate_current,
            capture_method_state_identity=self.method_state_identity,
        )
        if not compute:
            if self.fixed_z is None:
                raise TechnicalBoundary("shared fixed-z is unavailable")
            value.use_fixed_z(self.fixed_z)
        return value


def _serializable_result(result: ArmExecutionResult, ledger: Mapping[str, Any]) -> dict[str, Any]:
    derived = None
    if result.derived_endpoint is not None:
        derived = {
            "arm": result.derived_endpoint.arm,
            "status": result.derived_endpoint.status,
            "state_version": result.derived_endpoint.state_version,
            "factor_count": len(result.derived_endpoint.deltas),
            "factor_identities": [item.build_identity for item in result.derived_endpoint.deltas],
            "first_hit": result.derived_endpoint.first_hit,
            "endpoint": dict(result.derived_endpoint.endpoint),
        }
    return {
        "arm": result.arm,
        "status": result.status,
        "request_count": result.request_count,
        "endpoint": dict(result.endpoint),
        "telemetry": dict(result.telemetry),
        "overlay": None if result.overlay is None else dict(result.overlay),
        "derived_endpoint": derived,
        "fixed_z_identity_sha256": result.fixed_z_identity_sha256,
        "adapter_ledger": dict(ledger),
        "scientific_promotion": False,
    }


def _reconcile_arm_accounting(
    payload: Mapping[str, Any],
    *,
    family: str,
    arm: ArmId,
    endpoint_evaluator_count: int,
    method_state_entry_sha256: str,
    method_state_after_sha256: str,
) -> dict[str, Any]:
    """Fail-close the counters that cross adapter/integrator/runtime layers.

    Stock Official solves are opaque to the thin adapter, so they are labelled
    as source-level logical layer counts rather than falsely reported as
    intercepted runtime calls.  Every dynamic solve is adapter-intercepted and
    therefore admits exact reconciliation.
    """

    if family not in {"MEMIT", "AlphaEdit"}:
        raise TechnicalBoundary("accounting writer family differs")
    if endpoint_evaluator_count not in (1, 2):
        raise TechnicalBoundary("arm endpoint evaluator count is not reconciled")
    if method_state_entry_sha256 != method_state_after_sha256:
        raise TechnicalBoundary("arm cache/COV content identity differs")
    telemetry = payload.get("telemetry")
    ledger = payload.get("adapter_ledger")
    endpoint = payload.get("endpoint")
    if not all(isinstance(value, Mapping) for value in (telemetry, ledger, endpoint)):
        raise TechnicalBoundary("arm accounting payload shape differs")
    assert isinstance(telemetry, Mapping)
    assert isinstance(ledger, Mapping)
    assert isinstance(endpoint, Mapping)
    model_forward_count = payload.get("model_forward_invocation_count")
    if (
        isinstance(model_forward_count, bool)
        or not isinstance(model_forward_count, int)
        or model_forward_count <= 0
    ):
        raise TechnicalBoundary("arm model-forward count is invalid")
    jvp_ledger = payload.get("jvp_ledger")
    if not isinstance(jvp_ledger, Mapping):
        raise TechnicalBoundary("arm JVP ledger is missing")
    jvp_forward_count = int(jvp_ledger.get("model_forward_invocation_count", -1))
    jvp_call_count = int(jvp_ledger.get("jvp_call_count", -1))
    if (
        jvp_forward_count < 0
        or jvp_call_count < 0
        or jvp_forward_count > model_forward_count
        or jvp_call_count != int(telemetry.get("jvp_call_count", 0))
    ):
        raise TechnicalBoundary("arm JVP/model-forward accounting differs")

    endpoint_write_count = int(endpoint.get("physical_write_count", -1))
    history_append_count = int(endpoint.get("history_append_count", -1))
    terminal_history_key_count = int(
        endpoint.get("terminal_history_key_capture_count", -1)
    )
    if endpoint_write_count not in (0, 1):
        raise TechnicalBoundary("arm authoritative write count differs")
    expected_history = 1 if family == "AlphaEdit" and endpoint_write_count == 1 else 0
    expected_terminal_history_keys = 5 if expected_history else 0
    if (
        history_append_count != expected_history
        or terminal_history_key_count != expected_terminal_history_keys
    ):
        raise TechnicalBoundary("arm history/key finalization count differs")

    overlay = payload.get("overlay")
    overlay_write_count: int | None = None
    shadow_materialization_count: int | None = None
    if overlay is not None:
        if not isinstance(overlay, Mapping):
            raise TechnicalBoundary("arm overlay receipt shape differs")
        overlay_write_count = int(overlay.get("physical_terminal_write_count", -1))
        shadow_materialization_count = int(overlay.get("shadow_materialization_count", -1))
        if (
            overlay_write_count != endpoint_write_count
            or shadow_materialization_count < endpoint_write_count
        ):
            raise TechnicalBoundary("overlay/write/materialization counts differ")

    if arm is ArmId.OFFICIAL:
        official_endpoint_expected = int(str(payload.get("status")) != "ENTRY_ALREADY_HIT")
        if (
            int(ledger.get("official_endpoint_count", -1)) != official_endpoint_expected
            or int(ledger.get("solve_count", -1)) != 0
            or int(ledger.get("layer_factorization_count", -1)) != 0
            or int(ledger.get("terminal_finalize_count", -1)) != 0
            or overlay is not None
        ):
            raise TechnicalBoundary("Official adapter accounting differs")
        solve_reconciliation: dict[str, Any] = {
            "status": "STOCK_SOURCE_LEVEL_LOGICAL_COUNT_NOT_INTERCEPTED",
            "expected_layer_solve_count": 0 if official_endpoint_expected == 0 else 5,
            "adapter_intercepted_solve_count": 0,
            "actual_torch_solve_invocation_count": "NOT_INSTRUMENTED_STOCK_SOURCE",
        }
    else:
        factor_build_count = int(telemetry.get("factor_build_count", -1))
        solve_count = int(ledger.get("solve_count", -1))
        key_count = int(ledger.get("key_capture_count", -1))
        factorization_count = int(ledger.get("layer_factorization_count", -1))
        overlay_factor_count = int(
            cast(Mapping[str, Any], overlay).get("factor_count", -1)
        ) if isinstance(overlay, Mapping) else -1
        nonzero_visit_count = int(telemetry.get("nonzero_action_visit_count", -1))
        if not (
            solve_count
            == key_count
            == factorization_count
            == factor_build_count
            and overlay_factor_count == nonzero_visit_count
            and int(ledger.get("official_endpoint_count", -1)) == 0
        ):
            raise TechnicalBoundary("dynamic build/key/solve/action accounting differs")
        primary_finalize_count = endpoint_write_count
        derived = payload.get("derived_endpoint")
        derived_temporary_finalize_count = 0
        derived_status = None
        if isinstance(derived, Mapping):
            derived_status = str(derived.get("status"))
            derived_endpoint = derived.get("endpoint")
            if isinstance(derived_endpoint, Mapping):
                derived_temporary_finalize_count = int(
                    derived_endpoint.get("temporary_observation_materialization_count", 0)
                )
        observed_finalize_count = int(ledger.get("terminal_finalize_count", -1))
        if derived_status != "ORBHit_WITHHELD_TECHNICAL" and observed_finalize_count != (
            primary_finalize_count + derived_temporary_finalize_count
        ):
            raise TechnicalBoundary("dynamic terminal/derived finalize accounting differs")
        solve_reconciliation = {
            "status": "EXACT_ADAPTER_INTERCEPTED",
            "factor_build_count": factor_build_count,
            "key_capture_count": key_count,
            "solve_count": solve_count,
            "nonzero_action_visit_count": nonzero_visit_count,
            "overlay_factor_count": overlay_factor_count,
        }

    return {
        "status": "EXACT_CROSS_LAYER_RECONCILIATION_PASS",
        "method_state_kind": (
            "ALPHA_CACHE_C_CONTENT_SHA256" if family == "AlphaEdit"
            else "MEMIT_STATIC_COV_CACHE_CONTENT_SHA256"
        ),
        "method_state_entry_sha256": method_state_entry_sha256,
        "method_state_after_sha256": method_state_after_sha256,
        "method_state_content_identity_equal": True,
        "endpoint_evaluator_count": endpoint_evaluator_count,
        "primary_endpoint_count": 1,
        "derived_endpoint_evaluator_count": endpoint_evaluator_count - 1,
        "solve": solve_reconciliation,
        "endpoint_physical_write_count": endpoint_write_count,
        "authoritative_terminal_commit_count": endpoint_write_count,
        "overlay_physical_terminal_write_count": overlay_write_count,
        "shadow_materialization_count": shadow_materialization_count,
        "history_append_count": history_append_count,
        "terminal_history_key_capture_count": terminal_history_key_count,
        "model_forward_invocation_count": model_forward_count,
        "jvp_model_forward_invocation_count": jvp_forward_count,
        "jvp_call_count": jvp_call_count,
    }


def _arm_dtype_scope(*, family: str, arm: ArmId) -> dict[str, Any]:
    if family not in {"MEMIT", "AlphaEdit"}:
        raise TechnicalBoundary("dtype writer family differs")
    official_memit_fp64_exception = arm is ArmId.OFFICIAL and family == "MEMIT"
    return {
        "model_parameters_and_storage": "torch.float32",
        "orbode_dynamic_writer_solve_and_controller": "torch.float32",
        "autocast_tf32_bf16_fp16_quantization": "DISABLED",
        "official_native_ephemeral_solve_dtype": (
            "torch.float64" if official_memit_fp64_exception else "torch.float32"
        ),
        "official_memit_ephemeral_fp64_exception": official_memit_fp64_exception,
        "model_storage_full_fp32": True,
        "orbode_dynamic_path_full_fp32": True,
        "unqualified_all_algorithm_full_fp32": not official_memit_fp64_exception,
        "numeric_storage_cast_count": 0,
    }


def _arm_run(
    family: FamilyRuntime,
    arm: Any,
    *,
    preamble_stop_at_first_hit: bool = False,
) -> dict[str, Any]:
    seal_eager_attention(family.model)
    family.reset_entry()
    method_state_entry_sha256 = family.method_state_identity()
    endpoint_evaluation_count_before = family.endpoint_evaluation_count
    adapter = family.adapter(compute=False)
    overlay = GroupedFP32Overlay(family.model, adapter.weight_names_by_layer)
    observer = TerminalResponseObserver(
        model=family.model,
        overlay=overlay,
        capture_terminal_graph=family.terminal_graph,
    )
    integrator = OrderedResponseIntegrator(
        adapter=adapter,
        overlay=overlay,
        jvp=observer,
        observe_semantic=family.observe_semantic,
        observe_transition=(getattr(family, "cumulative_observer", None).transition
                            if getattr(family, "cumulative_observer", None) is not None else None),
    )
    started = time.perf_counter()
    forward_count_before = _model_forward_count(family.model)
    try:
        _sync()
        result = (
            integrator.run_official(arm, family.fixed_z)
            if arm.arm is ArmId.OFFICIAL
            else integrator.run_dynamic(
                arm,
                family.fixed_z,
                preamble_stop_at_first_hit=preamble_stop_at_first_hit,
            )
        )
        _sync()
        elapsed = time.perf_counter() - started
        adapter.assert_entry_method_state()
        method_state_after_sha256 = family.method_state_identity()
        if method_state_after_sha256 != method_state_entry_sha256:
            raise TechnicalBoundary("arm terminal cache/COV content identity differs from entry")
        payload = _serializable_result(result, adapter.ledger.raw_free_payload())
        payload["wall_seconds"] = elapsed
        payload["jvp_ledger"] = asdict(observer.ledger)
        payload["model_forward_invocation_count"] = (
            _model_forward_count(family.model) - forward_count_before
        )
        evaluation_count = family.endpoint_evaluation_count - endpoint_evaluation_count_before
        payload["accounting_reconciliation"] = _reconcile_arm_accounting(
            payload,
            family=family.family,
            arm=arm.arm,
            endpoint_evaluator_count=evaluation_count,
            method_state_entry_sha256=method_state_entry_sha256,
            method_state_after_sha256=method_state_after_sha256,
        )
        payload["dtype_scope"] = _arm_dtype_scope(
            family=family.family, arm=arm.arm
        )
        return payload
    finally:
        family.reset_entry()
        seal_eager_attention(family.model)


_FD_EPSILONS = (2.0 ** -7, 2.0 ** -8, 2.0 ** -9)
_FD_PRIMARY_EPSILON = 2.0 ** -8
_FD_RELATIVE_TOLERANCE = 2.0 ** -5


def _fd_absolute_tolerance(primal: torch.Tensor, epsilon: float) -> float:
    scale = max(1.0, float(primal.detach().abs().max().item()))
    return 64.0 * torch.finfo(torch.float32).eps * scale / epsilon


def _physical_factor_fd(
    family: FamilyRuntime,
    overlay: GroupedFP32Overlay,
    build: LayerBuild,
    epsilon: float,
) -> torch.Tensor:
    """Preamble-only central FD through temporary physical factor writes."""

    parameter = family.parameters[build.weight_name]
    entry = parameter.detach().clone()
    left = build.left.to(device=parameter.device, dtype=torch.float32)
    right = build.right.to(device=parameter.device, dtype=torch.float32)
    plus: torch.Tensor | None = None
    minus: torch.Tensor | None = None
    with overlay.suspend(authoritative=False):
        try:
            with torch.no_grad():
                parameter.copy_(entry)
                parameter.addmm_(left, right.T, alpha=float(epsilon))
            plus = family.terminal_graph().detach().to(device="cpu", dtype=torch.float32)
            with torch.no_grad():
                parameter.copy_(entry)
                parameter.addmm_(left, right.T, alpha=-float(epsilon))
            minus = family.terminal_graph().detach().to(device="cpu", dtype=torch.float32)
        finally:
            with torch.no_grad():
                parameter.copy_(entry)
    assert plus is not None and minus is not None
    response = ((plus - minus) / (2.0 * epsilon)).contiguous()
    if not bool(torch.isfinite(response).all()):
        raise TechnicalBoundary("materialized factor FD is non-finite")
    return response


def _response_identity(
    reference: torch.Tensor,
    observed: torch.Tensor,
    *,
    absolute_tolerance: float,
) -> dict[str, Any]:
    left = reference.detach().double().flatten()
    right = observed.detach().double().flatten()
    if left.shape != right.shape or not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise TechnicalBoundary("response identity operands differ")
    left_norm = float(torch.linalg.vector_norm(left).item())
    right_norm = float(torch.linalg.vector_norm(right).item())
    relative_l2 = float(
        torch.linalg.vector_norm(left - right).item()
        / max(left_norm, right_norm, torch.finfo(torch.float64).tiny)
    )
    cosine = float(
        torch.dot(left, right).item()
        / max(left_norm * right_norm, torch.finfo(torch.float64).tiny)
    )
    maximum_absolute = float((left - right).abs().max().item()) if left.numel() else 0.0
    rounding_envelope = absolute_tolerance * math.sqrt(max(1, left.numel()))
    both_inactive = left_norm <= rounding_envelope and right_norm <= rounding_envelope
    sign_active = torch.maximum(left.abs(), right.abs()) > absolute_tolerance
    sign_agreement = float(
        (
            torch.sign(left[sign_active]) == torch.sign(right[sign_active])
        ).double().mean().item()
    ) if bool(sign_active.any()) else 1.0
    allclose = bool(torch.allclose(
        reference,
        observed,
        atol=absolute_tolerance,
        rtol=_FD_RELATIVE_TOLERANCE,
    ))
    if both_inactive:
        cosine = 1.0
        sign_agreement = 1.0
    receipt = {
        "maximum_absolute_error": maximum_absolute,
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "reference_l2_norm": left_norm,
        "observed_l2_norm": right_norm,
        "active_element_sign_agreement_fraction": sign_agreement,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": _FD_RELATIVE_TOLERANCE,
        "rounding_l2_envelope": rounding_envelope,
        "numerical_activity_status": (
            "ALL_ZERO_WITHIN_NUMERICAL_ENVELOPE_OBSERVED"
            if both_inactive
            else "NUMERICALLY_ACTIVE_RESPONSE"
        ),
        "allclose": allclose,
    }
    if not allclose or (
        not both_inactive
        and (not math.isfinite(cosine) or cosine <= 0.0 or sign_agreement != 1.0)
    ):
        raise TechnicalBoundary(f"virtual/materialized response identity failed: {receipt}")
    return receipt


def _changed_state_rebuild_observation(
    *,
    entry_terminal: torch.Tensor,
    changed_terminal: torch.Tensor,
    entry_keys_sha256: str,
    changed_build: LayerBuild,
) -> dict[str, Any]:
    """Bind a versioned rebuild without requiring a nonzero scientific effect."""

    if changed_build.built_state_version != 1:
        raise TechnicalBoundary("L5 rebuild did not consume the advanced state version")
    return {
        "terminal_changed_observed": (
            tensor_sha256(changed_terminal) != tensor_sha256(entry_terminal)
        ),
        "l5_keys_changed_observed": changed_build.keys_sha256 != entry_keys_sha256,
        "state_effect_comparison_policy": "NONBLOCKING_TELEMETRY_ONLY",
        "state_effect_comparison_gate_count": 0,
    }


def _stock_official_parity_receipt(
    wrapper_endpoint: Mapping[str, Any],
    direct_endpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the guarded O wrapper with the literal pinned stock endpoint.

    This is the authoritative stock ``R/n`` parity gate.  A dynamic QCL
    one-sweep execution is deliberately not used as the oracle: its
    factorized FP32 solve/GEMM association is mathematically equivalent but
    is not required to be byte-identical to the stock dense-RHS solve.
    """

    fields: dict[str, tuple[Any, Any, bool]] = {}
    for name in ("selected_weight_endpoint_sha256", "terminal_activation_sha256"):
        left = wrapper_endpoint.get(name)
        right = direct_endpoint.get(name)
        present = (
            name in wrapper_endpoint
            and name in direct_endpoint
            and isinstance(left, str)
            and isinstance(right, str)
            and len(left) == 64
            and len(right) == 64
        )
        fields[name] = (left, right, present)
    for name in ("semantic_observation", "evaluation"):
        present = name in wrapper_endpoint and name in direct_endpoint
        fields[f"{name}_sha256"] = (
            canonical_hash(wrapper_endpoint.get(name)),
            canonical_hash(direct_endpoint.get(name)),
            present,
        )
    receipt: dict[str, Any] = {
        "mode": "PINNED_STOCK_R_OVER_N_WRAPPER_VS_DIRECT",
        "dynamic_qcl_one_pass_used_as_stock_oracle": False,
    }
    for name, (wrapper_value, direct_value, present) in fields.items():
        receipt[f"wrapper_{name}"] = wrapper_value
        receipt[f"direct_{name}"] = direct_value
        receipt[f"{name}_present"] = present
        receipt[f"{name}_equal"] = present and wrapper_value == direct_value
    receipt["exact_parity"] = all(
        bool(value) for key, value in receipt.items() if key.endswith("_equal")
    )
    return receipt


def _single_request_preamble(
    *,
    family: FamilyRuntime,
    raw_request: Mapping[str, Any],
    endpoint_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Run P0–P3 on the sealed first request without recomputing native z."""

    if family.fixed_z is None:
        raise TechnicalBoundary("runtime preamble requires the cohort fixed-z artifact")
    request = family.requests[0]
    request_order_sha256 = canonical_hash([str(raw_request["request_sha256"])])
    inventory = build_compute_z_semantic_inventory(
        tokenizer=family.tokenizer,
        requests=[request],
        context_templates=family.contexts,
        request_order_sha256=request_order_sha256,
    )
    first_z = family.fixed_z.values[:, :1].detach().clone().contiguous()
    fixed_z = FixedZArtifact(
        values=first_z,
        identity_sha256=tensor_sha256(first_z),
        request_order_sha256=request_order_sha256,
        target_context_identity_sha256=inventory.identity_sha256,
    )
    single = FamilyRuntime(
        model=family.model,
        tokenizer=family.tokenizer,
        family=family.family,
        hparams=family.hparams,
        module=family.module,
        requests=(request,),
        endpoint_records=(endpoint_record,),
        request_order_sha256=request_order_sha256,
        contexts=family.contexts,
    )
    # The preamble is a view over the already sealed family state.  Re-running
    # preparation would replace AlphaEdit cache_c storage and invalidate the
    # cohort-level cache identity rather than testing it.
    if family._prepared_method_state_identity is None:
        raise TechnicalBoundary("runtime preamble received unprepared method state")
    single._alpha_cache_entry_is_zero = family._alpha_cache_entry_is_zero
    single._alpha_cache_entry = (
        None
        if family._alpha_cache_entry is None
        else family._alpha_cache_entry.detach().clone()
    )
    single._cov_versions = dict(family._cov_versions)
    single._prepared_method_state_identity = family._prepared_method_state_identity
    single.semantic_inventory = inventory
    single.fixed_z = fixed_z
    single.reset_entry()
    seal_eager_attention(single.model)

    # P0: wrapper/direct stock parity and fixed-state residual scaling.
    wrapper = _arm_run(single, canonical_arm_configs(sweeps=4)[0])
    single.reset_entry()
    if wrapper["status"] == "ENTRY_ALREADY_HIT":
        direct = dict(wrapper["endpoint"])
        direct["direct_status"] = "NOT_RUN_ENTRY_ALREADY_HIT_CONTRACT"
        if any(
            int(wrapper["telemetry"].get(field, -1)) != 0
            for field in ("factor_build_count", "physical_write_count", "history_append_count")
        ):
            raise TechnicalBoundary("Official entry-hit path performed writer work")
    else:
        direct = dict(single.run_official(fixed_z=fixed_z))
    single.reset_entry()
    wrapper_endpoint = dict(wrapper["endpoint"])
    official_parity = _stock_official_parity_receipt(wrapper_endpoint, direct)
    if not official_parity["exact_parity"]:
        raise TechnicalBoundary(
            f"Official wrapper/direct endpoint parity failed: {official_parity}"
        )
    terminal0 = single.terminal()
    full_build = single.build_layer(
        layer=4,
        current_terminal=terminal0,
        fixed_z=fixed_z,
        residual_denominator=1,
        state_version=0,
    )
    divided_build = single.build_layer(
        layer=4,
        current_terminal=terminal0,
        fixed_z=fixed_z,
        residual_denominator=5,
        state_version=0,
    )
    # build_layer divides R on the writer device, then stores the detached
    # factor on CPU.  Replaying `/ 5` on CPU compares different FP32 backends
    # and can differ by one ULP even when the production scaling is exact.
    scaling_expected = _writer_device_residual_division_replay(
        full_build.left,
        divisor=5,
        device=single.device,
    )
    scaling_error = float((divided_build.left - scaling_expected).abs().max().item())
    scaling_reference = float(scaling_expected.abs().max().item())
    scaling_relative_error = scaling_error / max(
        scaling_reference, torch.finfo(torch.float32).tiny
    )
    right_factor_bitwise_identity = torch.equal(divided_build.right, full_build.right)
    if not torch.equal(divided_build.left, scaling_expected) or not right_factor_bitwise_identity:
        raise TechnicalBoundary(
            "fixed-state B(R/n)=B(R)/n scaling parity failed: "
            f"writer_device={single.device}, max_abs={scaling_error}, "
            f"max_relative={scaling_relative_error}, "
            f"right_factor_bitwise_identity={right_factor_bitwise_identity}"
        )

    # P1: first nonzero response layer, three locked FD epsilons, and a
    # materialized-factor response identity at the primary epsilon.
    adapter = single.adapter(compute=False)
    overlay = GroupedFP32Overlay(single.model, adapter.weight_names_by_layer)
    observer = TerminalResponseObserver(
        model=single.model,
        overlay=overlay,
        capture_terminal_graph=single.terminal_graph,
    )
    selected_build: LayerBuild | None = None
    selected_observation: Any | None = None
    response_selection_status = "ALL_ZERO_WITHIN_NUMERICAL_ENVELOPE_OBSERVED"
    zero_response_candidates: list[dict[str, Any]] = []
    with overlay:
        adapter.bind_state_version(0)
        terminal = adapter.current_terminal(state_version=0)
        for position, layer in enumerate(adapter.layers):
            build = adapter.build_layer(
                layer=layer,
                residual_denominator=len(adapter.layers) - position,
                state_version=0,
                current_terminal=terminal,
                fixed_z=fixed_z,
            )
            observation = observer.observe(build, expected_state_version=0)
            if selected_build is None:
                # Deterministic layer-order fallback for a valid all-zero
                # response.  A later numerically active layer supersedes it.
                selected_build = build
                selected_observation = observation
            primary_atol = _fd_absolute_tolerance(observation.terminal, _FD_PRIMARY_EPSILON)
            direction_norm = float(torch.linalg.vector_norm(observation.response.double()).item())
            rounding_envelope = primary_atol * math.sqrt(max(1, observation.response.numel()))
            if direction_norm > rounding_envelope:
                selected_build = build
                selected_observation = observation
                response_selection_status = "FIRST_NUMERICALLY_ACTIVE_RESPONSE"
                break
            zero_response_candidates.append({
                "layer": layer,
                "direction_l2_norm": direction_norm,
                "rounding_l2_envelope": rounding_envelope,
            })
        if selected_build is None or selected_observation is None:
            raise TechnicalBoundary("P1 response layer inventory is empty")
        fd_receipts = []
        for epsilon in _FD_EPSILONS:
            absolute_tolerance = _fd_absolute_tolerance(selected_observation.terminal, epsilon)
            fd_receipts.append(asdict(observer.audit_central_difference(
                selected_build,
                expected_state_version=0,
                epsilon=epsilon,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=_FD_RELATIVE_TOLERANCE,
            )))
        materialized = _physical_factor_fd(
            single, overlay, selected_build, _FD_PRIMARY_EPSILON
        )
        virtual_materialized = _response_identity(
            selected_observation.response,
            materialized,
            absolute_tolerance=_fd_absolute_tolerance(
                selected_observation.terminal, _FD_PRIMARY_EPSILON
            ),
        )
    adapter.assert_entry_method_state()
    single.reset_entry()

    # P2: changed-state layer rebuild, exact-version consumption, repeated
    # same-weight factors, and virtual/shadow terminal + semantic parity.
    adapter2 = single.adapter(compute=False)
    overlay2 = GroupedFP32Overlay(single.model, adapter2.weight_names_by_layer)
    with overlay2:
        adapter2.bind_state_version(0)
        entry_terminal = adapter2.current_terminal(state_version=0)
        entry_l5 = adapter2.build_layer(
            layer=5,
            residual_denominator=1,
            state_version=0,
            current_terminal=entry_terminal,
            fixed_z=fixed_z,
        )
        l4 = adapter2.build_layer(
            layer=4,
            residual_denominator=1,
            state_version=0,
            current_terminal=entry_terminal,
            fixed_z=fixed_z,
        )
        overlay2.append(l4.overlay_delta(0.25))
        adapter2.bind_state_version(overlay2.state_version)
        changed_terminal = adapter2.current_terminal(state_version=overlay2.state_version)
        changed_l5 = adapter2.build_layer(
            layer=5,
            residual_denominator=1,
            state_version=overlay2.state_version,
            current_terminal=changed_terminal,
            fixed_z=fixed_z,
        )
        changed_state_observation = _changed_state_rebuild_observation(
            entry_terminal=entry_terminal,
            changed_terminal=changed_terminal,
            entry_keys_sha256=entry_l5.keys_sha256,
            changed_build=changed_l5,
        )
        overlay2.append(changed_l5.overlay_delta(0.25))
        adapter2.bind_state_version(overlay2.state_version)
        repeat_terminal = adapter2.current_terminal(state_version=overlay2.state_version)
        repeat_l4 = adapter2.build_layer(
            layer=4,
            residual_denominator=1,
            state_version=overlay2.state_version,
            current_terminal=repeat_terminal,
            fixed_z=fixed_z,
        )
        overlay2.append(repeat_l4.overlay_delta(0.25))
        adapter2.bind_state_version(overlay2.state_version)
        shadow = overlay2.materialize_shadow()
        virtual_terminal = adapter2.current_terminal(state_version=overlay2.state_version)
        virtual_semantic = single.observe_semantic()
        try:
            with overlay2.suspend(authoritative=False):
                try:
                    single._apply_shadow(shadow)
                    physical_terminal = single.terminal()
                    physical_semantic = single.observe_semantic()
                finally:
                    _restore_selected(single.parameters, single.w0)
        finally:
            adapter2.bind_state_version(overlay2.state_version)
        terminal_atol = 64.0 * torch.finfo(torch.float32).eps * max(
            1.0, float(virtual_terminal.abs().max().item())
        )
        terminal_identity = {
            "maximum_absolute_error": float((virtual_terminal - physical_terminal).abs().max().item()),
            "absolute_tolerance": terminal_atol,
            "relative_tolerance": _FD_RELATIVE_TOLERANCE,
            "allclose": bool(torch.allclose(
                virtual_terminal,
                physical_terminal,
                atol=terminal_atol,
                rtol=_FD_RELATIVE_TOLERANCE,
            )),
        }
        semantic_identity = {
            "strict_identity": (
                virtual_semantic.request_strict == physical_semantic.request_strict
                and virtual_semantic.strict_event_count == physical_semantic.strict_event_count
                and virtual_semantic.strict_tie_count == physical_semantic.strict_tie_count
            ),
            "target_logit_mean_absolute_error": abs(
                virtual_semantic.target_logit_mean - physical_semantic.target_logit_mean
            ),
            "maximum_other_logit_mean_absolute_error": abs(
                virtual_semantic.maximum_other_logit_mean - physical_semantic.maximum_other_logit_mean
            ),
        }
        if (
            not terminal_identity["allclose"]
            or not semantic_identity["strict_identity"]
            or semantic_identity["target_logit_mean_absolute_error"] > terminal_atol
            or semantic_identity["maximum_other_logit_mean_absolute_error"] > terminal_atol
        ):
            raise TechnicalBoundary("virtual/shadow activation or semantic-logit parity failed")
    adapter2.assert_entry_method_state()
    single.reset_entry()

    # P3 is observation-only: finite endpoints and predeclared step defects at
    # N={2,4,8}.  It does not choose N or impose an outcome/Cauchy threshold.
    resolutions: list[dict[str, Any]] = []
    horizon_n4: dict[str, Any] | None = None
    for sweeps in (2, 4, 8):
        run = _arm_run(single, canonical_arm_configs(sweeps=sweeps)[3])
        if sweeps == 4:
            horizon_n4 = run
        steps = list(run["telemetry"].get("steps", []))
        if any(
            not math.isfinite(float(step[field]))
            for step in steps
            for field in ("potential_before", "potential_after", "discretization_defect")
        ):
            raise NumericalMethodBoundary("P3 resolution smoke produced a non-finite defect")
        resolutions.append({
            "N": sweeps,
            "h": 1.0 / sweeps,
            "T": 1.0,
            "status": run["status"],
            "selected_weight_endpoint_sha256": run["endpoint"].get(
                "selected_weight_endpoint_sha256"
            ),
            "step_count": len(steps),
            "terminal_potential": None if not steps else steps[-1]["potential_after"],
            "maximum_discretization_defect": max(
                (float(step["discretization_defect"]) for step in steps), default=0.0
            ),
            "sum_discretization_defect": sum(
                float(step["discretization_defect"]) for step in steps
            ),
            "selection_influence_count": 0,
        })
    assert horizon_n4 is not None
    derived_first_hit = horizon_n4.get("derived_endpoint")
    if not isinstance(derived_first_hit, Mapping):
        raise TechnicalBoundary("P3 horizon run omitted derived ORBHit endpoint")
    try:
        direct_first_hit = _arm_run(
            single,
            canonical_arm_configs(sweeps=4)[3],
            preamble_stop_at_first_hit=True,
        )
        direct_endpoint = dict(direct_first_hit["endpoint"])
        derived_endpoint = dict(derived_first_hit.get("endpoint", {}))
        first_hit_identity = {
            "status": "ORBHit_IDENTITY_PASS",
            "orbfh_primary_gate_influence_count": 0,
            "direct_status": direct_first_hit["status"],
            "derived_status": derived_first_hit.get("status"),
            "selected_weight_endpoint_sha256_equal": (
                direct_endpoint.get("selected_weight_endpoint_sha256")
                == derived_endpoint.get("selected_weight_endpoint_sha256")
            ),
            "terminal_activation_sha256_equal": (
                direct_endpoint.get("terminal_activation_sha256")
                == derived_endpoint.get("terminal_activation_sha256")
            ),
            "semantic_observation_sha256_equal": (
                canonical_hash(direct_endpoint.get("semantic_observation"))
                == canonical_hash(derived_endpoint.get("semantic_observation"))
            ),
            "evaluation_sha256_equal": (
                canonical_hash(direct_endpoint.get("evaluation"))
                == canonical_hash(derived_endpoint.get("evaluation"))
            ),
        }
        if not all(
            bool(value)
            for key, value in first_hit_identity.items()
            if key.endswith("_equal")
        ):
            first_hit_identity["status"] = "ORBHit_WITHHELD_TECHNICAL_IDENTITY_MISMATCH"
    except TechnicalBoundary as exc:
        first_hit_identity = {
            "status": "ORBHit_WITHHELD_TECHNICAL_DIRECT_AUDIT",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "orbfh_primary_gate_influence_count": 0,
        }
    single.reset_entry()
    family.reset_entry()
    seal_eager_attention(family.model)
    return {
        "schema": "orbode.fast-runtime-preamble.v1",
        "status": "FAST_RUNTIME_PREAMBLE_PASS",
        "request_case_id": str(request["case_id"]),
        "request_sha256": str(raw_request["request_sha256"]),
        "fixed_z_source": "COHORT_FIXED_Z_COLUMN_0_NO_RECOMPUTE",
        "fixed_z_request_consumption_count": 1,
        "fixed_z_recompute_count": 0,
        "P0_official_scaling": {
            "stock_official_parity": official_parity,
            "dynamic_qcl_one_pass_stock_parity_claim": "NOT_APPLICABLE_DISTINCT_NUMERICAL_PATH",
            "residual_scaling_replay": "WRITER_DEVICE_FP32_DIVISION_THEN_CPU_STORAGE",
            "residual_scaling_replay_device": str(single.device),
            "residual_scaling_max_abs_error": scaling_error,
            "residual_scaling_max_relative_error": scaling_relative_error,
            "right_factor_bitwise_identity": right_factor_bitwise_identity,
        },
        "outcome_comparison_gate_policy": {
            "schema": "orbode.nonblocking-outcome-comparison-policy.v1",
            "classification": "NONBLOCKING_TELEMETRY_ONLY",
            "blocking_outcome_comparison_count": 0,
            "ours_vs_official_gate": False,
            "ours_vs_ours_gate": False,
            "resolution_match_monotonicity_convergence_gate": False,
            "action_magnitude_match_gate": False,
            "zero_correction_or_official_path_gate": False,
            "official_nonworse_outcome_gate": False,
            "stock_official_o_wrapper_direct_fidelity_gate": True,
            "technical_integrity_gates_retained": True,
            "scientific_selection_influence_count": 0,
        },
        "P1_jvp": {
            "selected_layer": selected_build.layer,
            "response_selection_status": response_selection_status,
            "epsilon_grid": list(_FD_EPSILONS),
            "primary_epsilon": _FD_PRIMARY_EPSILON,
            "absolute_tolerance_formula": (
                "64*eps_float32*max(1,max_abs(primal))/epsilon"
            ),
            "relative_tolerance": _FD_RELATIVE_TOLERANCE,
            "zero_response_candidates": zero_response_candidates,
            "finite_difference": fd_receipts,
            "virtual_materialized": virtual_materialized,
            "ledger": asdict(observer.ledger),
        },
        "P2_state_transaction": {
            "entry_l5_keys_sha256": entry_l5.keys_sha256,
            "changed_l5_keys_sha256": changed_l5.keys_sha256,
            "changed_l5_state_version": changed_l5.built_state_version,
            **changed_state_observation,
            "same_layer_repeated_factor_count": 2,
            "overlay": overlay2.receipt(),
            "terminal_identity": terminal_identity,
            "semantic_logit_identity": semantic_identity,
            "inner_persistent_mutation_count": 0,
            "algorithmic_retry_count": 0,
            "algorithmic_rollback_count": 0,
        },
        "P3_resolution_observation_only": resolutions,
        "P3_orbhit_direct_prefix_identity": first_hit_identity,
        "w0_pointer_bytes_restore_pass": (
            tensor_set_sha256(single.parameters) == single.w0_sha256
            and tensor_set_sha256(family.parameters) == family.w0_sha256
        ),
        "scientific_selection_influence_count": 0,
    }


def _load_stream(
    repo_root: Path,
    dataset: Path,
    round_indices: Sequence[int],
    *,
    wave: str,
) -> tuple[Any, tuple[tuple[dict[str, Any], ...], ...], dict[int, dict[str, Any]]]:
    from project.run_scripts.ode_bf.p1r52_b100x10_stream import load_p1r52_b100x10_batches, verify_p1r52_b100x10_stream

    seal = verify_p1r52_b100x10_stream(json.loads((repo_root / STREAM_SEAL_RELATIVE).read_text(encoding="utf-8")))
    if seal["root_digest"] != STREAM_ROOT or seal["all_request_order_sha256"] != ORDER_ROOT:
        raise TechnicalBoundary("sealed stream identity differs")
    batches = load_p1r52_b100x10_batches(dataset, seal)
    chosen = tuple(batches[index] for index in round_indices)
    if wave == "b1":
        if tuple(round_indices) != (0,) or len(chosen) != 1 or len(chosen[0]) != 100:
            raise TechnicalBoundary("B1 canonical source cohort differs")
        chosen = (tuple(chosen[0][:1]),)
    case_ids = {int(item["case_id"]) for batch in chosen for item in batch}
    return seal, chosen, _raw_records(dataset, case_ids)


def run_cell(
    *,
    repo_root: Path,
    easyedit_source_root: Path,
    easyedit_artifact_root: Path,
    hf_hub_cache: Path,
    output_root: Path,
    source_head: str,
    source_tree: str,
    cell_id: int,
    wave: str,
    round_indices: tuple[int, ...],
    run_token: str,
) -> dict[str, Any]:
    """Execute one model×writer cell, preserving every cohort at independent W0."""

    if not isinstance(run_token, str) or not run_token:
        raise TechnicalBoundary("launcher run token is empty or invalid")
    started = time.perf_counter()
    root = output_root.absolute()
    if root.exists() or root.is_symlink():
        raise TechnicalBoundary(f"create-once task root exists: {root}")
    root.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    failure_path = root / "failure-boundary.json"
    model: torch.nn.Module | None = None
    model_forward_handle: Any | None = None
    family_runtime: FamilyRuntime | None = None
    action_counts = {"model_load": 0, "gpu_job": 1, "slurm_submit": 0}
    progress: dict[str, Any] = {
        "stage": "TASK_ROOT_CREATED",
        "round_index": None,
        "arm": None,
        "completed_rounds": [],
        "completed_primary_arms_current_round": [],
        "official_reference_parity_pass": False,
    }
    try:
        cell = _cell(cell_id)
        if tuple(round_indices) != wave_rounds(wave):
            raise TechnicalBoundary("wave/round mapping differs")
        if _git(repo_root, "rev-parse", "HEAD") != source_head or _git(repo_root, "rev-parse", "HEAD^{tree}") != source_tree:
            raise TechnicalBoundary("queued source HEAD/tree drift")
        easyedit = _bootstrap_easyedit(easyedit_source_root)
        if easyedit["head"] != "14cea8245f06715684592ab55184939b99d70784" or not easyedit["tracked_clean"]:
            raise TechnicalBoundary("pinned stock EasyEdit identity differs")
        progress["stage"] = "SOURCE_BINDING_PASS"

        os.environ.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",
        })
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise TechnicalBoundary("runtime requires exactly one visible CUDA device")
        torch.cuda.set_device(0)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        snapshot = _model_snapshot(hf_hub_cache, cell.model_alias)
        _sync()
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map={"": "cuda:0"},
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.eval()
        attention_receipt = dict(seal_eager_attention(model))
        snapshot_config = (snapshot / "config.json").resolve(strict=True)
        config_stat = snapshot_config.stat()
        if not stat.S_ISREG(config_stat.st_mode):
            raise TechnicalBoundary("HF model config is not a regular file")
        attention_receipt.update({
            "hf_config_path": str(snapshot_config),
            "hf_config_sha256": hashlib.sha256(snapshot_config.read_bytes()).hexdigest(),
            "hf_config_bytes": int(config_stat.st_size),
        })
        model_forward_handle = _install_model_forward_counter(model)
        _sync()
        action_counts["model_load"] = 1
        load_seconds = time.perf_counter() - load_started
        dtype_receipt = assert_full_fp32(model)
        progress["stage"] = "MODEL_LOAD_FP32_ATTENTION_PASS"
        runtime_lock = asdict(ORBODERuntimeLock())

        hparams, hparams_path = _load_hparams(repo_root, cell.writer_family, cell.model_alias)
        hparams.device = 0
        # Keep the short, pinned hparams model name (for example
        # ``Meta-Llama-3-8B-Instruct``).  Stock get_cov maps that value to the
        # on-disk statistics basename.  The distinct HF repository/revision is
        # already sealed by ``snapshot`` and must not be substituted here.
        hparams.stats_dir = str(easyedit_artifact_root / "examples/data/stats")
        if cell.writer_family == "AlphaEdit":
            hparams.P_loc = str(easyedit_artifact_root / str(MODEL_BINDINGS[cell.model_alias]["projector"][0]))
        if tuple(int(item) for item in hparams.layers) != (4, 5, 6, 7, 8):
            raise TechnicalBoundary("writer layer inventory differs")
        module = _method_module(cell.writer_family)
        module.CONTEXT_TEMPLATES_CACHE = None
        with _model_name(model, str(hparams.model_name)):
            contexts = module.get_context_templates(model, tokenizer)
        if not isinstance(contexts, list) or not contexts:
            raise TechnicalBoundary("stock compute-z context builder failed")

        dataset = (easyedit_artifact_root / "data/counterfact/counterfact.json").resolve(strict=True)
        seal, batches, raw_map = _load_stream(
            repo_root, dataset, round_indices, wave=wave
        )
        round_payloads: list[dict[str, Any]] = []
        round_publication_receipts: list[dict[str, Any]] = []
        fixed_z_total = 0
        fixed_z_recompute = 0
        for round_index, batch in zip(round_indices, batches, strict=True):
            round_forward_count_before = _model_forward_count(model)
            progress.update(
                stage="ROUND_PREPARE",
                round_index=int(round_index),
                arm=None,
                completed_primary_arms_current_round=[],
            )
            requests = _official_requests(batch)
            request_order_sha256 = _request_order_identity(batch)
            expected_order = seal["batch_ordered_request_digest_v1"][round_index]
            # The seal stores both a method-independent ordered digest and the
            # canonical hash; retain both, but never reorder the loaded batch.
            endpoint_records = tuple(raw_map[int(item["case_id"])] for item in batch)
            family_runtime = FamilyRuntime(
                model=model,
                tokenizer=tokenizer,
                family=cell.writer_family,
                hparams=hparams,
                module=module,
                requests=requests,
                endpoint_records=endpoint_records,
                request_order_sha256=request_order_sha256,
                contexts=contexts,
            )
            family_runtime.prepare_method_state()
            family_runtime.reset_entry()
            z_adapter = family_runtime.adapter(compute=True)
            fixed_z = z_adapter.compute_fixed_z_once()
            fixed_z_total += fixed_z.request_count
            fixed_z_recompute += z_adapter.ledger.fixed_z_recompute_count
            family_runtime.pre_evaluation = family_runtime.evaluate_endpoint()
            runtime_preamble = None
            if wave in {"b1", "round0"} and round_index == 0:
                progress["stage"] = "B1_RUNTIME_PREAMBLE"
                runtime_preamble = _single_request_preamble(
                    family=family_runtime,
                    raw_request=batch[0],
                    endpoint_record=endpoint_records[0],
                )
                progress["official_reference_parity_pass"] = bool(
                    runtime_preamble.get("P0_official_scaling")
                )
            arms: list[dict[str, Any]] = []
            for config in canonical_arm_configs(sweeps=4)[:5]:
                progress.update(stage="PRIMARY_ARM", arm=config.arm.value)
                try:
                    arms.append(_arm_run(family_runtime, config))
                except BaseException as exc:
                    inner = getattr(exc, "orbode_progress", None)
                    if isinstance(inner, Mapping):
                        progress["integrator"] = dict(inner)
                    raise
                progress["completed_primary_arms_current_round"].append(config.arm.value)
            observed_order = tuple(item["arm"] for item in arms)
            if observed_order != PRIMARY_ARM_ORDER:
                raise TechnicalBoundary("primary arm execution order differs")
            derived = arms[3].get("derived_endpoint")
            if not isinstance(derived, dict) or derived.get("arm") != "ORBHit":
                raise TechnicalBoundary("ORBFH-derived ORBHit endpoint is missing")
            raw_round_payload = {
                "round_index": round_index,
                "request_count": len(batch),
                "case_ids": [int(item["case_id"]) for item in batch],
                "request_sha256": [str(item["request_sha256"]) for item in batch],
                "canonical_request_order_sha256": request_order_sha256,
                "sealed_batch_order_digest": expected_order,
                "fixed_z": {
                    "identity_sha256": fixed_z.identity_sha256,
                    "target_context_identity_sha256": fixed_z.target_context_identity_sha256,
                    "compute_count": fixed_z.request_count,
                    "recompute_count": 0,
                },
                "semantic": {
                    "inventory_sha256": family_runtime.semantic_inventory.identity_sha256,
                    "event_count": family_runtime.semantic_inventory.event_count,
                    "request_count": family_runtime.semantic_inventory.request_count,
                },
                "pre_evaluation": family_runtime.pre_evaluation,
                "runtime_preamble": runtime_preamble,
                "arms": arms,
                "w0_sha256": family_runtime.w0_sha256,
                "w0_pointer_bytes_restore_pass": tensor_set_sha256(family_runtime.parameters) == family_runtime.w0_sha256,
                "model_forward_invocation_count": (
                    _model_forward_count(model) - round_forward_count_before
                ),
            }
            progress["stage"] = "ROUND_CREATE_ONCE_PUBLICATION"
            round_payload = reduce_round_payload(
                raw_round_payload, expected_request_count=len(batch)
            )
            validate_round_publication(
                round_payload,
                expected_round_index=int(round_index),
                expected_request_count=len(batch),
            )
            round_relative_path = f"round-{round_index:02d}-result.json"
            round_file_sha256 = _create_once_json(
                root / round_relative_path, round_payload
            )
            round_payloads.append(round_payload)
            round_publication_receipts.append(
                {
                    "round_index": int(round_index),
                    "relative_path": round_relative_path,
                    "file_sha256": round_file_sha256,
                    "identity_sha256": str(round_payload["identity_sha256"]),
                }
            )
            progress["completed_rounds"].append(int(round_index))
            progress.update(stage="ROUND_TERMINAL_VALID", arm=None)

        request_count = sum(int(payload["request_count"]) for payload in round_payloads)
        primary_endpoint_count = request_count * len(PRIMARY_ARM_ORDER)
        canonical_ns_accounting = _canonical_ns_accounting(round_payloads)
        entry_already_hit_count = sum(
            int(
                arm_payload.get("mechanism_telemetry", {})
                .get("telemetry", {})
                .get("entry_semantic", {})
                .get("request_strict_count", 0)
            )
            for round_payload in round_payloads
            for arm_payload in round_payload["primary_endpoints"]
        )
        if fixed_z_total != request_count or fixed_z_recompute != 0:
            raise TechnicalBoundary("fixed-z request accounting differs")
        result = {
            "schema": RESULT_SCHEMA,
            "status": "TERMINAL_VALID",
            "cell_id": cell_id,
            "wave": wave,
            "rounds": list(round_indices),
            "model_alias": cell.model_alias,
            "writer_family": cell.writer_family,
            "run_token": run_token,
            "source_head": source_head,
            "source_tree": source_tree,
            "easyedit": easyedit,
            "hparams_path": str(hparams_path),
            "stream_root": STREAM_ROOT,
            "order_root": ORDER_ROOT,
            "request_count": request_count,
            "primary_arms": list(PRIMARY_ARM_ORDER),
            "derived_arms": ["ORBHit"],
            "primary_endpoint_count": primary_endpoint_count,
            "scientific_attempted_count": primary_endpoint_count,
            "terminal_valid_count": primary_endpoint_count,
            "entry_already_hit_count": entry_already_hit_count,
            "fast_runtime_preamble_count": sum(
                int(round_payload["runtime_preamble"] is not None)
                for round_payload in round_payloads
            ),
            "fast_runtime_preamble_status": (
                "PASS_THIS_WAVE"
                if wave in {"b1", "round0"}
                else "INHERITED_ROUND0_COMMON_GATE"
            ),
            "fixed_z_compute_count": fixed_z_total,
            "fixed_z_recompute_count": fixed_z_recompute,
            "full_fp32": True,
            "full_fp32_claim_scope": "MODEL_STORAGE_AND_ORBODE_DYNAMIC_PATH",
            "official_native_memit_ephemeral_fp64_solve_exception": (
                cell.writer_family == "MEMIT"
            ),
            "all_algorithm_solves_full_fp32": cell.writer_family != "MEMIT",
            "unqualified_full_fp32_claim": cell.writer_family != "MEMIT",
            "dtype_receipt": dtype_receipt,
            "runtime_lock": runtime_lock,
            "attention_backend_receipt": attention_receipt,
            "parameter_inventory": _parameter_inventory(model),
            "w0_pointer_bytes_restore_pass": True,
            "cache_integrity_pass": True,
            "transaction_integrity_pass": True,
            "jvp_integrity_pass": True,
            "overlay_integrity_pass": True,
            "technical_failure_count": 0,
            "scientific_failure_count": 0,
            "imputation_count": 0,
            "round_results": round_payloads,
            "round_publication_receipts": round_publication_receipts,
            "round_publication_count": len(round_payloads),
            "round_publication_identity_root": canonical_hash(
                [str(item["identity_sha256"]) for item in round_payloads]
            ),
            "round_publication_file_sha256_root": canonical_hash(
                [str(item["file_sha256"]) for item in round_publication_receipts]
            ),
            "literal_prompt_target_token_prediction_publication_count": 0,
            "canonical_ns_accounting": dict(canonical_ns_accounting),
            "timing": {
                "model_load_seconds": load_seconds,
                "job_total_seconds": time.perf_counter() - started,
            },
            "memory": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
            "action_counts": action_counts,
            "compute": {
                "model_forward_invocation_count": _model_forward_count(model),
                "model_forward_counter_authority": "TOP_LEVEL_FORWARD_PRE_HOOK",
                "official_propose_ordered_internal_logical_capture_count": "NOT_RESOLVED",
            },
            "scientific_promotion": False,
        }
        result_sha = _create_once_json(root / "result.json", result)
        progress["stage"] = "CELL_RESULT_PUBLISHED"
        terminal = _terminal_receipt(
            cell_id=cell_id,
            wave=wave,
            round_indices=round_indices,
            model_alias=cell.model_alias,
            writer_family=cell.writer_family,
            run_token=run_token,
            request_count=request_count,
            primary_endpoint_count=primary_endpoint_count,
            entry_already_hit_count=entry_already_hit_count,
            fixed_z_compute_count=fixed_z_total,
            source_head=source_head,
            source_tree=source_tree,
            result_sha256=result_sha,
            round_publication_identities=[
                str(item["identity_sha256"]) for item in round_payloads
            ],
            round_publication_file_sha256=[
                str(item["file_sha256"]) for item in round_publication_receipts
            ],
            canonical_ns_accounting=canonical_ns_accounting,
        )
        _create_once_json(root / "terminal-receipt.json", terminal)
        progress["stage"] = "CELL_TERMINAL_RECEIPT_PUBLISHED"
        if model_forward_handle is not None:
            model_forward_handle.remove()
            model_forward_handle = None
        return terminal
    except BaseException as exc:
        inner_progress = getattr(exc, "orbode_progress", None)
        if isinstance(inner_progress, Mapping) and "integrator" not in progress:
            progress["integrator"] = dict(inner_progress)
        restore = False
        if family_runtime is not None:
            try:
                family_runtime.reset_entry()
                restore = tensor_set_sha256(family_runtime.parameters) == family_runtime.w0_sha256
            except BaseException:
                restore = False
        failure_status, nonfinite_boundary = _classify_failure_status(exc)
        reference_parity = bool(progress.get("official_reference_parity_pass", False))
        failure = {
            "schema": "orbode.failure-boundary.v1",
            "status": failure_status,
            "cell_id": cell_id,
            "wave": wave,
            "rounds": list(round_indices),
            "source_head": source_head,
            "source_tree": source_tree,
            "run_token": run_token,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "progress": progress,
            "progressive_round_receipt_count": len(progress.get("completed_rounds", [])),
            "nonfinite_detected": nonfinite_boundary,
            "official_reference_parity_exclusion_evidence": reference_parity,
            "nonfinite_scientific_promotion_allowed": False,
            "nonfinite_required_oracle_evidence": {
                "clean_replay": False,
                "reference_solve_and_jvp": False,
                "fp64_observer_exclusion": False,
            },
            "w0_pointer_bytes_restore_pass": restore,
            "science_change_count": 0,
            "tolerance_change_count": 0,
            "threshold_change_count": 0,
            "action_counts": action_counts,
            "elapsed_seconds": time.perf_counter() - started,
            "model_forward_invocation_count": (
                None if model is None or not hasattr(model, "_orbode_model_forward_invocation_count")
                else _model_forward_count(model)
            ),
        }
        try:
            _create_once_json(failure_path, failure)
        except BaseException:
            pass
        if model_forward_handle is not None:
            model_forward_handle.remove()
        raise


__all__ = ["run_cell"]
