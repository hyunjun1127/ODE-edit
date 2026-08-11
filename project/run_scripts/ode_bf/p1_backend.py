"""Sequential P1 Native capture, mixed64 fields, covariance, and gradients."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import random
import struct
import threading
import time
import zipfile
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional

from .accounting import ComputeLedger, JointInitializationReceipt, LayerFactorReceipt
from .alpha_backend import (
    ALPHA_SOLVE_DTYPE,
    ALPHA_SOLVE_REFERENCE,
    AlphaDenseSolveReceipt,
    W64_ASSEMBLER_REFERENCE,
    W64_CAST_DTYPE,
    W64_PRIMARY_REFERENCE,
    canonical_alpha_fp32_solve,
)
from .contracts import BATCH_SIZE, ODEBFContractError, canonical_hash
from .functional import CumulativeBF16FunctionalTrial, WaypointFactor, tensor_sha256
from .request_digest import (
    ordered_request_digest_scalable_v1,
    ordered_request_digest_v1,
)
from .target_new_nll import (
    RoutingObjective,
    evaluate_routing_objective,
    select_locked_routing_objective,
)
from .woodbury import ProjectorCertificate, WoodburyCertificate, solve_alpha_woodbury


P1_NATIVE_REFERENCE = "N32 canonical source-order original-BF16 AlphaEdit"
P1_DYNAMIC_REFERENCE = "Native AlphaEdit-WB-mixed64-v1 explicit-residual-policy field"
FULL_CURRENT_RESIDUAL_DEFINITION = "full_current"
FULL_CURRENT_RESIDUAL_DIVISOR = 1
LEGACY_PRE_SHARED_RESIDUAL_DEFINITION = "legacy_remaining_layer_pre_share"
SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1 = (
    "SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1"
)


@dataclass(frozen=True, slots=True)
class SharedTerminalResidualInput:
    """Explicit request-wise terminal residual supplied by the cold controller.

    The backend treats these tensors as immutable scientific inputs.  In
    particular, this opt-in path never recaptures a writer-layer activation,
    recomputes ``z-H_l``, or applies a remaining-layer divisor.
    """

    residual: torch.Tensor
    terminal_current_z: torch.Tensor
    request_order_sha256: str
    policy_identity: str = SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1

    def __post_init__(self) -> None:
        tensors = (self.residual, self.terminal_current_z)
        if (
            self.policy_identity
            != SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1
            or len(self.request_order_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.request_order_sha256)
            or any(
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[1] != BATCH_SIZE
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or not value.is_contiguous()
                or value.requires_grad
                or not torch.isfinite(value).all()
                for value in tensors
            )
            or self.residual.shape != self.terminal_current_z.shape
        ):
            raise ODEBFContractError("shared terminal residual input differs")

    @property
    def residual_sha256(self) -> str:
        return tensor_sha256(self.residual)

    @property
    def terminal_current_z_sha256(self) -> str:
        return tensor_sha256(self.terminal_current_z)

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "policy_identity": self.policy_identity,
            "request_order_sha256": self.request_order_sha256,
            "residual_shape": list(self.residual.shape),
            "residual_dtype": str(self.residual.dtype),
            "residual_device": self.residual.device.type,
            "residual_sha256": self.residual_sha256,
            "terminal_current_z_sha256": self.terminal_current_z_sha256,
            "backend_recapture_count": 0,
            "z_minus_h_layer_recompute_count": 0,
            "remaining_layer_division_count": 0,
        }


def full_current_residual(
    target_state: torch.Tensor,
    current_z: torch.Tensor,
) -> torch.Tensor:
    """Return the unshared current residual for one non-Native layer arm."""

    if target_state.shape != current_z.shape or target_state.ndim != 2:
        raise ODEBFContractError("full-current residual geometry differs")
    residual = (
        target_state.detach().to(device="cpu", dtype=torch.float32)
        - current_z.detach().to(device="cpu", dtype=torch.float32)
    ).contiguous()
    if not torch.isfinite(residual).all():
        raise ODEBFContractError("full-current residual is non-finite")
    return residual


def residual_for_policy(
    target_state: torch.Tensor,
    current_z: torch.Tensor,
    *,
    residual_policy: str,
    remaining_layers: int,
) -> tuple[torch.Tensor, int]:
    """Apply one of the two explicitly attributed P1R4 residual policies."""

    if residual_policy == FULL_CURRENT_RESIDUAL_DEFINITION:
        if remaining_layers <= 0:
            raise ODEBFContractError("full-current layer inventory differs")
        return full_current_residual(target_state, current_z), 1
    if residual_policy == LEGACY_PRE_SHARED_RESIDUAL_DEFINITION:
        if (
            isinstance(remaining_layers, bool)
            or not isinstance(remaining_layers, int)
            or remaining_layers <= 0
        ):
            raise ODEBFContractError("legacy residual divisor differs")
        residual = (
            full_current_residual(target_state, current_z) / float(remaining_layers)
        ).contiguous()
        if not torch.isfinite(residual).all():
            raise ODEBFContractError("legacy pre-shared residual is non-finite")
        return residual, remaining_layers
    raise ODEBFContractError("non-Native residual policy is not attributed")


def remaining_layer_count(layer_count: int, layer_index: int) -> int:
    if (
        isinstance(layer_count, bool)
        or isinstance(layer_index, bool)
        or layer_count <= 0
        or layer_index < 0
        or layer_index >= layer_count
    ):
        raise ODEBFContractError("legacy remaining-layer coordinate differs")
    return layer_count - layer_index


@dataclass(slots=True)
class P1NativeCapture:
    native_candidates: dict[str, torch.Tensor]
    entry_weights: dict[str, torch.Tensor]
    entry_sha256: dict[str, str]
    direct_z: tuple[torch.Tensor, ...]
    direct_z_sha256: tuple[str, ...]
    native_keys_by_layer: dict[int, torch.Tensor]
    entry_current_z_by_layer: dict[int, torch.Tensor]
    current_z_by_layer: dict[int, torch.Tensor]
    dense_solve_receipts: tuple[AlphaDenseSolveReceipt, ...]
    initialization: JointInitializationReceipt
    request_order_sha256: str
    target_backward_count: int
    history_columns_by_layer: tuple[tuple[int, int], ...]

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "reference": P1_NATIVE_REFERENCE,
            "entry_sha256": dict(sorted(self.entry_sha256.items())),
            "native_candidate_sha256": {
                name: tensor_sha256(value) for name, value in sorted(self.native_candidates.items())
            },
            "direct_z_sha256": list(self.direct_z_sha256),
            "native_key_sha256": {
                str(layer): tensor_sha256(value)
                for layer, value in sorted(self.native_keys_by_layer.items())
            },
            "current_z_sha256": {
                str(layer): tensor_sha256(value)
                for layer, value in sorted(self.current_z_by_layer.items())
            },
            "entry_current_z_sha256": {
                str(layer): tensor_sha256(value)
                for layer, value in sorted(self.entry_current_z_by_layer.items())
            },
            "dense_solve_receipts": [asdict(item) for item in self.dense_solve_receipts],
            "initialization": asdict(self.initialization),
            "request_order_sha256": self.request_order_sha256,
            "target_backward_count": self.target_backward_count,
            "history_columns_by_layer": [list(item) for item in self.history_columns_by_layer],
        }


def _normalize_requests(
    requests: Sequence[Mapping[str, Any]],
    *,
    expected_batch_size: int = BATCH_SIZE,
) -> list[dict[str, Any]]:
    if (
        isinstance(expected_batch_size, bool)
        or not isinstance(expected_batch_size, int)
        or expected_batch_size <= 0
        or len(requests) != expected_batch_size
    ):
        raise ODEBFContractError("P1 Alpha backend joint batch size differs")
    normalized = copy.deepcopy(list(requests))
    identities: list[str] = []
    for request in normalized:
        identity = str(request.get("request_sha256", ""))
        if len(identity) != 64:
            raise ODEBFContractError("P1 Alpha request identity differs")
        identities.append(identity)
        target = str(request["target_new"])
        request["target_new"] = target if target.startswith(" ") else " " + target
        prompt = str(request["prompt"])
        subject = str(request["subject"])
        if "{}" not in prompt:
            if subject not in prompt:
                raise ODEBFContractError("P1 Alpha subject is absent from prompt")
            prompt = prompt.replace(subject, "{}")
        request["prompt"] = prompt
    if len(set(identities)) != expected_batch_size:
        raise ODEBFContractError("P1 Alpha joint request identities repeat")
    return normalized


def _validate_history_keys(
    history_keys_by_layer: Mapping[int, torch.Tensor],
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    normalized_layers = tuple(int(layer) for layer in layers)
    if set(history_keys_by_layer) != set(normalized_layers):
        raise ODEBFContractError("P1 Alpha solve history layer set differs")
    result: dict[int, torch.Tensor] = {}
    column_counts: set[int] = set()
    for layer in normalized_layers:
        value = history_keys_by_layer[layer]
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ODEBFContractError("P1 Alpha solve history key shape differs")
        if value.dtype not in (torch.float32, torch.float64) or not torch.isfinite(value).all():
            raise ODEBFContractError("P1 Alpha solve history key values differ")
        result[layer] = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        column_counts.add(value.shape[1])
    if len(column_counts) != 1 or next(iter(column_counts)) not in (0, 10, 20, 30):
        raise ODEBFContractError("P1 Alpha solve history is not a prior-B10 prefix")
    return result


def capture_p1_native_entry(
    model: Any,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    *,
    history_keys_by_layer: Mapping[int, torch.Tensor],
    mutation_lock: threading.RLock,
    ledger: ComputeLedger,
    residual_tolerance: float,
    expected_batch_size: int = BATCH_SIZE,
) -> P1NativeCapture:
    """Run one canonical joint-B10 N32 endpoint and restore its entry exactly."""

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main
    from easyeditor.util import nethook

    normalized = _normalize_requests(
        requests, expected_batch_size=expected_batch_size
    )
    request_order = ordered_request_digest_scalable_v1(
        [str(request["request_sha256"]) for request in normalized]
    )
    layers = tuple(int(layer) for layer in hparams.layers)
    history = _validate_history_keys(history_keys_by_layer, layers)
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in layers
    }
    if any(parameter.dtype is not torch.bfloat16 for parameter in weights.values()):
        raise ODEBFContractError("P1 Native touched parameter is not BF16")
    if (
        not isinstance(projector, torch.Tensor)
        or projector.ndim != 3
        or projector.shape[0] != len(layers)
        or projector.dtype is not torch.float32
        or projector.device.type != "cpu"
    ):
        raise ODEBFContractError("P1 Native projector contract differs")
    entry = {
        name: parameter.detach().to(device="cpu").clone()
        for name, parameter in weights.items()
    }
    entry_hashes = {name: tensor_sha256(value) for name, value in entry.items()}
    pointers = {name: parameter.data_ptr() for name, parameter in weights.items()}
    versions = {name: parameter._version for name, parameter in weights.items()}
    direct_z: list[torch.Tensor] = []
    keys: dict[int, torch.Tensor] = {}
    entry_current_z_by_layer: dict[int, torch.Tensor] = {}
    current_z_by_layer: dict[int, torch.Tensor] = {}
    receipts: list[AlphaDenseSolveReceipt] = []
    candidates: dict[str, torch.Tensor] = {}
    target_backward_count = 0
    original_backward = torch.autograd.backward
    z_layer = layers[-1]

    def counted_backward(*args: Any, **kwargs: Any) -> Any:
        nonlocal target_backward_count
        target_backward_count += 1
        ledger.increment("backward")
        ledger.increment("target_backward")
        return original_backward(*args, **kwargs)

    resolved_contexts = alpha_main.get_context_templates(model, tokenizer)
    if canonical_hash(resolved_contexts) != canonical_hash(list(contexts)):
        raise ODEBFContractError("P1 Native context identity differs")
    try:
        torch.autograd.backward = counted_backward
        with mutation_lock, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            for request in normalized:
                value = alpha_main.compute_z(
                    model, tokenizer, request, hparams, z_layer, resolved_contexts
                )
                direct_z.append(value.detach().to(device="cpu", dtype=torch.float32))
            if len(direct_z) != expected_batch_size:
                raise ODEBFContractError("P1 Native direct-z count differs")
            zs = torch.stack(direct_z, dim=1)
            # Capture prospective layer activations once from the common W0
            # entry before the canonical Native writer mutates any layer.  The
            # pinned Native solve below still uses its source-order final-z
            # activation at each sequential layer; these W0 values are solely
            # the n=0 field source for the isolated adaptive variants.
            for layer in layers:
                entry_current_z = alpha_main.get_module_input_output_at_words(
                    model,
                    tokenizer,
                    layer,
                    context_templates=[request["prompt"] for request in normalized],
                    words=[request["subject"] for request in normalized],
                    module_template=hparams.layer_module_tmp,
                    fact_token_strategy=hparams.fact_token,
                )[1].T
                entry_current_z_by_layer[layer] = entry_current_z.detach().to(
                    device="cpu", dtype=torch.float32
                )
                del entry_current_z
            for layer_index, layer in enumerate(layers):
                layer_keys = alpha_main.compute_ks(
                    model,
                    tokenizer,
                    normalized,
                    hparams,
                    layer,
                    resolved_contexts,
                ).T
                if layer_keys.shape[1] != expected_batch_size:
                    raise ODEBFContractError("P1 Native key path global batch differs")
                keys[layer] = layer_keys.detach().to(device="cpu", dtype=torch.float32)
                native_current_z = alpha_main.get_module_input_output_at_words(
                    model,
                    tokenizer,
                    z_layer,
                    context_templates=[request["prompt"] for request in normalized],
                    words=[request["subject"] for request in normalized],
                    module_template=hparams.layer_module_tmp,
                    fact_token_strategy=hparams.fact_token,
                )[1].T
                current_z_by_layer[layer] = native_current_z.detach().to(
                    device="cpu", dtype=torch.float32
                )
                # Canonical N32 keeps the pinned source-order final-z residual.
                # The separately captured per-layer activation is only the
                # prospective field source for the adaptive causal panel.
                residual = (zs.to(device=native_current_z.device) - native_current_z) / (
                    len(layers) - layer_index
                )
                history_keys = history[layer]
                if history_keys.shape[0] == 0:
                    history_keys = torch.empty(
                        (layer_keys.shape[0], 0), dtype=torch.float32
                    )
                if history_keys.shape[0] != layer_keys.shape[0]:
                    raise ODEBFContractError("P1 Native history/key dimension differs")
                history_device = history_keys.to(device=layer_keys.device, dtype=torch.float32)
                covariance = (
                    history_device @ history_device.T
                    if history_device.shape[1]
                    else torch.zeros(
                        (layer_keys.shape[0], layer_keys.shape[0]),
                        dtype=torch.float32,
                        device=layer_keys.device,
                    )
                )
                weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                parameter = weights[weight_name]
                solved = canonical_alpha_fp32_solve(
                    projector[layer_index],
                    layer_keys,
                    covariance,
                    residual,
                    layer=layer,
                    regularization=float(hparams.L2),
                    solve_device=parameter.device,
                    residual_tolerance=residual_tolerance,
                )
                receipts.append(solved.receipt)
                update = alpha_main.upd_matrix_match_shape(solved.update, parameter.shape)
                with torch.no_grad():
                    parameter.copy_(parameter + update.float())
                candidates[weight_name] = parameter.detach().to(device="cpu").clone()
                del (
                    layer_keys,
                    native_current_z,
                    residual,
                    history_device,
                    covariance,
                    update,
                    solved,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        torch.autograd.backward = original_backward
        with mutation_lock, torch.no_grad():
            for name, parameter in weights.items():
                parameter.copy_(entry[name].to(device=parameter.device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if (
        len(receipts) != len(layers)
        or len(candidates) != len(layers)
        or set(keys) != set(layers)
        or set(entry_current_z_by_layer) != set(layers)
        or set(current_z_by_layer) != set(layers)
        or target_backward_count <= 0
    ):
        raise ODEBFContractError("P1 Native capture receipt count differs")
    if any(
        parameter.data_ptr() != pointers[name]
        or tensor_sha256(parameter) != entry_hashes[name]
        or parameter._version <= versions[name]
        or parameter.grad is not None
        for name, parameter in weights.items()
    ):
        raise ODEBFContractError("P1 Native capture did not restore BF16 entry")
    factors: list[LayerFactorReceipt] = []
    for layer in layers:
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        residual_rows = weights[weight_name].shape[0]
        layer_keys = keys[layer]
        rank = int(torch.linalg.matrix_rank(layer_keys.float()).item())
        factors.append(
            LayerFactorReceipt(
                layer,
                tuple(layer_keys.shape),
                (residual_rows, expected_batch_size),
                rank,
                str(torch.float32),
                weights[weight_name].device.type,
            )
        )
    initialization = JointInitializationReceipt(
        1,
        expected_batch_size,
        expected_batch_size,
        expected_batch_size,
        0,
        len(layers),
        0,
        len(layers),
        len(layers),
        False,
        tuple(factors),
        request_order,
    )
    return P1NativeCapture(
        candidates,
        entry,
        entry_hashes,
        tuple(direct_z),
        tuple(tensor_sha256(value) for value in direct_z),
        keys,
        entry_current_z_by_layer,
        current_z_by_layer,
        tuple(receipts),
        initialization,
        request_order,
        target_backward_count,
        tuple((layer, history[layer].shape[1]) for layer in layers),
    )


class CandidateBF16FunctionalTrial(AbstractContextManager["CandidateBF16FunctionalTrial"]):
    """Evaluate exact complete BF16 candidates without mutating parameters."""

    verdict_eligible = True

    def __init__(
        self,
        model: torch.nn.Module,
        candidates: Mapping[str, torch.Tensor],
    ) -> None:
        if not candidates:
            raise ODEBFContractError("candidate BF16 trial target set is empty")
        self.model = model
        self.candidates = {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in candidates.items()
        }
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._parameters: list[tuple[str, torch.nn.Parameter, int, int, bool]] = []
        self._cpu_rng: torch.Tensor | None = None
        self._cuda_rng: dict[torch.device, torch.Tensor] = {}
        self._active = 0
        self.max_live_candidate_weights = 0

    @staticmethod
    def _resolve(
        model: torch.nn.Module,
        weight_name: str,
    ) -> tuple[torch.nn.Linear, torch.nn.Parameter]:
        if not weight_name.endswith(".weight"):
            raise ODEBFContractError("candidate BF16 trial target is not a weight")
        module_name = weight_name[: -len(".weight")]
        try:
            module = model.get_submodule(module_name)
            parameter = dict(model.named_parameters())[weight_name]
        except (AttributeError, KeyError) as exc:
            raise ODEBFContractError("candidate BF16 trial target is absent") from exc
        if type(module) is not torch.nn.Linear or module.weight is not parameter:
            raise ODEBFContractError("candidate BF16 trial requires an exact Linear")
        return module, parameter

    def _hook(self, candidate_cpu: torch.Tensor):
        def apply(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> torch.Tensor:
            del output
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise ODEBFContractError("candidate BF16 trial Linear input differs")
            hidden = inputs[0]
            if hidden.dtype is not torch.bfloat16:
                raise ODEBFContractError("candidate BF16 trial activation is not BF16")
            self._active += 1
            self.max_live_candidate_weights = max(
                self.max_live_candidate_weights, self._active
            )
            try:
                effective = candidate_cpu.to(device=hidden.device, dtype=torch.bfloat16)
                return torch_functional.linear(hidden, effective, module.bias)
            finally:
                del effective
                self._active -= 1

        return apply

    def __enter__(self) -> "CandidateBF16FunctionalTrial":
        self._cpu_rng = torch.get_rng_state().clone()
        try:
            for name, candidate in sorted(self.candidates.items()):
                module, parameter = self._resolve(self.model, name)
                if (
                    parameter.dtype is not torch.bfloat16
                    or candidate.dtype is not torch.bfloat16
                    or parameter.shape != candidate.shape
                    or parameter.grad is not None
                    or not torch.isfinite(candidate).all()
                ):
                    raise ODEBFContractError("candidate BF16 trial tensor contract differs")
                self._parameters.append(
                    (name, parameter, parameter.data_ptr(), parameter._version, parameter.requires_grad)
                )
                self._handles.append(module.register_forward_hook(self._hook(candidate)))
            devices = {
                parameter.device
                for _, parameter, _, _, _ in self._parameters
                if parameter.device.type == "cuda"
            }
            self._cuda_rng = {
                device: torch.cuda.get_rng_state(device).clone() for device in devices
            }
        except BaseException:
            for handle in reversed(self._handles):
                handle.remove()
            self._handles.clear()
            self._parameters.clear()
            self._cuda_rng.clear()
            self._cpu_rng = None
            raise
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        violations: list[str] = []
        for name, parameter, pointer, version, requires_grad in self._parameters:
            if parameter.data_ptr() != pointer:
                violations.append(f"{name}:pointer")
            if parameter._version != version:
                violations.append(f"{name}:version")
            if parameter.requires_grad != requires_grad or parameter.grad is not None:
                violations.append(f"{name}:grad")
        self._parameters.clear()
        if self._cpu_rng is not None:
            torch.set_rng_state(self._cpu_rng)
            self._cpu_rng = None
        for device, state in self._cuda_rng.items():
            torch.cuda.set_rng_state(state, device)
        self._cuda_rng.clear()
        if self._active != 0 or self.max_live_candidate_weights > 1:
            violations.append("candidate-weight-liveness")
        if violations:
            error = ODEBFContractError(
                "candidate BF16 trial mutated state: " + ",".join(violations)
            )
            if exc is not None:
                raise error from exc
            raise error
        return False


@dataclass(frozen=True, slots=True)
class NpyMemberLocation:
    archive: Path
    member: str
    dtype: str
    shape: tuple[int, ...]
    fortran_order: bool
    data_offset: int


def locate_stored_npy_member(archive: str | Path, member: str) -> NpyMemberLocation:
    """Locate an uncompressed NPY payload for read-only direct memmapping."""

    path = Path(archive).resolve(strict=True)
    with zipfile.ZipFile(path, "r") as handle:
        info = handle.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ODEBFContractError("pinned covariance NPY member is compressed")
        header_offset = int(info.header_offset)
    with path.open("rb") as raw:
        raw.seek(header_offset)
        header = raw.read(30)
        if len(header) != 30:
            raise ODEBFContractError("pinned covariance ZIP header is incomplete")
        signature, _, _, _, _, _, _, _, _, name_length, extra_length = struct.unpack(
            "<IHHHHHIIIHH", header
        )
        if signature != 0x04034B50:
            raise ODEBFContractError("pinned covariance ZIP signature differs")
        raw.seek(name_length + extra_length, 1)
        version = np.lib.format.read_magic(raw)
        shape, fortran, dtype = np.lib.format._read_array_header(raw, version)
        offset = raw.tell()
    return NpyMemberLocation(
        path,
        member,
        np.dtype(dtype).str,
        tuple(int(value) for value in shape),
        bool(fortran),
        offset,
    )


def _load_small_npy_member(archive: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(archive, "r") as handle:
        with handle.open(member, "r") as source:
            return np.asarray(np.load(source, allow_pickle=False))


@dataclass(frozen=True, slots=True)
class CovarianceActionReceipt:
    layer: int
    source_sha256: str
    source_size: int
    matrix_shape: tuple[int, int]
    sample_count: int
    q_shape: tuple[int, int]
    q_sha256: str
    action_sha256: str
    gram_sha256: str
    finite: bool
    wall_seconds: float


class PinnedCovarianceRegistry:
    """Read-only streaming actions against pinned uncompressed NPZ moments."""

    def __init__(
        self,
        *,
        easyedit_root: str | Path,
        covariance_spec: Mapping[str, Sequence[Any]],
    ) -> None:
        self.easyedit_root = Path(easyedit_root).resolve(strict=True)
        self._spec = {int(layer): tuple(value) for layer, value in covariance_spec.items()}
        self._locations: dict[int, NpyMemberLocation] = {}
        self._counts: dict[int, int] = {}

    def _metadata(self, layer: int) -> tuple[NpyMemberLocation, int, str, int]:
        if layer not in self._spec:
            raise ODEBFContractError("pinned covariance layer is absent")
        relative, size, digest = self._spec[layer]
        path = (self.easyedit_root / str(relative)).resolve(strict=True)
        path.relative_to(self.easyedit_root)
        if path.stat().st_size != int(size):
            raise ODEBFContractError("pinned covariance size differs")
        if layer not in self._locations:
            location = locate_stored_npy_member(path, "mom2.mom2.npy")
            if location.fortran_order or np.dtype(location.dtype) != np.dtype(np.float32):
                raise ODEBFContractError("pinned covariance NPY layout differs")
            count_value = _load_small_npy_member(path, "mom2.count.npy")
            if count_value.size != 1:
                raise ODEBFContractError("pinned covariance count differs")
            count = int(count_value.reshape(-1)[0])
            if count <= 0:
                raise ODEBFContractError("pinned covariance count is not positive")
            self._locations[layer] = location
            self._counts[layer] = count
        return self._locations[layer], self._counts[layer], str(digest), int(size)

    def action(
        self,
        layer: int,
        q: torch.Tensor,
        *,
        row_block: int = 256,
        expected_batch_size: int = BATCH_SIZE,
    ) -> tuple[torch.Tensor, torch.Tensor, CovarianceActionReceipt]:
        location, count, source_sha, source_size = self._metadata(layer)
        if (
            isinstance(expected_batch_size, bool)
            or not isinstance(expected_batch_size, int)
            or expected_batch_size <= 0
            or q.ndim != 2
            or q.shape[1] != expected_batch_size
            or not torch.isfinite(q).all()
        ):
            raise ODEBFContractError("pretrained covariance action Q differs")
        if location.shape != (q.shape[0], q.shape[0]):
            raise ODEBFContractError("pretrained covariance/Q geometry differs")
        if row_block <= 0:
            raise ODEBFContractError("pretrained covariance row block differs")
        q_cpu = q.detach().to(device="cpu", dtype=torch.float32).contiguous()
        q_numpy = q_cpu.numpy()
        matrix = np.memmap(
            location.archive,
            mode="r",
            dtype=np.dtype(location.dtype),
            offset=location.data_offset,
            shape=location.shape,
            order="C",
        )
        action = np.empty_like(q_numpy, dtype=np.float32)
        started = time.perf_counter()
        for start in range(0, location.shape[0], row_block):
            end = min(start + row_block, location.shape[0])
            action[start:end] = (matrix[start:end] @ q_numpy) / np.float32(count)
        gram = q_numpy.T.astype(np.float64) @ action.astype(np.float64)
        wall = time.perf_counter() - started
        del matrix
        action_tensor = torch.from_numpy(action.copy())
        gram_tensor = torch.from_numpy(gram.copy())
        finite = bool(torch.isfinite(action_tensor).all() and torch.isfinite(gram_tensor).all())
        if not finite:
            raise ODEBFContractError("pretrained covariance action is non-finite")
        receipt = CovarianceActionReceipt(
            layer,
            source_sha,
            source_size,
            (location.shape[0], location.shape[1]),
            count,
            tuple(q_cpu.shape),
            tensor_sha256(q_cpu),
            tensor_sha256(action_tensor),
            tensor_sha256(gram_tensor),
            finite,
            wall,
        )
        return action_tensor, gram_tensor, receipt


@dataclass(slots=True)
class P1LayerField:
    layer: int
    weight_name: str
    key: torch.Tensor
    projected_key: torch.Tensor
    residual: torch.Tensor
    residual_definition: str
    residual_divisor: int
    q: torch.Tensor
    factor: WaypointFactor
    factor_frobenius_sq: float
    covariance_action: torch.Tensor
    covariance_gram: torch.Tensor
    covariance_receipt: CovarianceActionReceipt
    woodbury_certificate: WoodburyCertificate
    history_action: torch.Tensor

    def __post_init__(self) -> None:
        valid_full = (
            self.residual_definition == FULL_CURRENT_RESIDUAL_DEFINITION
            and self.residual_divisor == FULL_CURRENT_RESIDUAL_DIVISOR
        )
        valid_legacy = (
            self.residual_definition == LEGACY_PRE_SHARED_RESIDUAL_DEFINITION
            and isinstance(self.residual_divisor, int)
            and not isinstance(self.residual_divisor, bool)
            and self.residual_divisor >= 1
        )
        valid_shared_terminal = (
            self.residual_definition
            == SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1
            and self.residual_divisor == FULL_CURRENT_RESIDUAL_DIVISOR
        )
        if not (valid_full or valid_legacy or valid_shared_terminal):
            raise ODEBFContractError("non-Native residual policy differs")
        if (
            self.factor.weight_name != self.weight_name
            or self.factor.layer != self.layer
            or not torch.equal(self.factor.left, self.residual)
            or not torch.equal(self.factor.right, self.q)
        ):
            raise ODEBFContractError("non-Native B_l=R_l Q_l^T identity differs")

    def factor_identity(self) -> str:
        return canonical_hash(
            {
                "weight_name_sha256": hashlib.sha256(
                    self.factor.weight_name.encode("utf-8")
                ).hexdigest(),
                "layer": self.factor.layer,
                "correction_cycle": self.factor.correction_cycle,
                "step_in_cycle": self.factor.step_in_cycle,
                "factor_ordinal": self.factor.factor_ordinal,
                "theta": self.factor.theta,
                "left_sha256": tensor_sha256(self.factor.left),
                "right_sha256": tensor_sha256(self.factor.right),
                "joint_batch": self.factor.joint_batch,
                "global_batch_size": self.factor.global_batch_size,
            }
        )

    def arm_identity(self) -> str:
        return canonical_hash(
            {
                "layer": self.layer,
                "key_sha256": tensor_sha256(self.key),
                "projected_key_sha256": tensor_sha256(self.projected_key),
                "residual_definition": self.residual_definition,
                "residual_divisor": self.residual_divisor,
                "current_residual_sha256": tensor_sha256(self.residual),
                "q_sha256": tensor_sha256(self.q),
                "factor_sha256": self.factor_identity(),
            }
        )

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "weight_name_sha256": hashlib.sha256(self.weight_name.encode("utf-8")).hexdigest(),
            "key_shape": list(self.key.shape),
            "key_sha256": tensor_sha256(self.key),
            "projected_key_sha256": tensor_sha256(self.projected_key),
            "residual_shape": list(self.residual.shape),
            "residual_sha256": tensor_sha256(self.residual),
            "residual_definition": self.residual_definition,
            "residual_divisor": self.residual_divisor,
            "current_residual_sha256": tensor_sha256(self.residual),
            "current_residual_frobenius_norm": float(
                torch.linalg.norm(self.residual.double())
            ),
            "q_shape": list(self.q.shape),
            "q_sha256": tensor_sha256(self.q),
            "factor_sha256": self.factor_identity(),
            "layer_arm_sha256": self.arm_identity(),
            "factor_frobenius_sq": self.factor_frobenius_sq,
            "covariance": asdict(self.covariance_receipt),
            "woodbury": asdict(self.woodbury_certificate),
            "history_action_shape": list(self.history_action.shape),
            "history_action_sha256": tensor_sha256(self.history_action),
        }


@dataclass(slots=True)
class P1DynamicField:
    accepted_waypoint: int
    request_order_sha256: str
    target_state: torch.Tensor
    current_z: torch.Tensor
    layers: tuple[P1LayerField, ...]
    identity_sha256: str
    model_forward_count: int
    processed_token_count: int

    def factors_by_weight(self) -> dict[str, tuple[WaypointFactor, ...]]:
        return {item.weight_name: (item.factor,) for item in self.layers}

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "reference": P1_DYNAMIC_REFERENCE,
            "accepted_waypoint": self.accepted_waypoint,
            "request_order_sha256": self.request_order_sha256,
            "target_state_sha256": tensor_sha256(self.target_state),
            "current_z_sha256": tensor_sha256(self.current_z),
            "layers": [item.raw_free_payload() for item in self.layers],
            "identity_sha256": self.identity_sha256,
            "model_forward_count": self.model_forward_count,
            "processed_token_count": self.processed_token_count,
        }


def _validate_dynamic_factor_capacity(
    value: float, *, allow_zero_capacity: bool
) -> float:
    if not isinstance(allow_zero_capacity, bool):
        raise ODEBFContractError("P1 zero-capacity policy is not boolean")
    observed = float(value)
    if (
        not math.isfinite(observed)
        or observed < 0.0
        or (observed == 0.0 and not allow_zero_capacity)
    ):
        raise ODEBFContractError("P1 dynamic arm has nonpositive capacity")
    return observed


def _virtual_context(
    model: torch.nn.Module,
    factors: Mapping[str, Sequence[WaypointFactor]],
) -> AbstractContextManager[Any]:
    return (
        CumulativeBF16FunctionalTrial(model, factors, row_block=64)
        if factors
        else nullcontext()
    )


def build_p1_dynamic_field(
    model: Any,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    *,
    target_state: torch.Tensor,
    accepted_waypoint: int,
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    history_solve_keys_by_layer: Mapping[int, torch.Tensor],
    history_risk_keys_by_layer: Mapping[int, torch.Tensor],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    residual_tolerance: float,
    ledger: ComputeLedger,
    residual_policy: str = FULL_CURRENT_RESIDUAL_DEFINITION,
    allow_zero_capacity: bool = False,
    shared_terminal_residual: SharedTerminalResidualInput | None = None,
    captured_keys_by_layer: Mapping[int, torch.Tensor] | None = None,
    allow_inner_empty_cache: bool = True,
    expected_batch_size: int = BATCH_SIZE,
) -> P1DynamicField:
    """Rebuild all layer arms at one accepted virtual joint state."""

    if not isinstance(allow_zero_capacity, bool) or not isinstance(
        allow_inner_empty_cache, bool
    ):
        raise ODEBFContractError("P1 zero-capacity policy is not boolean")

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main
    from easyeditor.util import nethook

    normalized = _normalize_requests(
        requests, expected_batch_size=expected_batch_size
    )
    order = ordered_request_digest_scalable_v1(
        [str(item["request_sha256"]) for item in normalized]
    )
    layers = tuple(int(layer) for layer in hparams.layers)
    if captured_keys_by_layer is not None and set(captured_keys_by_layer) != set(layers):
        raise ODEBFContractError("P1 captured key inventory differs")
    history_solve = _validate_history_keys(history_solve_keys_by_layer, layers)
    history_risk = _validate_history_keys(history_risk_keys_by_layer, layers)
    if (
        isinstance(expected_batch_size, bool)
        or not isinstance(expected_batch_size, int)
        or expected_batch_size <= 0
        or target_state.ndim != 2
        or target_state.shape[1] != expected_batch_size
    ):
        raise ODEBFContractError("P1 target state global batch differs")
    if not torch.isfinite(target_state).all():
        raise ODEBFContractError("P1 target state contains non-finite values")
    shared_policy = residual_policy == SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1
    if shared_policy != (shared_terminal_residual is not None):
        raise ODEBFContractError("shared terminal residual opt-in differs")
    if shared_terminal_residual is not None:
        if (
            shared_terminal_residual.request_order_sha256 != order
            or shared_terminal_residual.policy_identity != residual_policy
            or shared_terminal_residual.residual.shape != target_state.shape
            or shared_terminal_residual.terminal_current_z.shape != target_state.shape
            or target_state.dtype != torch.float32
            or target_state.device.type != "cpu"
            or not target_state.is_contiguous()
        ):
            raise ODEBFContractError("shared terminal residual identity differs")
    resolved_contexts = alpha_main.get_context_templates(model, tokenizer)
    if canonical_hash(resolved_contexts) != canonical_hash(list(contexts)):
        raise ODEBFContractError("P1 dynamic field context identity differs")
    device = next(model.parameters()).device
    layer_fields: list[P1LayerField] = []
    processed_tokens = 0
    current_z_by_layer: dict[int, torch.Tensor] = {}
    with _virtual_context(model, cumulative_factors_by_weight):
        shared_current_z: torch.Tensor | None = None
        if residual_policy == LEGACY_PRE_SHARED_RESIDUAL_DEFINITION:
            shared_current_z = alpha_main.get_module_input_output_at_words(
                model,
                tokenizer,
                layers[-1],
                context_templates=[item["prompt"] for item in normalized],
                words=[item["subject"] for item in normalized],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T.detach().to(device="cpu", dtype=torch.float32)
        elif shared_terminal_residual is not None:
            shared_current_z = shared_terminal_residual.terminal_current_z
        for layer_index, layer in enumerate(layers):
            current_z = (
                shared_current_z
                if shared_current_z is not None
                else alpha_main.get_module_input_output_at_words(
                    model,
                    tokenizer,
                    layer,
                    context_templates=[item["prompt"] for item in normalized],
                    words=[item["subject"] for item in normalized],
                    module_template=hparams.layer_module_tmp,
                    fact_token_strategy=hparams.fact_token,
                )[1].T.detach().to(device="cpu", dtype=torch.float32)
            )
            if current_z is None or current_z.shape != target_state.shape:
                raise ODEBFContractError("P1 target/current-z geometry differs")
            current_z_by_layer[layer] = current_z.clone()
            if shared_terminal_residual is not None:
                residual = shared_terminal_residual.residual.clone()
                residual_divisor = FULL_CURRENT_RESIDUAL_DIVISOR
            else:
                full_residual = full_current_residual(target_state, current_z)
            if residual_policy == FULL_CURRENT_RESIDUAL_DEFINITION:
                residual = full_residual.clone()
                residual_divisor = FULL_CURRENT_RESIDUAL_DIVISOR
            elif shared_terminal_residual is None:
                residual, residual_divisor = residual_for_policy(
                    target_state,
                    current_z,
                    residual_policy=residual_policy,
                    remaining_layers=remaining_layer_count(
                        len(layers), layer_index
                    ),
                )
            key = (
                alpha_main.compute_ks(
                    model,
                    tokenizer,
                    normalized,
                    hparams,
                    layer,
                    resolved_contexts,
                ).T.detach().to(device="cpu", dtype=torch.float32)
                if captured_keys_by_layer is None
                else captured_keys_by_layer[layer]
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
            if key.shape[1] != expected_batch_size:
                raise ODEBFContractError("P1 dynamic key global batch differs")
            history = history_solve[layer]
            if history.shape[0] == 0:
                history = torch.empty((key.shape[0], 0), dtype=torch.float32)
            risk = history_risk[layer]
            if risk.shape[0] == 0:
                risk = torch.empty((key.shape[0], 0), dtype=torch.float32)
            if history.shape[0] != key.shape[0] or risk.shape[0] != key.shape[0]:
                raise ODEBFContractError("P1 dynamic history/key dimension differs")
            p_device = projector[layer_index].to(device=device, dtype=torch.float32)
            k_device = key.to(device=device, dtype=torch.float32)
            history_device = history.to(device=device, dtype=torch.float32)
            solved = solve_alpha_woodbury(
                p_device,
                k_device,
                history_keys=history_device,
                regularization=float(hparams.L2),
                projector_certificate=ProjectorCertificate(
                    projector_sha256,
                    1.0,
                    1.0,
                    "artifact-unverified",
                    1.0e-10,
                ),
                residual_tolerance=residual_tolerance,
            )
            if not solved.certificate.passed or solved.certificate.alpha_linear_residual > residual_tolerance:
                raise ODEBFContractError("P1 W64 field certificate failed")
            q = solved.q.detach().to(device="cpu", dtype=W64_CAST_DTYPE).contiguous()
            projected = (p_device @ k_device).detach().to(device="cpu", dtype=torch.float32)
            covariance_action, covariance_gram, covariance_receipt = covariance_registry.action(
                layer, q, expected_batch_size=expected_batch_size
            )
            right_gram = q.T.to(dtype=torch.float64) @ q.to(dtype=torch.float64)
            left_gram = residual.T.to(dtype=torch.float64) @ residual.to(dtype=torch.float64)
            frobenius_sq = _validate_dynamic_factor_capacity(
                float(torch.sum(left_gram * right_gram)),
                allow_zero_capacity=allow_zero_capacity,
            )
            history_action = (
                residual.to(dtype=torch.float64)
                @ (q.to(dtype=torch.float64).T @ risk.to(dtype=torch.float64))
                if risk.shape[1]
                else torch.empty((residual.shape[0], 0), dtype=torch.float64)
            )
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            parameter = nethook.get_parameter(model, weight_name)
            if tuple(parameter.shape) != (residual.shape[0], q.shape[0]):
                raise ODEBFContractError("P1 dynamic arm does not match its parameter")
            factor = WaypointFactor(
                weight_name,
                layer,
                0,
                accepted_waypoint,
                0,
                1.0,
                residual.clone(),
                q.clone(),
                global_batch_size=expected_batch_size,
            )
            layer_fields.append(
                P1LayerField(
                    layer,
                    weight_name,
                    key,
                    projected,
                    residual.clone(),
                    residual_policy,
                    residual_divisor,
                    q,
                    factor,
                    frobenius_sq,
                    covariance_action,
                    covariance_gram,
                    covariance_receipt,
                    solved.certificate,
                    history_action,
                )
            )
            del p_device, k_device, history_device, solved, parameter
            if allow_inner_empty_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
    if shared_terminal_residual is not None:
        residual_hashes = tuple(tensor_sha256(item.residual) for item in layer_fields)
        key_hashes = tuple(tensor_sha256(item.key) for item in layer_fields)
        q_hashes = tuple(tensor_sha256(item.q) for item in layer_fields)
        weight_names = tuple(item.weight_name for item in layer_fields)
        if (
            len(set(residual_hashes)) != 1
            or residual_hashes[0] != shared_terminal_residual.residual_sha256
            or len(set(key_hashes)) != len(layers)
            or len(set(q_hashes)) != len(layers)
            or len(set(weight_names)) != len(layers)
        ):
            raise ODEBFContractError("shared terminal residual layer identity differs")
    payload = {
        "accepted_waypoint": accepted_waypoint,
        "request_order_sha256": order,
        "target_state_sha256": tensor_sha256(target_state),
        "current_z_by_layer": [
            (layer, tensor_sha256(current_z_by_layer[layer])) for layer in layers
        ],
        "layers": [item.raw_free_payload() for item in layer_fields],
        "history_version_columns": sorted(
            (layer, history_solve[layer].shape[1]) for layer in layers
        ),
        "reference": P1_DYNAMIC_REFERENCE,
        "residual_policy": residual_policy,
        "assembler": W64_ASSEMBLER_REFERENCE,
    }
    if shared_terminal_residual is not None:
        payload["shared_terminal_residual"] = (
            shared_terminal_residual.raw_free_payload()
        )
    model_forward_count = 0 if captured_keys_by_layer is not None else (
        1 + len(layers)
        if residual_policy == LEGACY_PRE_SHARED_RESIDUAL_DEFINITION
        else (
            len(layers)
            if residual_policy == SHARED_TERMINAL_FULL_RESIDUAL_DIVISOR_ONE_V1
            else 2 * len(layers)
        )
    )
    return P1DynamicField(
        accepted_waypoint,
        order,
        target_state.detach().to(device="cpu", dtype=torch.float32).clone(),
        current_z_by_layer[layers[-1]],
        tuple(layer_fields),
        canonical_hash(payload),
        model_forward_count,
        processed_tokens,
    )


def build_p1_frozen_field_from_capture(
    model: Any,
    hparams: Any,
    projector: torch.Tensor,
    capture: P1NativeCapture,
    *,
    target_state: torch.Tensor,
    accepted_waypoint: int,
    history_solve_keys_by_layer: Mapping[int, torch.Tensor],
    history_risk_keys_by_layer: Mapping[int, torch.Tensor],
    covariance_registry: PinnedCovarianceRegistry,
    projector_sha256: str,
    residual_tolerance: float,
    residual_policy: str = FULL_CURRENT_RESIDUAL_DEFINITION,
) -> P1DynamicField:
    """Construct the entry-frozen W64 field from the exact N32 capture."""

    from easyeditor.util import nethook

    layers = tuple(int(layer) for layer in hparams.layers)
    if accepted_waypoint != 0:
        raise ODEBFContractError("entry-frozen field must be created at waypoint zero")
    if (
        set(capture.native_keys_by_layer) != set(layers)
        or set(capture.current_z_by_layer) != set(layers)
        or set(capture.entry_current_z_by_layer) != set(layers)
    ):
        raise ODEBFContractError("entry-frozen capture layer inventory differs")
    history_solve = _validate_history_keys(history_solve_keys_by_layer, layers)
    history_risk = _validate_history_keys(history_risk_keys_by_layer, layers)
    if target_state.ndim != 2 or target_state.shape[1] != BATCH_SIZE:
        raise ODEBFContractError("entry-frozen target state is not [hidden,10]")
    device = next(model.parameters()).device
    layer_fields: list[P1LayerField] = []
    current_z_digest: list[tuple[int, str]] = []
    for layer_index, layer in enumerate(layers):
        key = capture.native_keys_by_layer[layer].detach().to(
            device="cpu", dtype=torch.float32
        )
        current_z_source = (
            capture.current_z_by_layer
            if residual_policy == LEGACY_PRE_SHARED_RESIDUAL_DEFINITION
            else capture.entry_current_z_by_layer
        )
        current_z = current_z_source[layer].detach().to(
            device="cpu", dtype=torch.float32
        )
        if current_z.shape != target_state.shape:
            raise ODEBFContractError("entry-frozen target/current-z geometry differs")
        if residual_policy == FULL_CURRENT_RESIDUAL_DEFINITION:
            residual = full_current_residual(target_state, current_z)
            residual_divisor = FULL_CURRENT_RESIDUAL_DIVISOR
        else:
            residual, residual_divisor = residual_for_policy(
                target_state,
                current_z,
                residual_policy=residual_policy,
                remaining_layers=remaining_layer_count(
                    len(layers), layer_index
                ),
            )
        history = history_solve[layer]
        risk = history_risk[layer]
        if history.shape[0] == 0:
            history = torch.empty((key.shape[0], 0), dtype=torch.float32)
        if risk.shape[0] == 0:
            risk = torch.empty((key.shape[0], 0), dtype=torch.float32)
        if history.shape[0] != key.shape[0] or risk.shape[0] != key.shape[0]:
            raise ODEBFContractError("entry-frozen history/key dimension differs")
        p_device = projector[layer_index].to(device=device, dtype=torch.float32)
        k_device = key.to(device=device, dtype=torch.float32)
        history_device = history.to(device=device, dtype=torch.float32)
        solved = solve_alpha_woodbury(
            p_device,
            k_device,
            history_keys=history_device,
            regularization=float(hparams.L2),
            projector_certificate=ProjectorCertificate(
                projector_sha256,
                1.0,
                1.0,
                "artifact-unverified",
                1.0e-10,
            ),
            residual_tolerance=residual_tolerance,
        )
        if (
            not solved.certificate.passed
            or solved.certificate.alpha_linear_residual > residual_tolerance
        ):
            raise ODEBFContractError("entry-frozen W64 certificate failed")
        q = solved.q.detach().to(device="cpu", dtype=W64_CAST_DTYPE).contiguous()
        projected = (p_device @ k_device).detach().to(device="cpu", dtype=torch.float32)
        covariance_action, covariance_gram, covariance_receipt = covariance_registry.action(
            layer, q
        )
        right_gram = q.T.to(dtype=torch.float64) @ q.to(dtype=torch.float64)
        left_gram = residual.T.to(dtype=torch.float64) @ residual.to(dtype=torch.float64)
        frobenius_sq = float(torch.sum(left_gram * right_gram))
        if not math.isfinite(frobenius_sq) or frobenius_sq <= 0.0:
            raise ODEBFContractError("entry-frozen arm has nonpositive capacity")
        history_action = (
            residual.to(dtype=torch.float64)
            @ (q.to(dtype=torch.float64).T @ risk.to(dtype=torch.float64))
            if risk.shape[1]
            else torch.empty((residual.shape[0], 0), dtype=torch.float64)
        )
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        parameter = nethook.get_parameter(model, weight_name)
        if tuple(parameter.shape) != (residual.shape[0], q.shape[0]):
            raise ODEBFContractError("entry-frozen arm does not match its parameter")
        factor = WaypointFactor(
            weight_name,
            layer,
            0,
            0,
            layer_index,
            1.0,
            residual.clone(),
            q.clone(),
        )
        layer_fields.append(
            P1LayerField(
                layer,
                weight_name,
                key,
                projected,
                residual,
                residual_policy,
                residual_divisor,
                q,
                factor,
                frobenius_sq,
                covariance_action,
                covariance_gram,
                covariance_receipt,
                solved.certificate,
                history_action,
            )
        )
        current_z_digest.append((layer, tensor_sha256(current_z)))
        del p_device, k_device, history_device, solved, parameter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "accepted_waypoint": 0,
        "request_order_sha256": capture.request_order_sha256,
        "target_state_sha256": tensor_sha256(target_state),
        "current_z_by_layer": current_z_digest,
        "layers": [item.raw_free_payload() for item in layer_fields],
        "history_version_columns": sorted(
            (layer, history_solve[layer].shape[1]) for layer in layers
        ),
        "reference": P1_DYNAMIC_REFERENCE,
        "field_policy": "entry-frozen-from-canonical-N32-capture",
        "residual_policy": residual_policy,
        "assembler": W64_ASSEMBLER_REFERENCE,
    }
    representative_z = capture.current_z_by_layer[layers[-1]].detach().to(
        device="cpu", dtype=torch.float32
    )
    return P1DynamicField(
        0,
        capture.request_order_sha256,
        target_state.detach().to(device="cpu", dtype=torch.float32).clone(),
        representative_z,
        tuple(layer_fields),
        canonical_hash(payload),
        0,
        0,
    )


def capture_committed_history_key_views(
    model: Any,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    hparams: Any,
    projector: torch.Tensor,
    contexts: Sequence[Sequence[str]],
    *,
    ledger: ComputeLedger,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], str]:
    """Capture post-commit raw solve keys and unweighted projected risk keys."""

    from easyeditor.models.alphaedit import AlphaEdit_main as alpha_main

    normalized = _normalize_requests(requests)
    resolved_contexts = alpha_main.get_context_templates(model, tokenizer)
    if canonical_hash(resolved_contexts) != canonical_hash(list(contexts)):
        raise ODEBFContractError("post-commit history context identity differs")
    layers = tuple(int(layer) for layer in hparams.layers)
    if projector.shape[0] != len(layers):
        raise ODEBFContractError("post-commit history projector inventory differs")
    device = next(model.parameters()).device
    solve_keys: dict[int, torch.Tensor] = {}
    risk_keys: dict[int, torch.Tensor] = {}
    for layer_index, layer in enumerate(layers):
        key = alpha_main.compute_ks(
            model,
            tokenizer,
            normalized,
            hparams,
            layer,
            resolved_contexts,
        ).T.detach().to(device="cpu", dtype=torch.float32)
        if key.shape[1] != BATCH_SIZE:
            raise ODEBFContractError("post-commit history key is not joint B10")
        projected = (
            projector[layer_index].to(device=device, dtype=torch.float32)
            @ key.to(device=device, dtype=torch.float32)
        ).detach().to(device="cpu", dtype=torch.float32)
        solve_keys[layer] = key
        risk_keys[layer] = projected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    identity = canonical_hash(
        {
            "request_order": ordered_request_digest_v1(
                [str(item["request_sha256"]) for item in normalized]
            ),
            "solve": [(layer, tensor_sha256(solve_keys[layer])) for layer in layers],
            "risk": [(layer, tensor_sha256(risk_keys[layer])) for layer in layers],
            "weighting": "unweighted-projected-key-view-at-use-time",
        }
    )
    return solve_keys, risk_keys, identity


class _CoefficientOverlay(AbstractContextManager["_CoefficientOverlay"]):
    def __init__(
        self,
        model: torch.nn.Module,
        layers: Sequence[P1LayerField],
        coefficients: torch.Tensor,
        *,
        target_state_variable: torch.Tensor | None = None,
        current_z: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.layers = tuple(layers)
        self.coefficients = coefficients
        self.target_state_variable = target_state_variable
        self.current_z = current_z
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, index: int):
        field = self.layers[index]

        def apply(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> torch.Tensor:
            if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                raise ODEBFContractError("P1 gradient overlay Linear contract differs")
            hidden = inputs[0]
            right = field.q.to(device=hidden.device, dtype=torch.float32)
            if self.target_state_variable is None:
                left = field.residual.to(device=hidden.device, dtype=torch.float32)
            else:
                if self.current_z is None:
                    raise ODEBFContractError("P1 target overlay lacks current-z")
                left = self.target_state_variable - self.current_z.to(
                    device=hidden.device, dtype=torch.float32
                )
            perturbation = (hidden.float() @ right) @ left.T
            result = output.float() + self.coefficients[index] * perturbation
            return result.to(dtype=output.dtype)

        return apply

    def __enter__(self) -> "_CoefficientOverlay":
        if self.coefficients.ndim != 1 or self.coefficients.numel() != len(self.layers):
            raise ODEBFContractError("P1 coefficient overlay dimension differs")
        for index, field in enumerate(self.layers):
            module_name = field.weight_name[: -len(".weight")]
            module = self.model.get_submodule(module_name)
            if type(module) is not torch.nn.Linear:
                raise ODEBFContractError("P1 gradient overlay target is not exact Linear")
            self._handles.append(module.register_forward_hook(self._hook(index)))
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, traceback
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        return False


def _controller_margin_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    device = next(model.parameters()).device
    llama = "llama" in str(getattr(model.config, "_name_or_path", "")).casefold()
    losses: list[torch.Tensor] = []
    processed = 0
    for request in requests:
        prefix = str(request["prompt"]).format(str(request["subject"]))
        target_new = str(request["target_new"])
        target_true = str(request["target_true"])
        prefix_length = len(tokenizer([prefix])["input_ids"][0])
        encoded = tokenizer(
            [f"{prefix} {target_new}", f"{prefix} {target_true}"],
            padding=True,
            return_tensors="pt",
        ).to(device)
        tokens = [tokenizer(f" {value}")["input_ids"] for value in (target_new, target_true)]
        if llama:
            tokens = [value[1:] for value in tokens]
            prefix_length -= 1
        logits = model(**encoded).logits
        if llama:
            logits = logits[:, 1:, :]
        choice_losses: list[torch.Tensor] = []
        for row, target_tokens in enumerate(tokens):
            if not target_tokens:
                raise ODEBFContractError("P1 controller target suffix is empty")
            token_losses = []
            for offset, token in enumerate(target_tokens):
                position = prefix_length + offset - 1
                token_losses.append(
                    -torch.log_softmax(logits[row, position, :].float(), dim=0)[int(token)]
                )
            choice_losses.append(torch.stack(token_losses).mean())
        losses.append(choice_losses[0] - choice_losses[1])
        attention = encoded.get("attention_mask")
        processed += int(attention.sum()) if attention is not None else int(encoded["input_ids"].numel())
    per_request = torch.stack(losses)
    return per_request.mean(), per_request, processed


@dataclass(frozen=True, slots=True)
class ControllerMarginReceipt:
    value: float
    value_sha256: str
    per_request_values: tuple[float, ...]
    per_request_value_sha256: str
    request_order_sha256: str
    model_forward_count: int
    processed_token_count: int
    generation_call_count: int


@dataclass(frozen=True, slots=True)
class TargetNewNLLReceipt:
    objective: str
    value: float
    value_sha256: str
    per_request_values: tuple[float, ...]
    per_request_value_sha256: str
    request_order_sha256: str
    suffix_token_counts: tuple[int, ...]
    target_span_sha256: str
    context_group_sizes: tuple[int, ...]
    context_count: int
    context_sha256: str
    model_forward_count: int
    processed_token_count: int
    generation_call_count: int
    target_old_access_count: int
    receipt_sha256: str


def evaluate_controller_margin(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
) -> ControllerMarginReceipt:
    normalized = _normalize_requests(requests)
    parameter_state = tuple(
        (parameter.data_ptr(), parameter._version, parameter.grad)
        for parameter in model.parameters()
    )
    with _virtual_context(model, cumulative_factors_by_weight), torch.no_grad():
        value, per_request, processed = _controller_margin_loss(
            model, tokenizer, normalized
        )
    observed_model_mean = float(value.detach().to(device="cpu", dtype=torch.float64))
    observed_per_request = tuple(
        float(item)
        for item in per_request.detach().to(device="cpu", dtype=torch.float64)
    )
    if (
        not math.isfinite(observed_model_mean)
        or len(observed_per_request) != BATCH_SIZE
        or not all(math.isfinite(item) for item in observed_per_request)
    ):
        raise ODEBFContractError("P1 controller margin is non-finite")
    observed = math.fsum(observed_per_request) / BATCH_SIZE
    if not math.isclose(
        observed,
        observed_model_mean,
        rel_tol=1.0e-6,
        abs_tol=1.0e-7,
    ):
        raise ODEBFContractError("P1 controller uniform mean differs")
    if tuple(
        (parameter.data_ptr(), parameter._version, parameter.grad)
        for parameter in model.parameters()
    ) != parameter_state:
        raise ODEBFContractError("P1 controller margin mutated model state")
    return ControllerMarginReceipt(
        observed,
        canonical_hash({"float64": observed}),
        observed_per_request,
        canonical_hash({"float64_by_ordinal": observed_per_request}),
        ordered_request_digest_v1(
            [str(item["request_sha256"]) for item in normalized]
        ),
        BATCH_SIZE,
        processed,
        0,
    )


def evaluate_routing_progress(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    objective: RoutingObjective | str,
    contexts: Sequence[Sequence[str]] | None = None,
) -> ControllerMarginReceipt | TargetNewNLLReceipt:
    """Evaluate the selected routing loss without changing margin control.

    The historical MARGIN branch is the existing function byte-for-byte.  The
    TARGET_NEW_NLL branch uses the request objects supplied to the gradient
    path directly, so its field derivative and candidate progress share one
    tokenizer/span contract and never access ``target_true``.
    """

    selected = select_locked_routing_objective(objective)
    if selected is RoutingObjective.MARGIN:
        return evaluate_controller_margin(
            model,
            tokenizer,
            requests,
            cumulative_factors_by_weight=cumulative_factors_by_weight,
        )
    parameter_state = tuple(
        (parameter.data_ptr(), parameter._version, parameter.grad)
        for parameter in model.parameters()
    )
    with _virtual_context(model, cumulative_factors_by_weight), torch.no_grad():
        result = evaluate_routing_objective(
            model,
            tokenizer,
            requests,
            objective=selected,
            contexts=contexts,
        )
    observed_per_request = tuple(
        float(item)
        for item in result.per_request_values.detach().to(
            device="cpu", dtype=torch.float64
        )
    )
    observed_model_mean = float(
        result.loss.detach().to(device="cpu", dtype=torch.float64)
    )
    observed = math.fsum(observed_per_request) / BATCH_SIZE
    if (
        len(observed_per_request) != BATCH_SIZE
        or not all(math.isfinite(item) for item in observed_per_request)
        or not math.isfinite(observed_model_mean)
        or not math.isclose(
            observed,
            observed_model_mean,
            rel_tol=1.0e-6,
            abs_tol=1.0e-7,
        )
        or result.target_true_suffix_token_counts != (None,) * BATCH_SIZE
    ):
        raise ODEBFContractError("target-new routing objective receipt differs")
    if tuple(
        (parameter.data_ptr(), parameter._version, parameter.grad)
        for parameter in model.parameters()
    ) != parameter_state:
        raise ODEBFContractError("target-new routing objective mutated model state")
    payload = {
        "objective": selected.value,
        "value_sha256": canonical_hash({"float64": observed}),
        "per_request_value_sha256": canonical_hash(
            {"float64_by_ordinal": observed_per_request}
        ),
        "request_order_sha256": result.request_order_sha256,
        "suffix_token_counts": list(result.suffix_token_counts),
        "target_span_sha256": result.target_span_sha256,
        "context_group_sizes": list(result.context_group_sizes),
        "context_count": result.context_count,
        "context_sha256": result.context_sha256,
        "model_forward_count": result.model_forward_count,
        "processed_token_count": result.processed_token_count,
        "generation_call_count": result.generation_call_count,
        "target_old_access_count": 0,
    }
    return TargetNewNLLReceipt(
        selected.value,
        observed,
        payload["value_sha256"],
        observed_per_request,
        payload["per_request_value_sha256"],
        result.request_order_sha256,
        result.suffix_token_counts,
        result.target_span_sha256,
        result.context_group_sizes,
        result.context_count,
        result.context_sha256,
        result.model_forward_count,
        result.processed_token_count,
        result.generation_call_count,
        0,
        canonical_hash(payload),
    )


@dataclass(frozen=True, slots=True)
class SignedProgressReceipt:
    field_sha256: str
    signed_progress: tuple[float, ...]
    excluded_nonpositive_layers: tuple[int, ...]
    raw_gradient_sha256: str
    model_forward_count: int
    processed_token_count: int
    fp32_gradient_only: bool
    routing_objective: str = RoutingObjective.MARGIN.value
    target_old_access_count: int = BATCH_SIZE
    objective_receipt_sha256: str | None = None
    routing_context_sha256: str | None = None
    routing_context_count: int = 1
    routing_context_group_sizes: tuple[int, ...] = (1,)
    routing_backward_count: int = 1
    mean_objective_value: float | None = None


def signed_progress_gradient(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    field: P1DynamicField,
    *,
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    ledger: ComputeLedger,
    objective: RoutingObjective | str = RoutingObjective.MARGIN,
    contexts: Sequence[Sequence[str]] | None = None,
) -> SignedProgressReceipt:
    selected = select_locked_routing_objective(objective)
    device = next(model.parameters()).device
    coefficients = torch.zeros(
        len(field.layers), dtype=torch.float32, device=device, requires_grad=True
    )
    parameter_state = tuple(
        (parameter.data_ptr(), parameter._version, parameter.grad)
        for parameter in model.parameters()
    )
    with _virtual_context(model, cumulative_factors_by_weight):
        with _CoefficientOverlay(model, field.layers, coefficients):
            if selected is RoutingObjective.MARGIN:
                loss, _, processed = _controller_margin_loss(
                    model, tokenizer, requests
                )
                objective_receipt_sha256 = None
                target_old_access_count = BATCH_SIZE
            else:
                objective_result = evaluate_routing_objective(
                    model,
                    tokenizer,
                    requests,
                    objective=selected,
                    contexts=contexts,
                    gradient_input=coefficients,
                )
                if (
                    objective_result.input_gradient is None
                    or objective_result.backward_count != BATCH_SIZE
                ):
                    raise ODEBFContractError(
                        "target-new routing streamed gradient differs"
                    )
                loss = objective_result.loss
                processed = objective_result.processed_token_count
                target_old_access_count = 0
                objective_receipt_sha256 = canonical_hash(
                    {
                        "objective": selected.value,
                        "request_order_sha256": (
                            objective_result.request_order_sha256
                        ),
                        "suffix_token_counts": list(
                            objective_result.suffix_token_counts
                        ),
                        "target_span_sha256": (
                            objective_result.target_span_sha256
                        ),
                        "context_group_sizes": list(
                            objective_result.context_group_sizes
                        ),
                        "context_count": objective_result.context_count,
                        "context_sha256": objective_result.context_sha256,
                        "target_old_access_count": 0,
                        "routing_backward_count": objective_result.backward_count,
                    }
                )
                gradient = objective_result.input_gradient
            if selected is RoutingObjective.MARGIN:
                gradient = torch.autograd.grad(
                    loss, coefficients, retain_graph=False
                )[0]
    raw = -gradient.detach().to(device="cpu", dtype=torch.float64)
    progress = tuple(float(value) for value in raw)
    excluded = tuple(
        field.layers[index].layer for index, value in enumerate(progress) if value <= 0.0
    )
    if len(excluded) == len(progress):
        raise ODEBFContractError("P1 controller has no positive signed-progress direction")
    if tuple(
        (parameter.data_ptr(), parameter._version, parameter.grad)
        for parameter in model.parameters()
    ) != parameter_state:
        raise ODEBFContractError("P1 FP32 gradient overlay mutated model state")
    routing_backward_count = (
        1
        if selected is RoutingObjective.MARGIN
        else objective_result.backward_count
    )
    ledger.increment("backward", routing_backward_count)
    return SignedProgressReceipt(
        field.identity_sha256,
        progress,
        excluded,
        tensor_sha256(gradient.detach()),
        BATCH_SIZE
        if selected is RoutingObjective.MARGIN
        else objective_result.model_forward_count,
        processed,
        True,
        selected.value,
        target_old_access_count,
        objective_receipt_sha256,
        None
        if selected is RoutingObjective.MARGIN
        else objective_result.context_sha256,
        1 if selected is RoutingObjective.MARGIN else objective_result.context_count,
        (1,)
        if selected is RoutingObjective.MARGIN
        else objective_result.context_group_sizes,
        routing_backward_count,
    )


@dataclass(frozen=True, slots=True)
class TargetVelocityReceipt:
    field_sha256: str
    velocity_sha256: str
    gradient_norm: float
    velocity_norm: float
    trust_radius: float
    target_backward_count: int
    processed_token_count: int


def write_aware_target_velocity(
    model: torch.nn.Module,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    field: P1DynamicField,
    coefficients: Sequence[float],
    *,
    cumulative_factors_by_weight: Mapping[str, Sequence[WaypointFactor]],
    target_base: torch.Tensor,
    native_target: torch.Tensor,
    trust_fraction: float,
    ledger: ComputeLedger,
) -> tuple[torch.Tensor, TargetVelocityReceipt]:
    device = next(model.parameters()).device
    coefficient_tensor = torch.tensor(
        tuple(float(value) for value in coefficients),
        dtype=torch.float32,
        device=device,
    )
    target = field.target_state.to(device=device, dtype=torch.float32).detach().requires_grad_(True)
    with _virtual_context(model, cumulative_factors_by_weight):
        with _CoefficientOverlay(
            model,
            field.layers,
            coefficient_tensor,
            target_state_variable=target,
            current_z=field.current_z,
        ):
            loss, _, processed = _controller_margin_loss(model, tokenizer, requests)
            gradient = torch.autograd.grad(loss, target, retain_graph=False)[0]
    raw_velocity = -gradient.detach().to(device="cpu", dtype=torch.float32)
    gradient_norm = float(torch.linalg.norm(raw_velocity))
    radius = float(torch.linalg.norm(native_target.float() - target_base.float())) * float(
        trust_fraction
    )
    if not math.isfinite(radius) or radius <= 0.0 or gradient_norm <= 0.0:
        raise ODEBFContractError("P1 target velocity/trust geometry is degenerate")
    velocity = raw_velocity * (radius / max(gradient_norm, torch.finfo(torch.float32).eps))
    ledger.increment("backward")
    ledger.increment("target_backward")
    receipt = TargetVelocityReceipt(
        field.identity_sha256,
        tensor_sha256(velocity),
        gradient_norm,
        float(torch.linalg.norm(velocity)),
        radius,
        1,
        processed,
    )
    return velocity, receipt
