"""ODE-edit-side exact transaction and terminal-net-write hooks."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch

from project.run_scripts.ode_edit_motivation.contracts import LowRankFactor
from project.run_scripts.ode_edit_motivation.hooks import resolve_parameter, tensor_sha256

from .contracts import MethodContractError, canonical_hash


class RollbackError(RuntimeError):
    """A rejected/failing trial could not restore its exact entry state."""


@dataclass(frozen=True, slots=True)
class FactorDirection:
    layer: int
    weight_name: str
    left: torch.Tensor = field(repr=False, compare=False)
    right: torch.Tensor = field(repr=False, compare=False)
    native_update_transposed: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int):
            raise MethodContractError("factor layer must be an integer")
        if not self.weight_name or self.left.ndim != 2 or self.right.ndim != 2:
            raise MethodContractError("factor direction shape/identity is invalid")
        if self.left.shape[1] != self.right.shape[1] or self.left.shape[1] <= 0:
            raise MethodContractError("factor direction ranks differ")
        if not torch.isfinite(self.left).all() or not torch.isfinite(self.right).all():
            raise MethodContractError("factor direction contains non-finite values")
        object.__setattr__(self, "left", self.left.detach().clone())
        object.__setattr__(self, "right", self.right.detach().clone())

    @classmethod
    def from_low_rank_factor(cls, layer: int, factor: LowRankFactor) -> "FactorDirection":
        return cls(
            layer=layer,
            weight_name=factor.weight_name,
            left=factor.left,
            right=factor.right,
            native_update_transposed=factor.native_update_transposed,
        )

    @property
    def direction_id(self) -> str:
        return canonical_hash(
            {
                "layer": self.layer,
                "weight_name": self.weight_name,
                "left_sha256": tensor_sha256(self.left),
                "right_sha256": tensor_sha256(self.right),
                "left_shape": list(self.left.shape),
                "right_shape": list(self.right.shape),
                "left_dtype": str(self.left.dtype),
                "right_dtype": str(self.right.dtype),
                "native_update_transposed": self.native_update_transposed,
            }
        )


@dataclass(slots=True)
class TorchCheckpoint:
    weight_names: tuple[str, ...]
    backups: dict[str, torch.Tensor] = field(repr=False)
    parameter_hashes: tuple[tuple[str, str], ...]
    cpu_rng: torch.Tensor = field(repr=False)
    cuda_rng: dict[torch.device, torch.Tensor] = field(repr=False)

    @property
    def state_id(self) -> str:
        return canonical_hash({"parameter_hashes": dict(self.parameter_hashes)})

    @classmethod
    def capture(
        cls, model: torch.nn.Module, weight_names: Sequence[str]
    ) -> "TorchCheckpoint":
        names = tuple(weight_names)
        if not names or len(names) != len(set(names)):
            raise MethodContractError("checkpoint weight names must be non-empty and unique")
        backups = {
            name: resolve_parameter(model, name).detach().clone() for name in names
        }
        hashes = tuple((name, tensor_sha256(backups[name])) for name in names)
        cuda_devices = {
            resolve_parameter(model, name).device
            for name in names
            if resolve_parameter(model, name).device.type == "cuda"
        }
        return cls(
            weight_names=names,
            backups=backups,
            parameter_hashes=hashes,
            cpu_rng=torch.get_rng_state().clone(),
            cuda_rng={
                device: torch.cuda.get_rng_state(device).clone()
                for device in cuda_devices
            },
        )

    def restore(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name in self.weight_names:
                resolve_parameter(model, name).copy_(self.backups[name])
        torch.set_rng_state(self.cpu_rng)
        for device, state in self.cuda_rng.items():
            torch.cuda.set_rng_state(state, device)
        self.assert_exact(model, include_rng=True)

    def assert_exact(self, model: torch.nn.Module, *, include_rng: bool) -> None:
        mismatches = []
        expected = dict(self.parameter_hashes)
        for name in self.weight_names:
            actual = tensor_sha256(resolve_parameter(model, name))
            if actual != expected[name]:
                mismatches.append(f"{name}: expected {expected[name]}, got {actual}")
        if include_rng and not torch.equal(torch.get_rng_state(), self.cpu_rng):
            mismatches.append("CPU RNG state differs")
        if include_rng:
            for device, state in self.cuda_rng.items():
                if not torch.equal(torch.cuda.get_rng_state(device), state):
                    mismatches.append(f"CUDA RNG state differs on {device}")
        if mismatches:
            raise RollbackError("checkpoint mismatch: " + "; ".join(mismatches))


def guarded_proposal_build(
    model: torch.nn.Module,
    weight_names: Sequence[str],
    build: Callable[[], Any],
) -> Any:
    """Require proposal construction to leave parameters and RNG bit-identical."""

    checkpoint = TorchCheckpoint.capture(model, weight_names)
    try:
        result = build()
        checkpoint.assert_exact(model, include_rng=True)
        return result
    except BaseException:
        checkpoint.restore(model)
        raise


class TorchFactorTrial:
    """Mutable dense-checkpoint trial retained as a CPU/reference oracle only."""

    def __init__(
        self,
        model: torch.nn.Module,
        directions: Sequence[FactorDirection],
        coefficients: Sequence[float],
        *,
        native_exact: bool = False,
    ) -> None:
        self.model = model
        self.directions = tuple(directions)
        self.coefficients = tuple(float(value) for value in coefficients)
        if (
            not self.directions
            or len(self.directions) != len(self.coefficients)
            or any(not math.isfinite(value) or value < 0.0 for value in self.coefficients)
        ):
            raise MethodContractError("trial directions/coefficients are invalid")
        names = tuple(direction.weight_name for direction in self.directions)
        if len(names) != len(set(names)):
            raise MethodContractError("one trial may write each layer weight only once")
        self.native_exact = bool(native_exact)
        self.checkpoint: TorchCheckpoint | None = None
        self._committed = False
        self._active = False
        self.applied_coefficients: tuple[float, ...] = ()

    def __enter__(self) -> "TorchFactorTrial":
        if self._active:
            raise RuntimeError("trial is already active")
        self.checkpoint = TorchCheckpoint.capture(
            self.model, tuple(direction.weight_name for direction in self.directions)
        )
        try:
            with torch.no_grad():
                for direction, coefficient in zip(
                    self.directions, self.coefficients, strict=True
                ):
                    parameter = resolve_parameter(self.model, direction.weight_name)
                    left = direction.left.to(parameter.device)
                    right = direction.right.to(parameter.device)
                    if self.native_exact and direction.native_update_transposed:
                        dense = (right @ left.transpose(0, 1)).transpose(0, 1)
                    else:
                        dense = left @ right.transpose(0, 1)
                    if tuple(dense.shape) != tuple(parameter.shape):
                        raise MethodContractError(
                            "factor update "
                            f"{tuple(dense.shape)} differs from {tuple(parameter.shape)}"
                        )
                    # Native/scalar freeze EasyEdit's terminal *net write*,
                    # including its float32 cast, before alpha is applied.
                    # Adaptive unit directions multiply y before the cast.
                    update = (
                        dense.float() * coefficient
                        if self.native_exact
                        else (dense * coefficient).float()
                    )
                    # The coefficient returned by the controller is multiplied
                    # exactly once.  No norm/share rescaling follows this line.
                    parameter[...] += update
            self.applied_coefficients = self.coefficients
        except BaseException:
            assert self.checkpoint is not None
            self.checkpoint.restore(self.model)
            raise
        self._active = True
        return self

    def commit(self) -> None:
        if not self._active:
            raise RuntimeError("cannot commit an inactive trial")
        self._committed = True

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        if not self._active:
            return False
        assert self.checkpoint is not None
        if exc is not None or not self._committed:
            self.checkpoint.restore(self.model)
        self._active = False
        return False


def _apply_factor_update_(
    parameter: torch.nn.Parameter,
    direction: FactorDirection,
    coefficient: float,
    *,
    native_exact: bool,
    row_block: int,
) -> None:
    """Apply one accepted update without constructing a full adaptive delta."""

    left = direction.left.to(parameter.device)
    right = direction.right.to(parameter.device)
    if tuple(parameter.shape) != (left.shape[0], right.shape[0]):
        raise MethodContractError("accepted-write factor/weight shapes differ")
    if native_exact:
        dense = (
            (right @ left.transpose(0, 1)).transpose(0, 1)
            if direction.native_update_transposed
            else left @ right.transpose(0, 1)
        )
        parameter.add_((dense.float() * coefficient).to(parameter.dtype))
        return
    # MEMIT factors/models use <=float32 precision.  Preserve the existing
    # float32 product/cast order there, while allowing float64 CPU identity
    # fixtures to remain a tight mathematical oracle.
    compute_dtype = torch.float64 if parameter.dtype is torch.float64 else torch.float32
    left_compute = left.to(dtype=compute_dtype)
    right_compute_t = right.to(dtype=compute_dtype).transpose(0, 1)
    for start in range(0, int(parameter.shape[0]), row_block):
        update = (
            left_compute[start : start + row_block] @ right_compute_t
        ) * coefficient
        parameter[start : start + row_block].add_(update.to(parameter.dtype))


def apply_accepted_factors(
    model: torch.nn.Module,
    directions: Sequence[FactorDirection],
    coefficients: Sequence[float],
    *,
    native_exact: bool = False,
    row_block: int = 64,
) -> tuple[float, ...]:
    """Persist one accepted coefficient vector inside an outer transaction.

    This function deliberately takes no per-step dense backup.  Its caller
    must hold the single edit-entry :class:`TorchCheckpoint` and restore that
    checkpoint if any layer update raises.  Rejected trials never call this
    function, and accepted updates remain row-blocked.
    """

    locked = tuple(directions)
    applied = tuple(float(value) for value in coefficients)
    if (
        not locked
        or len(locked) != len(applied)
        or any(not math.isfinite(value) or value < 0.0 for value in applied)
        or isinstance(row_block, bool)
        or not isinstance(row_block, int)
        or row_block <= 0
    ):
        raise MethodContractError("accepted-write directions/coefficients are invalid")
    names = tuple(direction.weight_name for direction in locked)
    if len(names) != len(set(names)):
        raise MethodContractError("accepted write weights repeat")
    parameters = tuple(resolve_parameter(model, name) for name in names)
    if any(parameter.grad is not None for parameter in parameters):
        raise MethodContractError("accepted-write target weight .grad must be None")
    for parameter, direction in zip(parameters, locked, strict=True):
        if tuple(parameter.shape) != (
            direction.left.shape[0],
            direction.right.shape[0],
        ):
            raise MethodContractError("accepted-write factor/weight shapes differ")
    pointers = tuple(parameter.data_ptr() for parameter in parameters)
    with torch.no_grad():
        for parameter, direction, coefficient in zip(
            parameters, locked, applied, strict=True
        ):
            _apply_factor_update_(
                parameter,
                direction,
                coefficient,
                native_exact=native_exact,
                row_block=row_block,
            )
    if any(
        parameter.data_ptr() != pointer
        for parameter, pointer in zip(parameters, pointers, strict=True)
    ):
        raise MethodContractError("accepted write replaced parameter storage")
    if any(parameter.grad is not None for parameter in parameters):
        raise MethodContractError("accepted write materialized target weight .grad")
    return applied


def terminal_net_c_energy(
    model: torch.nn.Module,
    entry: TorchCheckpoint,
    covariance_by_layer: Mapping[int, torch.Tensor],
    weight_by_layer: Mapping[int, str],
    *,
    row_block: int = 64,
) -> dict[int, float]:
    """Measure each layer's terminal net write, independent of subdivision."""

    if set(weight_by_layer.values()) != set(entry.weight_names):
        raise MethodContractError("terminal energy weights differ from entry checkpoint")
    result: dict[int, float] = {}
    for layer in sorted(weight_by_layer):
        try:
            name = weight_by_layer[layer]
            covariance = covariance_by_layer[layer]
        except KeyError as exc:
            raise MethodContractError("terminal energy covariance mapping is incomplete") from exc
        current = resolve_parameter(model, name).detach()
        delta = current - entry.backups[name].to(current.device)
        if (
            delta.ndim != 2
            or covariance.ndim != 2
            or covariance.shape[0] != covariance.shape[1]
            or delta.shape[1] != covariance.shape[0]
        ):
            raise MethodContractError("terminal net-write covariance geometry differs")
        metric = covariance.detach().to(device=delta.device, dtype=torch.float32)
        total = torch.zeros((), device=delta.device, dtype=torch.float64)
        with torch.inference_mode():
            for start in range(0, int(delta.shape[0]), row_block):
                block = delta[start : start + row_block].float()
                total += torch.sum(block.double() * (block @ metric).double())
        value = float(total.cpu())
        if not math.isfinite(value) or value < -1e-12:
            raise MethodContractError("terminal net-write C energy is invalid")
        result[int(layer)] = max(0.0, value)
    return result
