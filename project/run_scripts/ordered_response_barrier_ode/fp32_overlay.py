"""Grouped FULL-FP32 virtual overlay and one-time shadow materialization."""

from __future__ import annotations

import hashlib
import math
import secrets
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from project.run_scripts.ode_edit_motivation.hooks import resolve_module, resolve_parameter

from .contracts import NumericalMethodBoundary, StaleStateBoundary, TechnicalBoundary


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_set_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"orbode.tensor-set.v1\0")
    for name in sorted(values):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_sha256(values[name]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class OverlayDelta:
    """One block-local detached direction in canonical weight orientation."""

    weight_name: str
    layer: int
    left: torch.Tensor
    right: torch.Tensor
    coefficient: float
    built_state_version: int
    build_identity: str

    def __post_init__(self) -> None:
        if (
            not self.weight_name.endswith(".weight")
            or isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.left.ndim != 2
            or self.right.ndim != 2
            or self.left.shape[1] != self.right.shape[1]
            or self.left.shape[1] <= 0
            or self.left.dtype is not torch.float32
            or self.right.dtype is not torch.float32
            or self.left.requires_grad
            or self.right.requires_grad
            or not bool(torch.isfinite(self.left).all())
            or not bool(torch.isfinite(self.right).all())
            or not math.isfinite(self.coefficient)
            or self.coefficient < 0.0
            or isinstance(self.built_state_version, bool)
            or self.built_state_version < 0
            or len(self.build_identity) != 64
        ):
            raise TechnicalBoundary("overlay delta contract differs")
        object.__setattr__(self, "left", self.left.detach().clone().contiguous())
        object.__setattr__(self, "right", self.right.detach().clone().contiguous())

    @property
    def rank(self) -> int:
        return int(self.left.shape[1])

    def with_coefficient(self, coefficient: float) -> "OverlayDelta":
        return OverlayDelta(
            self.weight_name,
            self.layer,
            self.left,
            self.right,
            coefficient,
            self.built_state_version,
            self.build_identity,
        )


@dataclass(frozen=True, slots=True)
class SweepEntryToken:
    """Unforgeable-in-process authority for JAC's one-sweep frozen field.

    A token can only be issued at the exact current overlay version and is
    valid until :meth:`GroupedFP32Overlay.close_sweep` is called.  This avoids
    the former ``built_version <= current_version`` escape hatch: a caller
    cannot relabel an arbitrary stale factor as a Jacobi field.
    """

    overlay_identity: str
    sweep: int
    state_version: int
    nonce: str


class GroupedFP32Overlay(AbstractContextManager["GroupedFP32Overlay"]):
    """Observe a cumulative low-rank W(t) while live parameter bytes stay at W0.

    Hooks evaluate ``linear(x, W0) + sum alpha * (x R) L.T``.  All selected
    layers are active in the same forward, so downstream propagation sees the
    exact joint virtual state.  No dense effective weight is kept hot.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        weight_names_by_layer: Mapping[int, str],
    ) -> None:
        if not weight_names_by_layer:
            raise TechnicalBoundary("overlay target inventory is empty")
        ordered = tuple(sorted((int(layer), str(name)) for layer, name in weight_names_by_layer.items()))
        if len({layer for layer, _ in ordered}) != len(ordered) or len({name for _, name in ordered}) != len(ordered):
            raise TechnicalBoundary("overlay layer/weight inventory contains duplicates")
        self.model = model
        self.weight_names_by_layer = dict(ordered)
        self._parameters: dict[str, torch.nn.Parameter] = {}
        self._entry_pointer: dict[str, int] = {}
        self._entry_version: dict[str, int] = {}
        self._entry_sha256: dict[str, str] = {}
        self._deltas: list[OverlayDelta] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.state_version = 0
        self.forward_hook_calls = 0
        self.shadow_materialization_count = 0
        self.physical_write_count = 0
        self._active = False
        self._sweep_token: SweepEntryToken | None = None

        for _, name in ordered:
            parameter = resolve_parameter(model, name)
            if parameter.ndim != 2 or parameter.dtype is not torch.float32 or parameter.grad is not None:
                raise TechnicalBoundary(f"overlay target is not an idle FP32 matrix: {name}")
            self._parameters[name] = parameter
            self._entry_pointer[name] = int(parameter.data_ptr())
            self._entry_version[name] = int(parameter._version)
            self._entry_sha256[name] = tensor_sha256(parameter)
        identity = hashlib.sha256(b"orbode.overlay.v1\0")
        for layer, name in ordered:
            identity.update(f"{layer}:{name}:{self._entry_sha256[name]}\0".encode("utf-8"))
        self._overlay_identity = identity.hexdigest()

    @property
    def deltas(self) -> tuple[OverlayDelta, ...]:
        return tuple(self._deltas)

    @property
    def factor_count(self) -> int:
        return len(self._deltas)

    @property
    def accumulated_rank(self) -> int:
        return sum(delta.rank for delta in self._deltas)

    @property
    def entry_selected_sha256(self) -> str:
        return tensor_set_sha256(self._parameters)

    @property
    def virtual_state_identity_sha256(self) -> str:
        """Identity of W0 plus the ordered low-rank coefficient ledger.

        This is intentionally not advertised as a dense endpoint byte hash;
        dense shadow bytes are materialized exactly once at terminal.
        """

        digest = hashlib.sha256(b"orbode.virtual-state.v1\0")
        digest.update(self._overlay_identity.encode("ascii"))
        digest.update(b"\0")
        for delta in self._deltas:
            digest.update(delta.build_identity.encode("ascii"))
            digest.update(b":")
            digest.update(float(delta.coefficient).hex().encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _hook(self, weight_name: str):
        def apply(module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> torch.Tensor:
            del module
            if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
                raise TechnicalBoundary("overlay rewrite module has unexpected IO")
            hidden = inputs[0]
            if hidden.dtype is not torch.float32 or output.dtype is not torch.float32:
                raise TechnicalBoundary("overlay activation left FULL_FP32")
            result = output
            for delta in self._deltas:
                if delta.weight_name != weight_name or delta.coefficient == 0.0:
                    continue
                left = delta.left.to(device=hidden.device, dtype=torch.float32)
                right = delta.right.to(device=hidden.device, dtype=torch.float32)
                result = result + float(delta.coefficient) * ((hidden @ right) @ left.transpose(0, 1))
            self.forward_hook_calls += 1
            return result

        return apply

    def _install_handles(self) -> None:
        if self._handles:
            raise TechnicalBoundary("overlay hooks are already installed")
        try:
            for _, weight_name in sorted(self.weight_names_by_layer.items()):
                module = resolve_module(self.model, weight_name.removesuffix(".weight"))
                if getattr(module, "weight", None) is not self._parameters[weight_name]:
                    raise TechnicalBoundary(
                        f"overlay module/parameter binding differs: {weight_name}"
                    )
                self._handles.append(module.register_forward_hook(self._hook(weight_name)))
        except BaseException:
            self._remove_handles()
            raise

    def _remove_handles(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "GroupedFP32Overlay":
        if self._active:
            raise TechnicalBoundary("grouped overlay is already active")
        self.assert_w0_unchanged(full_bytes=True)
        self._install_handles()
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, traceback
        self._remove_handles()
        self._active = False
        self._sweep_token = None
        try:
            self.assert_w0_unchanged(full_bytes=True)
        except BaseException as guard:
            if exc is not None:
                raise guard from exc
            raise
        return False

    @contextmanager
    def suspend(self, *, authoritative: bool):
        """Temporarily expose physical W for the terminal transaction only.

        The terminal callback must apply, observe, and restore its shadow while
        this context is open.  Both sides are byte-checked at W0, so a callback
        cannot accidentally evaluate ``shadow + overlay`` or leak an edit into
        the next arm.
        """

        if type(authoritative) is not bool:
            raise TechnicalBoundary("terminal suspension authority must be boolean")
        if not self._active or not self._handles:
            raise TechnicalBoundary("overlay suspension requires installed active hooks")
        self.assert_w0_unchanged(full_bytes=True)
        self._remove_handles()
        try:
            yield
        finally:
            violations: list[str] = []
            for name, parameter in self._parameters.items():
                if int(parameter.data_ptr()) != self._entry_pointer[name]:
                    violations.append(f"{name}:pointer")
                if parameter.dtype is not torch.float32 or parameter.grad is not None:
                    violations.append(f"{name}:dtype-or-grad")
                if tensor_sha256(parameter) != self._entry_sha256[name]:
                    violations.append(f"{name}:bytes")
            if violations:
                raise TechnicalBoundary(
                    "terminal transaction did not restore W0: " + ",".join(violations)
                )
            # A sanctioned copy-in/copy-back advances Tensor._version even
            # though pointer and bytes are restored.  Rebase the cheap guard
            # only after the full-byte terminal boundary succeeds.
            self._entry_version = {
                name: int(parameter._version) for name, parameter in self._parameters.items()
            }
            self.physical_write_count += int(authoritative)
            self._install_handles()

    def seal_sweep_entry(self, sweep: int) -> SweepEntryToken:
        if not self._active or self._sweep_token is not None:
            raise TechnicalBoundary("JAC sweep entry may be sealed exactly once while active")
        if isinstance(sweep, bool) or not isinstance(sweep, int) or sweep < 0:
            raise TechnicalBoundary("JAC sweep ordinal is invalid")
        token = SweepEntryToken(
            overlay_identity=self._overlay_identity,
            sweep=sweep,
            state_version=self.state_version,
            nonce=secrets.token_hex(16),
        )
        self._sweep_token = token
        return token

    def close_sweep(self, token: SweepEntryToken) -> None:
        if token is not self._sweep_token:
            raise StaleStateBoundary("JAC sweep token is not the active exact token")
        self._sweep_token = None

    def assert_w0_unchanged(self, *, full_bytes: bool) -> None:
        violations: list[str] = []
        for name, parameter in self._parameters.items():
            if int(parameter.data_ptr()) != self._entry_pointer[name]:
                violations.append(f"{name}:pointer")
            if int(parameter._version) != self._entry_version[name]:
                violations.append(f"{name}:version")
            if parameter.dtype is not torch.float32 or parameter.grad is not None:
                violations.append(f"{name}:dtype-or-grad")
            if full_bytes and tensor_sha256(parameter) != self._entry_sha256[name]:
                violations.append(f"{name}:bytes")
        if violations:
            raise TechnicalBoundary("live W0 mutated under virtual overlay: " + ",".join(violations))

    def append(
        self,
        delta: OverlayDelta,
        *,
        sweep_token: SweepEntryToken | None = None,
    ) -> int:
        if not self._active:
            raise TechnicalBoundary("overlay append requires an active observation context")
        expected_name = self.weight_names_by_layer.get(delta.layer)
        if expected_name != delta.weight_name:
            raise TechnicalBoundary("overlay delta layer/weight binding differs")
        parameter = self._parameters[delta.weight_name]
        if (delta.left.shape[0], delta.right.shape[0]) != tuple(parameter.shape):
            raise TechnicalBoundary("overlay factor does not match weight shape")
        allowed = (
            delta.built_state_version == self.state_version
            or (
                sweep_token is not None
                and sweep_token is self._sweep_token
                and sweep_token.overlay_identity == self._overlay_identity
                and delta.built_state_version == sweep_token.state_version
            )
        )
        if not allowed:
            raise StaleStateBoundary(
                f"field built at v{delta.built_state_version} consumed at v{self.state_version}"
            )
        if delta.coefficient == 0.0:
            return self.state_version
        self._deltas.append(delta)
        self.state_version += 1
        return self.state_version

    def prefix(self, count: int) -> tuple[OverlayDelta, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(self._deltas):
            raise TechnicalBoundary("overlay prefix count is invalid")
        return tuple(self._deltas[:count])

    def grouped(self, deltas: Iterable[OverlayDelta] | None = None) -> dict[str, tuple[OverlayDelta, ...]]:
        selected = self._deltas if deltas is None else list(deltas)
        result: dict[str, list[OverlayDelta]] = {name: [] for name in self._parameters}
        for delta in selected:
            if delta.weight_name not in result:
                raise TechnicalBoundary("overlay group contains an unknown weight")
            result[delta.weight_name].append(delta)
        return {name: tuple(items) for name, items in result.items()}

    def materialize_shadow(
        self,
        *,
        deltas: Iterable[OverlayDelta] | None = None,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Materialize one detached endpoint without touching authoritative W0."""

        grouped = self.grouped(deltas)
        shadow: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, parameter in self._parameters.items():
                value = parameter.detach().to(device=device, dtype=torch.float32).clone()
                for delta in grouped[name]:
                    left = delta.left.to(device=device, dtype=torch.float32)
                    right = delta.right.to(device=device, dtype=torch.float32)
                    value.addmm_(left, right.transpose(0, 1), alpha=float(delta.coefficient))
                if not bool(torch.isfinite(value).all()):
                    raise NumericalMethodBoundary("shadow endpoint is non-finite")
                shadow[name] = value.contiguous()
        self.shadow_materialization_count += 1
        self.assert_w0_unchanged(full_bytes=False)
        return shadow

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "orbode.grouped-fp32-overlay.v1",
            "state_version": self.state_version,
            "factor_count": self.factor_count,
            "accumulated_rank": self.accumulated_rank,
            "forward_hook_calls": self.forward_hook_calls,
            "shadow_materialization_count": self.shadow_materialization_count,
            "physical_inner_write_count": 0,
            "physical_terminal_write_count": self.physical_write_count,
            "w0_pointer_version_bytes_unchanged": True,
            "entry_component_sha256": dict(sorted(self._entry_sha256.items())),
            "entry_selected_sha256": tensor_set_sha256(self._parameters),
        }


__all__ = [
    "GroupedFP32Overlay",
    "OverlayDelta",
    "SweepEntryToken",
    "tensor_set_sha256",
    "tensor_sha256",
]
