"""Method-generic bindings to stock MEMIT/AlphaEdit engineering primitives.

This module does not implement either writer equation.  Production bindings
provide callbacks backed by the pinned Official source (or an already verified
ODE-side bridge); the adapter only enforces fixed-z, state-version, factor and
method-state boundaries shared by both families.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from .contracts import StaleStateBoundary, TechnicalBoundary
from .fp32_overlay import OverlayDelta, tensor_sha256


def _digest_fields(*values: str) -> str:
    digest = hashlib.sha256(b"orbode.adapter-fields.v1\0")
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FixedZArtifact:
    values: torch.Tensor
    identity_sha256: str
    request_order_sha256: str
    target_context_identity_sha256: str

    def __post_init__(self) -> None:
        if (
            self.values.ndim != 2
            or self.values.dtype is not torch.float32
            or self.values.requires_grad
            or not bool(torch.isfinite(self.values).all())
            or any(len(value) != 64 for value in (
                self.identity_sha256,
                self.request_order_sha256,
                self.target_context_identity_sha256,
            ))
        ):
            raise TechnicalBoundary("fixed native z artifact contract differs")
        object.__setattr__(self, "values", self.values.detach().clone().contiguous())
        if tensor_sha256(self.values) != self.identity_sha256:
            raise TechnicalBoundary("fixed native z tensor identity differs")

    @property
    def request_count(self) -> int:
        return int(self.values.shape[1])


@dataclass(frozen=True, slots=True)
class LayerBuild:
    layer: int
    weight_name: str
    left: torch.Tensor
    right: torch.Tensor
    built_state_version: int
    residual_denominator: int
    residual_sha256: str
    keys_sha256: str
    solver_identity: str
    solve_backward_error: float | None = None
    factor_rank: int = field(init=False)
    build_identity: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or not self.weight_name.endswith(".weight")
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
            or isinstance(self.built_state_version, bool)
            or self.built_state_version < 0
            or isinstance(self.residual_denominator, bool)
            or self.residual_denominator <= 0
            or any(len(value) != 64 for value in (self.residual_sha256, self.keys_sha256))
            or not self.solver_identity
            or (
                self.solve_backward_error is not None
                and (self.solve_backward_error < 0.0 or not torch.isfinite(torch.tensor(self.solve_backward_error)))
            )
        ):
            raise TechnicalBoundary("Official-form layer build contract differs")
        object.__setattr__(self, "left", self.left.detach().clone().contiguous())
        object.__setattr__(self, "right", self.right.detach().clone().contiguous())
        object.__setattr__(self, "factor_rank", int(self.left.shape[1]))
        object.__setattr__(
            self,
            "build_identity",
            _digest_fields(
                str(self.layer),
                self.weight_name,
                tensor_sha256(self.left),
                tensor_sha256(self.right),
                str(self.built_state_version),
                str(self.residual_denominator),
                self.residual_sha256,
                self.keys_sha256,
                self.solver_identity,
            ),
        )

    def overlay_delta(self, coefficient: float) -> OverlayDelta:
        return OverlayDelta(
            weight_name=self.weight_name,
            layer=self.layer,
            left=self.left,
            right=self.right,
            coefficient=coefficient,
            built_state_version=self.built_state_version,
            build_identity=self.build_identity,
        )


@dataclass(slots=True)
class AdapterLedger:
    fixed_z_compute_count: int = 0
    fixed_z_recompute_count: int = 0
    terminal_capture_count: int = 0
    key_capture_count: int = 0
    layer_factorization_count: int = 0
    solve_count: int = 0
    official_endpoint_count: int = 0
    terminal_finalize_count: int = 0
    inner_history_append_count: int = 0
    inner_cache_mutation_count: int = 0
    dynamic_z_count: int = 0

    def raw_free_payload(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in self.__dataclass_fields__}


class FixedZCompute(Protocol):
    def __call__(self) -> FixedZArtifact: ...


class TerminalCapture(Protocol):
    def __call__(self) -> torch.Tensor: ...


class LayerBuilder(Protocol):
    def __call__(
        self,
        *,
        layer: int,
        current_terminal: torch.Tensor,
        fixed_z: FixedZArtifact,
        residual_denominator: int,
        state_version: int,
    ) -> LayerBuild: ...


class OfficialEndpointRunner(Protocol):
    def __call__(self, *, fixed_z: FixedZArtifact) -> Mapping[str, Any]: ...


class TerminalFinalizer(Protocol):
    def __call__(
        self,
        *,
        arm: str,
        terminal_state_version: int,
        shadow_weights: Mapping[str, torch.Tensor],
        deltas: Sequence[OverlayDelta],
        derived_observation_only: bool,
    ) -> Mapping[str, Any]: ...


class CurrentEndpointEvaluator(Protocol):
    def __call__(self, *, arm: str, status: str, state_version: int) -> Mapping[str, Any]: ...


class MethodStateGuard(Protocol):
    def __call__(self) -> str: ...


class MethodAdapter:
    """Runtime interface consumed by the science-only ordered integrator."""

    family: str
    layers: tuple[int, ...]
    weight_names_by_layer: Mapping[int, str]
    ledger: AdapterLedger

    def bind_state_version(self, state_version: int) -> None:
        raise NotImplementedError

    def compute_fixed_z_once(self) -> FixedZArtifact:
        raise NotImplementedError

    def current_terminal(self, *, state_version: int) -> torch.Tensor:
        raise NotImplementedError

    def build_layer(
        self,
        *,
        layer: int,
        residual_denominator: int,
        state_version: int,
        current_terminal: torch.Tensor,
        fixed_z: FixedZArtifact,
    ) -> LayerBuild:
        raise NotImplementedError

    def run_official(self, *, fixed_z: FixedZArtifact) -> Mapping[str, Any]:
        raise NotImplementedError

    def finalize_terminal(
        self,
        *,
        arm: str,
        state_version: int,
        shadow_weights: Mapping[str, torch.Tensor],
        deltas: Sequence[OverlayDelta],
        derived_observation_only: bool = False,
    ) -> Mapping[str, Any]:
        raise NotImplementedError

    def evaluate_current(self, *, arm: str, status: str, state_version: int) -> Mapping[str, Any]:
        raise NotImplementedError

    def method_state_identity(self) -> str:
        raise NotImplementedError


class CallbackMethodAdapter(MethodAdapter):
    """Thin guarded composition over stock-method callbacks."""

    def __init__(
        self,
        *,
        family: str,
        layers: Sequence[int],
        weight_names_by_layer: Mapping[int, str],
        compute_fixed_z: FixedZCompute,
        capture_terminal: TerminalCapture,
        build_official_form_layer: LayerBuilder,
        run_official_endpoint: OfficialEndpointRunner,
        finalize_terminal_state: TerminalFinalizer,
        evaluate_current_endpoint: CurrentEndpointEvaluator,
        capture_method_state_identity: MethodStateGuard,
    ) -> None:
        normalized_layers = tuple(int(layer) for layer in layers)
        if (
            family not in {"MEMIT", "AlphaEdit"}
            or normalized_layers != (4, 5, 6, 7, 8)
            or set(weight_names_by_layer) != set(normalized_layers)
            or len(set(weight_names_by_layer.values())) != len(normalized_layers)
        ):
            raise TechnicalBoundary("method adapter family/layer inventory differs")
        self.family = family
        self.layers = normalized_layers
        self.weight_names_by_layer = dict(weight_names_by_layer)
        self._compute_fixed_z = compute_fixed_z
        self._capture_terminal = capture_terminal
        self._build_layer = build_official_form_layer
        self._run_official = run_official_endpoint
        self._finalize = finalize_terminal_state
        self._evaluate_current = evaluate_current_endpoint
        self._capture_method_state = capture_method_state_identity
        self._fixed_z: FixedZArtifact | None = None
        self._expected_state_version = 0
        self._entry_method_state = self._capture_method_state()
        if len(self._entry_method_state) != 64:
            raise TechnicalBoundary("method entry-state identity is not SHA-256")
        self.ledger = AdapterLedger()

    def bind_state_version(self, state_version: int) -> None:
        if isinstance(state_version, bool) or not isinstance(state_version, int) or state_version < 0:
            raise TechnicalBoundary("adapter state version is invalid")
        self._expected_state_version = state_version

    def _assert_state_version(self, state_version: int) -> None:
        if state_version != self._expected_state_version:
            raise StaleStateBoundary(
                f"adapter expected v{self._expected_state_version}, observed v{state_version}"
            )

    def compute_fixed_z_once(self) -> FixedZArtifact:
        if self._fixed_z is not None:
            self.ledger.fixed_z_recompute_count += 1
            raise StaleStateBoundary("native fixed z recomputation is forbidden")
        artifact = self._compute_fixed_z()
        if not isinstance(artifact, FixedZArtifact):
            raise TechnicalBoundary("stock compute-z binding returned an invalid artifact")
        self._fixed_z = artifact
        self.ledger.fixed_z_compute_count += artifact.request_count
        return artifact

    def use_fixed_z(self, artifact: FixedZArtifact) -> None:
        """Bind the once-computed cohort artifact to a fresh arm adapter."""

        if self._fixed_z is not None or not isinstance(artifact, FixedZArtifact):
            raise TechnicalBoundary("fixed-z arm binding is repeated or invalid")
        self._fixed_z = artifact

    def current_terminal(self, *, state_version: int) -> torch.Tensor:
        self._assert_state_version(state_version)
        value = self._capture_terminal()
        self.ledger.terminal_capture_count += 1
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.dtype is not torch.float32
            or value.requires_grad
            or not bool(torch.isfinite(value).all())
            or (self._fixed_z is not None and value.shape != self._fixed_z.values.shape)
        ):
            raise TechnicalBoundary("shared terminal activation capture differs")
        return value.detach().clone().contiguous()

    def build_layer(
        self,
        *,
        layer: int,
        residual_denominator: int,
        state_version: int,
        current_terminal: torch.Tensor,
        fixed_z: FixedZArtifact,
    ) -> LayerBuild:
        self._assert_state_version(state_version)
        if fixed_z is not self._fixed_z or layer not in self.layers:
            raise StaleStateBoundary("writer requested a different target/layer")
        build = self._build_layer(
            layer=layer,
            current_terminal=current_terminal,
            fixed_z=fixed_z,
            residual_denominator=residual_denominator,
            state_version=state_version,
        )
        if (
            not isinstance(build, LayerBuild)
            or build.layer != layer
            or build.weight_name != self.weight_names_by_layer[layer]
            or build.built_state_version != state_version
            or build.residual_denominator != residual_denominator
        ):
            raise TechnicalBoundary("Official-form writer build differs from request")
        self.ledger.key_capture_count += 1
        self.ledger.layer_factorization_count += 1
        self.ledger.solve_count += 1
        return build

    def run_official(self, *, fixed_z: FixedZArtifact) -> Mapping[str, Any]:
        if fixed_z is not self._fixed_z:
            raise TechnicalBoundary("Official endpoint target differs from shared fixed z")
        self._assert_state_version(0)
        result = self._run_official(fixed_z=fixed_z)
        self.ledger.official_endpoint_count += 1
        return dict(result)

    def finalize_terminal(
        self,
        *,
        arm: str,
        state_version: int,
        shadow_weights: Mapping[str, torch.Tensor],
        deltas: Sequence[OverlayDelta],
        derived_observation_only: bool = False,
    ) -> Mapping[str, Any]:
        self._assert_state_version(state_version)
        if set(shadow_weights) != set(self.weight_names_by_layer.values()):
            raise TechnicalBoundary("terminal shadow weight inventory differs")
        if any(delta.weight_name not in shadow_weights for delta in deltas):
            raise TechnicalBoundary("terminal delta inventory differs")
        result = dict(self._finalize(
            arm=arm,
            terminal_state_version=state_version,
            shadow_weights=shadow_weights,
            deltas=tuple(deltas),
            derived_observation_only=derived_observation_only,
        ))
        self.ledger.terminal_finalize_count += 1
        if self.family == "AlphaEdit":
            expected = 0 if derived_observation_only else 1
            if int(result.get("history_append_count", -1)) != expected:
                raise TechnicalBoundary("AlphaEdit terminal history append count differs")
        elif int(result.get("history_append_count", -1)) != 0:
            raise TechnicalBoundary("native MEMIT cannot append AlphaEdit history")
        if int(result.get("inner_history_append_count", -1)) != 0 or int(result.get("inner_cache_mutation_count", -1)) != 0:
            raise TechnicalBoundary("inner method state mutated")
        return result

    def evaluate_current(self, *, arm: str, status: str, state_version: int) -> Mapping[str, Any]:
        self._assert_state_version(state_version)
        result = dict(self._evaluate_current(arm=arm, status=status, state_version=state_version))
        if int(result.get("physical_write_count", -1)) != 0 or int(result.get("history_append_count", -1)) != 0:
            raise TechnicalBoundary("current/no-op endpoint evaluation mutated writer state")
        return result

    def method_state_identity(self) -> str:
        value = self._capture_method_state()
        if len(value) != 64:
            raise TechnicalBoundary("method state identity is not SHA-256")
        return value

    def assert_entry_method_state(self) -> None:
        if self.method_state_identity() != self._entry_method_state:
            raise TechnicalBoundary("entry committed cache/history state changed")


__all__ = [
    "AdapterLedger",
    "CallbackMethodAdapter",
    "FixedZArtifact",
    "LayerBuild",
    "MethodAdapter",
]
