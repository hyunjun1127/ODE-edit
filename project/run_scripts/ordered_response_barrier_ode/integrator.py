"""Science-only ordered FP32 overlay integrator.

Model-, tokenizer-, writer- and evaluator-specific operations are injected via
the guarded adapter/JVP/semantic interfaces.  There is one loop for both
Official writer families and no arm-specific duplicated runtime.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

from .adapters import FixedZArtifact, LayerBuild, MethodAdapter
from .contracts import (
    ArmConfig,
    ArmId,
    FrozenAnchor,
    ORBODEContractError,
    RefreshPolicy,
    ResidualPolicy,
    ResponsePolicy,
    ScientificBoundary,
    TechnicalBoundary,
    build_frozen_anchor,
    normalized_potential,
    response_statistics,
)
from .fp32_overlay import GroupedFP32Overlay, OverlayDelta, SweepEntryToken, tensor_set_sha256
from .semantic import SemanticObservation
from .telemetry import ArmTelemetry, build_step_record, semantic_payload
from .terminal_jvp import TerminalJVPResult, TerminalResponseObserver


@dataclass(frozen=True, slots=True)
class DerivedEndpoint:
    arm: str
    status: str
    state_version: int
    deltas: tuple[OverlayDelta, ...]
    first_hit: Mapping[str, int] | None
    endpoint: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ArmExecutionResult:
    arm: str
    status: str
    request_count: int
    endpoint: Mapping[str, Any]
    telemetry: Mapping[str, Any]
    overlay: Mapping[str, Any] | None
    derived_endpoint: DerivedEndpoint | None
    fixed_z_identity_sha256: str
    scientific_promotion: bool = False


def _factor_frobenius(build: LayerBuild) -> float:
    left = build.left.detach().double()
    right = build.right.detach().double()
    squared = torch.sum((left.transpose(0, 1) @ left) * (right.transpose(0, 1) @ right))
    value = float(torch.sqrt(squared.clamp_min(0.0)).item())
    if not math.isfinite(value):
        raise ScientificBoundary("official-form factor Frobenius norm is non-finite")
    return value


def _residual_denominator(config: ArmConfig, layer_position: int, layer_count: int) -> int:
    if config.residual_policy is ResidualPolicy.CURRENT_REMAINING:
        return layer_count - layer_position
    if config.residual_policy is ResidualPolicy.CURRENT_FULL:
        return 1
    raise ORBODEContractError("dynamic integrator received Official residual policy")


def _anchor_semantic_observation(
    anchor: FrozenAnchor, semantic: SemanticObservation
) -> dict[str, int | str]:
    if len(semantic.request_strict) != anchor.request_count:
        raise TechnicalBoundary("semantic/target request denominator differs")
    invalid = [
        index
        for index, active in enumerate(anchor.active.tolist())
        if not active and not semantic.request_strict[index]
    ]
    return {
        "anchor_active_request_count": int(anchor.active.sum().item()),
        "anchor_zero_request_count": int((~anchor.active).sum().item()),
        "anchor_zero_semantic_miss_count": len(invalid),
        "anchor_zero_semantic_miss_policy": "NONBLOCKING_SCIENTIFIC_OBSERVATION",
    }


def terminal_flow_status(
    *,
    stalled: bool,
    terminal_hit: bool,
    first_hit_latched: bool,
    direct_first_hit_audit: bool = False,
) -> str:
    """Name terminal state independently from the observation-only hit latch."""

    if direct_first_hit_audit:
        if not terminal_hit or not first_hit_latched:
            raise ORBODEContractError("direct first-hit audit stopped without a strict hit")
        return "FIRST_HIT_AUDIT"
    if stalled:
        if terminal_hit:
            return "FLOW_STALLED_TERMINAL_HIT"
        if first_hit_latched:
            return "FLOW_STALLED_AFTER_TRANSIENT_HIT"
        return "FLOW_STALLED_TERMINAL_MISS"
    if terminal_hit:
        return "HORIZON_SEMANTIC_HIT_OBSERVED"
    if first_hit_latched:
        return "HORIZON_TRANSIENT_HIT_TERMINAL_MISS"
    return "HORIZON_SEMANTIC_MISS"


@contextmanager
def _progressive_failure_context(
    progress: dict[str, int], telemetry: ArmTelemetry, overlay: GroupedFP32Overlay
) -> Iterator[None]:
    try:
        yield
    except BaseException as exc:
        if not hasattr(exc, "orbode_progress"):
            setattr(exc, "orbode_progress", {
                "sweep": int(progress["sweep"]),
                "layer": int(progress["layer"]),
                "visit_ordinal": int(progress["visit_ordinal"]),
                "completed_step_count": len(telemetry.steps),
                "completed_prefix_factor_count": int(overlay.factor_count),
                "nonzero_action_visit_count": int(telemetry.nonzero_action_visit_count),
                "zero_action_visit_count": int(telemetry.zero_action_visit_count),
                "first_hit": telemetry.first_hit,
            })
        raise


class OrderedResponseIntegrator:
    def __init__(
        self,
        *,
        adapter: MethodAdapter,
        overlay: GroupedFP32Overlay,
        jvp: TerminalResponseObserver,
        observe_semantic: Callable[[], SemanticObservation],
    ) -> None:
        self.adapter = adapter
        self.overlay = overlay
        self.jvp = jvp
        self.observe_semantic = observe_semantic

    def run_official(self, config: ArmConfig, fixed_z: FixedZArtifact) -> ArmExecutionResult:
        if config.arm is not ArmId.OFFICIAL:
            raise ORBODEContractError("Official bypass requires O config")
        self.adapter.bind_state_version(0)
        semantic = self.observe_semantic()
        if len(semantic.request_strict) != fixed_z.request_count:
            raise ScientificBoundary("Official entry semantic/fixed-z denominator differs")
        if semantic.all_strict:
            status = "ENTRY_ALREADY_HIT"
            payload = dict(self.adapter.evaluate_current(
                arm=config.arm.value,
                status=status,
                state_version=0,
            ))
            return ArmExecutionResult(
                arm=config.arm.value,
                status=status,
                request_count=fixed_z.request_count,
                endpoint=payload,
                telemetry={
                    "schema": "orbode.official-bypass.v1",
                    "entry_semantic": semantic_payload(semantic),
                    "terminal_semantic": semantic_payload(semantic),
                    "native_compute_z_bypassed_with_shared_fixed_z": True,
                    "dynamic_layer_visit_count": 0,
                    "factor_build_count": 0,
                    "physical_write_count": 0,
                    "history_append_count": 0,
                    "terminal_net_frobenius": 0.0,
                    "terminal_net_frobenius_squared": 0.0,
                    "sum_step_action_frobenius_squared": 0.0,
                    "resolution_stable_path_frobenius_squared": 0.0,
                    "native_creg_action_status": "TELEMETRY_WITHHELD",
                },
                overlay=None,
                derived_endpoint=None,
                fixed_z_identity_sha256=fixed_z.identity_sha256,
            )
        payload = dict(self.adapter.run_official(fixed_z=fixed_z))
        terminal_semantic = payload.get("semantic_observation")
        if not isinstance(terminal_semantic, Mapping):
            raise TechnicalBoundary("Official endpoint omitted terminal semantic observation")
        return ArmExecutionResult(
            arm=config.arm.value,
            status=str(payload.get("status", "TERMINAL_VALID")),
            request_count=fixed_z.request_count,
            endpoint=payload,
            telemetry={
                "schema": "orbode.official-bypass.v1",
                "entry_semantic": semantic_payload(semantic),
                "terminal_semantic": dict(terminal_semantic),
                "native_compute_z_bypassed_with_shared_fixed_z": True,
                "dynamic_layer_visit_count": 0,
                "factor_build_count": 5,
                "physical_write_count": int(payload.get("physical_write_count", 1)),
                "history_append_count": int(payload.get("history_append_count", 0)),
                "terminal_net_frobenius": float(payload["terminal_net_frobenius"]),
                "terminal_net_frobenius_squared": float(
                    payload["terminal_net_frobenius_squared"]
                ),
                "sum_step_action_frobenius_squared": float(
                    payload["terminal_net_frobenius_squared"]
                ),
                "resolution_stable_path_frobenius_squared": float(
                    payload["terminal_net_frobenius_squared"]
                ),
                "native_creg_action_status": "TELEMETRY_WITHHELD",
            },
            overlay=None,
            derived_endpoint=None,
            fixed_z_identity_sha256=fixed_z.identity_sha256,
        )

    def _build_and_response(
        self,
        *,
        config: ArmConfig,
        layer: int,
        layer_position: int,
        fixed_z: FixedZArtifact,
        terminal: torch.Tensor,
        state_version: int,
    ) -> tuple[LayerBuild, TerminalJVPResult | None, float, Any | None, torch.Tensor]:
        denominator = _residual_denominator(config, layer_position, len(self.adapter.layers))
        command_residual = (
            (fixed_z.values - terminal) / float(denominator)
        ).detach().to(device="cpu", dtype=torch.float32).contiguous()
        build = self.adapter.build_layer(
            layer=layer,
            residual_denominator=denominator,
            state_version=state_version,
            current_terminal=terminal,
            fixed_z=fixed_z,
        )
        if config.response_policy is ResponsePolicy.UNIT:
            return build, None, 1.0, None, command_residual
        if config.response_policy is not ResponsePolicy.CURRENT_RESPONSE:
            raise ORBODEContractError("dynamic integrator response policy differs")
        observed = self.jvp.observe(build, expected_state_version=state_version)
        if not torch.equal(observed.terminal, terminal):
            raise ScientificBoundary("terminal JVP primal differs from current terminal bytes")
        statistics = response_statistics(
            anchor=self._anchor,
            terminal=terminal,
            response=observed.response,
        )
        return build, observed, statistics.u, statistics, command_residual

    def _record_and_advance(
        self,
        *,
        config: ArmConfig,
        sweep: int,
        layer: int,
        visit_ordinal: int,
        terminal_before: torch.Tensor,
        command_residual: torch.Tensor,
        build: LayerBuild,
        jvp: TerminalJVPResult | None,
        u: float,
        statistics: Any | None,
        semantic_before: SemanticObservation,
        sweep_token: SweepEntryToken | None,
        telemetry: ArmTelemetry,
    ) -> tuple[torch.Tensor, SemanticObservation, bool]:
        before_potential, _ = normalized_potential(self._anchor, terminal_before)
        coefficient = config.step_size * u
        acted = coefficient != 0.0
        if acted:
            self.overlay.append(build.overlay_delta(coefficient), sweep_token=sweep_token)
            self.adapter.bind_state_version(self.overlay.state_version)
            terminal_after = self.adapter.current_terminal(state_version=self.overlay.state_version)
            semantic_after = self.observe_semantic()
            telemetry.nonzero_action_visit_count += 1
        else:
            terminal_after = terminal_before
            semantic_after = semantic_before
            telemetry.zero_action_visit_count += 1
        after_potential, _ = normalized_potential(self._anchor, terminal_after)
        telemetry.steps.append(
            build_step_record(
                arm=config.arm,
                sweep=sweep,
                layer=layer,
                visit_ordinal=visit_ordinal,
                built_state_version=build.built_state_version,
                resulting_state_version=self.overlay.state_version,
                residual_denominator=build.residual_denominator,
                build_identity=build.build_identity,
                residual_sha256=build.residual_sha256,
                keys_sha256=build.keys_sha256,
                solver_identity=build.solver_identity,
                factor_rank=build.factor_rank,
                u=u,
                h=config.step_size,
                entry_potential=self._entry_potential,
                potential_before=before_potential,
                potential_after=after_potential,
                target=self._anchor.target,
                command_residual=command_residual,
                anchor_scales=self._anchor.scales,
                anchor_active=self._anchor.active,
                terminal_before=terminal_before,
                terminal_after=terminal_after,
                direction_frobenius=_factor_frobenius(build),
                response=None if jvp is None else jvp.response,
                statistics=statistics,
                semantic=semantic_after,
                virtual_state_identity_sha256=self.overlay.virtual_state_identity_sha256,
            )
        )
        telemetry.factor_build_count += 1
        telemetry.key_capture_count += 1
        telemetry.terminal_capture_count += int(acted)
        telemetry.jvp_call_count += int(jvp is not None)
        return terminal_after, semantic_after, acted

    def run_dynamic(
        self,
        config: ArmConfig,
        fixed_z: FixedZArtifact,
        *,
        preamble_stop_at_first_hit: bool = False,
    ) -> ArmExecutionResult:
        if config.arm in {ArmId.OFFICIAL, ArmId.ORDERED_RESPONSE_FIRST_HIT}:
            raise ORBODEContractError("dynamic primary arm config differs")
        if preamble_stop_at_first_hit and config.arm is not ArmId.ORDERED_RESPONSE_FIXED_HORIZON:
            raise ORBODEContractError("direct first-hit audit is ORBFH preamble-only")
        self.adapter.bind_state_version(0)
        terminal = self.adapter.current_terminal(state_version=0)
        self._anchor = build_frozen_anchor(
            target=fixed_z.values,
            entry_terminal=terminal,
            target_sha256=fixed_z.identity_sha256,
            request_order_sha256=fixed_z.request_order_sha256,
        )
        self._entry_potential, _ = normalized_potential(self._anchor, terminal)
        semantic = self.observe_semantic()
        anchor_observation = _anchor_semantic_observation(self._anchor, semantic)
        telemetry = ArmTelemetry(
            arm=config.arm.value,
            entry_semantic=semantic_payload(semantic),
            anchor_active_request_count=int(
                anchor_observation["anchor_active_request_count"]
            ),
            anchor_zero_request_count=int(
                anchor_observation["anchor_zero_request_count"]
            ),
            anchor_zero_semantic_miss_count=int(
                anchor_observation["anchor_zero_semantic_miss_count"]
            ),
        )
        if semantic.all_strict:
            status = "ENTRY_ALREADY_HIT"
            endpoint = dict(self.adapter.evaluate_current(
                arm=config.arm.value, status=status, state_version=0
            ))
            telemetry.terminal_status = status
            telemetry.terminal_semantic = semantic_payload(semantic)
            telemetry.terminal_net_frobenius = float(
                endpoint.get("terminal_net_frobenius", 0.0)
            )
            telemetry.terminal_net_frobenius_squared = float(
                endpoint.get("terminal_net_frobenius_squared", 0.0)
            )
            return ArmExecutionResult(
                arm=config.arm.value,
                status=status,
                request_count=fixed_z.request_count,
                endpoint=endpoint,
                telemetry=telemetry.payload(),
                overlay=self.overlay.receipt(),
                derived_endpoint=(
                    DerivedEndpoint(
                        ArmId.ORDERED_RESPONSE_FIRST_HIT.value,
                        status,
                        0,
                        (),
                        None,
                        dict(endpoint),
                    )
                    if config.arm is ArmId.ORDERED_RESPONSE_FIXED_HORIZON else None
                ),
                fixed_z_identity_sha256=fixed_z.identity_sha256,
            )

        first_hit_deltas: tuple[OverlayDelta, ...] | None = None
        visit = 0
        stalled = False
        direct_first_hit_stopped = False
        progress = {"sweep": -1, "layer": -1, "visit_ordinal": -1}
        with self.overlay, _progressive_failure_context(progress, telemetry, self.overlay):
            for sweep in range(config.sweeps):
                sweep_acted = False
                if config.refresh_policy is RefreshPolicy.PER_SWEEP:
                    token = self.overlay.seal_sweep_entry(sweep)
                    self.adapter.bind_state_version(token.state_version)
                    frozen_terminal = self.adapter.current_terminal(state_version=token.state_version)
                    frozen: list[
                        tuple[LayerBuild, TerminalJVPResult | None, float, Any | None, torch.Tensor]
                    ] = []
                    for position, layer in enumerate(self.adapter.layers):
                        progress.update(sweep=sweep, layer=layer, visit_ordinal=visit)
                        frozen.append(self._build_and_response(
                            config=config,
                            layer=layer,
                            layer_position=position,
                            fixed_z=fixed_z,
                            terminal=frozen_terminal,
                            state_version=token.state_version,
                        ))
                    for position, layer in enumerate(self.adapter.layers):
                        progress.update(sweep=sweep, layer=layer, visit_ordinal=visit)
                        build, observed, u, statistics, command_residual = frozen[position]
                        terminal, semantic, acted = self._record_and_advance(
                            config=config,
                            sweep=sweep,
                            layer=layer,
                            visit_ordinal=visit,
                            terminal_before=terminal,
                            command_residual=command_residual,
                            build=build,
                            jvp=observed,
                            u=u,
                            statistics=statistics,
                            semantic_before=semantic,
                            sweep_token=token,
                            telemetry=telemetry,
                        )
                        visit += 1
                        sweep_acted |= acted
                        if semantic.all_strict and telemetry.first_hit is None:
                            telemetry.latch_hit(
                                sweep=sweep,
                                layer=layer,
                                state_version=self.overlay.state_version,
                                prefix_length=self.overlay.factor_count,
                            )
                            first_hit_deltas = self.overlay.prefix(self.overlay.factor_count)
                            if preamble_stop_at_first_hit:
                                direct_first_hit_stopped = True
                                break
                    self.overlay.close_sweep(token)
                else:
                    for position, layer in enumerate(self.adapter.layers):
                        progress.update(sweep=sweep, layer=layer, visit_ordinal=visit)
                        self.adapter.bind_state_version(self.overlay.state_version)
                        terminal = self.adapter.current_terminal(state_version=self.overlay.state_version)
                        build, observed, u, statistics, command_residual = self._build_and_response(
                            config=config,
                            layer=layer,
                            layer_position=position,
                            fixed_z=fixed_z,
                            terminal=terminal,
                            state_version=self.overlay.state_version,
                        )
                        terminal, semantic, acted = self._record_and_advance(
                            config=config,
                            sweep=sweep,
                            layer=layer,
                            visit_ordinal=visit,
                            terminal_before=terminal,
                            command_residual=command_residual,
                            build=build,
                            jvp=observed,
                            u=u,
                            statistics=statistics,
                            semantic_before=semantic,
                            sweep_token=None,
                            telemetry=telemetry,
                        )
                        visit += 1
                        sweep_acted |= acted
                        if semantic.all_strict and telemetry.first_hit is None:
                            telemetry.latch_hit(
                                sweep=sweep,
                                layer=layer,
                                state_version=self.overlay.state_version,
                                prefix_length=self.overlay.factor_count,
                            )
                            first_hit_deltas = self.overlay.prefix(self.overlay.factor_count)
                            if preamble_stop_at_first_hit:
                                direct_first_hit_stopped = True
                                break
                if direct_first_hit_stopped:
                    break
                if not sweep_acted:
                    stalled = True
                    break

            if first_hit_deltas is None and config.arm is ArmId.ORDERED_RESPONSE_FIXED_HORIZON:
                first_hit_deltas = self.overlay.deltas
            if self.overlay.factor_count == 0:
                status = "FLOW_STALLED_ZERO_ACTION"
                endpoint = dict(self.adapter.evaluate_current(
                    arm=config.arm.value, status=status, state_version=0
                ))
            else:
                status = terminal_flow_status(
                    stalled=stalled,
                    terminal_hit=semantic.all_strict,
                    first_hit_latched=telemetry.first_hit is not None,
                    direct_first_hit_audit=direct_first_hit_stopped,
                )
                shadow = self.overlay.materialize_shadow()
                with self.overlay.suspend(authoritative=True):
                    endpoint = dict(self.adapter.finalize_terminal(
                        arm=config.arm.value,
                        state_version=self.overlay.state_version,
                        shadow_weights=shadow,
                        deltas=self.overlay.deltas,
                        derived_observation_only=False,
                    ))
                endpoint.setdefault("selected_weight_endpoint_sha256", tensor_set_sha256(shadow))
                telemetry.materialization_count = 1
                telemetry.physical_write_count = int(endpoint.get("physical_write_count", 1))
                telemetry.history_append_count = int(endpoint.get("history_append_count", 0))
            telemetry.terminal_status = status
            telemetry.terminal_semantic = semantic_payload(semantic)
            telemetry.terminal_net_frobenius = float(
                endpoint.get("terminal_net_frobenius", 0.0)
            )
            telemetry.terminal_net_frobenius_squared = float(
                endpoint.get("terminal_net_frobenius_squared", 0.0)
            )
            derived = None
            if config.arm is ArmId.ORDERED_RESPONSE_FIXED_HORIZON:
                assert first_hit_deltas is not None
                if not first_hit_deltas:
                    derived_endpoint_payload = dict(self.adapter.evaluate_current(
                        arm=ArmId.ORDERED_RESPONSE_FIRST_HIT.value,
                        status="HORIZON_SEMANTIC_MISS",
                        state_version=0,
                    ))
                elif len(first_hit_deltas) == self.overlay.factor_count:
                    # A horizon miss has the same physical endpoint as ORBFH;
                    # reuse the already observed bytes instead of a second
                    # materialization while retaining distinct provenance.
                    derived_endpoint_payload = dict(endpoint)
                    derived_endpoint_payload["derived_endpoint_reused"] = True
                else:
                    try:
                        first_hit_shadow = self.overlay.materialize_shadow(deltas=first_hit_deltas)
                        with self.overlay.suspend(authoritative=False):
                            derived_endpoint_payload = dict(self.adapter.finalize_terminal(
                                arm=ArmId.ORDERED_RESPONSE_FIRST_HIT.value,
                                state_version=self.overlay.state_version,
                                shadow_weights=first_hit_shadow,
                                deltas=first_hit_deltas,
                                derived_observation_only=True,
                            ))
                        derived_endpoint_payload.setdefault(
                            "selected_weight_endpoint_sha256", tensor_set_sha256(first_hit_shadow)
                        )
                    except TechnicalBoundary as exc:
                        # ORBHit is an observation-only derived prefix.  Its
                        # materialization/identity failure must not invalidate
                        # the already terminal-valid ORBFH primary endpoint.
                        derived_endpoint_payload = {
                            "status": "ORBHit_WITHHELD_TECHNICAL",
                            "exception_type": type(exc).__name__,
                            "exception": str(exc),
                            "orbfh_primary_gate_influence_count": 0,
                            "physical_write_count": 0,
                            "history_append_count": 0,
                        }
                derived = DerivedEndpoint(
                    arm=ArmId.ORDERED_RESPONSE_FIRST_HIT.value,
                    status=(
                        "ORBHit_WITHHELD_TECHNICAL"
                        if derived_endpoint_payload.get("status") == "ORBHit_WITHHELD_TECHNICAL"
                        else ("FIRST_HIT" if telemetry.first_hit is not None else "HORIZON_SEMANTIC_MISS")
                    ),
                    state_version=len(first_hit_deltas),
                    deltas=tuple(first_hit_deltas),
                    first_hit=telemetry.first_hit,
                    endpoint=derived_endpoint_payload,
                )
            return ArmExecutionResult(
                arm=config.arm.value,
                status=status,
                request_count=fixed_z.request_count,
                endpoint=endpoint,
                telemetry=telemetry.payload(),
                overlay=self.overlay.receipt(),
                derived_endpoint=derived,
                fixed_z_identity_sha256=fixed_z.identity_sha256,
            )


__all__ = [
    "ArmExecutionResult",
    "DerivedEndpoint",
    "OrderedResponseIntegrator",
    "terminal_flow_status",
]
