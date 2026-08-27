"""BGODE-R3 G2 normalized-factor live Euler trajectories."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from project.run_scripts.barrier_guided_ode.r2.actuator_guard import propose_validated_ordered
from project.run_scripts.barrier_guided_ode.s1_alphaedit_runtime import (
    AtomicWeightTrajectory,
    BGODETargetAuthority,
)
from project.run_scripts.ode_edit_motivation.alphaedit_proposal_adapter import AlphaEditProposalAdapter
from project.run_scripts.ode_edit_motivation.hooks import resolve_parameter

from .actuators import (
    NodeBuildLedger,
    NormalizedActuatorBasis,
    release_adapter_build,
)
from .errors import NumericalBoundary, R3ScientificBoundary
from .events import FineEventEvaluation, FineEventLayout, evaluate_fine_events
from .g1_jvp import R3SerialForwardJVPBackend
from .g1_probe import _event_payload, _jsonable, _solution_payload
from .g2_convergence import G2Endpoint, block_distance
from .moments import aggregate_fine_event_moments, differential_identity
from .reference import W0ConditionalSeal, barrier_decomposition, conditional_q_kl
from .solver import ControllerArm, barrier_attribution, solve_factor_space


@dataclass(frozen=True, slots=True)
class G2TrajectoryResult:
    payload: Mapping[str, Any]
    endpoint: G2Endpoint
    endpoint_blocks: tuple[torch.Tensor, ...]


def _capture_blocks(model: torch.nn.Module, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
    blocks = tuple(
        resolve_parameter(model, name).detach().cpu().to(dtype=torch.float32).clone().contiguous()
        for name in names
    )
    if any(block.requires_grad or not bool(torch.isfinite(block).all()) for block in blocks):
        raise R3ScientificBoundary("G2 physical endpoint capture is not detached finite FP32")
    return blocks


def _layer_delta_rows(
    names: Sequence[str],
    before: tuple[torch.Tensor, ...],
    after: tuple[torch.Tensor, ...],
) -> tuple[tuple[dict[str, float | str], ...], float]:
    if len(names) != len(before) or len(before) != len(after):
        raise R3ScientificBoundary("G2 layer delta inventories differ")
    rows: list[dict[str, float | str]] = []
    energy = 0.0
    for name, left, right in zip(names, before, after, strict=True):
        difference = right.to(dtype=torch.float64) - left.to(dtype=torch.float64)
        value = float(torch.sum(difference.square()).item())
        if not math.isfinite(value) or value < 0.0:
            raise R3ScientificBoundary("G2 actual layer delta energy is invalid")
        energy += value
        rows.append(
            {
                "weight_name": name,
                "actual_delta_frobenius": math.sqrt(value),
                "actual_delta_energy": value,
            }
        )
    if energy > 0.0:
        for row in rows:
            row["actual_delta_energy_share"] = float(row["actual_delta_energy"]) / energy
    else:
        for row in rows:
            row["actual_delta_energy_share"] = 0.0
    return tuple(rows), math.sqrt(energy)


def _event_odds(value: FineEventEvaluation) -> dict[str, float]:
    target = float(value.target_log_probability)
    source = float(value.source_log_probability)
    return {
        "target_source_log_odds": target - source,
        "target_log_probability": target,
        "source_log_probability": source,
    }


def run_g2_trajectory(
    *,
    arm: ControllerArm,
    step_count: int,
    horizon: float,
    runtime: Any,
    adapter: AlphaEditProposalAdapter,
    layout: FineEventLayout,
    q0: W0ConditionalSeal,
    authority0: BGODETargetAuthority,
    snapshot_factory: Callable[[], Any],
    weight_names: tuple[str, ...],
    tokenization: Any,
    endpoint_observer: Callable[[], Mapping[str, Any]],
    solver_prefix: str,
) -> G2TrajectoryResult:
    if arm not in (ControllerArm.PLAIN, ControllerArm.FISHER, ControllerArm.FULL):
        raise R3ScientificBoundary("G2 arm is outside the locked matrix")
    if step_count not in (4, 8, 16, 32):
        raise R3ScientificBoundary("G2 step count is outside the locked matrix")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise R3ScientificBoundary("G2 horizon is invalid")
    step_size = horizon / step_count
    device = next(runtime.model.parameters()).device
    build_ledger = NodeBuildLedger()
    authority = authority0
    nodes: list[dict[str, Any]] = []
    path_length = 0.0
    integrated_kinetic_energy = 0.0
    initial_blocks = _capture_blocks(runtime.model, weight_names)
    trajectory_started = time.perf_counter()
    with AtomicWeightTrajectory(runtime.model, weight_names) as trajectory:
        for node_index in range(step_count):
            node_started = time.perf_counter()
            entry_time = node_index * step_size
            current_snapshot = snapshot_factory()
            authority.assert_authorizes(
                direct_z=adapter.direct_z,
                snapshot=current_snapshot,
                target_token_ids=adapter.target_token_ids,
            )
            torch.cuda.synchronize(device)
            build_started = time.perf_counter()
            validated = propose_validated_ordered(
                adapter,
                origin_lineage=authority,
                construction="genuine-p-inside-solve",
                solver_suffix=f"{solver_prefix}/{arm.value}/N{step_count}/node-{node_index}",
            )
            torch.cuda.synchronize(device)
            build_wall = time.perf_counter() - build_started
            raw = validated.build.proposal
            basis = NormalizedActuatorBasis.from_factors(raw.factors)
            normalized = basis.normalized_proposal(raw)
            build_ledger.publish(basis, validation_call_count=validated.validation_call_count)
            backend = R3SerialForwardJVPBackend(runtime.model, tokenization)
            try:
                observations = backend.observe_all_prefix_fd(layout=layout, proposal=normalized)
            except NumericalBoundary as error:
                raise NumericalBoundary(
                    str(error),
                    receipt={
                        **error.receipt,
                        "arm": arm.value,
                        "step_count": step_count,
                        "node_index": node_index,
                        "completed_node_count": len(nodes),
                        "physical_action_count": len(nodes),
                        "current_node_write_count": 0,
                        "failure_stage": "ALL_PREFIX_NORMALIZED_JVP_CENTRAL_FD",
                    },
                ) from error
            entry_event = evaluate_fine_events(layout, observations)
            moments = aggregate_fine_event_moments(
                entry_event,
                seal=q0,
                reference_time=entry_time,
            )
            torch.cuda.synchronize(device)
            controller_started = time.perf_counter()
            if arm is ControllerArm.FULL:
                fisher = solve_factor_space(
                    moments.factor,
                    moments.objective_offset,
                    moments.equality_directions,
                    moments.equality_rates,
                    arm=ControllerArm.FISHER,
                )
                selected = solve_factor_space(
                    moments.factor,
                    moments.objective_offset,
                    moments.equality_directions,
                    moments.equality_rates,
                    arm=ControllerArm.FULL,
                )
                attribution = asdict(
                    barrier_attribution(moments.factor, moments.objective_offset, fisher, selected)
                )
            else:
                selected = solve_factor_space(
                    moments.factor,
                    moments.objective_offset,
                    moments.equality_directions,
                    moments.equality_rates,
                    arm=arm,
                )
                attribution = None
            retained = backend.validate_retained(selected)
            torch.cuda.synchronize(device)
            controller_wall = time.perf_counter() - controller_started
            beta = selected.coefficient(step_size)
            raw_coefficients = basis.raw_coefficients_fp32(beta)
            before_blocks = _capture_blocks(runtime.model, weight_names)
            action = trajectory.apply(normalized, beta.to(dtype=torch.float32))
            after_blocks = _capture_blocks(runtime.model, weight_names)
            layer_rows, actual_step_norm = _layer_delta_rows(
                weight_names, before_blocks, after_blocks
            )
            del before_blocks
            path_length += actual_step_norm
            integrated_kinetic_energy += actual_step_norm * actual_step_norm / step_size
            child = snapshot_factory()
            authority = authority.derive(
                proposal=normalized,
                coefficients=beta.to(dtype=torch.float32),
                child_snapshot=child,
                applied_hashes=dict(action.parameter_hashes),
            )
            exit_event = evaluate_fine_events(layout, backend.observe_primal(layout))
            entry_barrier = barrier_decomposition(q0, entry_event, time=entry_time)
            exit_barrier = barrier_decomposition(q0, exit_event, time=entry_time + step_size)
            gradient_dot = float(torch.dot(moments.moving_gradient, selected.velocity).item())
            fisher_kinetic = float(
                torch.dot(
                    moments.factor @ selected.velocity,
                    moments.factor @ selected.velocity,
                ).item()
            )
            if any(not math.isfinite(value) for value in (gradient_dot, fisher_kinetic)):
                raise R3ScientificBoundary("G2 node objective telemetry is non-finite")
            retained_receipt = release_adapter_build(adapter, validated.build)
            build_ledger.release_node()
            nodes.append(
                {
                    "node": node_index,
                    "entry_time": entry_time,
                    "exit_time": entry_time + step_size,
                    "step_size": step_size,
                    "entry_event": _event_payload(entry_event),
                    "exit_event": _event_payload(exit_event),
                    "entry_odds": _event_odds(entry_event),
                    "exit_odds": _event_odds(exit_event),
                    "entry_q0_conditional_kl": conditional_q_kl(q0, entry_event),
                    "exit_q0_conditional_kl": conditional_q_kl(q0, exit_event),
                    "barrier_entry": asdict(entry_barrier),
                    "barrier_exit": asdict(exit_barrier),
                    "differential_identity": differential_identity(moments, selected.velocity),
                    "solution": _solution_payload(selected),
                    "barrier_attribution": attribution,
                    "gradient_dot_velocity": gradient_dot,
                    "fisher_kinetic_term": fisher_kinetic,
                    "velocity": [float(value) for value in selected.velocity.tolist()],
                    "beta": [float(value) for value in beta.tolist()],
                    "raw_coefficients": [float(value) for value in raw_coefficients.tolist()],
                    "normalized_beta_norm": basis.block_frobenius(beta),
                    "ordered_prefix_mismatch_beta_based": list(
                        basis.ordered_prefix_mismatch(beta)
                    ),
                    "actual_physical_step_norm": actual_step_norm,
                    "actual_per_layer_delta": list(layer_rows),
                    "physical_action": asdict(action),
                    "ordered_dictionary": {
                        **asdict(validated.dictionary),
                        "factor_sha256": list(validated.factor_sha256),
                        "validation_call_count": validated.validation_call_count,
                        "lightweight_retention": retained_receipt,
                    },
                    "retained_direction_fd": list(retained),
                    "jvp_compute": asdict(backend.ledger),
                    "prefix_fd": [asdict(value) for value in backend.prefix_fd_receipts],
                    "dictionary_build_wall_seconds": build_wall,
                    "controller_wall_seconds": controller_wall,
                    "node_total_wall_seconds": time.perf_counter() - node_started,
                    "fixed_z_recompute_count": 0,
                    "history_append_count": 0,
                    "retained_factor_count_after_release": build_ledger.retained_factor_count,
                }
            )
            del after_blocks, observations, entry_event, moments, normalized, raw, basis, backend
            torch.cuda.empty_cache()
        endpoint_event = evaluate_fine_events(
            layout,
            R3SerialForwardJVPBackend(runtime.model, tokenization).observe_primal(layout),
        )
        endpoint_blocks = _capture_blocks(runtime.model, weight_names)
        terminal_displacement = block_distance(initial_blocks, endpoint_blocks)
        terminal_observation = dict(endpoint_observer())
        endpoint = G2Endpoint(
            step_count=step_count,
            terminal_displacement=terminal_displacement,
            target_logit=float(_event_payload(endpoint_event)["target_logit"]),
            q0_conditional_kl=conditional_q_kl(q0, endpoint_event),
        )
        payload = {
            "arm": arm.value,
            "step_count": step_count,
            "horizon": horizon,
            "step_size": step_size,
            "nodes": nodes,
            "terminal_event": _event_payload(endpoint_event),
            "terminal_odds": _event_odds(endpoint_event),
            "terminal_q0_conditional_kl": endpoint.q0_conditional_kl,
            "terminal_displacement": terminal_displacement,
            "physical_path_length": path_length,
            "integrated_kinetic_energy": integrated_kinetic_energy,
            "terminal_observation": terminal_observation,
            "dictionary_build_count": build_ledger.ordered_build_count,
            "dictionary_validation_count": build_ledger.validation_call_count,
            "retained_factor_count": build_ledger.retained_factor_count,
            "physical_write_count": step_count,
            "physical_layer_apply_count": 5 * step_count,
            "fixed_z_recompute_count": 0,
            "history_append_count": 0,
            "controller_evaluator_influence_count": 0,
            "trajectory_wall_seconds": time.perf_counter() - trajectory_started,
        }
    if build_ledger.live_factor_count != 0 or build_ledger.retained_factor_count != 0:
        raise R3ScientificBoundary("G2 trajectory retained live factors after restore")
    return G2TrajectoryResult(_jsonable(payload), endpoint, endpoint_blocks)


__all__ = ["G2TrajectoryResult", "run_g2_trajectory"]
