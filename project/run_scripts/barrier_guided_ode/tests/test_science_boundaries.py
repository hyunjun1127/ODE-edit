from __future__ import annotations

import pytest
import torch

from project.run_scripts.barrier_guided_ode.alphaedit_actuator_interface import (
    ordered_predictor_mismatch,
)
from project.run_scripts.barrier_guided_ode.errors import (
    BGODEScientificBoundary,
    NativeBypassBoundary,
    NumericalRankBoundary,
    UnsupportedBatchBoundary,
)
from project.run_scripts.barrier_guided_ode.event_moments import (
    aggregate_event_score_moments,
)
from project.run_scripts.barrier_guided_ode.event_trie import (
    EventDistribution,
    EventKind,
    EventSpec,
)
from project.run_scripts.barrier_guided_ode.policies import explicit_native_bypass
from project.run_scripts.barrier_guided_ode.rayleighian_controller import (
    solve_equality_rayleighian,
)
from project.run_scripts.barrier_guided_ode.telemetry import ExecutionBoundaryReceipt


def _distribution(probabilities: torch.Tensor) -> EventDistribution:
    events = (
        EventSpec(EventKind.TARGET, (0,), None),
        EventSpec(EventKind.SOURCE, (1,), None),
        EventSpec(EventKind.DEPARTURE, (), 2),
        EventSpec(EventKind.DEPARTURE, (), 3),
    )
    logs = probabilities.log()
    return EventDistribution(
        events=events,
        log_probabilities=logs,
        target_index=0,
        source_index=1,
        normalization_log_residual=abs(float(torch.logsumexp(logs, 0))),
    )


def test_bgode_r1_batch_fixed_target_history_and_leakage_boundaries():
    receipt = ExecutionBoundaryReceipt(
        batch_size=1,
        z_compute_count=1,
        z_recompute_count=0,
        euler_node_count=4,
        history_append_inside_euler_count=0,
        terminal_history_append_count=1,
        controller_evaluator_input_count=0,
        heldout_decision_influence_count=0,
        length_normalization_count=0,
        progress_penalty_count=0,
    )
    assert receipt.z_compute_count == 1
    with pytest.raises(UnsupportedBatchBoundary, match="batch size"):
        ExecutionBoundaryReceipt(
            batch_size=2,
            z_compute_count=1,
            z_recompute_count=0,
            euler_node_count=1,
            history_append_inside_euler_count=0,
            terminal_history_append_count=0,
            controller_evaluator_input_count=0,
            heldout_decision_influence_count=0,
            length_normalization_count=0,
            progress_penalty_count=0,
        )
    with pytest.raises(BGODEScientificBoundary, match="history"):
        ExecutionBoundaryReceipt(
            batch_size=1,
            z_compute_count=1,
            z_recompute_count=0,
            euler_node_count=1,
            history_append_inside_euler_count=1,
            terminal_history_append_count=0,
            controller_evaluator_input_count=0,
            heldout_decision_influence_count=0,
            length_normalization_count=0,
            progress_penalty_count=0,
        )


def test_event_moment_interface_rejects_shared_batched_controller():
    distribution = _distribution(
        torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float32)
    )
    scores = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5], [-0.25, -0.875]],
        dtype=torch.float32,
    )
    with pytest.raises(UnsupportedBatchBoundary, match="B>1"):
        aggregate_event_score_moments(
            distribution,
            initial=distribution,
            reference_log_probabilities=distribution.log_probabilities,
            scores=scores,
            batch_size=100,
        )


def test_native_bypass_is_explicit_and_exact():
    receipt = explicit_native_bypass(
        batch_size=1,
        euler_steps=1,
        barrier_enabled=False,
        factor_count=5,
    )
    assert torch.equal(receipt.coefficients, torch.ones(5, dtype=torch.float32))
    assert receipt.rho == 1.0
    with pytest.raises(NativeBypassBoundary, match="N=1"):
        explicit_native_bypass(
            batch_size=1,
            euler_steps=2,
            barrier_enabled=False,
            factor_count=5,
        )


def test_pair_mass_equalities_imply_target_up_source_down():
    p_y, p_s = 0.2, 0.3
    s_y = torch.tensor([1.0, 0.0], dtype=torch.float32)
    s_s = torch.tensor([0.0, 1.0], dtype=torch.float32)
    a = s_y - s_s
    b = p_y * s_y + p_s * s_s
    solution = solve_equality_rayleighian(
        torch.eye(2, dtype=torch.float32),
        torch.zeros(2, dtype=torch.float32),
        torch.stack((a, b), dim=1),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    target_derivative = p_y * torch.dot(s_y, solution.velocity)
    source_derivative = p_s * torch.dot(s_s, solution.velocity)
    assert target_derivative.item() > 0
    assert source_derivative.item() < 0
    assert (target_derivative + source_derivative).item() == pytest.approx(0.0, abs=2e-6)


def test_pair_mass_infeasibility_is_not_repaired_by_subdivision():
    a = torch.tensor([1.0, 0.0], dtype=torch.float32)
    rank_deficient = torch.stack((a, 2.0 * a), dim=1)
    for h in (1.0, 0.5, 0.125, 0.01):
        with pytest.raises(NumericalRankBoundary, match="rank deficient"):
            solve_equality_rayleighian(
                torch.eye(2, dtype=torch.float32),
                torch.zeros(2, dtype=torch.float32),
                rank_deficient,
                torch.tensor([h, 0.0], dtype=torch.float32),
            )


def _toy_quantities(base: torch.Tensor, features: torch.Tensor, weight: torch.Tensor, time: float):
    initial = torch.softmax(base, 0)
    current = torch.softmax(base + features @ weight, 0)
    scores = features - (current[:, None] * features).sum(0, keepdim=True)
    fisher = scores.transpose(0, 1) @ (current[:, None] * scores)
    fisher = 0.5 * (fisher + fisher.transpose(0, 1))
    a = features[0] - features[1]
    r0 = base[0] - base[1]
    c0 = initial[0] + initial[1]
    reference = initial.clone()
    reference[0] = c0 * torch.sigmoid(r0 + time)
    reference[1] = c0 * torch.sigmoid(-(r0 + time))
    gradient = -(reference[:, None] * scores).sum(0)
    return initial, current, scores, fisher, a, reference, gradient


def _toy_kl(base, features, initial, weight, time):
    current = torch.softmax(base + features @ weight, 0)
    r0 = base[0] - base[1]
    c0 = initial[0] + initial[1]
    reference = initial.clone()
    reference[0] = c0 * torch.sigmoid(r0 + time)
    reference[1] = c0 * torch.sigmoid(-(r0 + time))
    return torch.sum(reference * (torch.log(reference) - torch.log(current)))


def test_off_reference_full_barrier_can_be_load_bearing_but_is_not_assumed():
    base = torch.tensor(
        [0.26454088, 0.10676964, 0.02467090, 0.24852693], dtype=torch.float32
    )
    features = torch.tensor(
        [
            [-0.45190597, -0.16613023],
            [-1.52276850, 0.38168392],
            [-1.02760863, -0.56305277],
            [-0.89229053, -0.05825018],
        ],
        dtype=torch.float32,
    )
    weight = torch.tensor([0.75588560, 0.74742413], dtype=torch.float32)
    time, h = 0.4, 0.08
    initial, _, _, fisher, a, _, gradient = _toy_quantities(
        base, features, weight, time
    )
    full = solve_equality_rayleighian(fisher, gradient, a, 1.0).velocity
    fisher_only = solve_equality_rayleighian(
        fisher, torch.zeros_like(gradient), a, 1.0
    ).velocity
    plain = a / torch.dot(a, a)
    values = {
        "full": _toy_kl(base, features, initial, weight + h * full, time + h),
        "fisher": _toy_kl(
            base, features, initial, weight + h * fisher_only, time + h
        ),
        "plain": _toy_kl(base, features, initial, weight + h * plain, time + h),
    }
    assert torch.linalg.vector_norm(full - fisher_only).item() > 0.01
    assert values["full"].item() < values["fisher"].item()
    assert values["full"].item() < values["plain"].item()


def test_dynamic_relinearization_differs_only_for_state_dependent_geometry():
    base = torch.tensor(
        [0.26454088, 0.10676964, 0.02467090, 0.24852693], dtype=torch.float32
    )
    features = torch.tensor(
        [
            [-0.45190597, -0.16613023],
            [-1.52276850, 0.38168392],
            [-1.02760863, -0.56305277],
            [-0.89229053, -0.05825018],
        ],
        dtype=torch.float32,
    )
    start = torch.tensor([0.75588560, 0.74742413], dtype=torch.float32)
    start_time, total_time, count = 0.4, 0.32, 4
    initial, _, _, fisher, a, _, gradient = _toy_quantities(
        base, features, start, start_time
    )
    frozen_velocity = solve_equality_rayleighian(fisher, gradient, a, 1.0).velocity
    one_step = start + total_time * frozen_velocity
    frozen_split = start.clone()
    for _ in range(count):
        frozen_split += total_time / count * frozen_velocity
    assert torch.allclose(one_step, frozen_split, atol=2e-6, rtol=2e-6)

    dynamic = start.clone()
    for index in range(count):
        _, _, _, local_fisher, local_a, _, local_gradient = _toy_quantities(
            base, features, dynamic, start_time + index * total_time / count
        )
        velocity = solve_equality_rayleighian(
            local_fisher, local_gradient, local_a, 1.0
        ).velocity
        dynamic += total_time / count * velocity
    assert torch.linalg.vector_norm(dynamic - frozen_split).item() > 1e-3


def test_fixed_time_nonlinear_euler_converges_under_subcycling():
    base = torch.tensor(
        [0.26454088, 0.10676964, 0.02467090, 0.24852693], dtype=torch.float32
    )
    features = torch.tensor(
        [
            [-0.45190597, -0.16613023],
            [-1.52276850, 0.38168392],
            [-1.02760863, -0.56305277],
            [-0.89229053, -0.05825018],
        ],
        dtype=torch.float32,
    )
    start = torch.tensor([0.75588560, 0.74742413], dtype=torch.float32)
    start_time, total_time = 0.4, 0.32

    def integrate(count: int) -> torch.Tensor:
        state = start.clone()
        h = total_time / count
        for index in range(count):
            _, _, _, fisher, a, _, gradient = _toy_quantities(
                base, features, state, start_time + index * h
            )
            velocity = solve_equality_rayleighian(fisher, gradient, a, 1.0).velocity
            state += h * velocity
        return state

    reference = integrate(128)
    errors = [torch.linalg.vector_norm(integrate(count) - reference).item() for count in (1, 2, 4, 8, 16)]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))


def test_ordered_unit_prefix_mismatch_is_observed_not_erased_by_outer_h():
    unit = torch.tensor([[1.0, -0.5], [0.25, 0.75]], dtype=torch.float32)
    effective = 0.1 * torch.tensor([[0.3, -0.2], [0.4, 0.1]], dtype=torch.float32)
    mismatch = ordered_predictor_mismatch(unit, effective)
    assert mismatch.difference_norm > 0
    assert mismatch.relative_to_unit_prefix > 0.5
