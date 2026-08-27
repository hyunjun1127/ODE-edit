from __future__ import annotations

import pytest
import torch

from project.run_scripts.barrier_guided_ode.errors import NumericalRankBoundary
from project.run_scripts.barrier_guided_ode.event_moments import (
    aggregate_event_score_moments,
)
from project.run_scripts.barrier_guided_ode.event_trie import (
    EventDistribution,
    EventKind,
    EventSpec,
)
from project.run_scripts.barrier_guided_ode.rayleighian_controller import (
    solve_equality_rayleighian,
)
from project.run_scripts.barrier_guided_ode.reference_path import (
    single_coordinate_reference,
)


def _categorical_distribution(probabilities: torch.Tensor) -> EventDistribution:
    assert probabilities.dtype == torch.float32
    events = (
        EventSpec(EventKind.TARGET, (0,), None),
        EventSpec(EventKind.SOURCE, (1,), None),
        *(EventSpec(EventKind.DEPARTURE, (), index) for index in range(2, len(probabilities))),
    )
    logs = probabilities.log()
    return EventDistribution(
        events=events,
        log_probabilities=logs,
        target_index=0,
        source_index=1,
        normalization_log_residual=abs(float(torch.logsumexp(logs, 0))),
    )


def _centered_scores(probabilities: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    return raw - (probabilities[:, None] * raw).sum(dim=0, keepdim=True)


def test_event_scores_match_autograd_jacobian_and_finite_difference():
    base = torch.tensor([0.2, -0.1, 0.4, 0.0], dtype=torch.float32)
    raw = torch.tensor(
        [[0.7, -0.2], [-0.3, 0.9], [0.4, 0.5], [-0.6, -0.1]],
        dtype=torch.float32,
    )
    theta = torch.zeros(2, dtype=torch.float32, requires_grad=True)

    def log_prob(value):
        return torch.log_softmax(base + raw @ value, dim=0)

    jacobian = torch.autograd.functional.jacobian(log_prob, theta).detach()
    expected = _centered_scores(torch.softmax(base, 0), raw)
    assert torch.allclose(jacobian, expected, atol=2e-7, rtol=2e-7)
    direction = torch.tensor([0.35, -0.8], dtype=torch.float32)
    epsilon = 1e-3
    finite_difference = (
        log_prob((epsilon * direction).detach())
        - log_prob((-epsilon * direction).detach())
    ) / (2.0 * epsilon)
    assert torch.allclose(
        finite_difference, expected @ direction, atol=8e-5, rtol=8e-5
    )


def test_score_centering_fisher_psd_g0_and_moving_anchor_equivalence():
    probabilities = torch.tensor([0.18, 0.22, 0.27, 0.33], dtype=torch.float32)
    initial = _categorical_distribution(probabilities)
    raw = torch.tensor(
        [[1.2, -0.3], [-0.7, 0.8], [0.5, 1.1], [-0.4, -0.6]],
        dtype=torch.float32,
    )
    scores = _centered_scores(probabilities, raw)
    reference = single_coordinate_reference(initial, time=0.6)
    moments = aggregate_event_score_moments(
        initial,
        initial=initial,
        reference_log_probabilities=reference.log_probabilities,
        scores=scores,
    )
    assert torch.linalg.vector_norm(moments.score_expectation).item() < 2e-7
    assert torch.linalg.vector_norm(moments.anchor_gradient).item() < 2e-7
    assert moments.fisher.symmetry_residual == 0.0
    assert moments.fisher.minimum_eigenvalue > -1e-7
    assert moments.moving_minus_anchor_parallel_residual < 3e-7

    moving = solve_equality_rayleighian(
        moments.fisher.matrix,
        moments.moving_gradient,
        moments.progress_sensitivity,
        1.0,
    )
    anchor = solve_equality_rayleighian(
        moments.fisher.matrix,
        moments.anchor_gradient,
        moments.progress_sensitivity,
        1.0,
    )
    assert torch.allclose(moving.velocity, anchor.velocity, atol=3e-5, rtol=3e-5)
    assert torch.dot(moments.progress_sensitivity, moving.velocity).item() == pytest.approx(
        1.0, abs=2e-5
    )

    # Replacing equality with a quadratic progress penalty destroys the
    # optimizer equivalence because a gradient shift along a is no longer a
    # constant on the feasible set.
    penalty = 2.0
    a = moments.progress_sensitivity
    hessian = moments.fisher.matrix + penalty * torch.outer(a, a)
    penalized_moving = torch.linalg.solve(
        hessian, penalty * a - moments.moving_gradient
    )
    penalized_anchor = torch.linalg.solve(
        hessian, penalty * a - moments.anchor_gradient
    )
    assert not torch.allclose(penalized_moving, penalized_anchor, atol=1e-5, rtol=1e-5)


def _direct_kkt(fisher, gradient, directions, rates):
    if directions.ndim == 1:
        directions = directions[:, None]
    if rates.ndim == 0:
        rates = rates[None]
    zeros = torch.zeros(
        (directions.shape[1], directions.shape[1]), dtype=torch.float32
    )
    matrix = torch.cat(
        (
            torch.cat((fisher, -directions), dim=1),
            torch.cat((directions.transpose(0, 1), zeros), dim=1),
        ),
        dim=0,
    )
    rhs = torch.cat((-gradient, rates))
    return torch.linalg.lstsq(matrix, rhs).solution[: fisher.shape[0]]


def test_kkt_matches_direct_full_rank_and_multi_equality():
    fisher = torch.tensor([[2.0, 0.3], [0.3, 1.4]], dtype=torch.float32)
    gradient = torch.tensor([0.2, -0.4], dtype=torch.float32)
    direction = torch.tensor([1.1, -0.2], dtype=torch.float32)
    rate = torch.tensor(1.0, dtype=torch.float32)
    solution = solve_equality_rayleighian(fisher, gradient, direction, rate)
    direct = _direct_kkt(fisher, gradient, direction, rate)
    assert torch.allclose(solution.velocity, direct, atol=2e-5, rtol=2e-5)
    assert solution.receipt.matrix_rank == 2
    assert solution.receipt.equality_rank == 1

    directions = torch.eye(2, dtype=torch.float32)
    rates = torch.tensor([0.4, -0.7], dtype=torch.float32)
    multi = solve_equality_rayleighian(fisher, gradient, directions, rates)
    assert torch.allclose(multi.velocity, rates, atol=2e-5, rtol=2e-5)
    assert multi.receipt.equality_count == 2


def test_singular_feasible_kkt_and_rank_range_failures():
    fisher = torch.diag(torch.tensor([2.0, 0.0], dtype=torch.float32))
    feasible = solve_equality_rayleighian(
        fisher,
        torch.tensor([0.5, 0.0], dtype=torch.float32),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        1.0,
    )
    assert feasible.receipt.matrix_rank == 1
    assert feasible.velocity.tolist() == pytest.approx([1.0, 0.0], abs=2e-6)
    with pytest.raises(NumericalRankBoundary, match="gradient is outside"):
        solve_equality_rayleighian(
            fisher,
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            torch.tensor([1.0, 0.0], dtype=torch.float32),
            1.0,
        )
    with pytest.raises(NumericalRankBoundary, match="direction is outside"):
        solve_equality_rayleighian(
            fisher,
            torch.tensor([0.5, 0.0], dtype=torch.float32),
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            1.0,
        )
    with pytest.raises(NumericalRankBoundary, match="rank deficient"):
        solve_equality_rayleighian(
            torch.eye(2, dtype=torch.float32),
            torch.zeros(2, dtype=torch.float32),
            torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float32),
            torch.tensor([1.0, 2.0], dtype=torch.float32),
        )


def test_beta_scales_linearly_and_vanishes_with_h():
    solution = solve_equality_rayleighian(
        torch.eye(2, dtype=torch.float32),
        torch.tensor([0.1, -0.2], dtype=torch.float32),
        torch.tensor([1.0, 0.5], dtype=torch.float32),
        1.0,
    )
    beta1 = solution.finite_step(0.25)
    beta2 = solution.finite_step(0.125)
    assert torch.equal(beta1, 2.0 * beta2)
    assert torch.linalg.vector_norm(solution.finite_step(1e-6)).item() < 1e-5


def test_fixed_time_affine_field_one_step_equals_frozen_euler():
    solution = solve_equality_rayleighian(
        torch.tensor([[1.7, 0.2], [0.2, 0.9]], dtype=torch.float32),
        torch.tensor([0.3, -0.1], dtype=torch.float32),
        torch.tensor([0.8, -0.4], dtype=torch.float32),
        1.0,
    )
    total_time = 0.8
    one_step = solution.finite_step(total_time)
    for count in (2, 4, 8, 16):
        subcycled = sum(
            (solution.finite_step(total_time / count) for _ in range(count)),
            start=torch.zeros_like(one_step),
        )
        assert torch.allclose(subcycled, one_step, atol=2e-6, rtol=2e-6)
