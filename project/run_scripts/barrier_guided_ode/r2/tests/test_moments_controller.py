from __future__ import annotations

import pytest
import torch

from project.run_scripts.barrier_guided_ode.r2.controller import (
    solve_plain_minimum_norm,
    solve_two_equality_rayleighian,
)
from project.run_scripts.barrier_guided_ode.r2.errors import (
    ActuatorEqualityInfeasible,
    NumericalImplementationBoundary,
)
from project.run_scripts.barrier_guided_ode.r2.events import (
    PrefixDirectionalObservation,
    PrefixEventLayout,
    evaluate_prefix_events,
)
from project.run_scripts.barrier_guided_ode.r2.moments import (
    aggregate_r2_event_moments,
    barrier_decomposition,
)


def _event_pair() -> tuple[object, object]:
    layout = PrefixEventLayout.build(
        source_tokens=(1, 2),
        target_tokens=(1, 3),
        output_vocabulary_size=6,
        tokenizer_vocabulary_size=7,
    )
    initial_observations = {}
    current_observations = {}
    for ordinal, prefix in enumerate(layout.internal_prefixes):
        base = torch.tensor([0.2, -0.4, 0.7, -0.1, 0.3, -0.8], dtype=torch.float32).roll(ordinal)
        tangent = torch.stack(
            (
                torch.tensor([0.4, -0.2, 0.3, -0.5, 0.1, 0.2]),
                torch.tensor([-0.1, 0.6, -0.4, 0.2, 0.3, -0.2]),
                torch.tensor([0.2, 0.1, -0.3, 0.5, -0.6, 0.4]),
                torch.tensor([-0.3, 0.2, 0.4, -0.1, 0.5, -0.4]),
                torch.tensor([0.5, -0.3, 0.2, 0.1, -0.2, -0.1]),
            ),
            dim=1,
        ).to(torch.float32)
        initial_observations[prefix] = PrefixDirectionalObservation(base, tangent)
        current_observations[prefix] = PrefixDirectionalObservation(
            (base + 0.04 * tangent[:, 0] - 0.02 * tangent[:, 1]).contiguous(),
            tangent,
        )
    return (
        evaluate_prefix_events(layout, initial_observations),
        evaluate_prefix_events(layout, current_observations),
    )


def test_moments_centering_pair_mass_theorem_and_barrier_decomposition() -> None:
    initial, current = _event_pair()
    moments = aggregate_r2_event_moments(current, initial=initial, reference_time=0.3)
    torch.testing.assert_close(moments.score_expectation, torch.zeros(5, dtype=torch.float64), rtol=0, atol=3e-15)
    torch.testing.assert_close(moments.fisher, moments.fisher.T, rtol=0, atol=0)
    assert float(torch.linalg.eigvalsh(moments.fisher).min()) >= -2e-14
    solution = solve_plain_minimum_norm(moments.equality_directions, moments.equality_rates)
    rates = moments.equality_directions.T @ solution.velocity
    torch.testing.assert_close(rates, torch.tensor([1.0, 0.0], dtype=torch.float64), rtol=0, atol=2e-13)
    p_y = moments.target_probability
    p_s = moments.source_probability
    c = moments.pair_mass
    target_derivative = p_y * (float(torch.dot(current.scores[current.layout.target_index], solution.velocity)))
    source_derivative = p_s * (float(torch.dot(current.scores[current.layout.source_index], solution.velocity)))
    assert target_derivative == pytest.approx(p_y * p_s / c, abs=3e-13)
    assert source_derivative == pytest.approx(-p_y * p_s / c, abs=3e-13)
    assert target_derivative + source_derivative == pytest.approx(0.0, abs=3e-13)
    decomposition = barrier_decomposition(initial, current, time=0.3)
    assert decomposition.moving_reference_kl >= 0.0
    assert decomposition.pair_mass_drift_kl >= 0.0
    assert decomposition.outside_conditional_event_kl >= 0.0


def test_moving_and_anchor_gradients_have_same_optimizer_under_equalities() -> None:
    initial, current = _event_pair()
    moments = aggregate_r2_event_moments(current, initial=initial, reference_time=0.25)
    assert moments.moving_minus_anchor_parallel_residual < 3e-15
    moving = solve_two_equality_rayleighian(
        moments.fisher,
        moments.moving_gradient,
        moments.equality_directions,
        moments.equality_rates,
    )
    anchor = solve_two_equality_rayleighian(
        moments.fisher,
        moments.anchor_gradient,
        moments.equality_directions,
        moments.equality_rates,
    )
    torch.testing.assert_close(moving.velocity, anchor.velocity, rtol=2e-11, atol=2e-11)


def test_full_rank_kkt_matches_direct_augmented_solve() -> None:
    fisher = torch.tensor(
        [[2.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.8]],
        dtype=torch.float64,
    )
    gradient = torch.tensor([0.3, -0.1, 0.2], dtype=torch.float64)
    directions = torch.tensor([[1.0, 0.1], [0.2, 1.0], [-0.3, 0.4]], dtype=torch.float64)
    rates = torch.tensor([1.0, 0.0], dtype=torch.float64)
    solution = solve_two_equality_rayleighian(fisher, gradient, directions, rates)
    augmented = torch.cat(
        (
            torch.cat((fisher, -directions), dim=1),
            torch.cat((directions.T, torch.zeros((2, 2), dtype=torch.float64)), dim=1),
        ),
        dim=0,
    )
    direct = torch.linalg.solve(augmented, torch.cat((-gradient, rates)))[:3]
    torch.testing.assert_close(solution.velocity, direct, rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(directions.T @ solution.velocity, rates, rtol=0, atol=2e-13)
    assert solution.physical_coefficient_fp32(0.125).dtype == torch.float32


def test_singular_feasible_and_genuinely_infeasible_boundaries() -> None:
    fisher = torch.diag(torch.tensor([1.0, 2.0, 0.0], dtype=torch.float64))
    gradient = torch.tensor([0.1, -0.2, 0.0], dtype=torch.float64)
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    rates = torch.tensor([1.0, 0.0], dtype=torch.float64)
    feasible = solve_two_equality_rayleighian(fisher, gradient, directions, rates)
    assert feasible.receipt.matrix_rank == 2
    torch.testing.assert_close(directions.T @ feasible.velocity, rates, rtol=0, atol=2e-13)
    with pytest.raises(ActuatorEqualityInfeasible) as caught:
        solve_two_equality_rayleighian(
            fisher,
            gradient,
            torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
            rates,
        )
    assert caught.value.receipt["equality_rank"] == 1


def test_outside_range_is_numerical_boundary_with_full_spectrum() -> None:
    fisher = torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.2]], dtype=torch.float64)
    with pytest.raises(NumericalImplementationBoundary) as caught:
        solve_two_equality_rayleighian(
            fisher,
            torch.zeros(3, dtype=torch.float64),
            directions,
            torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
    assert caught.value.receipt["eigenvalues"] == (0.0, 1.0, 1.0)
    assert caught.value.receipt["range_residual_directions"][1] > 0.0


def test_fp32_cutoff_failure_becomes_fp64_feasible_without_tuning() -> None:
    diagonal = torch.tensor([1.0, 1e-8, 0.0], dtype=torch.float64)
    fisher = torch.diag(diagonal)
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
    rates = torch.tensor([1.0, 0.0], dtype=torch.float64)
    fp32_cutoff = 64.0 * torch.finfo(torch.float32).eps * 3
    assert int((diagonal.to(torch.float32) > fp32_cutoff).sum()) == 1
    solution = solve_two_equality_rayleighian(
        fisher,
        torch.zeros(3, dtype=torch.float64),
        directions,
        rates,
    )
    assert solution.receipt.matrix_rank == 2
    assert solution.receipt.matrix_rank_tolerance < 1e-8
    torch.testing.assert_close(directions.T @ solution.velocity, rates, rtol=0, atol=2e-13)
