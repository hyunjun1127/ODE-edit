from __future__ import annotations

import pytest
import torch

from project.run_scripts.barrier_guided_ode.r3.actuators import NodeBuildLedger, NormalizedActuatorBasis
from project.run_scripts.barrier_guided_ode.r3.errors import EqualityInfeasible
from project.run_scripts.barrier_guided_ode.r3.solver import (
    ControllerArm,
    barrier_attribution,
    solve_factor_space,
    validate_retained_direction_scores,
)
from project.run_scripts.barrier_guided_ode.r3.events import FineEventLayout, PrefixDirectionalObservation, evaluate_fine_events
from project.run_scripts.barrier_guided_ode.r3.moments import aggregate_fine_event_moments
from project.run_scripts.barrier_guided_ode.r3.reference import W0ConditionalSeal
from project.run_scripts.ode_edit_motivation.contracts import LowRankFactor


def _factor() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [[0.5, -0.2, 0.1], [-0.1, 0.6, 0.3], [0.2, 0.1, -0.4], [-0.3, -0.2, 0.5]],
        dtype=torch.float64,
    )
    y = torch.tensor([-0.7, -0.2, -0.4, -0.3], dtype=torch.float64)
    c = torch.tensor([[1.0, 0.1], [0.2, 0.9], [-0.3, 0.4]], dtype=torch.float64)
    d = torch.tensor([1.0, 0.0], dtype=torch.float64)
    return x, y, c, d


def test_factor_space_matches_dense_kkt() -> None:
    x, y, c, d = _factor()
    solution = solve_factor_space(x, y, c, d, arm=ControllerArm.FULL)
    gram = x.T @ x
    gradient = x.T @ y
    augmented = torch.cat(
        (
            torch.cat((gram, -c), dim=1),
            torch.cat((c.T, torch.zeros((2, 2), dtype=torch.float64)), dim=1),
        ),
        dim=0,
    )
    direct = torch.linalg.solve(augmented, torch.cat((-gradient, d)))[:3]
    torch.testing.assert_close(solution.velocity, direct, rtol=2e-12, atol=2e-12)
    assert solution.receipt.equality_residual <= solution.receipt.equality_tolerance
    assert solution.receipt.stationarity_residual <= solution.receipt.stationarity_tolerance


def test_redundant_equality_is_consistent_without_mode_change() -> None:
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, -0.4]], dtype=torch.float64)
    c = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    d = torch.tensor([1.0, 0.0], dtype=torch.float64)
    result = solve_factor_space(x, torch.zeros(3, dtype=torch.float64), c, d, arm=ControllerArm.FISHER)
    assert result.receipt.equality.raw_rank == 1
    torch.testing.assert_close(c.T @ result.velocity, d, rtol=0, atol=2e-15)


def test_inconsistent_equality_fails_closed() -> None:
    x = torch.eye(3, dtype=torch.float64)
    c = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    with pytest.raises(EqualityInfeasible):
        solve_factor_space(
            x,
            torch.zeros(3, dtype=torch.float64),
            c,
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            arm=ControllerArm.FISHER,
        )


def test_three_event_rank_two_has_full_equal_fisher() -> None:
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=torch.float64)
    c = torch.eye(2, dtype=torch.float64)
    d = torch.tensor([1.0, 0.0], dtype=torch.float64)
    fisher = solve_factor_space(x, torch.zeros(3, dtype=torch.float64), c, d, arm=ControllerArm.FISHER)
    full = solve_factor_space(x, torch.tensor([-0.4, -0.7, -0.2], dtype=torch.float64), c, d, arm=ControllerArm.FULL)
    torch.testing.assert_close(full.velocity, fisher.velocity, rtol=0, atol=0)


def test_full_barrier_local_derivative_dominates_fisher() -> None:
    x, y, c, d = _factor()
    fisher = solve_factor_space(x, y, c, d, arm=ControllerArm.FISHER)
    full = solve_factor_space(x, y, c, d, arm=ControllerArm.FULL)
    receipt = barrier_attribution(x, y, fisher, full)
    assert receipt.gradient_dot_full <= receipt.gradient_dot_fisher + 2e-12
    assert receipt.identity_residual < 2e-12
    assert receipt.correction_fisher_energy >= 0.0


def test_retained_factor_directions_match_symmetric_difference_scores() -> None:
    x, y, c, d = _factor()
    full = solve_factor_space(x, y, c, d, arm=ControllerArm.FULL)
    scores = x / torch.sqrt(torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64))[:, None]
    epsilon = 1.0 / 512.0

    def event_logs(coefficient: torch.Tensor) -> torch.Tensor:
        # Synthetic affine log-event chart used only to validate retained modes.
        return scores @ coefficient

    columns = []
    for index in range(scores.shape[1]):
        direction = torch.zeros(scores.shape[1], dtype=torch.float64)
        direction[index] = epsilon
        columns.append((event_logs(direction) - event_logs(-direction)) / (2.0 * epsilon))
    symmetric = torch.stack(columns, dim=1)
    receipt = validate_retained_direction_scores(
        scores,
        symmetric,
        full.retained_directions,
        absolute_tolerance=2e-13,
        relative_tolerance=2e-13,
    )
    assert receipt.allclose


def test_t0_zero_gradient_full_is_numerically_identical_to_fisher() -> None:
    x, _, c, d = _factor()
    zero = torch.zeros(x.shape[0], dtype=torch.float64)
    fisher = solve_factor_space(x, zero, c, d, arm=ControllerArm.FISHER)
    full = solve_factor_space(x, zero, c, d, arm=ControllerArm.FULL)
    assert torch.equal(full.velocity, fisher.velocity)


def test_t0_sealed_reference_has_zero_gradient_and_full_equals_fisher() -> None:
    layout = FineEventLayout.build(
        source_tokens=(1, 2), target_tokens=(1, 3), output_vocabulary_size=6, tokenizer_vocabulary_size=7
    )
    tangent = torch.tensor(
        [[0.2, -0.1, 0.3], [-0.2, 0.5, 0.1], [0.3, -0.4, 0.2], [-0.1, 0.2, -0.5], [0.4, 0.1, -0.2], [-0.3, -0.3, 0.1]],
        dtype=torch.float32,
    )
    observations = {
        prefix: PrefixDirectionalObservation(
            torch.linspace(-1.0, 1.0, 6, dtype=torch.float32).roll(index),
            tangent.roll(index, dims=0),
        )
        for index, prefix in enumerate(layout.internal_prefixes)
    }
    initial = evaluate_fine_events(layout, observations)
    seal = W0ConditionalSeal.capture(initial)
    moments = aggregate_fine_event_moments(initial, seal=seal, reference_time=0.0)
    assert float(torch.linalg.vector_norm(moments.moving_gradient)) < 2e-15
    fisher = solve_factor_space(
        moments.factor, moments.objective_offset, moments.equality_directions, moments.equality_rates, arm=ControllerArm.FISHER
    )
    full = solve_factor_space(
        moments.factor, moments.objective_offset, moments.equality_directions, moments.equality_rates, arm=ControllerArm.FULL
    )
    torch.testing.assert_close(full.velocity, fisher.velocity, rtol=0, atol=2e-14)


def _dense(factor: LowRankFactor) -> torch.Tensor:
    value = factor.left @ factor.right.T
    return value.T if factor.native_update_transposed else value


def test_normalized_actuator_coordinates_preserve_dense_write_and_block_norm() -> None:
    factors = (
        LowRankFactor("layer.0.weight", torch.tensor([[2.0], [0.0]]), torch.tensor([[1.5], [-0.5]]), "a" * 64, False),
        LowRankFactor("layer.1.weight", torch.tensor([[0.0], [3.0]]), torch.tensor([[0.4], [0.8]]), "b" * 64, False),
    )
    basis = NormalizedActuatorBasis.from_factors(factors)
    beta = torch.tensor([0.3, -0.7], dtype=torch.float64)
    raw = basis.raw_coefficients_fp32(beta)
    for original, normalized, coefficient, normalized_coefficient in zip(factors, basis.normalized_factors, raw, beta, strict=True):
        torch.testing.assert_close(
            coefficient * _dense(original),
            float(normalized_coefficient) * _dense(normalized),
            rtol=2e-7,
            atol=2e-7,
        )
    dense_norm = torch.sqrt(sum(torch.sum((float(value) * _dense(factor)).double().square()) for value, factor in zip(beta, basis.normalized_factors, strict=True)))
    assert basis.block_frobenius(beta) == pytest.approx(float(dense_norm), rel=2e-7)
    mismatch = basis.ordered_prefix_mismatch(beta)
    assert mismatch[1]["applied_prefix_frobenius"] == pytest.approx(abs(float(beta[0])))


def test_node_build_ledger_releases_all_factors() -> None:
    factor = LowRankFactor("layer.weight", torch.eye(2), torch.eye(2), "c" * 64, False)
    basis = NormalizedActuatorBasis.from_factors((factor,))
    ledger = NodeBuildLedger()
    ledger.publish(basis, validation_call_count=1)
    assert ledger.live_factor_count == 1
    ledger.release_node()
    assert ledger.live_factor_count == 0 and ledger.retained_factor_count == 0
