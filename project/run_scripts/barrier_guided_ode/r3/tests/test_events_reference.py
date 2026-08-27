from __future__ import annotations

import pytest
import torch

from project.run_scripts.barrier_guided_ode.r3.events import (
    EventRowMode,
    FineEventLayout,
    PrefixDirectionalObservation,
    evaluate_fine_events,
)
from project.run_scripts.barrier_guided_ode.r3.moments import (
    aggregate_fine_event_moments,
    differential_identity,
)
from project.run_scripts.barrier_guided_ode.r3.reference import W0ConditionalSeal, barrier_decomposition
from project.run_scripts.barrier_guided_ode.r3.solver import ControllerArm, solve_factor_space


def observations(layout: FineEventLayout, *, shift: float = 0.0) -> dict[tuple[int, ...], PrefixDirectionalObservation]:
    result = {}
    vocabulary = layout.output_vocabulary_size
    base_tangents = torch.stack(
        (
            torch.linspace(-0.4, 0.5, vocabulary),
            torch.sin(torch.arange(vocabulary, dtype=torch.float32)),
            torch.cos(torch.arange(vocabulary, dtype=torch.float32)),
            torch.linspace(0.3, -0.2, vocabulary),
            torch.tensor([(index % 3) / 5.0 - 0.2 for index in range(vocabulary)]),
        ),
        dim=1,
    ).float()
    for ordinal, prefix in enumerate(layout.internal_prefixes):
        logits = torch.linspace(-2.2, 1.7, vocabulary, dtype=torch.float32).roll(ordinal)
        logits = (logits + shift * base_tangents[:, 0]).contiguous()
        result[prefix] = PrefixDirectionalObservation(logits, base_tangents.roll(ordinal, dims=0).contiguous())
    return result


@pytest.mark.parametrize(
    ("source", "target", "topology"),
    (
        ((1, 2), (3, 4), "unequal-non-prefix"),
        ((1, 2), (1, 3), "common-prefix-divergence"),
        ((1,), (1, 2, 3), "source-prefix-target"),
        ((2, 3, 4), (2,), "target-prefix-source"),
    ),
)
def test_four_canonical_topologies_normalize_and_match_reference(
    source: tuple[int, ...], target: tuple[int, ...], topology: str
) -> None:
    layout = FineEventLayout.build(
        source_tokens=source,
        target_tokens=target,
        output_vocabulary_size=7,
        tokenizer_vocabulary_size=9,
    )
    assert layout.topology == topology
    streamed = evaluate_fine_events(layout, observations(layout), reducer="streaming")
    reference = evaluate_fine_events(layout, observations(layout), reducer="bruteforce")
    torch.testing.assert_close(streamed.log_probabilities, reference.log_probabilities, rtol=0, atol=3e-14)
    assert streamed.scores is not None and reference.scores is not None
    torch.testing.assert_close(streamed.scores, reference.scores, rtol=0, atol=3e-13)
    assert float(streamed.probabilities.sum()) == pytest.approx(1.0, abs=3e-13)


def test_source_prefix_keeps_atomic_barrier_identity_and_aggregates_only_equality() -> None:
    layout = FineEventLayout.build(
        source_tokens=(1,), target_tokens=(1, 2, 3), output_vocabulary_size=8, tokenizer_vocabulary_size=9
    )
    assert len(layout.source_event_indices) == 7
    assert all(EventRowMode(int(layout.event_modes[index])) is EventRowMode.ATOMIC_DEPARTURE for index in layout.source_event_indices)
    evaluated = evaluate_fine_events(layout, observations(layout))
    seal = W0ConditionalSeal.capture(evaluated)
    moments = aggregate_fine_event_moments(evaluated, seal=seal, reference_time=0.0)
    torch.testing.assert_close(moments.source_score, evaluated.group_score(layout.source_event_indices), rtol=0, atol=2e-16)


def test_target_prefix_collapses_one_target_macro() -> None:
    layout = FineEventLayout.build(
        source_tokens=(2, 4, 5), target_tokens=(2,), output_vocabulary_size=8, tokenizer_vocabulary_size=10
    )
    assert len(layout.target_event_indices) == 1
    target = layout.target_event_indices[0]
    assert EventRowMode(int(layout.event_modes[target])) is EventRowMode.COMPLEMENT_MACRO
    assert layout.macro_excluded_tokens[target] == (4,)


def test_w0_q0_is_immutable_and_log_space_objective_matches_direct_formula() -> None:
    layout = FineEventLayout.build(
        source_tokens=(1, 2), target_tokens=(1, 3), output_vocabulary_size=7, tokenizer_vocabulary_size=8
    )
    initial = evaluate_fine_events(layout, observations(layout))
    current = evaluate_fine_events(layout, observations(layout, shift=0.17))
    seal = W0ConditionalSeal.capture(initial)
    before = seal.tensor_identity
    moments = aggregate_fine_event_moments(current, seal=seal, reference_time=0.4)
    reference = seal.reference_log_probabilities(0.4)
    direct = -reference.exp() / torch.sqrt(current.probabilities)
    torch.testing.assert_close(moments.objective_offset, direct, rtol=2e-14, atol=2e-14)
    seal.assert_immutable()
    assert seal.tensor_identity == before


def test_target_excluded_and_anchored_kl_decompositions() -> None:
    layout = FineEventLayout.build(
        source_tokens=(1, 2), target_tokens=(1, 4), output_vocabulary_size=7, tokenizer_vocabulary_size=8
    )
    initial = evaluate_fine_events(layout, observations(layout))
    current = evaluate_fine_events(layout, observations(layout, shift=0.23))
    seal = W0ConditionalSeal.capture(initial)
    value = barrier_decomposition(seal, current, time=0.6)
    assert value.moving_kl == pytest.approx(value.bernoulli_target_kl + value.conditional_kl_weighted, abs=4e-14)
    assert value.anchored_excess == pytest.approx(
        (1.0 - seal.initial_target_probability)
        * (value.conditional_kl_weighted / (1.0 - seal.reference_log_probabilities(0.6)[seal.target_index].exp().item())),
        abs=5e-14,
    )


def test_reference_tilt_normalizes_and_preserves_non_target_q0() -> None:
    layout = FineEventLayout.build(
        source_tokens=(1, 2), target_tokens=(3, 4), output_vocabulary_size=7, tokenizer_vocabulary_size=9
    )
    initial = evaluate_fine_events(layout, observations(layout))
    seal = W0ConditionalSeal.capture(initial)
    for time in (0.0, 0.25, 2.0, 5.0):
        reference = seal.reference_log_probabilities(time)
        assert float(torch.logsumexp(reference, dim=0)) == pytest.approx(0.0, abs=4e-15)
        non_target_mass = torch.logsumexp(reference[list(seal.non_target_indices)], dim=0)
        torch.testing.assert_close(reference[list(seal.non_target_indices)] - non_target_mass, seal.log_q0, rtol=0, atol=3e-15)


def test_equalities_imply_target_source_signs_and_closed_form_rates() -> None:
    layout = FineEventLayout.build(
        source_tokens=(1, 2), target_tokens=(1, 3), output_vocabulary_size=8, tokenizer_vocabulary_size=9
    )
    initial = evaluate_fine_events(layout, observations(layout))
    seal = W0ConditionalSeal.capture(initial)
    moments = aggregate_fine_event_moments(initial, seal=seal, reference_time=0.0)
    solution = solve_factor_space(
        moments.factor,
        torch.zeros_like(moments.objective_offset),
        moments.equality_directions,
        moments.equality_rates,
        arm=ControllerArm.FISHER,
    )
    identity = differential_identity(moments, solution.velocity)
    assert identity["target_probability_rate"] > 0.0
    assert identity["source_probability_rate"] < 0.0
    assert identity["target_probability_rate"] == pytest.approx(identity["target_closed_form_rate"], abs=2e-13)
    assert identity["source_probability_rate"] == pytest.approx(identity["source_closed_form_rate"], abs=2e-13)


def test_event_scores_match_symmetric_directional_difference() -> None:
    layout = FineEventLayout.build(
        source_tokens=(1, 2), target_tokens=(1, 3), output_vocabulary_size=7, tokenizer_vocabulary_size=8
    )
    base = observations(layout)
    evaluated = evaluate_fine_events(layout, base)
    assert evaluated.scores is not None
    epsilon = 1.0 / 256.0

    def shifted(sign: float) -> dict[tuple[int, ...], PrefixDirectionalObservation]:
        result = {}
        for prefix, value in base.items():
            assert value.tangent_logits is not None
            result[prefix] = PrefixDirectionalObservation(
                (value.logits + sign * epsilon * value.tangent_logits[:, 0]).contiguous()
            )
        return result

    plus = evaluate_fine_events(layout, shifted(1.0))
    minus = evaluate_fine_events(layout, shifted(-1.0))
    difference = (plus.log_probabilities - minus.log_probabilities) / (2.0 * epsilon)
    torch.testing.assert_close(difference, evaluated.scores[:, 0], rtol=3e-3, atol=1e-4)


def test_fine_fisher_adds_conditional_covariance_over_aggregate_event() -> None:
    probabilities = torch.tensor([0.2, 0.3, 0.1, 0.4], dtype=torch.float64)
    scores = torch.tensor([[1.0, 0.0], [-0.5, 0.7], [0.2, -1.0], [-0.175, -0.275]], dtype=torch.float64)
    # Center exactly before comparing fine and a grouped collateral macro.
    scores = scores - (probabilities[:, None] * scores).sum(dim=0)
    fine = scores.T @ (probabilities[:, None] * scores)
    group = (1, 2, 3)
    p_c = probabilities[list(group)].sum()
    mean_c = (probabilities[list(group), None] * scores[list(group)]).sum(dim=0) / p_c
    aggregate = probabilities[0] * torch.outer(scores[0], scores[0]) + p_c * torch.outer(mean_c, mean_c)
    conditional_covariance = (
        probabilities[list(group), None, None]
        * torch.stack([torch.outer(scores[index] - mean_c, scores[index] - mean_c) for index in group])
    ).sum(dim=0)
    torch.testing.assert_close(fine - aggregate, conditional_covariance, rtol=0, atol=3e-16)
    assert float(torch.linalg.eigvalsh(fine - aggregate).min()) >= -4e-16


def test_rare_events_remain_log_stable_when_vocabularies_differ() -> None:
    layout = FineEventLayout.build(
        source_tokens=(2, 4),
        target_tokens=(2, 5, 7),
        output_vocabulary_size=11,
        tokenizer_vocabulary_size=13,
    )
    values = observations(layout)
    rare = {}
    for index, (prefix, value) in enumerate(values.items()):
        logits = value.logits
        if index == len(values) - 1:
            logits = torch.full_like(logits, -80.0)
            logits[0] = 80.0
        rare[prefix] = PrefixDirectionalObservation(logits, value.tangent_logits)
    evaluated = evaluate_fine_events(layout, rare)
    assert torch.isfinite(evaluated.log_probabilities).all()
    assert float(evaluated.log_probabilities.min()) < -100.0
