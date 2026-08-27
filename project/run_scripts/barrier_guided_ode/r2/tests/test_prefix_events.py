from __future__ import annotations

import math
import warnings

import pytest
import torch

from project.run_scripts.barrier_guided_ode.r2.errors import PrefixEventBoundary
from project.run_scripts.barrier_guided_ode.r2.events import (
    PrefixDirectionalObservation,
    PrefixEventLayout,
    evaluate_prefix_events,
)


def _observations(
    layout: PrefixEventLayout,
    *,
    actuator_count: int = 3,
    rare: bool = False,
) -> dict[tuple[int, ...], PrefixDirectionalObservation]:
    result: dict[tuple[int, ...], PrefixDirectionalObservation] = {}
    for ordinal, prefix in enumerate(layout.internal_prefixes):
        logits = torch.linspace(-2.0, 2.0, layout.output_vocabulary_size, dtype=torch.float32)
        logits = logits.roll(ordinal % layout.output_vocabulary_size)
        if rare and ordinal == len(layout.internal_prefixes) - 1:
            logits = torch.full_like(logits, -80.0)
            logits[0] = 80.0
        tangent = torch.stack(
            (
                torch.linspace(-0.5, 0.5, layout.output_vocabulary_size),
                torch.cos(torch.arange(layout.output_vocabulary_size, dtype=torch.float32)),
                torch.sin(torch.arange(layout.output_vocabulary_size, dtype=torch.float32)),
            ),
            dim=1,
        )[:, :actuator_count]
        result[prefix] = PrefixDirectionalObservation(logits, tangent.contiguous())
    return result


@pytest.mark.parametrize(
    ("source", "target", "relation"),
    (
        ((1, 2), (1, 3), "non-prefix"),
        ((1,), (1, 2, 3), "source-prefix-of-target"),
        ((2, 3, 4), (2,), "target-prefix-of-source"),
        ((1, 2), (3, 4, 1), "non-prefix"),
    ),
)
def test_four_topologies_streaming_matches_tiny_vocab_reference(
    source: tuple[int, ...],
    target: tuple[int, ...],
    relation: str,
) -> None:
    layout = PrefixEventLayout.build(
        source_tokens=source,
        target_tokens=target,
        output_vocabulary_size=7,
        tokenizer_vocabulary_size=9,
    )
    assert layout.prefix_relation == relation
    observations = _observations(layout)
    streaming = evaluate_prefix_events(layout, observations, implementation="streaming")
    reference = evaluate_prefix_events(layout, observations, implementation="bruteforce")
    torch.testing.assert_close(streaming.log_probabilities, reference.log_probabilities, rtol=0, atol=2e-14)
    assert streaming.scores is not None and reference.scores is not None
    torch.testing.assert_close(streaming.scores, reference.scores, rtol=0, atol=2e-13)
    assert abs(float(streaming.probabilities.sum()) - 1.0) < 2e-13
    assert streaming.score_centering_norm is not None
    assert streaming.score_centering_norm < 2e-13


def test_rare_event_is_stable_and_vocabularies_may_differ() -> None:
    layout = PrefixEventLayout.build(
        source_tokens=(2, 4),
        target_tokens=(2, 5, 7),
        output_vocabulary_size=11,
        tokenizer_vocabulary_size=13,
    )
    evaluated = evaluate_prefix_events(layout, _observations(layout, rare=True))
    assert torch.isfinite(evaluated.log_probabilities).all()
    assert float(evaluated.log_probabilities.min()) < -100.0
    assert evaluated.normalization_log_residual <= evaluated.normalization_tolerance


def test_directional_scores_match_central_difference() -> None:
    layout = PrefixEventLayout.build(
        source_tokens=(1, 2),
        target_tokens=(1, 3, 4),
        output_vocabulary_size=7,
        tokenizer_vocabulary_size=8,
    )
    observations = _observations(layout, actuator_count=1)
    evaluated = evaluate_prefix_events(layout, observations)
    assert evaluated.scores is not None
    epsilon = 2e-3

    def shifted(sign: float) -> dict[tuple[int, ...], PrefixDirectionalObservation]:
        result: dict[tuple[int, ...], PrefixDirectionalObservation] = {}
        for prefix, observation in observations.items():
            assert observation.tangent_logits is not None
            logits = observation.logits + sign * epsilon * observation.tangent_logits[:, 0]
            result[prefix] = PrefixDirectionalObservation(logits.detach().contiguous())
        return result

    plus = evaluate_prefix_events(layout, shifted(1.0))
    minus = evaluate_prefix_events(layout, shifted(-1.0))
    finite_difference = (plus.log_probabilities - minus.log_probabilities) / (2.0 * epsilon)
    torch.testing.assert_close(finite_difference, evaluated.scores[:, 0], rtol=3e-4, atol=3e-5)


def test_directional_scores_match_forward_mode_jvp() -> None:
    layout = PrefixEventLayout.build(
        source_tokens=(1, 2),
        target_tokens=(1, 3),
        output_vocabulary_size=6,
        tokenizer_vocabulary_size=7,
    )
    observations = _observations(layout, actuator_count=1)
    evaluated = evaluate_prefix_events(layout, observations)
    assert evaluated.scores is not None
    prefixes = layout.internal_prefixes
    primal = tuple(observations[prefix].logits.to(torch.float64) for prefix in prefixes)
    tangent = tuple(
        observations[prefix].tangent_logits[:, 0].to(torch.float64)  # type: ignore[index]
        for prefix in prefixes
    )

    def independent_event_logs(root_logits: torch.Tensor, branch_logits: torch.Tensor) -> torch.Tensor:
        root = torch.log_softmax(root_logits, dim=0)
        branch = torch.log_softmax(branch_logits, dim=0)
        root_mask = torch.ones(6, dtype=torch.bool)
        root_mask[1] = False
        branch_mask = torch.ones(6, dtype=torch.bool)
        branch_mask[[2, 3]] = False
        return torch.stack(
            (
                root[1] + branch[3],
                root[1] + branch[2],
                torch.logsumexp(root[root_mask], dim=0),
                root[1] + torch.logsumexp(branch[branch_mask], dim=0),
            )
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated.*", category=DeprecationWarning)
        _, jvp = torch.func.jvp(independent_event_logs, primal, tangent)
    torch.testing.assert_close(jvp, evaluated.scores[:, 0], rtol=2e-13, atol=2e-13)


def test_batch_equal_tokens_and_invalid_vocab_fail_close() -> None:
    with pytest.raises(PrefixEventBoundary, match="exactly one"):
        PrefixEventLayout.build(
            source_tokens=(1,),
            target_tokens=(2,),
            output_vocabulary_size=4,
            tokenizer_vocabulary_size=4,
            batch_size=2,
        )
    with pytest.raises(PrefixEventBoundary, match="equal"):
        PrefixEventLayout.build(
            source_tokens=(1, 2),
            target_tokens=(1, 2),
            output_vocabulary_size=4,
            tokenizer_vocabulary_size=4,
        )
    with pytest.raises(PrefixEventBoundary, match="invalid"):
        PrefixEventLayout.build(
            source_tokens=(4,),
            target_tokens=(1,),
            output_vocabulary_size=6,
            tokenizer_vocabulary_size=4,
        )


def test_no_termination_token_is_part_of_layout_identity() -> None:
    layout = PrefixEventLayout.build(
        source_tokens=(1,),
        target_tokens=(1, 2),
        output_vocabulary_size=5,
        tokenizer_vocabulary_size=5,
    )
    identities = "\n".join(event.identity for event in layout.events).lower()
    assert "eot" not in identities and "eos" not in identities and "delimiter" not in identities
    assert math.isfinite(float(len(layout.events)))
