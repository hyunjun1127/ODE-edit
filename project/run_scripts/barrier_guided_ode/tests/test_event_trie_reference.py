from __future__ import annotations

import math

import pytest
import torch

from project.run_scripts.barrier_guided_ode.errors import (
    EventPartitionBoundary,
    UnsupportedBatchBoundary,
)
from project.run_scripts.barrier_guided_ode.event_trie import (
    JointFirstDepartureTrie,
    TerminationConvention,
)
from project.run_scripts.barrier_guided_ode.reference_path import (
    forward_kl,
    pair_mass_ceiling_diagnostic,
    single_coordinate_decomposition,
    single_coordinate_reference,
)


def _prefix_logits(trie: JointFirstDepartureTrie, *, extreme: bool = False):
    result = {}
    for index, prefix in enumerate(trie.internal_prefixes):
        if extreme:
            row = torch.linspace(-900.0, 900.0, trie.vocabulary_size, dtype=torch.float32)
            row = torch.roll(row, index)
        else:
            row = torch.arange(trie.vocabulary_size, dtype=torch.float32) * 0.17
            row = torch.roll(row, index)
        result[prefix] = row
    return result


@pytest.mark.parametrize(
    ("source", "target", "relation"),
    [
        ((0,), (0, 1), "source-prefix-of-target"),
        ((0, 1), (0,), "target-prefix-of-source"),
        ((0, 1), (0, 2, 1), "non-prefix"),
    ],
)
@pytest.mark.parametrize("boundary", [4, 5])
def test_joint_partition_is_disjoint_exhaustive_and_log_stable(
    source, target, relation, boundary
):
    trie = JointFirstDepartureTrie(
        source_tokens=source,
        target_tokens=target,
        termination=TerminationConvention(f"boundary-{boundary}", boundary),
        vocabulary_size=6,
    )
    distribution = trie.evaluate_bruteforce(_prefix_logits(trie, extreme=True))
    assert trie.prefix_relation == relation
    assert len({event.identity for event in distribution.events}) == len(distribution.events)
    assert torch.logsumexp(distribution.log_probabilities, 0).abs().item() < 2e-5
    assert distribution.normalization_log_residual < 2e-5
    # Every internal-node token is either a trie edge or exactly one departure.
    departures = {
        (event.prefix, event.token)
        for event in distribution.events
        if event.token is not None
    }
    for prefix, children in trie.child_tokens_by_prefix.items():
        observed = {token for candidate, token in departures if candidate == prefix}
        assert observed == set(range(trie.vocabulary_size)) - set(children)


def test_event_partition_rejects_equal_tokenization_embedded_boundary_and_batches():
    termination = TerminationConvention("eot", 4)
    with pytest.raises(EventPartitionBoundary, match="equal tokenization"):
        JointFirstDepartureTrie(
            source_tokens=(0, 1),
            target_tokens=(0, 1),
            termination=termination,
            vocabulary_size=5,
        )
    with pytest.raises(EventPartitionBoundary, match="cannot occur"):
        JointFirstDepartureTrie(
            source_tokens=(0, 4),
            target_tokens=(0, 1),
            termination=termination,
            vocabulary_size=5,
        )
    with pytest.raises(UnsupportedBatchBoundary, match="B=1"):
        JointFirstDepartureTrie(
            source_tokens=(0,),
            target_tokens=(1,),
            termination=termination,
            vocabulary_size=5,
            batch_size=2,
        )


def test_prefix_logits_reject_batch_dimension_greater_than_one():
    trie = JointFirstDepartureTrie(
        source_tokens=(0,),
        target_tokens=(1,),
        termination=TerminationConvention("eot", 4),
        vocabulary_size=5,
    )
    logits = _prefix_logits(trie)
    logits[()] = torch.zeros((2, 5), dtype=torch.float32)
    with pytest.raises(UnsupportedBatchBoundary, match="batch dimension one"):
        trie.evaluate_bruteforce(logits)


def _distribution():
    trie = JointFirstDepartureTrie(
        source_tokens=(0,),
        target_tokens=(0, 1),
        termination=TerminationConvention("eot", 4),
        vocabulary_size=5,
    )
    return trie.evaluate_bruteforce(_prefix_logits(trie))


def test_forward_kl_reference_and_on_manifold_decomposition():
    initial = _distribution()
    time = 0.7
    reference = single_coordinate_reference(initial, time=time)
    iy, is_ = initial.target_index, initial.source_index
    progress = (
        reference.log_probabilities[iy]
        - reference.log_probabilities[is_]
        - (initial.log_probabilities[iy] - initial.log_probabilities[is_])
    )
    assert progress.item() == pytest.approx(time, abs=2e-6)
    outside = torch.ones(len(initial.events), dtype=torch.bool)
    outside[iy] = False
    outside[is_] = False
    assert torch.equal(
        reference.log_probabilities[outside], initial.log_probabilities[outside]
    )
    exact = single_coordinate_decomposition(
        initial, reference.log_probabilities, time=time
    )
    assert exact.exact_progress_residual < 2e-6
    assert exact.anchored_kl == pytest.approx(exact.unavoidable_pair_cost, abs=2e-6)
    assert exact.anchored_excess == pytest.approx(exact.moving_reference_kl, abs=2e-6)
    assert exact.moving_reference_kl == pytest.approx(0.0, abs=2e-6)

    # Keep pair mass/log odds exact but move conditional outside mass.
    current = reference.log_probabilities.exp()
    outside_indices = outside.nonzero(as_tuple=False).flatten()
    assert len(outside_indices) >= 2
    total = current[outside_indices].sum()
    weights = torch.arange(1, len(outside_indices) + 1, dtype=torch.float32)
    weights = weights / weights.sum()
    current[outside_indices] = total * weights
    current_log = current.log()
    drifted = single_coordinate_decomposition(initial, current_log, time=time)
    assert drifted.exact_progress_residual < 2e-6
    assert drifted.outside_conditional_drift > 0
    assert drifted.anchored_excess == pytest.approx(
        drifted.moving_reference_kl, rel=2e-5, abs=2e-6
    )


def test_single_coordinate_reference_is_forward_kl_optimum():
    initial = _distribution()
    time = 0.55
    reference = single_coordinate_reference(initial, time=time)
    optimum = forward_kl(initial.log_probabilities, reference.log_probabilities)
    p0 = initial.probabilities
    iy, is_ = initial.target_index, initial.source_index
    ratio = torch.exp(
        reference.log_probabilities[iy] - reference.log_probabilities[is_]
    )
    outside = torch.ones_like(p0, dtype=torch.bool)
    outside[iy] = False
    outside[is_] = False
    q0 = p0[outside] / p0[outside].sum()
    for pair_mass in (0.08, 0.2, 0.45, 0.75):
        candidate = torch.empty_like(p0)
        c = torch.tensor(pair_mass, dtype=torch.float32)
        candidate[iy] = c * ratio / (1.0 + ratio)
        candidate[is_] = c / (1.0 + ratio)
        # Deliberately retain the best outside conditional; varying c alone is
        # enough to show the reference pair mass is the optimum.
        candidate[outside] = (1.0 - c) * q0
        value = forward_kl(initial.log_probabilities, candidate.log())
        assert value.item() >= optimum.item() - 3e-6


def test_anchored_difference_equivalence_fails_off_manifold():
    initial = _distribution()
    time = 0.9
    values = single_coordinate_decomposition(
        initial, initial.log_probabilities, time=time
    )
    assert values.exact_progress_residual == pytest.approx(time, abs=2e-6)
    assert not math.isclose(
        values.anchored_excess,
        values.moving_reference_kl,
        rel_tol=1e-4,
        abs_tol=1e-5,
    )


def test_pair_mass_ceiling_is_diagnostic_without_threshold():
    initial = _distribution()
    native_log = initial.log_probabilities.clone()
    iy, is_ = initial.target_index, initial.source_index
    native = native_log.exp()
    outside = torch.ones_like(native, dtype=torch.bool)
    outside[iy] = False
    outside[is_] = False
    # Increase pair mass while preserving a normalized categorical endpoint.
    target_pair = torch.tensor(0.55, dtype=torch.float32)
    native[iy] = target_pair * 0.8
    native[is_] = target_pair * 0.2
    native[outside] *= (1.0 - target_pair) / native[outside].sum()
    from project.run_scripts.barrier_guided_ode.event_trie import EventDistribution

    endpoint = EventDistribution(
        events=initial.events,
        log_probabilities=native.log(),
        target_index=iy,
        source_index=is_,
        normalization_log_residual=abs(float(torch.logsumexp(native.log(), 0))),
    )
    diagnostic = pair_mass_ceiling_diagnostic(initial, endpoint)
    assert diagnostic.matched_target_probability / diagnostic.native_target_probability == pytest.approx(
        diagnostic.pair_mass_ratio, rel=2e-5
    )
    assert diagnostic.matched_target_probability <= diagnostic.initial_pair_mass
    assert not hasattr(diagnostic, "pass_threshold")


def test_two_coordinate_single_kl_identity_is_not_reused():
    # On a two-coordinate manifold the anchored outside term has weight
    # (1-c0), whereas KL(pi*_t || pi_W) has weight (1-ct).
    c0 = torch.tensor(0.2, dtype=torch.float32)
    ct = torch.tensor(0.6, dtype=torch.float32)
    q0 = torch.tensor([0.8, 0.2], dtype=torch.float32)
    qw = torch.tensor([0.5, 0.5], dtype=torch.float32)
    outside_kl = torch.sum(q0 * (torch.log(q0) - torch.log(qw)))
    anchored_excess = (1.0 - c0) * outside_kl
    moving_reference_kl = (1.0 - ct) * outside_kl
    assert not torch.isclose(anchored_excess, moving_reference_kl)


def test_forward_kl_never_uses_length_normalization():
    initial = _distribution()
    assert forward_kl(initial.log_probabilities, initial.log_probabilities).item() == pytest.approx(0.0)
