from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import torch

from project.run_scripts.barrier_guided_ode.r3.events import (
    FineEventLayout,
    PrefixDirectionalObservation,
    evaluate_fine_events,
)
from project.run_scripts.ode_edit_motivation.contracts import (
    LowRankFactor,
    MemitFactorProposal,
    ParameterRecord,
    ProposalSemantics,
    SnapshotManifest,
)
from project.run_scripts.ode_edit_motivation.hooks import tensor_sha256

from ..alphaedit_basis import FixedAlphaEditBasis
from ..barrier_projection import project_nominal_velocity
from ..event_distribution import TargetExcludedReference
from ..functional_observer import FixedBasisObserver


def _snapshot(factors: tuple[LowRankFactor, ...]) -> SnapshotManifest:
    return SnapshotManifest(
        model_id="toy",
        context_id="ctx",
        request_ids=("r",),
        hparams_sha256="h",
        parameters=tuple(
            ParameterRecord(factor.weight_name, "0" * 64, factor.weight_shape, "torch.float32")
            for factor in factors
        ),
    )


def _proposal(factors: tuple[LowRankFactor, ...]) -> MemitFactorProposal:
    return MemitFactorProposal(
        snapshot=_snapshot(factors),
        factors=factors,
        semantics=ProposalSemantics.ORDERED_GAUSS_SEIDEL,
        solver_name="toy",
    )


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(7, 3)
        self.l1 = torch.nn.Linear(3, 3, bias=False)
        self.l2 = torch.nn.Linear(3, 7, bias=False)
        self.config = SimpleNamespace(use_cache=False)

    def forward(self, *, input_ids: torch.Tensor, use_cache: bool = False):
        hidden = torch.tanh(self.l1(self.embed(input_ids)))
        return SimpleNamespace(logits=self.l2(hidden))


class FixedBasisCoreTests(unittest.TestCase):
    def test_native_theta_maps_to_unit_raw_coefficients(self) -> None:
        factors = tuple(
            LowRankFactor(
                weight_name=f"w{i}.weight",
                left=torch.tensor([[1.0 + i], [2.0 + i]]),
                right=torch.tensor([[0.5], [1.5], [2.5]]),
                expected_weight_sha256="0" * 64,
                native_update_transposed=True,
            )
            for i in range(5)
        )
        basis = FixedAlphaEditBasis.capture(_proposal(factors), factor_sha256=("a" * 64,) * 5)
        self.assertTrue(torch.equal(basis.raw_coefficients(basis.theta_ae), torch.ones(5)))
        self.assertEqual(basis.captured_build_count, 1)

    def test_projection_contract(self) -> None:
        nominal = torch.tensor([1.0, 2.0], dtype=torch.float64)
        gradient = torch.tensor([2.0, 0.0], dtype=torch.float64)
        velocity, receipt = project_nominal_velocity(nominal, gradient)
        self.assertTrue(receipt.projection_active)
        self.assertLessEqual(float(torch.dot(gradient, velocity)), 1e-12)
        unchanged, receipt2 = project_nominal_velocity(nominal, -gradient)
        self.assertTrue(torch.equal(unchanged, nominal))
        self.assertFalse(receipt2.projection_active)

    def test_first_departure_target_excluded_kl(self) -> None:
        layout = FineEventLayout.build(
            source_tokens=(1, 2),
            target_tokens=(3,),
            output_vocabulary_size=5,
            tokenizer_vocabulary_size=5,
        )
        primal = {
            prefix: PrefixDirectionalObservation(logits=torch.linspace(-1, 1, 5, dtype=torch.float32))
            for prefix in layout.internal_prefixes
        }
        initial = evaluate_fine_events(layout, primal)
        reference = TargetExcludedReference.capture(initial)
        scored = {
            prefix: PrefixDirectionalObservation(
                logits=torch.linspace(-0.8, 1.2, 5, dtype=torch.float32),
                tangent_logits=torch.stack(
                    (torch.linspace(-1, 1, 5), torch.linspace(1, -1, 5)), dim=1
                ).float(),
            )
            for prefix in layout.internal_prefixes
        }
        current = evaluate_fine_events(layout, scored)
        barrier, gradient = reference.evaluate(current)
        self.assertGreaterEqual(barrier, 0.0)
        self.assertEqual(tuple(gradient.shape), (2,))
        self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_combined_functional_state_and_jvp(self) -> None:
        torch.manual_seed(4)
        model = ToyModel().eval()
        factors = (
            LowRankFactor("l1.weight", torch.randn(3, 1), torch.randn(3, 1), "0" * 64),
            LowRankFactor("l2.weight", torch.randn(7, 1), torch.randn(3, 1), "0" * 64),
        )
        proposal = _proposal(factors)
        tokenization = SimpleNamespace(prompt_token_ids=(1, 2))
        observer = FixedBasisObserver(model, tokenization, proposal)
        theta = torch.tensor([0.2, -0.3], dtype=torch.float64)
        function = observer._function(())
        functional = function(theta.float()).detach()
        clone = copy.deepcopy(model)
        with torch.no_grad():
            clone.l1.weight.add_(0.2 * (factors[0].left @ factors[0].right.T))
            clone.l2.weight.add_(-0.3 * (factors[1].left @ factors[1].right.T))
        explicit = clone(input_ids=torch.tensor([[1, 2]])).logits[0, -1]
        self.assertTrue(torch.allclose(functional, explicit, atol=2e-6, rtol=2e-6))
        direction = torch.tensor([1.0, -0.5], dtype=torch.float32)
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(theta.float(), direction)
            _, tangent = torch.autograd.forward_ad.unpack_dual(function(dual))
        epsilon = 1e-3
        finite = (function(theta.float() + epsilon * direction) - function(theta.float() - epsilon * direction)) / (2 * epsilon)
        self.assertTrue(torch.allclose(tangent, finite, atol=2e-3, rtol=2e-3))

    def test_functional_observation_does_not_mutate_live_weights(self) -> None:
        model = ToyModel().eval()
        factors = (
            LowRankFactor("l1.weight", torch.randn(3, 1), torch.randn(3, 1), "0" * 64),
            LowRankFactor("l2.weight", torch.randn(7, 1), torch.randn(3, 1), "0" * 64),
        )
        before = {name: tensor_sha256(value) for name, value in model.named_parameters()}
        observer = FixedBasisObserver(model, SimpleNamespace(prompt_token_ids=(1,)), _proposal(factors))
        function = observer._function(())
        function(torch.tensor([0.4, -0.2], dtype=torch.float32))
        after = {name: tensor_sha256(value) for name, value in model.named_parameters()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
