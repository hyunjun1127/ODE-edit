from __future__ import annotations

import types
import unittest
from types import SimpleNamespace

import torch

from project.run_scripts.ode_edit_motivation.alphaedit_proposal_adapter import (
    AlphaEditProposalAdapter,
)
from project.run_scripts.ode_edit_motivation.contracts import (
    ContractError,
    EditRequest,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
)
from project.run_scripts.ode_edit_motivation.hooks import capture_snapshot, tensor_sha256


class _Toy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2, bias=False) for _ in range(2)]
        )
        with torch.no_grad():
            self.layers[0].weight.copy_(torch.eye(2))
            self.layers[1].weight.copy_(2.0 * torch.eye(2))


class _Lineage:
    def __init__(
        self,
        expected_state: str,
        *,
        model: torch.nn.Module,
        weight_names: tuple[str, ...],
        fail_on_call: int | None = None,
    ) -> None:
        self.expected_state = expected_state
        self.model = model
        self.weight_names = weight_names
        self.fail_on_call = fail_on_call
        self.live_parameter_states: list[tuple[str, ...]] = []

    def assert_authorizes(self, *, direct_z, snapshot, target_token_ids) -> None:
        if snapshot.state_id != self.expected_state:
            raise AssertionError("unexpected origin state")
        self.live_parameter_states.append(
            tuple(
                tensor_sha256(dict(self.model.named_parameters())[name])
                for name in self.weight_names
            )
        )
        if self.fail_on_call == len(self.live_parameter_states):
            raise ContractError("injected descendant target rejection")


def _install_one_layer_stub(
    adapter: AlphaEditProposalAdapter,
    *,
    weight_names: tuple[str, ...],
    observed_states: list[str],
) -> None:
    def one_layer(
        self,
        *,
        snapshot,
        layer,
        denominator,
        construction,
        suffix,
        current_z=None,
    ):
        observed_states.append(snapshot.state_id)
        name = weight_names[layer]
        factor = LowRankFactor(
            weight_name=name,
            left=torch.tensor([[0.1], [0.0]], dtype=torch.float32),
            right=torch.tensor([[1.0], [0.0]], dtype=torch.float32),
            expected_weight_sha256=snapshot.parameter(name).sha256,
        )
        proposal = MemitFactorProposal(
            snapshot=snapshot,
            factors=(factor,),
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name=f"toy/{suffix}",
            residual_denominator=denominator,
        )
        return proposal, 0.0

    adapter._one_layer = types.MethodType(one_layer, adapter)


class AlphaEditProposalAdapterTests(unittest.TestCase):
    def test_ordered_construction_measures_descendant_and_restores_w0(self) -> None:
        model = _Toy()
        request = EditRequest(
            case_id="1",
            prompt="{} is",
            subject="A",
            target_new="B",
        )
        weight_names = ("layers.0.weight", "layers.1.weight")
        hparams = SimpleNamespace(value=1)

        def snapshot():
            return capture_snapshot(
                model,
                model_id="toy/model@" + "a" * 40,
                requests=(request,),
                context_id="b" * 64,
                hparams=hparams,
                weight_names=weight_names,
                provenance_ids=("c" * 64,),
            )

        origin = snapshot()
        before = {
            name: tensor_sha256(dict(model.named_parameters())[name])
            for name in weight_names
        }
        adapter = object.__new__(AlphaEditProposalAdapter)
        adapter.model = model
        adapter.config = SimpleNamespace(layers=(0, 1))
        adapter.direct_z = object()
        adapter.target_token_ids = torch.tensor([1], dtype=torch.long)
        adapter._last_builds = []
        adapter._snapshot = snapshot

        observed_states: list[str] = []

        _install_one_layer_stub(
            adapter,
            weight_names=weight_names,
            observed_states=observed_states,
        )
        lineage = _Lineage(
            origin.state_id,
            model=model,
            weight_names=weight_names,
        )
        build = AlphaEditProposalAdapter.propose_ordered(
            adapter,
            origin_lineage=lineage,
            construction="genuine-p-inside-solve",
            solver_suffix="unit",
        )

        self.assertEqual(build.proposal.semantics, ProposalSemantics.ORDERED_GAUSS_SEIDEL)
        self.assertEqual(len(build.proposal.factors), 2)
        self.assertEqual(observed_states[0], origin.state_id)
        self.assertNotEqual(observed_states[1], origin.state_id)
        self.assertEqual(len(lineage.live_parameter_states), 3)
        self.assertEqual(lineage.live_parameter_states[0], tuple(before.values()))
        self.assertNotEqual(
            lineage.live_parameter_states[1], lineage.live_parameter_states[0]
        )
        self.assertNotEqual(
            lineage.live_parameter_states[2], lineage.live_parameter_states[1]
        )
        self.assertEqual(
            {
                name: tensor_sha256(dict(model.named_parameters())[name])
                for name in weight_names
            },
            before,
        )
        for factor in build.proposal.factors:
            self.assertEqual(
                factor.expected_weight_sha256,
                origin.parameter(factor.weight_name).sha256,
            )

    def test_descendant_target_rejection_rolls_back_active_layer(self) -> None:
        model = _Toy()
        request = EditRequest(
            case_id="2",
            prompt="{} is",
            subject="A",
            target_new="B",
        )
        weight_names = ("layers.0.weight", "layers.1.weight")
        hparams = SimpleNamespace(value=1)

        def snapshot():
            return capture_snapshot(
                model,
                model_id="toy/model@" + "a" * 40,
                requests=(request,),
                context_id="b" * 64,
                hparams=hparams,
                weight_names=weight_names,
                provenance_ids=("c" * 64,),
            )

        origin = snapshot()
        before = tuple(
            tensor_sha256(dict(model.named_parameters())[name])
            for name in weight_names
        )
        adapter = object.__new__(AlphaEditProposalAdapter)
        adapter.model = model
        adapter.config = SimpleNamespace(layers=(0, 1))
        adapter.direct_z = object()
        adapter.target_token_ids = torch.tensor([1], dtype=torch.long)
        adapter._last_builds = []
        adapter._snapshot = snapshot
        observed_states: list[str] = []
        _install_one_layer_stub(
            adapter,
            weight_names=weight_names,
            observed_states=observed_states,
        )
        lineage = _Lineage(
            origin.state_id,
            model=model,
            weight_names=weight_names,
            fail_on_call=2,
        )

        with self.assertRaisesRegex(
            ContractError, "injected descendant target rejection"
        ):
            AlphaEditProposalAdapter.propose_ordered(
                adapter,
                origin_lineage=lineage,
                construction="genuine-p-inside-solve",
                solver_suffix="unit-reject",
            )

        self.assertEqual(len(observed_states), 1)
        self.assertEqual(len(lineage.live_parameter_states), 2)
        self.assertNotEqual(
            lineage.live_parameter_states[1], lineage.live_parameter_states[0]
        )
        self.assertEqual(
            tuple(
                tensor_sha256(dict(model.named_parameters())[name])
                for name in weight_names
            ),
            before,
        )


if __name__ == "__main__":
    unittest.main()
