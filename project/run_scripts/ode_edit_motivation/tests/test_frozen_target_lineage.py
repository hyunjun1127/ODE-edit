import unittest

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    ContextManifest,
    EditRequest,
    FileRecord,
)
from project.run_scripts.ode_edit_motivation.direct_z import FrozenDirectZ
from project.run_scripts.ode_edit_motivation.frozen_target_lineage import (
    FrozenTargetLineage,
)
from project.run_scripts.ode_edit_motivation.hooks import (
    TemporaryLowRankApplication,
    capture_snapshot,
    tensor_sha256,
)
from project.run_scripts.ode_edit_motivation.contracts import (
    ContractError,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
)
from project.run_scripts.ode_edit_motivation.mv1_calibration import scale_proposal
class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = torch.nn.Linear(2, 2, bias=False)
        self.layer5 = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.layer4.weight.copy_(torch.eye(2))
            self.layer5.weight.copy_(2.0 * torch.eye(2))


class FrozenTargetLineageTests(unittest.TestCase):
    def setUp(self):
        self.model = ToyModel()
        self.request = EditRequest.from_mapping(
            {
                "case_id": "case-1",
                "prompt": "{} lives in",
                "subject": "Ada",
                "target_new": "London",
            }
        )
        self.contexts = ContextManifest.freeze([["{}"]], source="lineage-test")
        self.hparams = {"layers": [4, 5]}
        self.names = ("layer4.weight", "layer5.weight")
        self.origin = capture_snapshot(
            self.model,
            model_id="toy",
            requests=(self.request,),
            context_id=self.contexts.manifest_id,
            hparams=self.hparams,
            weight_names=self.names,
            provenance_ids=("b" * 64,),
        )
        values = torch.tensor([[1.0], [2.0]])
        self.direct_z = FrozenDirectZ(
            values=values,
            source_snapshot_id=self.origin.snapshot_id,
            source_state_id=self.origin.state_id,
            model_id=self.origin.model_id,
            context_id=self.origin.context_id,
            request_ids=self.origin.request_ids,
            z_layer=5,
            tensor_sha256=tensor_sha256(values),
            artifact=FileRecord(path="/local/direct-z.pt", sha256="a" * 64, size=64),
        )
        self.target_ids = torch.tensor([7, 8], dtype=torch.int64)
        factors = tuple(
            LowRankFactor(
                weight_name=name,
                left=torch.tensor([[1.0], [0.0]]),
                right=torch.tensor([[0.25], [0.0]]),
                expected_weight_sha256=self.origin.parameter(name).sha256,
            )
            for name in self.names
        )
        self.proposal = MemitFactorProposal(
            snapshot=self.origin,
            factors=factors,
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="toy",
            residual_denominator=2,
        )

    def lineage(self):
        return FrozenTargetLineage.start(
            direct_z=self.direct_z,
            origin_snapshot=self.origin,
            target_token_ids=self.target_ids,
        )

    def test_h0_sham_requires_identical_state_and_preserves_target(self):
        lineage = self.lineage()
        sham = lineage.derive_h0(snapshot=self.origin)
        sham.assert_authorizes(
            direct_z=self.direct_z,
            snapshot=self.origin,
            target_token_ids=self.target_ids,
        )
        self.assertEqual(sham.terminal_state_id, self.origin.state_id)
        self.assertEqual(sham.to_dict()["hops"][0]["step_scale"], 0.0)

    def test_partial_descendant_uses_observed_applied_hashes(self):
        lineage = self.lineage()
        partial = scale_proposal(
            self.proposal,
            0.5,
            solver_suffix="partial-h-1-2",
        )
        with TemporaryLowRankApplication(self.model, partial) as applied:
            child = capture_snapshot(
                self.model,
                model_id="toy",
                requests=(self.request,),
                context_id=self.contexts.manifest_id,
                hparams=self.hparams,
                weight_names=self.names,
                provenance_ids=("b" * 64,),
            )
            derived = lineage.derive_partial(
                parent_snapshot=self.origin,
                child_snapshot=child,
                application=applied,
                full_step_proposal=self.proposal,
            )
            derived.assert_authorizes(
                direct_z=self.direct_z,
                snapshot=child,
                target_token_ids=self.target_ids,
            )
            self.assertNotEqual(child.state_id, self.origin.state_id)
        # Outer trajectory application must still be bit-exact after lineage use.
        for record in self.origin.parameters:
            parameter = dict(self.model.named_parameters())[record.name]
            self.assertEqual(tensor_sha256(parameter), record.sha256)

    def test_sparse_score_mix_descendant_proves_changed_subset_and_unchanged_rest(self):
        sparse = MemitFactorProposal(
            snapshot=self.origin,
            factors=(self.proposal.factors[0],),
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="toy-sparse-score-mix",
            residual_denominator=2,
        )
        partial = scale_proposal(sparse, 0.5, solver_suffix="partial-h-1-2")
        with TemporaryLowRankApplication(self.model, partial) as applied:
            child = capture_snapshot(
                self.model,
                model_id="toy",
                requests=(self.request,),
                context_id=self.contexts.manifest_id,
                hparams=self.hparams,
                weight_names=self.names,
                provenance_ids=("b" * 64,),
            )
            derived = self.lineage().derive_partial(
                parent_snapshot=self.origin,
                child_snapshot=child,
                application=applied,
                full_step_proposal=sparse,
            )
            derived.assert_authorizes(
                direct_z=self.direct_z,
                snapshot=child,
                target_token_ids=self.target_ids,
            )
            self.assertEqual(
                child.parameter(self.names[1]).sha256,
                self.origin.parameter(self.names[1]).sha256,
            )

    def test_hash_target_and_descendant_mismatches_fail_closed(self):
        lineage = self.lineage()
        with self.assertRaises(ContractError):
            lineage.assert_target_tokens(torch.tensor([7, 9], dtype=torch.int64))

        self.direct_z.values[0, 0] = 9.0
        with self.assertRaises(ContractError):
            lineage.assert_authorizes(
                direct_z=self.direct_z,
                snapshot=self.origin,
                target_token_ids=self.target_ids,
            )

    def test_noncanonical_application_scale_is_rejected(self):
        partial = scale_proposal(
            self.proposal,
            0.5,
            solver_suffix="partial-h-1-2",
        )
        with TemporaryLowRankApplication(
            self.model,
            partial,
            scale=0.5,
        ) as applied:
            child = capture_snapshot(
                self.model,
                model_id="toy",
                requests=(self.request,),
                context_id=self.contexts.manifest_id,
                hparams=self.hparams,
                weight_names=self.names,
                provenance_ids=("b" * 64,),
            )
            with self.assertRaises(ContractError):
                self.lineage().derive_partial(
                    parent_snapshot=self.origin,
                    child_snapshot=child,
                    application=applied,
                    full_step_proposal=self.proposal,
                )

    def test_wrong_direction_wrong_h_and_second_hop_are_rejected(self):
        lineage = self.lineage()
        wrong_direction = self.make_wrong_direction()
        wrong_h = scale_proposal(
            self.proposal,
            0.25,
            solver_suffix="wrong-h-1-4",
        )
        for proposal in (wrong_direction, wrong_h):
            with self.subTest(solver_name=proposal.solver_name):
                with TemporaryLowRankApplication(self.model, proposal) as applied:
                    child = capture_snapshot(
                        self.model,
                        model_id="toy",
                        requests=(self.request,),
                        context_id=self.contexts.manifest_id,
                        hparams=self.hparams,
                        weight_names=self.names,
                        provenance_ids=("b" * 64,),
                    )
                    with self.assertRaises(ContractError):
                        lineage.derive_partial(
                            parent_snapshot=self.origin,
                            child_snapshot=child,
                            application=applied,
                            full_step_proposal=self.proposal,
                        )
        sham = lineage.derive_h0(snapshot=self.origin)
        with self.assertRaises(ContractError):
            sham.derive_h0(snapshot=self.origin)

    def make_wrong_direction(self):
        factors = tuple(
            LowRankFactor(
                weight_name=factor.weight_name,
                left=-0.5 * factor.left,
                right=factor.right,
                expected_weight_sha256=factor.expected_weight_sha256,
                native_update_transposed=factor.native_update_transposed,
            )
            for factor in self.proposal.factors
        )
        return MemitFactorProposal(
            snapshot=self.origin,
            factors=factors,
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="wrong-direction",
            residual_denominator=2,
        )


if __name__ == "__main__":
    unittest.main()
