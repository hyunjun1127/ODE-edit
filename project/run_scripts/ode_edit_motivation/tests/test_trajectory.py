import math
import types
import unittest

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    ContextManifest,
    ContractError,
    EditRequest,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
)
from project.run_scripts.ode_edit_motivation.hooks import (
    TemporaryLowRankApplication,
    capture_snapshot,
    tensor_sha256,
)
from project.run_scripts.ode_edit_motivation.frozen_target_lineage import (
    FrozenTargetLineage,
)
from project.run_scripts.ode_edit_motivation.mv1_calibration import (
    ActionDirection,
    scale_proposal,
)
from project.run_scripts.ode_edit_motivation.trajectory import (
    build_central_probe_panel,
    evaluate_temporary_trajectory,
    proposal_direction_hash,
    transport_fixed_actions,
    transport_fixed_proposal,
    validate_matched_c_budget,
)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = torch.nn.Linear(2, 2, bias=False)
        self.layer5 = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.layer4.weight.copy_(torch.eye(2))
            self.layer5.weight.copy_(2.0 * torch.eye(2))


class TrajectoryPrimitiveTests(unittest.TestCase):
    def setUp(self):
        self.model = ToyModel()
        self.request = EditRequest.from_mapping(
            {
                "case_id": "one",
                "prompt": "{} lives in",
                "subject": "Ada",
                "target_new": "London",
            }
        )
        self.contexts = ContextManifest.freeze([["{}"]], source="trajectory-test")
        self.names = ("layer4.weight", "layer5.weight")
        self.hparams = {"layers": [4, 5]}
        self.origin = self.snapshot()
        self.proposal = self.make_proposal(self.origin)
        self.covariances = {4: torch.eye(2), 5: torch.eye(2)}
        self.layer_by_weight = {"layer4.weight": 4, "layer5.weight": 5}

    def snapshot(self):
        return capture_snapshot(
            self.model,
            model_id="toy",
            requests=(self.request,),
            context_id=self.contexts.manifest_id,
            hparams=self.hparams,
            weight_names=self.names,
        )

    def make_proposal(self, snapshot, *, scale=1.0):
        factors = tuple(
            LowRankFactor(
                weight_name=name,
                left=torch.tensor([[scale], [0.0]]),
                right=torch.tensor([[0.5], [0.0]]),
                expected_weight_sha256=snapshot.parameter(name).sha256,
            )
            for name in self.names
        )
        return MemitFactorProposal(
            snapshot=snapshot,
            factors=factors,
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="toy",
            residual_denominator=2,
        )

    def descendant(self):
        first = scale_proposal(self.proposal, 0.5, solver_suffix="partial")
        application = TemporaryLowRankApplication(self.model, first)
        application.__enter__()
        child = self.snapshot()
        lineage = self.lineage_origin().derive_partial(
            parent_snapshot=self.origin,
            child_snapshot=child,
            application=application,
            full_step_proposal=self.proposal,
        )
        return first, application, child, lineage

    def lineage_origin(self):
        return FrozenTargetLineage(
            model_id=self.origin.model_id,
            context_id=self.origin.context_id,
            request_ids=self.origin.request_ids,
            hparams_sha256=self.origin.hparams_sha256,
            origin_snapshot_id=self.origin.snapshot_id,
            origin_state_id=self.origin.state_id,
            direct_z_tensor_sha256="a" * 64,
            direct_z_artifact_sha256="b" * 64,
            direct_z_artifact_size=1,
            target_token_sha256="c" * 64,
            target_token_shape=(1,),
            target_token_dtype="torch.int64",
            origin_parameter_hashes=tuple(
                (record.name, record.sha256)
                for record in self.origin.parameters
            ),
        )

    def test_fixed_transport_keeps_tensor_direction_and_rebinds_hashes(self):
        first, application, child, lineage = self.descendant()
        try:
            transported = transport_fixed_proposal(
                self.proposal,
                descendant_snapshot=child,
                frozen_target_lineage=lineage,
                solver_suffix="fixed",
            )
            self.assertEqual(
                proposal_direction_hash(transported),
                proposal_direction_hash(self.proposal),
            )
            self.assertEqual(transported.entry_snapshot_id, child.state_id)
            for factor in transported.factors:
                self.assertEqual(
                    factor.expected_weight_sha256,
                    child.parameter(factor.weight_name).sha256,
                )
        finally:
            application.__exit__(None, None, None)

    def test_transport_action_order_is_exact(self):
        first, application, child, lineage = self.descendant()
        try:
            single_proposals = tuple(
                MemitFactorProposal(
                    snapshot=self.origin,
                    factors=(factor,),
                    semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
                    solver_name=f"toy-single-{index}",
                    residual_denominator=2,
                )
                for index, factor in enumerate(self.proposal.factors)
            )
            actions = (
                ActionDirection("layer_4", single_proposals[0], 1.0),
                ActionDirection("layer_5", single_proposals[1], 1.0),
            )
            transported = transport_fixed_actions(
                actions,
                descendant_snapshot=child,
                frozen_target_lineage=lineage,
                solver_suffix="fixed",
            )
            self.assertEqual(
                tuple(action.action_id for action in transported),
                ("layer_4", "layer_5"),
            )
        finally:
            application.__exit__(None, None, None)

    def test_fixed_transport_rejects_unrelated_descendant_with_valid_lineage(self):
        _first, application, child, lineage = self.descendant()
        application.__exit__(None, None, None)
        unrelated_step = scale_proposal(
            self.proposal,
            0.25,
            solver_suffix="unrelated",
        )
        with TemporaryLowRankApplication(self.model, unrelated_step):
            unrelated = self.snapshot()
            with self.assertRaises(ContractError):
                transport_fixed_proposal(
                    self.proposal,
                    descendant_snapshot=unrelated,
                    frozen_target_lineage=lineage,
                    solver_suffix="must-reject",
                )

        with TemporaryLowRankApplication(self.model, unrelated_step):
            unrelated_origin = self.snapshot()
        unrelated_w0_proposal = self.make_proposal(unrelated_origin)
        with self.assertRaises(ContractError):
            transport_fixed_proposal(
                unrelated_w0_proposal,
                descendant_snapshot=child,
                frozen_target_lineage=lineage,
                solver_suffix="wrong-w0-must-reject",
            )

    def test_central_panel_calls_each_signed_probe_once(self):
        actions = (
            ActionDirection("layer_4", self.proposal, 1.0),
            ActionDirection("layer_5", self.proposal, 1.0),
        )
        calls = []

        def evaluate(action, distance):
            calls.append((action.action_id, distance))
            slope = 2.0 if action.action_id == "layer_4" else -1.0
            return types.SimpleNamespace(
                utility=slope * distance,
                context_utility=(slope * distance, 2.0 * slope * distance),
            )

        panel = build_central_probe_panel(
            actions,
            epsilon=0.25,
            evaluate=evaluate,
            expected_action_ids=("layer_4", "layer_5"),
        )
        self.assertEqual(panel.evaluation_count, 4)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertAlmostEqual(panel.scores["layer_4"], 2.0)
        self.assertAlmostEqual(panel.scores["layer_5"], -1.0)
        self.assertRegex(panel.panel_hash, r"^[0-9a-f]{64}$")

    def test_central_panel_rejects_nonfinite_and_wrong_order(self):
        actions = (ActionDirection("layer_4", self.proposal, 1.0),)
        with self.assertRaises(ContractError):
            build_central_probe_panel(
                actions,
                epsilon=math.inf,
                evaluate=lambda *_args: None,
                expected_action_ids=("layer_4",),
            )
        with self.assertRaises(ContractError):
            build_central_probe_panel(
                actions,
                epsilon=0.1,
                evaluate=lambda *_args: None,
                expected_action_ids=("layer_5",),
            )

    def test_matched_c_budget_requires_exact_arm_order_and_finite_energy(self):
        # Each factor has energy .25 and there are two layers => .5.
        proposals = {
            "A": self.proposal,
            "B": self.proposal,
            "C": self.proposal,
        }
        observed = validate_matched_c_budget(
            proposals,
            expected_c_energy=0.5,
            covariance_by_layer=self.covariances,
            layer_by_weight=self.layer_by_weight,
            exact_action_ids=("A", "B", "C"),
        )
        self.assertEqual(tuple(observed), ("A", "B", "C"))
        poisoned = self.make_proposal(self.origin, scale=2.0)
        with self.assertRaises(ContractError):
            validate_matched_c_budget(
                {"A": self.proposal, "B": poisoned, "C": self.proposal},
                expected_c_energy=0.5,
                covariance_by_layer=self.covariances,
                layer_by_weight=self.layer_by_weight,
                exact_action_ids=("A", "B", "C"),
            )

    def test_two_step_branch_rolls_back_state_and_rng_on_success_and_fatal(self):
        first = scale_proposal(self.proposal, 0.5, solver_suffix="partial")
        with TemporaryLowRankApplication(self.model, first) as application:
            child = self.snapshot()
            lineage = self.lineage_origin().derive_partial(
                parent_snapshot=self.origin,
                child_snapshot=child,
                application=application,
                full_step_proposal=self.proposal,
            )
        second = transport_fixed_proposal(
            self.proposal,
            descendant_snapshot=child,
            frozen_target_lineage=lineage,
            solver_suffix="second",
        )
        cpu_rng = torch.get_rng_state().clone()

        value = evaluate_temporary_trajectory(
            model=self.model,
            origin_snapshot=self.origin,
            evaluator=lambda: float(torch.rand(()).item()),
            cpu_rng=cpu_rng,
            cuda_rng=None,
            first_proposal=first,
            descendant_snapshot=child,
            second_proposal=second,
        )
        self.assertTrue(math.isfinite(value))
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        for record in self.origin.parameters:
            self.assertEqual(
                tensor_sha256(dict(self.model.named_parameters())[record.name]),
                record.sha256,
            )

        def fatal():
            torch.rand(())
            raise RuntimeError("fatal")

        with self.assertRaisesRegex(RuntimeError, "fatal"):
            evaluate_temporary_trajectory(
                model=self.model,
                origin_snapshot=self.origin,
                evaluator=fatal,
                cpu_rng=cpu_rng,
                cuda_rng=None,
                first_proposal=first,
                descendant_snapshot=child,
                second_proposal=second,
            )
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))
        for record in self.origin.parameters:
            self.assertEqual(
                tensor_sha256(dict(self.model.named_parameters())[record.name]),
                record.sha256,
            )


if __name__ == "__main__":
    unittest.main()
