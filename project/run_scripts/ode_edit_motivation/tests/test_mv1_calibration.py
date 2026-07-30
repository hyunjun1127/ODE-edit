import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
from project.run_scripts.ode_edit_motivation.mv0_fidelity import (
    MV0Error,
    TeacherForcedResult,
)
from project.run_scripts.ode_edit_motivation.mv1_calibration import (
    ACTION_UNIFORM,
    FRACTIONS,
    _feature_hash,
    _mv1_slurm_state,
    assert_exact_action_contract,
    build_exact_teacher_batch,
    build_unit_c_actions,
    commit_feature_actions,
    proposal_c_energy,
    rewrite_metrics,
    scale_proposal,
    select_analytic_action,
)


class _ExactTokenizer:
    padding_side = "right"
    pad_token_id = 0

    def __init__(self, *, merge_boundary=False):
        self.merge_boundary = merge_boundary

    def encode(self, text, *, add_special_tokens):
        if text == " London":
            return [7, 8]
        if text.endswith(" London"):
            suffix = [9, 8] if self.merge_boundary else [7, 8]
            prefix = [1, 2] if add_special_tokens else [2]
            return [*prefix, *suffix]
        raise AssertionError(f"unexpected fake-tokenizer input: {text!r}")


class _WriterSpy:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def write(self, event, payload):
        self.events.append((self.name, "write", event, payload["fraction"]))

    def sync(self):
        self.events.append((self.name, "sync"))


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Linear(2, 2, bias=False)
        self.layer1 = torch.nn.Linear(2, 2, bias=False)


class MV1CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.model = _ToyModel()
        self.model.eval()
        self.request = EditRequest.from_mapping(
            {
                "case_id": "case-1",
                "prompt": "{} lives in",
                "subject": "Ada",
                "target_new": "London",
            }
        )
        contexts = ContextManifest.freeze([["{}"]], source="unit-test")
        snapshot = capture_snapshot(
            self.model,
            model_id="toy",
            requests=(self.request,),
            context_id=contexts.manifest_id,
            hparams={"layers": [0, 1]},
            weight_names=("layer0.weight", "layer1.weight"),
        )
        factors = (
            LowRankFactor(
                "layer0.weight",
                torch.tensor([[1.0], [2.0]]),
                torch.tensor([[2.0], [1.0]]),
                snapshot.parameter("layer0.weight").sha256,
            ),
            LowRankFactor(
                "layer1.weight",
                torch.tensor([[3.0], [1.0]]),
                torch.tensor([[1.0], [4.0]]),
                snapshot.parameter("layer1.weight").sha256,
            ),
        )
        self.proposal = MemitFactorProposal(
            snapshot=snapshot,
            factors=factors,
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="unit-test/synchronous",
            residual_denominator=2,
        )
        self.covariances = {0: torch.eye(2), 1: torch.eye(2)}
        self.layers = {"layer0.weight": 0, "layer1.weight": 1}

    def test_equal_c_actions_have_unit_energy_and_rollback(self):
        actions = build_unit_c_actions(
            self.proposal,
            self.covariances,
            self.layers,
        )

        self.assertEqual(
            tuple(action.action_id for action in actions),
            ("layer_0", "layer_1", ACTION_UNIFORM),
        )
        before = {
            name: tensor_sha256(parameter)
            for name, parameter in self.model.named_parameters()
        }
        for action in actions:
            energy = proposal_c_energy(
                action.proposal,
                self.covariances,
                self.layers,
            )
            self.assertAlmostEqual(energy, 1.0, places=5)
            with TemporaryLowRankApplication(
                self.model,
                scale_proposal(action.proposal, 0.125, solver_suffix="test"),
            ):
                self.assertNotEqual(
                    tensor_sha256(self.model.layer0.weight)
                    if action.action_id != "layer_1"
                    else tensor_sha256(self.model.layer1.weight),
                    before[
                        "layer0.weight"
                        if action.action_id != "layer_1"
                        else "layer1.weight"
                    ],
                )
            self.assertEqual(
                {
                    name: tensor_sha256(parameter)
                    for name, parameter in self.model.named_parameters()
                },
                before,
            )

    def test_exact_teacher_batch_preserves_multitoken_suffix(self):
        contexts = ContextManifest.freeze([["{}"]], source="unit-test")

        batch = build_exact_teacher_batch(
            _ExactTokenizer(),
            self.request,
            contexts,
        )

        self.assertEqual(batch.target_ids.tolist(), [7, 8])
        self.assertEqual(batch.input_ids.tolist(), [[1, 2, 7]])
        self.assertEqual(batch.attention_mask.tolist(), [[1, 1, 1]])
        self.assertEqual(batch.input_lengths, (3,))

    def test_exact_teacher_batch_fails_closed_on_boundary_merge(self):
        contexts = ContextManifest.freeze([["{}"]], source="unit-test")

        with self.assertRaises(MV0Error):
            build_exact_teacher_batch(
                _ExactTokenizer(merge_boundary=True),
                self.request,
                contexts,
            )

    def test_exact_action_contract_rejects_missing_factor(self):
        actions = build_unit_c_actions(
            self.proposal,
            self.covariances,
            self.layers,
        )

        with self.assertRaises(MV0Error):
            assert_exact_action_contract(
                self.proposal,
                self.proposal,
                actions,
                expected_factor_names=("layer0.weight",),
                expected_action_ids=("layer_0", ACTION_UNIFORM),
                covariance_by_layer=self.covariances,
                layer_by_weight=self.layers,
            )

    def test_each_fraction_gets_a_distinct_receipt_after_both_streams_sync(self):
        events = []
        request_id = self.request.request_id
        feature_records = {}
        committed_actions = {}
        for fraction in FRACTIONS:
            feature = {
                "case_id": self.request.case_id,
                "request_id": request_id,
                "fraction": fraction,
                "feature_hash": f"feature-{fraction}",
            }
            action = {
                "case_id": self.request.case_id,
                "request_id": request_id,
                "fraction": fraction,
                "feature_hash": feature["feature_hash"],
                "commitment_hash": f"commitment-{fraction}",
                "action_id": ACTION_UNIFORM,
            }
            feature_records[fraction] = feature
            committed_actions[fraction] = action
        with tempfile.TemporaryDirectory() as temporary:
            receipt_names = []
            for fraction in FRACTIONS:
                event_offset = len(events)
                receipt_name, receipt_sha256 = commit_feature_actions(
                    feature_writer=_WriterSpy(events, "feature"),
                    action_writer=_WriterSpy(events, "action"),
                    receipt_root=Path(temporary),
                    case_id=self.request.case_id,
                    request_id=request_id,
                    feature_records={fraction: feature_records[fraction]},
                    committed_actions={fraction: committed_actions[fraction]},
                )
                receipt_names.append(receipt_name)
                self.assertTrue((Path(temporary) / receipt_name).is_file())
                self.assertRegex(receipt_sha256, r"^[0-9a-f]{64}$")
                self.assertEqual(
                    events[event_offset:],
                    [
                        ("feature", "write", "mv1_feature", fraction),
                        (
                            "action",
                            "write",
                            "mv1_action_commitment",
                            fraction,
                        ),
                        ("feature", "sync"),
                        ("action", "sync"),
                    ],
                )
            self.assertEqual(len(set(receipt_names)), len(FRACTIONS))

    def test_fraction_distances_match_native_energy(self):
        energy = proposal_c_energy(
            self.proposal,
            self.covariances,
            self.layers,
        )
        self.assertGreater(energy, 0)
        distances = [torch.sqrt(torch.tensor(value * energy)).item() for value in FRACTIONS]
        self.assertEqual(distances, sorted(distances))
        self.assertAlmostEqual(
            (distances[-1] / distances[0]) ** 2,
            FRACTIONS[-1] / FRACTIONS[0],
            places=5,
        )

    def test_analytic_selection_uses_fallback_for_near_tie(self):
        scores = {"layer_0": 1.0, "layer_1": 0.0, ACTION_UNIFORM: 0.99}
        self.assertEqual(select_analytic_action(scores), "layer_0")
        self.assertEqual(
            select_analytic_action(scores, near_tie_tolerance=0.02),
            ACTION_UNIFORM,
        )
        with self.assertRaises(ContractError):
            select_analytic_action({"layer_0": float("nan"), ACTION_UNIFORM: 0.0})

    def test_rewrite_metrics_persists_scalars_not_token_ids(self):
        teacher = TeacherForcedResult(
            logits=torch.tensor(
                [
                    [[0.0, 2.0, 1.0]],
                    [[0.0, 3.0, 1.0]],
                ]
            ),
            nll=0.25,
            context_nll=(0.3, 0.2),
            target_token_count=1,
        )

        metrics = rewrite_metrics(teacher, torch.tensor([1]))
        payload = metrics.compact()

        self.assertTrue(metrics.exact_satisfied)
        self.assertEqual(metrics.target_token_count, 1)
        self.assertEqual(len(metrics.context_utility), 2)
        self.assertNotIn("target_ids", payload)
        self.assertNotIn("predicted_ids", payload)

    def test_feature_firewall_and_slurm_identity(self):
        with self.assertRaises(MV0Error):
            _feature_hash({"prompt": "forbidden"})
        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "20001",
                "SLURM_JOB_NAME": "odeedit_mv1_c0p_pair_v1",
                "SLURMD_NODENAME": "devbox",
            },
            clear=False,
        ):
            state = _mv1_slurm_state(
                "llama3-8b-inst",
                "mv1_llama_c0p_v1",
                3,
            )
            self.assertEqual(state["job_id"], "20001")
            with self.assertRaises(MV0Error):
                _mv1_slurm_state(
                    "qwen2.5-7b-inst",
                    "mv1_qwen_c0_v1",
                    20,
                )


if __name__ == "__main__":
    unittest.main()
