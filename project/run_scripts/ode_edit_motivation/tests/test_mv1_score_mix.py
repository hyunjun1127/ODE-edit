import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    ContextManifest,
    ContractError,
    EditRequest,
    ExpectedFileIdentity,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
)
from project.run_scripts.ode_edit_motivation.hooks import capture_snapshot
from project.run_scripts.ode_edit_motivation.manifests import (
    CounterFactSelectionManifest,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import (
    SanitizedJsonlWriter,
)
from project.run_scripts.ode_edit_motivation.mv1_calibration import (
    ACTION_UNIFORM,
    MV1Error,
    _feature_hash,
    build_unit_c_actions,
    proposal_c_energy,
)
from project.run_scripts.ode_edit_motivation.mv1_score_mix import (
    D0_ENVELOPE,
    D1_ENVELOPE,
    EXPECTED_OUTCOME_ACTIONS,
    SCORE_MIX_ACTION,
    SCORE_MIX_Q,
    _execution_envelope,
    _mv1mix_slurm_state,
    build_parser,
    build_score_mix_direction,
    commit_score_mix_action,
    derive_score_mix_decision,
    select_score_mix_cases,
)


class _FiveLayerToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for layer in range(4, 9):
            setattr(self, f"layer{layer}", torch.nn.Linear(2, 2, bias=False))


class ScoreMixTests(unittest.TestCase):
    def setUp(self):
        self.layers = tuple(range(4, 9))
        self.model = _FiveLayerToy().eval()
        self.request = EditRequest.from_mapping(
            {
                "case_id": "score-mix-case",
                "prompt": "{} lives in",
                "subject": "Ada",
                "target_new": "London",
            }
        )
        contexts = ContextManifest.freeze([["{}"]], source="score-mix-test")
        weight_names = tuple(f"layer{layer}.weight" for layer in self.layers)
        snapshot = capture_snapshot(
            self.model,
            model_id="toy",
            requests=(self.request,),
            context_id=contexts.manifest_id,
            hparams={"layers": list(self.layers)},
            weight_names=weight_names,
        )
        factors = tuple(
            LowRankFactor(
                weight_name=name,
                left=torch.tensor([[1.0], [0.0]]),
                right=torch.tensor([[1.0], [0.0]]),
                expected_weight_sha256=snapshot.parameter(name).sha256,
            )
            for name in weight_names
        )
        self.proposal = MemitFactorProposal(
            snapshot=snapshot,
            factors=factors,
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="score-mix-test",
            residual_denominator=5,
        )
        self.covariances = {layer: torch.eye(2) for layer in self.layers}
        self.layer_by_weight = {
            name: layer for name, layer in zip(weight_names, self.layers)
        }

    def test_positive_relu_weights_known_answer_and_actual_unit_c(self):
        scores = {
            "layer_4": 3.0,
            "layer_5": 4.0,
            "layer_6": -2.0,
            "layer_7": 0.0,
            "layer_8": -1.0,
        }

        decision = derive_score_mix_decision(scores, layers=self.layers)
        self.assertEqual(decision.controller_branch, "positive-relu-l2")
        self.assertEqual(decision.weights, (0.6, 0.8, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(decision.predicted_score, 5.0)

        unit_actions = build_unit_c_actions(
            self.proposal,
            self.covariances,
            self.layer_by_weight,
        )
        mixed = build_score_mix_direction(
            synchronous=self.proposal,
            unit_actions=unit_actions,
            decision=decision,
            covariance_by_layer=self.covariances,
            layer_by_weight=self.layer_by_weight,
        )
        self.assertEqual(mixed.action_id, SCORE_MIX_ACTION)
        self.assertEqual(
            tuple(factor.weight_name for factor in mixed.proposal.factors),
            ("layer4.weight", "layer5.weight"),
        )
        self.assertAlmostEqual(
            proposal_c_energy(
                mixed.proposal,
                self.covariances,
                self.layer_by_weight,
            ),
            1.0,
            places=6,
        )

    def test_all_nonpositive_fallback_and_exact_tie_choose_lower_layer(self):
        decision = derive_score_mix_decision(
            {
                "layer_4": -2.0,
                "layer_5": -0.5,
                "layer_6": -0.5,
                "layer_7": -3.0,
                "layer_8": -1.0,
            },
            layers=self.layers,
        )

        self.assertEqual(
            decision.controller_branch,
            "all-nonpositive-max-onehot",
        )
        self.assertEqual(decision.weights, (0.0, 1.0, 0.0, 0.0, 0.0))
        self.assertEqual(decision.predicted_score, -0.5)

    def _valid_feature_action(self):
        scores = {
            **{f"layer_{layer}": float(layer) for layer in self.layers},
            ACTION_UNIFORM: 0.25,
        }
        feature = {
            "case_id": self.request.case_id,
            "request_id": self.request.request_id,
            "q": SCORE_MIX_Q,
            "q_label": "q_1_256",
            "native_c_energy": 64.0,
            "operational_c_distance": 0.5,
            "probe_c_distance": 0.125,
            "action_scores": scores,
            "context_action_scores": {
                key: [value] for key, value in scores.items()
            },
            "feature_policy": "synthetic-central-fd",
        }
        feature["feature_hash"] = _feature_hash(feature)
        action = {
            "case_id": self.request.case_id,
            "request_id": self.request.request_id,
            "q": SCORE_MIX_Q,
            "q_label": "q_1_256",
            "feature_hash": feature["feature_hash"],
            "action_id": SCORE_MIX_ACTION,
            "layer_weights": {
                "layer_4": 1.0,
                "layer_5": 0.0,
                "layer_6": 0.0,
                "layer_7": 0.0,
                "layer_8": 0.0,
            },
            "predicted_score": 4.0,
            "actual_unit_c_energy": 1.0,
            "controller": "five-single-slopes/relu-l2-else-max-onehot",
            "controller_branch": "positive-relu-l2",
            "tie_policy": "exact-tie-lower-layer",
        }
        action["commitment_hash"] = _feature_hash(action)
        return feature, action

    def test_commitment_is_outcome_free_durable_and_firewalled(self):
        feature, action = self._valid_feature_action()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = root / "receipts"
            receipts.mkdir()
            with (
                SanitizedJsonlWriter(
                    root / "features.jsonl",
                    "test-run",
                ) as feature_writer,
                SanitizedJsonlWriter(
                    root / "actions.jsonl",
                    "test-run",
                ) as action_writer,
            ):
                receipt_name, receipt_hash = commit_score_mix_action(
                    feature_writer=feature_writer,
                    action_writer=action_writer,
                    receipt_root=receipts,
                    feature=feature,
                    action=action,
                )

            self.assertRegex(receipt_hash, r"^[0-9a-f]{64}$")
            receipt = json.loads((receipts / receipt_name).read_text())
            self.assertFalse(receipt["outcomes_observed_before_commitment"])
            persisted = (
                (root / "features.jsonl").read_text()
                + (root / "actions.jsonl").read_text()
                + (receipts / receipt_name).read_text()
            )
            for forbidden in (
                '"prompt"',
                '"subject"',
                '"target_new"',
                '"progress"',
                '"nll_reduction"',
            ):
                self.assertNotIn(forbidden, persisted)

        poisoned_action = dict(action)
        poisoned_action["progress"] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.joinpath("receipts").mkdir()
            with (
                SanitizedJsonlWriter(
                    root / "features.jsonl",
                    "test-run",
                ) as feature_writer,
                SanitizedJsonlWriter(
                    root / "actions.jsonl",
                    "test-run",
                ) as action_writer,
            ):
                with self.assertRaises(ContractError):
                    commit_score_mix_action(
                        feature_writer=feature_writer,
                        action_writer=action_writer,
                        receipt_root=root / "receipts",
                        feature=feature,
                        action=poisoned_action,
                    )

    @staticmethod
    def _selection():
        return CounterFactSelectionManifest(
            seed="test",
            source_sha256=ExpectedFileIdentity(
                sha256="a" * 64,
                size=1,
            ).sha256,
            source_size=1,
            source_row_count=100,
            calibration=tuple(f"c{index}" for index in range(20)),
            confirmatory=tuple(f"x{index}" for index in range(60)),
            untouched=tuple(f"u{index}" for index in range(20)),
        )

    def test_only_exact_postpilot_d0_and_d1_slices_are_allowed(self):
        selection = self._selection()

        self.assertEqual(
            select_score_mix_cases(selection, start=3, count=5),
            tuple(f"c{index}" for index in range(3, 8)),
        )
        self.assertEqual(
            select_score_mix_cases(selection, start=8, count=12),
            tuple(f"c{index}" for index in range(8, 20)),
        )
        with self.assertRaises(ContractError):
            select_score_mix_cases(selection, start=0, count=5)
        with self.assertRaises(ContractError):
            select_score_mix_cases(selection, start=3, count=12)

    def test_exact_run_slurm_and_cli_envelopes(self):
        self.assertEqual(
            _execution_envelope(
                "llama3-8b-inst",
                D0_ENVELOPE.llama_run_id,
                3,
                5,
            ),
            D0_ENVELOPE,
        )
        self.assertEqual(
            _execution_envelope(
                "qwen2.5-7b-inst",
                D1_ENVELOPE.qwen_run_id,
                8,
                12,
            ),
            D1_ENVELOPE,
        )
        with self.assertRaises(MV1Error):
            _execution_envelope(
                "llama3-8b-inst",
                D1_ENVELOPE.llama_run_id,
                3,
                5,
            )

        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "30001",
                "SLURM_JOB_NAME": D0_ENVELOPE.job_name,
                "SLURMD_NODENAME": "devbox",
            },
            clear=True,
        ):
            state = _mv1mix_slurm_state(
                "qwen2.5-7b-inst",
                D0_ENVELOPE.qwen_run_id,
                3,
                5,
            )
            self.assertTrue(state["under_slurm"])
            with self.assertRaises(MV1Error):
                _mv1mix_slurm_state(
                    "qwen2.5-7b-inst",
                    D1_ENVELOPE.qwen_run_id,
                    8,
                    12,
                )

        parsed = build_parser().parse_args(
            [
                "--easyedit-root",
                "/tmp/easyedit",
                "--model",
                "llama3-8b-inst",
                "--run-id",
                D0_ENVELOPE.llama_run_id,
                "--start-index",
                "3",
                "--cases",
                "5",
            ]
        )
        self.assertEqual((parsed.start, parsed.cases), (3, 5))
        self.assertEqual(
            EXPECTED_OUTCOME_ACTIONS,
            (
                "score_mix",
                "uniform",
                "ordered_global_alpha",
                "native_memit_full",
                "no_op_replay",
            ),
        )


if __name__ == "__main__":
    unittest.main()
