from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from project.run_scripts.ode_edit_motivation.contracts import (
    EditRequest,
    ExpectedFileIdentity,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
)
from project.run_scripts.ode_edit_motivation.gpu_runtime import load_fixed_model
from project.run_scripts.ode_edit_motivation.manifests import (
    DEFAULT_SELECTION_SEED,
    build_counterfact_selection,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import SanitizedJsonlWriter
from project.run_scripts.ode_edit_motivation.mv1_calibration import _feature_hash
from project.run_scripts.ode_edit_motivation.quarter_step_refresh import (
    QSTEP_BRANCH_ORDER,
    QSTEP_CASE_COUNT,
    QSTEP_JOB_NAME,
    QSTEP_K,
    QSTEP_RUN_IDS,
    QSTEP_STREAM_SCHEMA,
    _combine_steps,
    _commit_action,
    _execution_envelope,
    _run_quarter_step_event,
    _slurm_state,
    _validate_execution_mode,
    select_quarter_step_cases,
)


class QuarterStepEnvelopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_ids = tuple(f"case-{index}" for index in range(250))
        identity = ExpectedFileIdentity(sha256="a" * 64, size=123)
        self.canonical = build_counterfact_selection(
            self.case_ids,
            source_identity=identity,
            seed=DEFAULT_SELECTION_SEED,
        )

    def test_fresh_rank_slice_is_deterministic_and_disjoint(self) -> None:
        selected = select_quarter_step_cases(
            self.case_ids, canonical=self.canonical
        )
        repeated = select_quarter_step_cases(
            tuple(reversed(self.case_ids)), canonical=self.canonical
        )
        ranked = tuple(
            sorted(
                self.case_ids,
                key=lambda case_id: (
                    hashlib.sha256(
                        DEFAULT_SELECTION_SEED.encode("utf-8")
                        + b"\0"
                        + case_id.encode("utf-8")
                    ).digest(),
                    case_id,
                ),
            )
        )
        self.assertEqual(selected.case_ids, ranked[112:124])
        self.assertEqual(selected.manifest_id, repeated.manifest_id)
        self.assertEqual(len(selected.case_ids), QSTEP_CASE_COUNT)
        self.assertFalse(set(selected.case_ids).intersection(ranked[:112]))

    def test_fixed_pair_identity_and_test_seams(self) -> None:
        for model_alias, run_id in QSTEP_RUN_IDS.items():
            _execution_envelope(model_alias, run_id)
        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "99100",
                "SLURM_JOB_NAME": QSTEP_JOB_NAME,
                "SLURMD_NODENAME": "devbox",
            },
            clear=True,
        ):
            state = _slurm_state(
                "llama3-8b-inst", QSTEP_RUN_IDS["llama3-8b-inst"]
            )
            self.assertTrue(state["under_slurm"])

        def injected_loader(_alias):
            return object()

        def injected_event(**_kwargs):
            return {}

        self.assertTrue(
            _validate_execution_mode(
                slurm_state={"under_slurm": True},
                model_loader=load_fixed_model,
                event_runner=_run_quarter_step_event,
            )
        )
        self.assertFalse(
            _validate_execution_mode(
                slurm_state={"under_slurm": False},
                model_loader=injected_loader,
                event_runner=injected_event,
            )
        )
        self.assertEqual(QSTEP_K, 4)
        self.assertEqual(len(QSTEP_BRANCH_ORDER), 13)


class QuarterStepCommitTests(unittest.TestCase):
    def request(self) -> EditRequest:
        return EditRequest.from_mapping(
            {
                "case_id": "one",
                "prompt": "{} lives in",
                "subject": "Ada",
                "target_new": "London",
            }
        )

    def test_feature_action_receipt_is_exclusive(self) -> None:
        request = self.request()
        feature = {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "target_identity": {"origin_lineage_id": "a" * 64},
            "paths": {"policy_locked": True},
        }
        feature["feature_hash"] = _feature_hash(feature)
        action = {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "feature_hash": feature["feature_hash"],
            "branch_order": list(QSTEP_BRANCH_ORDER),
            "path_action_hashes": {"fixed": ["b" * 64]},
            "per_hop_c_energy": 1.0,
        }
        action["commitment_hash"] = _feature_hash(action)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = root / "receipts"
            receipts.mkdir()
            with (
                SanitizedJsonlWriter(
                    root / "features.jsonl",
                    "quarter-test",
                    schema_version=QSTEP_STREAM_SCHEMA,
                ) as feature_writer,
                SanitizedJsonlWriter(
                    root / "actions.jsonl",
                    "quarter-test",
                    schema_version=QSTEP_STREAM_SCHEMA,
                ) as action_writer,
            ):
                name, digest = _commit_action(
                    feature_writer=feature_writer,
                    action_writer=action_writer,
                    receipt_root=receipts,
                    feature=feature,
                    action=action,
                )
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            receipt = json.loads((receipts / name).read_text())
            self.assertTrue(receipt["all_paths_committed_before_outcomes"])
            self.assertEqual(tuple(receipt["branch_order"]), QSTEP_BRANCH_ORDER)

    def test_sparse_steps_combine_as_zero_for_missing_layers(self) -> None:
        request = self.request()
        from project.run_scripts.ode_edit_motivation.contracts import SnapshotManifest
        from project.run_scripts.ode_edit_motivation.hooks import capture_snapshot

        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False),
            torch.nn.Linear(2, 2, bias=False),
        )
        snapshot = capture_snapshot(
            model,
            model_id="toy",
            requests=(request,),
            context_id="c" * 64,
            hparams={"layers": [0]},
            weight_names=("0.weight", "1.weight"),
            provenance_ids=("d" * 64,),
        )
        self.assertIsInstance(snapshot, SnapshotManifest)
        factors = tuple(
            LowRankFactor(
                weight_name=name,
                left=torch.ones(2, 1),
                right=torch.ones(2, 1),
                expected_weight_sha256=snapshot.parameter(name).sha256,
            )
            for name in ("0.weight", "1.weight")
        )
        origin = MemitFactorProposal(
            snapshot=snapshot,
            factors=factors,
            semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
            solver_name="origin",
            residual_denominator=1,
        )
        steps = tuple(
            MemitFactorProposal(
                snapshot=snapshot,
                factors=(factor,),
                semantics=ProposalSemantics.SYNCHRONOUS_FROZEN_SNAPSHOT,
                solver_name=f"step-{index}",
                residual_denominator=1,
            )
            for index, factor in enumerate(factors)
        )
        combined = _combine_steps(origin, steps, solver_suffix="sparse-two")
        self.assertEqual(len(combined.factors), 2)
        self.assertEqual(combined.factors[0].left.shape[1], 1)


if __name__ == "__main__":
    unittest.main()
