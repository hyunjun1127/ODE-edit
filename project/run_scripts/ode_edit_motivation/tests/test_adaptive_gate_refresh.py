from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts.ode_edit_motivation.adaptive_gate_refresh import (
    AGATE_BRANCH_ORDER,
    AGATE_CASE_COUNT,
    AGATE_CONTROLLED_NFE,
    AGATE_JOB_NAMES,
    AGATE_PROBE_PANEL_COUNT,
    AGATE_RANK_START,
    AGATE_RANK_STOP,
    AGATE_RUN_IDS,
    RECEIPT_SCHEMA,
    STREAM_SCHEMA,
    TRACKS,
    _execution_envelope,
    _slurm_state,
    select_adaptive_gate_cases,
)
from project.run_scripts.ode_edit_motivation.contracts import (
    ContractError,
    ExpectedFileIdentity,
    sha256_bytes,
)
from project.run_scripts.ode_edit_motivation.manifests import (
    DEFAULT_SELECTION_SEED,
    build_counterfact_selection,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import SanitizedJsonlWriter
from project.run_scripts.ode_edit_motivation.mv1_calibration import _feature_hash
from project.run_scripts.ode_edit_motivation.mv1_score_mix import _assert_outcome_free
from project.run_scripts.ode_edit_motivation.quarter_step_refresh import _commit_action


class AdaptiveGateEnvelopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case_ids = tuple(f"case-{index}" for index in range(250))
        self.canonical = build_counterfact_selection(
            self.case_ids,
            source_identity=ExpectedFileIdentity(sha256="a" * 64, size=123),
            seed=DEFAULT_SELECTION_SEED,
        )

    def test_fresh_rank_slice_is_deterministic_and_disjoint(self) -> None:
        selected = select_adaptive_gate_cases(
            self.case_ids,
            canonical=self.canonical,
        )
        repeated = select_adaptive_gate_cases(
            tuple(reversed(self.case_ids)),
            canonical=self.canonical,
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
        self.assertEqual(
            selected.case_ids,
            ranked[AGATE_RANK_START:AGATE_RANK_STOP],
        )
        self.assertEqual(selected.manifest_id, repeated.manifest_id)
        self.assertEqual(len(selected.case_ids), AGATE_CASE_COUNT)
        self.assertFalse(
            set(selected.case_ids).intersection(ranked[:AGATE_RANK_START])
        )

    def test_track_model_run_and_slurm_identity_are_fixed(self) -> None:
        for track in TRACKS:
            for model_alias, run_id in AGATE_RUN_IDS[track].items():
                _execution_envelope(track, model_alias, run_id)
                with mock.patch.dict(
                    os.environ,
                    {
                        "SLURM_JOB_ID": "99100",
                        "SLURM_JOB_NAME": AGATE_JOB_NAMES[track],
                        "SLURMD_NODENAME": "devbox",
                    },
                    clear=True,
                ):
                    state = _slurm_state(track, model_alias, run_id)
                self.assertTrue(state["under_slurm"])
                self.assertEqual(state["job_name"], AGATE_JOB_NAMES[track])

    def test_compute_and_endpoint_arm_counts_are_locked(self) -> None:
        self.assertEqual(AGATE_CASE_COUNT, 8)
        self.assertEqual(AGATE_PROBE_PANEL_COUNT, 13)
        self.assertEqual(AGATE_CONTROLLED_NFE, 164)
        self.assertEqual(len(AGATE_BRANCH_ORDER), 7)

    def test_adaptive_commit_uses_own_branch_and_outcome_free_schema(self) -> None:
        with self.assertRaisesRegex(ContractError, "outcome field"):
            _assert_outcome_free({"outcomes_unseen_at_commit": True})
        feature = {
            "case_id": "one",
            "request_id": "1" * 64,
            "target_identity": {"origin_lineage_id": "2" * 64},
            "paths": {"gate": "committed"},
        }
        feature["feature_hash"] = _feature_hash(feature)
        action = {
            "case_id": "one",
            "request_id": "1" * 64,
            "feature_hash": feature["feature_hash"],
            "branch_order": list(AGATE_BRANCH_ORDER),
            "path_action_hashes": {"gated": ["3" * 64]},
            "per_hop_c_energy": 0.25,
            "evaluation_unseen_at_commit": True,
        }
        action["commitment_hash"] = _feature_hash(action)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = root / "receipts"
            receipts.mkdir()
            with (
                SanitizedJsonlWriter(
                    root / "features.jsonl", "agate-test", schema_version=STREAM_SCHEMA
                ) as feature_writer,
                SanitizedJsonlWriter(
                    root / "actions.jsonl", "agate-test", schema_version=STREAM_SCHEMA
                ) as action_writer,
            ):
                name, digest = _commit_action(
                    feature_writer=feature_writer,
                    action_writer=action_writer,
                    receipt_root=receipts,
                    feature=feature,
                    action=action,
                    expected_branch_order=AGATE_BRANCH_ORDER,
                    feature_event="adaptive_gate_feature",
                    action_event="adaptive_gate_action_commitment",
                    receipt_schema=RECEIPT_SCHEMA,
                )
            receipt = json.loads((receipts / name).read_text())
            self.assertEqual(receipt["schema_version"], RECEIPT_SCHEMA)
            self.assertEqual(tuple(receipt["branch_order"]), AGATE_BRANCH_ORDER)
            self.assertEqual(digest, sha256_bytes((receipts / name).read_bytes()))


if __name__ == "__main__":
    unittest.main()
