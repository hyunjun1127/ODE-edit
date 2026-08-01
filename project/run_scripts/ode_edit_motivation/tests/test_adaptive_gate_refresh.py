from __future__ import annotations

import hashlib
import os
import unittest
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
    TRACKS,
    _execution_envelope,
    _slurm_state,
    select_adaptive_gate_cases,
)
from project.run_scripts.ode_edit_motivation.contracts import ExpectedFileIdentity
from project.run_scripts.ode_edit_motivation.manifests import (
    DEFAULT_SELECTION_SEED,
    build_counterfact_selection,
)


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


if __name__ == "__main__":
    unittest.main()
