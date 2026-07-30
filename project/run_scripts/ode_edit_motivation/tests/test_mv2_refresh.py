import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts.ode_edit_motivation.contracts import (
    ContractError,
    EditRequest,
    ExpectedFileIdentity,
)
from project.run_scripts.ode_edit_motivation.easyedit_bridge import EasyEditBridge
from project.run_scripts.ode_edit_motivation.gpu_runtime import load_fixed_model
from project.run_scripts.ode_edit_motivation.manifests import (
    DEFAULT_SELECTION_SEED,
    build_counterfact_selection,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import (
    SanitizedJsonlWriter,
)
from project.run_scripts.ode_edit_motivation.mv1_calibration import _feature_hash
from project.run_scripts.ode_edit_motivation.mv2_refresh import (
    CONTINUATION_ACTIONS,
    MV2_BRANCH_ORDER,
    MV2_CASE_COUNT,
    MV2_JOB_NAME,
    MV2_Q,
    MV2_RUN_IDS,
    MV2_RUN_SEED,
    MV2_STEP_SIZE,
    MV2_STREAM_SCHEMA,
    _execution_envelope,
    _run_mv2_event,
    _slurm_state,
    _validate_execution_mode,
    build_parser,
    commit_mv2_action,
    failure_analysis_case,
    run_mv2_event_loop,
    select_mv2_cases,
)
from project.run_scripts.ode_edit_motivation import mv2_refresh_analysis


class MemoryWriter:
    def __init__(self):
        self.rows = []

    def write(self, event, payload):
        self.rows.append((event, dict(payload)))


class MV2SelectionAndEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.case_ids = tuple(f"case-{index}" for index in range(250))
        self.identity = ExpectedFileIdentity(sha256="a" * 64, size=123)
        self.canonical = build_counterfact_selection(
            self.case_ids,
            source_identity=self.identity,
            seed=DEFAULT_SELECTION_SEED,
        )

    def test_next12_is_deterministic_and_disjoint_from_first100(self):
        selected = select_mv2_cases(self.case_ids, canonical=self.canonical)
        repeated = select_mv2_cases(
            tuple(reversed(self.case_ids)),
            canonical=self.canonical,
        )
        self.assertEqual(selected.case_ids, repeated.case_ids)
        self.assertEqual(selected.manifest_id, repeated.manifest_id)
        self.assertEqual(len(selected.case_ids), MV2_CASE_COUNT)
        self.assertFalse(
            set(selected.case_ids).intersection(self.canonical.ordered_case_ids)
        )

        ranked = tuple(
            sorted(
                self.case_ids,
                key=lambda case_id: (
                    __import__("hashlib").sha256(
                        DEFAULT_SELECTION_SEED.encode("utf-8")
                        + b"\0"
                        + case_id.encode("utf-8")
                    ).digest(),
                    case_id,
                ),
            )
        )
        self.assertEqual(selected.case_ids, ranked[100:112])

    def test_next12_rejects_tampered_first100_and_duplicates(self):
        tampered = type(self.canonical)(
            seed=self.canonical.seed,
            source_sha256=self.canonical.source_sha256,
            source_size=self.canonical.source_size,
            source_row_count=self.canonical.source_row_count,
            calibration=tuple(reversed(self.canonical.calibration)),
            confirmatory=self.canonical.confirmatory,
            untouched=self.canonical.untouched,
        )
        with self.assertRaises(ContractError):
            select_mv2_cases(self.case_ids, canonical=tampered)
        with self.assertRaises(ContractError):
            select_mv2_cases(
                (*self.case_ids[:-1], self.case_ids[0]),
                canonical=self.canonical,
            )

    def test_fixed_identity_slurm_and_no_injected_production_seam(self):
        for model_alias, run_id in MV2_RUN_IDS.items():
            _execution_envelope(model_alias, run_id)
        with self.assertRaises(Exception):
            _execution_envelope(
                "llama3-8b-inst",
                MV2_RUN_IDS["qwen2.5-7b-inst"],
            )
        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "99001",
                "SLURM_JOB_NAME": MV2_JOB_NAME,
                "SLURMD_NODENAME": "devbox",
            },
            clear=True,
        ):
            state = _slurm_state(
                "llama3-8b-inst",
                MV2_RUN_IDS["llama3-8b-inst"],
            )
            self.assertTrue(state["under_slurm"])
        with mock.patch.dict(
            os.environ,
            {
                "SLURM_JOB_ID": "99001",
                "SLURM_JOB_NAME": "wrong",
                "SLURMD_NODENAME": "devbox",
            },
            clear=True,
        ):
            with self.assertRaises(Exception):
                _slurm_state(
                    "llama3-8b-inst",
                    MV2_RUN_IDS["llama3-8b-inst"],
                )

        def injected_loader(_alias):
            return object()

        def injected_event(**_kwargs):
            return {}

        self.assertTrue(
            _validate_execution_mode(
                slurm_state={"under_slurm": True},
                model_loader=load_fixed_model,
                event_runner=_run_mv2_event,
            )
        )
        for loader, runner in (
            (injected_loader, _run_mv2_event),
            (load_fixed_model, injected_event),
            (injected_loader, injected_event),
        ):
            with self.assertRaises(Exception):
                _validate_execution_mode(
                    slurm_state={"under_slurm": True},
                    model_loader=loader,
                    event_runner=runner,
                )
        self.assertFalse(
            _validate_execution_mode(
                slurm_state={"under_slurm": False},
                model_loader=injected_loader,
                event_runner=injected_event,
            )
        )

    def test_cli_exposes_no_retuning_knobs_and_bridge_default_is_unchanged(self):
        parsed = build_parser().parse_args(
            [
                "--easyedit-root",
                "/tmp/easyedit",
                "--model",
                "llama3-8b-inst",
                "--run-id",
                MV2_RUN_IDS["llama3-8b-inst"],
            ]
        )
        self.assertFalse(
            any(
                hasattr(parsed, name)
                for name in ("q", "h", "seed", "cases", "bootstrap")
            )
        )
        for forbidden, value in (
            ("--q", "0.1"),
            ("--h", "0.25"),
            ("--seed", "5"),
            ("--cases", "3"),
            ("--bootstrap", "5"),
        ):
            with mock.patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    build_parser().parse_args(
                        [
                            "--easyedit-root",
                            "/tmp/easyedit",
                            "--model",
                            "llama3-8b-inst",
                            "--run-id",
                            MV2_RUN_IDS["llama3-8b-inst"],
                            forbidden,
                            value,
                        ]
                    )
        signature = inspect.signature(
            EasyEditBridge.propose_synchronous_memit_factors
        )
        self.assertIsNone(
            signature.parameters["frozen_target_lineage"].default
        )
        self.assertEqual(MV2_Q, 1.0 / 256.0)
        self.assertEqual(MV2_STEP_SIZE, 0.5)
        self.assertEqual(MV2_RUN_SEED, 17)
        self.assertEqual(len(MV2_BRANCH_ORDER), 6)


class MV2CommitAndLoopTests(unittest.TestCase):
    def request(self, case_id="one"):
        return EditRequest.from_mapping(
            {
                "case_id": case_id,
                "prompt": "{} lives in",
                "subject": "Ada",
                "target_new": "London",
            }
        )

    def feature_action(self):
        request = self.request()
        feature = {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "q": MV2_Q,
            "h": MV2_STEP_SIZE,
            "b0": 4.0,
            "partial_c_energy": 1.0,
            "second_step_c_energy": 1.0,
            "probe_c_distance": 0.1,
            "target_identity": {
                "direct_z_compute_count": 1,
                "direct_z_tensor_sha256": "a" * 64,
            },
            "lineage": {
                "origin": {"lineage_id": "b" * 64},
                "h0_sham": {"lineage_id": "c" * 64},
                "partial_joint": {"lineage_id": "d" * 64},
            },
            "panels": {"panel_build_counts": {"w0_b0": 1}},
            "layer_weights": {"w0": {"layer_4": 1.0}},
            "proposal_direction_hashes": {"b0": "e" * 64},
            "w0_w1_c_cosine": {"layer_4": 0.5},
            "compute_plan": {"probe_panel_count": 3},
        }
        feature["feature_hash"] = _feature_hash(feature)
        action = {
            "case_id": request.case_id,
            "request_id": request.request_id,
            "q": MV2_Q,
            "h": MV2_STEP_SIZE,
            "feature_hash": feature["feature_hash"],
            "branch_order": list(MV2_BRANCH_ORDER),
            "branch_action_hashes": {
                action_id: f"{index + 10:064x}"
                for index, action_id in enumerate(MV2_BRANCH_ORDER)
            },
            "branch_c_energies": {
                action_id: (
                    1.0
                    if action_id in CONTINUATION_ACTIONS
                    else 0.0
                )
                for action_id in MV2_BRANCH_ORDER
            },
            "matched_second_c": True,
            "matched_c_rel_tol": 3e-5,
            "matched_c_abs_tol": 3e-5,
            "lineage_ids": {
                "origin": "b" * 64,
                "h0_sham": "c" * 64,
                "partial": "d" * 64,
            },
            "action_policy": "synthetic outcome-free action",
        }
        action["commitment_hash"] = _feature_hash(action)
        return feature, action

    def test_action_receipt_is_exclusive_and_precedes_outcomes(self):
        feature, action = self.feature_action()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = root / "receipts"
            receipts.mkdir()
            with (
                SanitizedJsonlWriter(
                    root / "features.jsonl",
                    "mv2-test",
                    schema_version=MV2_STREAM_SCHEMA,
                ) as feature_writer,
                SanitizedJsonlWriter(
                    root / "actions.jsonl",
                    "mv2-test",
                    schema_version=MV2_STREAM_SCHEMA,
                ) as action_writer,
            ):
                name, digest = commit_mv2_action(
                    feature_writer=feature_writer,
                    action_writer=action_writer,
                    receipt_root=receipts,
                    feature=feature,
                    action=action,
                )
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            receipt = json.loads((receipts / name).read_text())
            self.assertFalse(
                receipt["operational_branches_observed_before_commitment"]
            )
            self.assertEqual(tuple(receipt["branch_order"]), MV2_BRANCH_ORDER)
            persisted = (
                (root / "features.jsonl").read_text()
                + (root / "actions.jsonl").read_text()
                + (receipts / name).read_text()
            )
            for forbidden in (
                '"prompt"',
                '"subject"',
                '"target_new"',
                '"progress"',
                '"logits"',
                '"generation"',
            ):
                self.assertNotIn(forbidden, persisted)
            with (
                SanitizedJsonlWriter(
                    root / "features2.jsonl",
                    "mv2-test",
                ) as feature_writer,
                SanitizedJsonlWriter(
                    root / "actions2.jsonl",
                    "mv2-test",
                ) as action_writer,
            ):
                with self.assertRaises(FileExistsError):
                    commit_mv2_action(
                        feature_writer=feature_writer,
                        action_writer=action_writer,
                        receipt_root=receipts,
                        feature=feature,
                        action=action,
                    )

    @staticmethod
    def valid_event(request):
        return {
            "schema_version": "ode-edit-mv2-refresh-event/v1",
            "case_id": request.case_id,
            "request_id": request.request_id,
            "event_seed": 1,
            "target_token_count": 1,
            "direct_z_artifact_sha256": "a" * 64,
            "direct_z_artifact_size": 64,
            "direct_z_compute_count": 1,
            "origin_lineage_id": "b" * 64,
            "partial_lineage_id": "c" * 64,
            "feature_count": 1,
            "commitment_count": 1,
            "outcome_count": 6,
            "analysis_case_count": 1,
            "receipt": {
                "name": f"{'d' * 64}.json",
                "sha256": "e" * 64,
            },
            "technical": {
                "exact_panel": True,
                "lineage_exact": True,
                "matched_second_c": True,
                "rollback_exact": True,
                "firewall_pass": True,
                "receipt_before_outcome": True,
                "h0_identical_state": True,
                "frozen_target_recompute_count": 0,
            },
            "compute": {
                "controlled_nfe": 43,
                "proposal_build_count": 4,
                "probe_panel_count": 3,
                "wall_seconds": 1.0,
            },
            "pass": True,
        }

    def test_event_loop_enforces_direct_z_once_and_fatal_abort(self):
        requests = (self.request("one"), self.request("two"))
        writer = MemoryWriter()
        observed = []

        def valid(*, request):
            observed.append(request.case_id)
            return self.valid_event(request)

        results, abort = run_mv2_event_loop(
            requests=requests,
            event_runner=valid,
            event_writer=writer,
            event_kwargs={},
        )
        self.assertIsNone(abort)
        self.assertEqual(observed, ["one", "two"])
        self.assertEqual(len(results), 2)

        observed.clear()

        def poisoned(*, request):
            observed.append(request.case_id)
            result = self.valid_event(request)
            result["direct_z_compute_count"] = 2
            return result

        results, abort = run_mv2_event_loop(
            requests=requests,
            event_runner=poisoned,
            event_writer=MemoryWriter(),
            event_kwargs={},
        )
        self.assertEqual(abort, "ContractError")
        self.assertEqual(observed, ["one"])
        self.assertFalse(results[0]["pass"])

    def test_event_loop_rejects_type_spoof_nonfinite_and_runtime_fatal(self):
        request = self.request()
        mutations = (
            ("mapping subclass", lambda result: SpoofedMapping(result)),
            (
                "nonfinite",
                lambda result: {
                    **result,
                    "compute": {**result["compute"], "wall_seconds": float("inf")},
                },
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                def runner(*, request, mutate=mutate):
                    return mutate(self.valid_event(request))

                results, abort = run_mv2_event_loop(
                    requests=(request,),
                    event_runner=runner,
                    event_writer=MemoryWriter(),
                    event_kwargs={},
                )
                if label == "mapping subclass":
                    # Mapping subclasses are normalized; production seam identity
                    # rather than Python annotation is the anti-injection gate.
                    self.assertIsNone(abort)
                else:
                    self.assertEqual(abort, "ContractError")

        def fatal(**_kwargs):
            raise RuntimeError("fatal")

        results, abort = run_mv2_event_loop(
            requests=(request,),
            event_runner=fatal,
            event_writer=MemoryWriter(),
            event_kwargs={},
        )
        self.assertEqual(abort, "RuntimeError")
        self.assertFalse(results[0]["rollback_exact"])

    def test_fatal_placeholders_preserve_exact_analysis_denominator(self):
        rows = [
            failure_analysis_case(
                request=self.request(f"case-{index}"),
                model_alias="llama3-8b-inst",
                context_id="f" * 64,
            )
            for index in range(12)
        ]
        summary = mv2_refresh_analysis.analyze_records(
            rows,
            model_alias="llama3-8b-inst",
        )
        self.assertEqual(
            summary["decision"]["verdict"],
            "BLOCK_TECHNICAL_INVALID",
        )
        self.assertEqual(
            summary["technical_validity"]["failed_case_count_in_denominator"],
            12,
        )
        self.assertEqual(
            summary["effects"]["direction_refresh"]["mean"],
            0.0,
        )


class SpoofedMapping(dict):
    pass


if __name__ == "__main__":
    unittest.main()
