from __future__ import annotations

import hashlib
import inspect
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts.ode_bf.contracts import FIXED_K, ODEBFContractError
from project.run_scripts.ode_bf.first_hit import FeasibilityVerdict
from project.run_scripts.ode_bf.p1_diagnostics import (
    P1DiagnosticRecorder,
    diagnostic_receipt_links,
    first_false_terminal_component,
)
from project.run_scripts.ode_bf.p1_runtime import (
    _arm_local_infeasibility,
    _replay_entry,
    _run_nonnative_rollout,
    _run_terminal_component_diagnostic,
    write_p1_failure_once,
)
from project.run_scripts.session04_ode_bf_p1_diag_dry_plan import (
    JOB_NAMES,
    build_plan,
)
from project.run_scripts import session04_ode_bf_submit_p1_diag as submit_diag


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class DiagnosticFixture:
    @staticmethod
    def component(*, passed: bool = True) -> dict[str, object]:
        return {
            "value": 0.0002,
            "budget": 0.001,
            "slack": 0.0008,
            "raw_pass": passed,
            "value_available": True,
        }

    @staticmethod
    def trust(*, passed: bool = True) -> dict[str, object]:
        return {
            "value": 0.02,
            "radius": 0.2,
            "radius_squared": 0.04,
            "slack": 0.02,
            "raw_pass": passed,
            "value_available": True,
        }

    @staticmethod
    def risk(label: str, *, passed: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "decision_rule": (
                "mean-and-smooth-max"
                if label == "h"
                else "uniform-mean-positive-part"
            ),
            "sample_count": 10,
            "mean_positive_damage": 0.0002,
            "smooth_max_positive_damage": 0.0003,
            "raw_max_positive_damage": 0.002,
            "signed_mean_damage": -0.0001,
            "budget": 0.001,
            "raw_pass": passed,
            "sample_order_sha256": _digest(f"sample-{label}"),
        }
        if label == "p":
            payload["baseline"] = {
                "baseline_kind": "outer_entry",
                "outer_entry_snapshot_sha256": _digest("outer-entry"),
                "anchor_population_sha256": _digest("population"),
                "sample_order_sha256": _digest("sample-p"),
                "entry_kl_identity_sha256": _digest("entry-kl"),
                "cache_receipt_sha256": _digest("cache"),
                "item_count": 10,
            }
        return payload

    @classmethod
    def trial(
        cls,
        slot: int,
        trial: int,
        *,
        accepted: bool = False,
    ) -> dict[str, object]:
        beta = (1.0, 0.5, 0.25)[trial]
        return {
            "beta": beta,
            "field_sha256": _digest("field"),
            "raw_velocity_sha256": _digest("raw"),
            "proposal_sha256": _digest(f"proposal-{slot}-{trial}"),
            "entry_snapshot_sha256": _digest(f"entry-{slot}"),
            "trial_snapshot_sha256": _digest(f"trial-{slot}-{trial}"),
            "predicted_beta_progress": 0.2 * beta,
            "actual_signed_progress": 0.1 * beta,
            "trust_ratio": 0.5,
            "progress_pass": True,
            "requested_progress": 0.2,
            "maximum_feasible_progress": 0.8,
            "solver": {
                "routing_problem_sha256": _digest("problem"),
                "projection": {
                    "projection_status": "FEASIBLE",
                    "raw_velocity_sha256": _digest("raw"),
                },
                "raw_certificate": {
                    "solver": "SLSQP",
                    "passed": True,
                    "maximum_primal_violation": 0.0,
                },
            },
            "structural_h": cls.component(),
            "structural_p": cls.component(),
            "trust": cls.trust(),
            "functional_h": cls.risk("h"),
            "functional_p": cls.risk("p"),
            "authoritative_bf16_pass": True,
            "first_rejecting_component": None if accepted else "functional_p",
            "accepted": accepted,
            "official_success": {
                "numerator": 4,
                "denominator": 10,
                "official_aggregate": 0.4,
                "joint_exact_success": False,
                "success_vector_sha256": _digest("success-vector"),
            },
            "progress_telemetry": {
                "per_request_signed_progress_sha256": _digest("progress"),
                "improved_vector_sha256": _digest("improved"),
                "harm_vector_sha256": _digest("harm"),
                "improved_count": 9,
                "harm_count": 1,
            },
            "purity": {
                "state_before_sha256": _digest(f"state-{slot}"),
                "state_after_sha256": _digest(f"state-{slot}"),
                "history_before_sha256": _digest("history"),
                "history_after_sha256": _digest("history"),
                "sampler_before_sha256": _digest("sampler"),
                "sampler_after_sha256": _digest("sampler"),
            },
        }

    @classmethod
    def populate_k8(cls, recorder: P1DiagnosticRecorder) -> None:
        for slot in range(FIXED_K):
            trial_hashes = [
                recorder.write_trial(
                    slot_index=slot,
                    trial_ordinal=trial,
                    payload=cls.trial(slot, trial),
                )
                for trial in range(3)
            ]
            recorder.write_slot(
                slot_index=slot,
                payload={
                    "accepted": False,
                    "accepted_beta": 0.0,
                    "accepted_t": 0.0,
                    "selected_snapshot_sha256": _digest("entry"),
                    "rejected_field_reuse": True,
                    "trial_receipt_sha256": trial_hashes,
                    "state_before_sha256": _digest("entry"),
                    "state_after_sha256": _digest("entry"),
                    "history_before_sha256": _digest("history"),
                    "history_after_sha256": _digest("history"),
                    "sampler_before_sha256": _digest("sampler"),
                    "sampler_after_sha256": _digest("sampler"),
                },
            )

    @classmethod
    def terminal(
        cls,
        recorder: P1DiagnosticRecorder,
        *,
        false_components: tuple[str, ...] = ("terminal_p",),
    ) -> dict[str, object]:
        inputs = {
            name: name not in false_components
            for name in (
                "structural_h",
                "structural_p",
                "trust",
                "terminal_h",
                "terminal_p",
            )
        }
        return {
            "selected_stage": 8,
            "selected_snapshot_sha256": _digest("entry"),
            "structural_h": cls.component(passed=inputs["structural_h"]),
            "structural_p": cls.component(passed=inputs["structural_p"]),
            "trust": cls.trust(passed=inputs["trust"]),
            "terminal_h": cls.risk("h", passed=inputs["terminal_h"]),
            "terminal_p": cls.risk("p", passed=inputs["terminal_p"]),
            "boolean_inputs": inputs,
            "first_false_component": first_false_terminal_component(inputs),
            "selector": {
                "status": "NO_EXACT_BATCH_HIT",
                "selected_stage": 8,
                "joint_first_exact_hit_step": None,
                "all_exact_hit_steps": [],
                "fixed_budget_terminal_success_count": 4,
                "fixed_budget_terminal_feasible": True,
                "max_official_success_count": 4,
                "earliest_max_success_step": 1,
                "executed_slots": 8,
                "accepted_slots": 0,
                "rejected_slots": 8,
            },
            "slot_receipt_sha256": recorder.slot_hashes,
            "arm_local_infeasibility": _arm_local_infeasibility(
                FeasibilityVerdict(
                    inputs["structural_h"],
                    inputs["structural_p"],
                    inputs["trust"],
                    inputs["terminal_h"],
                    inputs["terminal_p"],
                    True,
                )
            ),
        }


class P1TerminalDiagnosticTests(unittest.TestCase):
    def test_all_nonnative_arms_publish_independent_terminal_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            observed: dict[str, str | None] = {}
            false_by_arm = {
                "F_G": ("terminal_p",),
                "F_BF": (),
                "R_BF": ("structural_p",),
            }
            for arm in ("F_G", "F_BF", "R_BF"):
                recorder = P1DiagnosticRecorder(
                    raw / "diagnostics" / f"arm-{arm}",
                    arm=arm,
                )
                DiagnosticFixture.populate_k8(recorder)
                recorder.write_terminal(
                    DiagnosticFixture.terminal(
                        recorder,
                        false_components=false_by_arm[arm],
                    )
                )
                observed[arm] = json.loads(
                    (
                        raw
                        / "diagnostics"
                        / f"arm-{arm}"
                        / f"arm-{arm}-terminal-components.json"
                    ).read_text()
                )["first_false_component"]
            self.assertEqual(
                observed,
                {"F_G": "terminal_p", "F_BF": None, "R_BF": "structural_p"},
            )
            self.assertEqual(len(diagnostic_receipt_links(raw)), 99)

    def test_actual_k8_failure_fixture_retains_24_trials_and_8_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            recorder = P1DiagnosticRecorder(raw / "diagnostics", arm="F_G")
            DiagnosticFixture.populate_k8(recorder)
            terminal_hash = recorder.write_terminal(
                DiagnosticFixture.terminal(recorder)
            )
            self.assertEqual(len(recorder.trial_hashes), 24)
            self.assertEqual(len(recorder.slot_hashes), 8)
            self.assertEqual(len(terminal_hash), 64)
            links = diagnostic_receipt_links(raw)
            self.assertEqual(len(links), 33)
            value = json.loads(
                (raw / "diagnostics/arm-F_G-terminal-components.json").read_text()
            )
            self.assertEqual(value["first_false_component"], "terminal_p")
            with self.assertRaisesRegex(ODEBFContractError, "terminal verifier"):
                raise ODEBFContractError("terminal verifier fixture")
            self.assertEqual(len(diagnostic_receipt_links(raw)), 33)

    def test_independent_false_components_and_multi_failure_order(self) -> None:
        names = (
            "structural_h",
            "structural_p",
            "trust",
            "terminal_h",
            "terminal_p",
        )
        for name in names:
            with self.subTest(name=name):
                inputs = {item: item != name for item in names}
                self.assertEqual(first_false_terminal_component(inputs), name)
                classified = _arm_local_infeasibility(
                    FeasibilityVerdict(
                        inputs["structural_h"],
                        inputs["structural_p"],
                        inputs["trust"],
                        inputs["terminal_h"],
                        inputs["terminal_p"],
                        True,
                    )
                )
                self.assertIsNotNone(classified)
                self.assertEqual(classified["first_false_component"], name)
        inputs = {item: item not in ("structural_p", "terminal_h") for item in names}
        self.assertEqual(first_false_terminal_component(inputs), "structural_p")
        self.assertIsNone(first_false_terminal_component({item: True for item in names}))

    def test_mid_slot_and_after_k8_crashes_leave_valid_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            recorder = P1DiagnosticRecorder(raw / "diagnostics", arm="F_G")
            hashes = [
                recorder.write_trial(
                    slot_index=0,
                    trial_ordinal=trial,
                    payload=DiagnosticFixture.trial(0, trial),
                )
                for trial in range(3)
            ]
            recorder.write_slot(
                slot_index=0,
                payload={
                    "accepted": False,
                    "accepted_beta": 0.0,
                    "accepted_t": 0.0,
                    "selected_snapshot_sha256": _digest("entry"),
                    "rejected_field_reuse": True,
                    "trial_receipt_sha256": hashes,
                    "state_before_sha256": _digest("entry"),
                    "state_after_sha256": _digest("entry"),
                    "history_before_sha256": _digest("history"),
                    "history_after_sha256": _digest("history"),
                    "sampler_before_sha256": _digest("sampler"),
                    "sampler_after_sha256": _digest("sampler"),
                },
            )
            recorder.write_trial(
                slot_index=1,
                trial_ordinal=0,
                payload=DiagnosticFixture.trial(1, 0),
            )
            self.assertEqual(len(diagnostic_receipt_links(raw)), 5)

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            recorder = P1DiagnosticRecorder(raw / "diagnostics", arm="F_G")
            DiagnosticFixture.populate_k8(recorder)
            self.assertEqual(len(diagnostic_receipt_links(raw)), 32)
            self.assertFalse(
                (raw / "diagnostics/arm-F_G-terminal-components.json").exists()
            )

    def test_terminal_atomic_failure_keeps_k8_prefix_and_no_temp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            recorder = P1DiagnosticRecorder(raw / "diagnostics", arm="F_G")
            DiagnosticFixture.populate_k8(recorder)
            with mock.patch(
                "project.run_scripts.ode_bf.p1_diagnostics.os.replace",
                side_effect=OSError("injected terminal replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    recorder.write_terminal(DiagnosticFixture.terminal(recorder))
            self.assertEqual(len(diagnostic_receipt_links(raw)), 32)
            self.assertFalse(list((raw / "diagnostics").glob(".*.tmp")))

    def test_failure_receipt_links_every_flushed_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            raw = output / "raw"
            recorder = P1DiagnosticRecorder(raw / "diagnostics", arm="F_G")
            recorder.write_trial(
                slot_index=0,
                trial_ordinal=0,
                payload=DiagnosticFixture.trial(0, 0),
            )
            digest, failure = write_p1_failure_once(
                output,
                ODEBFContractError("sanitized fixture"),
                repo_root=Path(__file__).resolve().parents[4],
            )
            self.assertEqual(len(digest), 64)
            self.assertEqual(len(failure["diagnostic_receipt_sha256"]), 1)
            self.assertNotIn("sanitized fixture", json.dumps(failure))

    def test_raw_key_nonfinite_duplicate_and_schema_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            recorder = P1DiagnosticRecorder(raw / "diagnostics", arm="F_G")
            payload = DiagnosticFixture.trial(0, 0)
            payload["solver"] = {"prompt": "forbidden"}
            with self.assertRaisesRegex(ODEBFContractError, "forbidden raw key"):
                recorder.write_trial(slot_index=0, trial_ordinal=0, payload=payload)

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            recorder = P1DiagnosticRecorder(raw / "diagnostics", arm="F_G")
            payload = DiagnosticFixture.trial(0, 0)
            payload["actual_signed_progress"] = math.nan
            with self.assertRaisesRegex(ODEBFContractError, "non-finite"):
                recorder.write_trial(slot_index=0, trial_ordinal=0, payload=payload)

        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            recorder = P1DiagnosticRecorder(raw / "diagnostics", arm="F_G")
            payload = DiagnosticFixture.trial(0, 0)
            recorder.write_trial(slot_index=0, trial_ordinal=0, payload=payload)
            with self.assertRaises(ODEBFContractError):
                recorder.write_trial(slot_index=0, trial_ordinal=0, payload=payload)

    def test_scientific_files_and_decision_order_remain_frozen(self) -> None:
        root = Path(__file__).resolve().parents[1]
        expected = {
            "routing.py": "9218a0a0e25098b85658640a6d280d40bdafcd80d9d824f96122d0c19b1809ff",
            "barriers.py": "b3d59cf7db3c3cc00e318115228777518ce6731c4797ec589a298319ce18c9f9",
            "p1_backend.py": "a5208ba7ab9ee31c0606420bf5b5b301f23947bb6e58e4d4f86f6e655f1743eb",
            "p1_controller.py": "1015f02a81921abec1208bf892339266af4f415077dbbf7c050b6b5f581d8b3b",
            "p1_evaluator.py": "a574b7566b4fc5ee1ff76fdcd19feb829d5eec8b7158b2f91542344c416b0381",
            "p1_state.py": "700aa763361748cb1b089d5d0d228832ac405b9fdbb1bdf281e6beeaf174e9a8",
            "transaction.py": "95540b9e2df393b8aa0d59973a3a553272b64495b682e11829977d956966db7d",
        }
        for name, digest in expected.items():
            observed = hashlib.sha256((root / name).read_bytes()).hexdigest()
            self.assertEqual(observed, digest, name)

        source = inspect.getsource(_run_nonnative_rollout)
        self.assertLess(source.index("if diagnostic_stop_at_terminal:"), source.index(
            'measure("selected_endpoint_rewrite_verdict")'
        ))
        self.assertIn("rho_accept=lock.rho_accept", source)
        self.assertIn("functional.pretrained.passed", source)
        self.assertIn("_arm_local_infeasibility(terminal_feasibility)", source)
        self.assertNotIn("P1 all-history/terminal-P verifier failed", source)
        replay_entry = inspect.getsource(_replay_entry)
        self.assertIn("outer_entry_p_cache.select", replay_entry)
        self.assertNotIn("capture_pretrained_entry_kl", replay_entry)
        self.assertLess(
            replay_entry.index("with _virtual_context(model, factors):"),
            replay_entry.index("outer_entry_p_cache.select"),
        )
        self.assertIn(
            '"baseline_kind": "outer_entry"',
            source,
        )

        diagnostic = inspect.getsource(_run_terminal_component_diagnostic)
        self.assertNotIn("_run_native_batch", diagnostic)
        self.assertNotIn("load_counterfact_cases_after_freeze", diagnostic)
        self.assertNotIn("transaction.commit", diagnostic)
        self.assertIn("diagnostic_stop_at_terminal=True", diagnostic)
        self.assertIn("P1Arm.F_BF", diagnostic)
        self.assertIn("P1Arm.R_BF", diagnostic)
        self.assertLess(diagnostic.index("P1Arm.F_G"), diagnostic.index("P1Arm.F_BF"))
        self.assertLess(diagnostic.index("P1Arm.F_BF"), diagnostic.index("P1Arm.R_BF"))

    def test_dry_plan_is_b10_1_all_arm_diagnostic_and_resource_locked(self) -> None:
        self.assertEqual(
            JOB_NAMES,
            {
                "llama3-8b-inst": "odebf_s04_p1r3diag_llama",
                "qwen2.5-7b-inst": "odebf_s04_p1r3diag_qwen",
            },
        )
        first = build_plan("a" * 40)
        second = build_plan("a" * 40)
        self.assertEqual(first, second)
        self.assertEqual(first["edit_batch_size"], 10)
        self.assertEqual(first["sequential_batch_count"], 1)
        self.assertEqual(
            first["arms"], ["N32_NATIVE", "F_G", "F_BF", "R_BF"]
        )
        self.assertEqual(first["functional_p_baseline_kind"], "outer_entry")
        self.assertTrue(first["arm_local_infeasibility"])
        self.assertEqual(first["k_resolution"], 8)
        self.assertEqual(first["trials_per_slot"], 3)
        self.assertEqual(first["history_append_count"], 0)
        for job in first["jobs"]:
            self.assertEqual((job["gpu"], job["cpu"]), (1, 8))
            self.assertEqual(job["memory_mib"], 65_000)
            self.assertEqual(job["time"], "04:00:00")

    def test_submit_scope_and_existing_p1r2_immutability(self) -> None:
        self.assertEqual(len(submit_diag.ALLOWED_CHANGED_PATHS), 11)
        self.assertEqual(len(submit_diag._p1r2_immutability_gate()), 16)
        frozen = submit_diag._frozen_semantics_gate()
        self.assertEqual(len(frozen), 12)
        self.assertEqual(
            frozen["project/run_scripts/ode_bf/locks/numerical_lock_p1r2.json"],
            "0cdb4ff528f0eddea7b40b9d36a103fa372dca8433da8a0a2cf36f77ad2fa903",
        )

    def test_runtime_firewall_allows_only_preserved_cuda_helper(self) -> None:
        self.assertEqual(
            len(submit_diag._assert_p1_runtime_ast_firewall()),
            64,
        )
        valid = (
            "from project.run_scripts.ode_alloc.p1_runtime import (\n"
            "    _prepare_p1_cuda_runtime as "
            "_prepare_preserved_one_device_cuda_runtime,\n"
            ")\n"
        )
        invalid_alias = valid.replace(
            "_prepare_p1_cuda_runtime",
            "unexpected_helper",
            1,
        )
        invalid_foreign = (
            "from project.run_scripts.ode_alloc.p1_evaluator import forbidden\n"
        )
        for source, message in (
            (invalid_alias, "CUDA helper import contract"),
            (invalid_foreign, "forbidden foreign/session import"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "runtime.py"
                path.write_text(source, encoding="utf-8")
                with self.assertRaisesRegex(ODEBFContractError, message):
                    submit_diag._assert_p1_runtime_ast_firewall(path)

    def test_sbatch_is_server2_one_gpu_four_hour_offline_and_diag_only(self) -> None:
        path = Path(__file__).resolve().parents[2] / "session04_ode_bf_p1_diag.sbatch"
        source = path.read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --nodelist=server2",
            "#SBATCH --mem=65000M",
            "#SBATCH --time=04:00:00",
            "#SBATCH --export=NONE",
            "export HF_HUB_OFFLINE=1",
            "export TRANSFORMERS_OFFLINE=1",
            "session04_ode_bf_p1_diag",
        ):
            self.assertIn(expected, source)
        self.assertNotIn("#SBATCH --array", source)

    def test_submit_entrypoint_preserves_systemexit_and_sanitizes_exception(self) -> None:
        stderr = io.StringIO()
        with mock.patch.object(submit_diag, "main", return_value=0):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    submit_diag._entrypoint()
        self.assertEqual(caught.exception.code, 0)
        self.assertEqual(stderr.getvalue(), "")

        stderr = io.StringIO()
        with mock.patch.object(submit_diag, "main", side_effect=SystemExit(7)):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    submit_diag._entrypoint()
        self.assertEqual(caught.exception.code, 7)
        self.assertEqual(stderr.getvalue(), "")

        stderr = io.StringIO()
        with mock.patch.object(
            submit_diag,
            "main",
            side_effect=RuntimeError("private endpoint must remain hidden"),
        ):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(RuntimeError):
                    submit_diag._entrypoint()
        value = json.loads(stderr.getvalue())
        self.assertEqual(value["status"], "PRE_SUBMIT_OR_PARTIAL_HOLD")
        self.assertNotIn("private endpoint", stderr.getvalue())
        self.assertFalse(value["retry_or_resubmit"])


if __name__ == "__main__":
    unittest.main()
