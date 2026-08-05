from __future__ import annotations

import ast
import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from project.run_scripts import session05_ode_bf_target_new_nll as entry
from project.run_scripts import session05_ode_bf_target_new_nll_dry_plan as dry
from project.run_scripts import session05_ode_bf_submit_target_new_nll as submit
from project.run_scripts.ode_bf import p1_adaptive_runtime as runtime
from project.run_scripts.ode_bf import routing
from project.run_scripts.ode_bf.accounting import ComputeLedger
from project.run_scripts.ode_bf.artifacts import ODEBFArtifactGuard, load_rooted_json
from project.run_scripts.ode_bf.contracts import BATCH_SIZE, ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1_backend import (
    ControllerMarginReceipt,
    TargetNewNLLReceipt,
    signed_progress_gradient,
)
from project.run_scripts.ode_bf.tests.test_p1_backend_controller import _field
from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
from project.run_scripts.ode_bf.p1_runtime import (
    P1OutputRootCollision,
    _controller_progress_telemetry,
    write_p1_failure_once,
)
from project.run_scripts.ode_bf.p1_target_new_panel import (
    TARGET_NEW_INSTRUCTION_ID,
    TARGET_NEW_PANEL_LABELS,
    TARGET_NEW_RESULT_TOKEN,
    expected_target_new_result_name,
    forecast_target_new_panel,
    target_new_panel_specs,
    validate_target_new_lock,
)
from project.run_scripts.ode_bf.target_new_nll import (
    RoutingObjective,
    RoutingObjectiveResult,
)
from project.run_scripts.ode_bf.request_digest import ordered_request_digest_v1
from project.run_scripts.ode_bf.routing import (
    QuadraticBarrier,
    RoutingProblem,
    verify_backtracked_candidate,
)


ROOT = Path(__file__).resolve().parents[4]


def _target_receipt(values: tuple[float, ...]) -> TargetNewNLLReceipt:
    mean = sum(values) / len(values)
    payload = canonical_hash({"values": values})
    return TargetNewNLLReceipt(
        RoutingObjective.TARGET_NEW_NLL.value,
        mean,
        canonical_hash({"mean": mean}),
        values,
        canonical_hash({"values": values}),
        "a" * 64,
        (1,) * BATCH_SIZE,
        "b" * 64,
        (1, 5),
        6,
        "f" * 64,
        BATCH_SIZE * 6,
        100,
        0,
        0,
        payload,
    )


class TargetNewPanelContractTests(unittest.TestCase):
    def test_panel_is_common_and_fr_a16_is_preforecast_excluded(self) -> None:
        specs = target_new_panel_specs()
        self.assertEqual(tuple(item.label for item in specs), TARGET_NEW_PANEL_LABELS)
        self.assertEqual(
            tuple(item.routing_objective for item in specs),
            (
                RoutingObjective.MARGIN,
                RoutingObjective.TARGET_NEW_NLL,
            ),
        )
        plan = dry.build_plan("c" * 40, repository_root=ROOT)
        self.assertEqual(plan["panel_labels"], list(TARGET_NEW_PANEL_LABELS))
        self.assertFalse(plan["scientific_promotion_authorized"])
        self.assertFalse(plan["model_load"])
        self.assertFalse(plan["slurm_submit"])
        self.assertEqual(plan["server1_gpu_cap"], 3)
        telemetry = plan["stepwise_layer_routing_telemetry"]
        self.assertEqual(telemetry["candidate_layer_ids"], [4, 5, 6, 7, 8])
        self.assertEqual(
            telemetry["defined_for_labels"],
            ["FR-A8-MARGIN", "FR-A8-NEWNLL", "FR-A16-NEWNLL"],
        )
        self.assertTrue(telemetry["observation_only"])
        self.assertEqual(telemetry["controller_dependency_count"], 0)
        self.assertFalse(telemetry["fr_a16_executed"])
        self.assertEqual(
            plan["fr_a16_newnll"],
            {
                "executed": False,
                "forecast_seconds": 161_920,
                "allocation_seconds": 86_400,
                "exclusion_reason": (
                    "SIX_CONTEXT_A16_EXCEEDS_LOCKED_24H_ENVELOPE"
                ),
                "outcome_metric_used": False,
            },
        )
        for job in plan["jobs"]:
            self.assertFalse(job["forecast"]["fr_a16_included"])
            self.assertEqual(job["forecast"]["routing_context_count"], 6)
            self.assertEqual(
                job["forecast"]["fr_a16_forecast_seconds"], 161_920
            )
            self.assertTrue(job["forecast"]["fits_same_envelope"])
            self.assertLessEqual(job["forecast"]["conservative_gpu_peak_mib"], 65_000)
            self.assertLessEqual(job["forecast"]["conservative_host_peak_mib"], 65_000)
            self.assertLessEqual(job["forecast"]["conservative_time_seconds"], 86_400)

    def test_lock_validates_exact_panel_and_fails_closed_on_objective_change(self) -> None:
        lock_path = ROOT / "project/run_scripts/ode_bf/locks/numerical_lock_s05_target_new_nll.json"
        value, _ = load_rooted_json(
            lock_path,
            expected_schema="ode-edit-s05-ode-bf-target-new-nll-routing-numerical-lock/v1",
        )
        arguments = {
            "controller_identity_sha256": P1ControllerLock().identity(),
            "stream_root_digest": "a3e2fbf27e94c3ace4e048027abf1f08f215715388dacc3cf85bb452e8e89157",
            "population_root_digest": "49f3b2674d5fc649e8e17982769a8be7efde70e92b68791b550100fef627ee4b",
        }
        validate_target_new_lock(value, **arguments)
        changed = copy.deepcopy(value)
        changed["routing_objective_by_label"]["FR-A8-NEWNLL"] = "MARGIN"
        with self.assertRaisesRegex(ODEBFContractError, "numerical lock"):
            validate_target_new_lock(changed, **arguments)
        for path, replacement in (
            (("routing_loss", "nll_old_routing_influence"), 1),
            (("adaptive_pseudo_time", "rho_accept"), 0.2),
            (("target_state_objective",), "TARGET_NEW_NLL"),
        ):
            changed = copy.deepcopy(value)
            cursor = changed
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = replacement
            with self.assertRaisesRegex(ODEBFContractError, "numerical lock"):
                validate_target_new_lock(changed, **arguments)

    def test_target_new_progress_is_objective_typed_and_old_target_independent(self) -> None:
        entry_receipt = _target_receipt((2.0,) * BATCH_SIZE)
        trial_receipt = _target_receipt((1.5,) * BATCH_SIZE)
        observed = _controller_progress_telemetry(entry_receipt, trial_receipt)
        self.assertEqual(observed.actual_signed_progress, 0.5)
        self.assertEqual(observed.per_request_improved_bits, (1,) * BATCH_SIZE)
        margin = ControllerMarginReceipt(
            2.0,
            "c" * 64,
            (2.0,) * BATCH_SIZE,
            "d" * 64,
            "a" * 64,
            BATCH_SIZE,
            200,
            0,
        )
        with self.assertRaisesRegex(ODEBFContractError, "geometry"):
            _controller_progress_telemetry(margin, trial_receipt)

    def test_old_only_apparent_gain_is_rejected_and_new_nll_gain_is_accepted(self) -> None:
        zero = np.zeros(2, dtype=np.float64)
        barrier = QuadraticBarrier(
            "historical",
            0.0,
            zero,
            np.zeros((2, 2), dtype=np.float64),
            1.0,
            "layer-local-diagonal",
        )
        pretrained = QuadraticBarrier(
            "pretrained",
            0.0,
            zero,
            np.zeros((2, 2), dtype=np.float64),
            1.0,
            "layer-local-diagonal",
        )
        problem = RoutingProblem(
            np.ones(2, dtype=np.float64),
            np.eye(2, dtype=np.float64),
            np.eye(2, dtype=np.float64),
            2.0,
            np.ones(2, dtype=np.float64),
            0.5,
            1.0e-8,
            barrier,
            pretrained,
        )
        velocity = np.asarray((0.5, 0.5), dtype=np.float64)
        old_only = verify_backtracked_candidate(
            problem,
            velocity,
            beta=1.0,
            actual_signed_progress=0.0,
            functional_h_pass=True,
            functional_p_pass=True,
            authoritative_bf16_pass=True,
        )
        genuine = verify_backtracked_candidate(
            problem,
            velocity,
            beta=1.0,
            actual_signed_progress=0.5,
            functional_h_pass=True,
            functional_p_pass=True,
            authoritative_bf16_pass=True,
        )
        self.assertFalse(old_only.progress_pass)
        self.assertFalse(old_only.accepted)
        self.assertTrue(genuine.progress_pass)
        self.assertTrue(genuine.accepted)

    def test_signed_gradient_keeps_full_vector_and_never_reads_old_target(self) -> None:
        field, _ = _field()
        identities = tuple(f"{index + 1:064x}" for index in range(BATCH_SIZE))
        requests = tuple(
            {
                "request_sha256": identity,
                "target_true": object(),
            }
            for identity in identities
        )
        captured: dict[str, torch.Tensor] = {}

        class Overlay:
            def __init__(self, model, layers, coefficients):
                del model, layers
                captured["coefficients"] = coefficients

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback
                return False

        def objective(
            model,
            tokenizer,
            observed_requests,
            *,
            objective,
            contexts,
            gradient_input,
        ):
            del model, tokenizer
            self.assertIs(objective, RoutingObjective.TARGET_NEW_NLL)
            self.assertEqual(tuple(observed_requests), requests)
            self.assertEqual(tuple(len(group) for group in contexts), (1, 5))
            coefficients = captured["coefficients"]
            self.assertIs(gradient_input, coefficients)
            signed = coefficients.new_tensor((1.0, -2.0, 3.0, 0.0, -4.0))
            loss = (-(coefficients * signed).sum()).detach()
            per_request = loss.expand(BATCH_SIZE)
            spans = tuple(f"{index + 101:064x}" for index in range(BATCH_SIZE))
            return RoutingObjectiveResult(
                RoutingObjective.TARGET_NEW_NLL,
                loss,
                per_request,
                (1,) * BATCH_SIZE,
                (None,) * BATCH_SIZE,
                identities,
                ordered_request_digest_v1(identities),
                spans,
                "e" * 64,
                (1, 5),
                6,
                "f" * 64,
                BATCH_SIZE * 6,
                BATCH_SIZE * 6,
                0,
                -signed,
                BATCH_SIZE,
            )

        model = torch.nn.Linear(1, 1, bias=False)
        with mock.patch(
            "project.run_scripts.ode_bf.p1_backend._CoefficientOverlay",
            Overlay,
        ), mock.patch(
            "project.run_scripts.ode_bf.p1_backend.evaluate_routing_objective",
            objective,
        ):
            receipt = signed_progress_gradient(
                model,
                object(),
                requests,
                field,
                cumulative_factors_by_weight={},
                ledger=ComputeLedger(),
                objective=RoutingObjective.TARGET_NEW_NLL,
                contexts=(("{}",), tuple(f"context-{index} {{}}" for index in range(5))),
            )
        self.assertEqual(receipt.signed_progress, (1.0, -2.0, 3.0, 0.0, -4.0))
        self.assertEqual(receipt.excluded_nonpositive_layers, (5, 7, 8))
        self.assertEqual(receipt.target_old_access_count, 0)
        self.assertEqual(receipt.routing_objective, "TARGET_NEW_NLL")
        self.assertIsNotNone(receipt.objective_receipt_sha256)
        self.assertEqual(receipt.routing_context_group_sizes, (1, 5))
        self.assertEqual(receipt.routing_context_count, 6)
        self.assertEqual(receipt.routing_context_sha256, "f" * 64)
        self.assertEqual(receipt.model_forward_count, BATCH_SIZE * 6)
        self.assertEqual(receipt.routing_backward_count, BATCH_SIZE)

    def test_default_recorder_envelope_is_legacy_exact_while_custom_is_namespaced(self) -> None:
        captured: list[dict[str, object]] = []

        def write_once(path: Path, value: dict[str, object]) -> str:
            del path
            captured.append(value)
            return canonical_hash(value)

        with tempfile.TemporaryDirectory() as directory:
            legacy = runtime.AdaptiveReceiptRecorder(
                Path(directory), runtime.AdaptiveVariant.FR_A8, write_once
            )
            legacy.field({"x": 1})
            self.assertNotIn("clock_variant", captured[-1])
            self.assertEqual(
                captured[-1]["schema"],
                "ode-edit-s04-ode-bf-p1r4-adaptive-receipt/v1",
            )
            custom = runtime.AdaptiveReceiptRecorder(
                Path(directory),
                runtime.AdaptiveVariant.FR_A8,
                write_once,
                variant_label="FR-A8-NEWNLL",
                instruction_id=TARGET_NEW_INSTRUCTION_ID,
                receipt_schema="ode-edit-s05-ode-bf-target-new-nll-routing-receipt/v1",
            )
            custom.field({"routing_objective": "TARGET_NEW_NLL"})
            custom.solver({"status": "SOLVER_CERTIFICATE_PASSED"})
            self.assertEqual(captured[-1]["clock_variant"], "FR-A8")
            self.assertEqual(captured[-1]["variant"], "FR-A8-NEWNLL")
            self.assertEqual(len(custom.links()["solver"]), 1)

    def test_solver_observer_persists_failed_certificate_before_raise(self) -> None:
        zero = np.zeros(2, dtype=np.float64)
        barrier = QuadraticBarrier(
            "historical",
            0.0,
            zero,
            np.zeros((2, 2), dtype=np.float64),
            1.0,
            "layer-local-diagonal",
        )
        problem = RoutingProblem(
            np.ones(2, dtype=np.float64),
            np.eye(2, dtype=np.float64),
            np.eye(2, dtype=np.float64),
            2.0,
            np.ones(2, dtype=np.float64),
            0.5,
            1.0e-8,
            barrier,
            barrier,
        )
        observed = []

        def write_once(path: Path, value: dict[str, object]) -> str:
            del path
            observed.append(value)
            return canonical_hash(value)

        failed = type(
            "FailedSolve",
            (),
            {"success": False, "nit": 1, "x": np.zeros(2, dtype=np.float64)},
        )()
        with tempfile.TemporaryDirectory() as directory:
            recorder = runtime.AdaptiveReceiptRecorder(
                Path(directory),
                runtime.AdaptiveVariant.FR_A8,
                write_once,
                variant_label="FR-A8-NEWNLL",
                instruction_id=TARGET_NEW_INSTRUCTION_ID,
                receipt_schema=(
                    "ode-edit-s05-ode-bf-target-new-nll-routing-receipt/v1"
                ),
            )
            with mock.patch.object(routing, "minimize", return_value=failed):
                with self.assertRaisesRegex(
                    ODEBFContractError, "solver certificate"
                ):
                    routing.solve_raw_velocity(
                        problem,
                        certificate_observer=runtime._solver_certificate_observer(
                            recorder=recorder,
                            problem=problem,
                            accepted_index=0,
                            solve_role="RAW_ROUTING",
                            routing_objective=RoutingObjective.TARGET_NEW_NLL,
                        ),
                    )
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0]["status"], "SOLVER_CERTIFICATE_FAILED")
        self.assertEqual(
            observed[0]["first_false_component"], "solver_success"
        )

    def test_solver_observation_failure_cannot_change_solver_result(self) -> None:
        zero = np.zeros(2, dtype=np.float64)
        barrier = QuadraticBarrier(
            "historical",
            0.0,
            zero,
            np.zeros((2, 2), dtype=np.float64),
            1.0,
            "layer-local-diagonal",
        )
        problem = RoutingProblem(
            np.ones(2, dtype=np.float64),
            np.eye(2, dtype=np.float64),
            np.eye(2, dtype=np.float64),
            2.0,
            np.ones(2, dtype=np.float64),
            0.5,
            1.0e-8,
            barrier,
            barrier,
        )
        expected = routing.solve_raw_velocity(problem)

        def broken_write(path: Path, value: dict[str, object]) -> str:
            del path, value
            raise OSError("fixture telemetry write failure")

        with tempfile.TemporaryDirectory() as directory:
            recorder = runtime.AdaptiveReceiptRecorder(
                Path(directory),
                runtime.AdaptiveVariant.FR_A8,
                broken_write,
                variant_label="FR-A8-NEWNLL",
                instruction_id=TARGET_NEW_INSTRUCTION_ID,
                receipt_schema=(
                    "ode-edit-s05-ode-bf-target-new-nll-routing-receipt/v1"
                ),
            )
            observed = routing.solve_raw_velocity(
                problem,
                certificate_observer=runtime._solver_certificate_observer(
                    recorder=recorder,
                    problem=problem,
                    accepted_index=0,
                    solve_role="RAW_ROUTING",
                    routing_objective=RoutingObjective.TARGET_NEW_NLL,
                ),
            )
            np.testing.assert_array_equal(observed.values, expected.values)
            self.assertEqual(observed.problem_identity, expected.problem_identity)
            self.assertEqual(observed.velocity_identity, expected.velocity_identity)
            self.assertEqual(
                observed.maximum_feasible_progress,
                expected.maximum_feasible_progress,
            )
            self.assertEqual(observed.certificate, expected.certificate)
            self.assertEqual(len(recorder.observation_failures), 2)
            with self.assertRaisesRegex(
                ODEBFContractError, "observation-only"
            ):
                recorder.assert_observation_complete()

    def test_a1_termination_taxonomy_and_accepted_p_receipt_are_explicit(self) -> None:
        self.assertEqual(
            runtime._termination_label(
                status="SAME_STATE_RETRY_EXHAUSTED", exact_hit=False
            ),
            "SAME_STATE_RETRY_EXHAUSTED",
        )
        self.assertEqual(
            runtime._termination_label(
                status="MIN_DT_EXHAUSTED", exact_hit=False
            ),
            "MIN_DT_EXHAUSTED",
        )
        self.assertEqual(
            runtime._termination_label(status="TAU_COMPLETE", exact_hit=True),
            "EXACT_HIT",
        )
        self.assertEqual(
            runtime._termination_label(status="TAU_COMPLETE", exact_hit=False),
            "TERMINAL_INFEASIBLE",
        )
        accepted_source = inspect.getsource(runtime._append_accepted_snapshot)
        self.assertIn('"structural_functional"', accepted_source)
        self.assertIn("outcome.structural_payload", accepted_source)

    def test_s05_rejects_handoff_local_artifact_without_weakening_other_guards(self) -> None:
        guard = ODEBFArtifactGuard(
            ROOT,
            ROOT / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
            "llama3-8b-inst",
            require_held_ode_alloc=False,
        )
        self.assertFalse(guard.require_held_ode_alloc)
        runtime_source = inspect.getsource(
            __import__(
                "project.run_scripts.ode_bf.p1_runtime",
                fromlist=["run_p1"],
            ).run_p1
        )
        self.assertIn(
            "require_held_ode_alloc=not target_new_routing_mode",
            runtime_source,
        )
        submit_source = inspect.getsource(submit._pre_submit)
        self.assertIn("require_held_ode_alloc=False", submit_source)

    def test_target_velocity_call_has_no_routing_objective_and_old_nll_is_diagnostic_only(self) -> None:
        source = inspect.getsource(runtime._build_active_field)
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "write_aware_target_velocity"
        ]
        self.assertEqual(len(calls), 1)
        self.assertNotIn("objective", {item.arg for item in calls[0].keywords})
        self.assertIn('"target_state_objective": "MARGIN_LOCKED"', source)

        trial_source = inspect.getsource(runtime._run_trial)
        trial_tree = ast.parse(trial_source)
        verdict_calls = [
            node
            for node in ast.walk(trial_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "verify_backtracked_candidate"
        ]
        self.assertEqual(len(verdict_calls), 1)
        verdict_text = ast.unparse(verdict_calls[0])
        self.assertNotIn("canonical_rewrite", verdict_text)
        self.assertNotIn("nll_old", verdict_text)
        self.assertIn("progress.actual_signed_progress", verdict_text)

    def test_entry_parser_and_namespaces_are_distinct_and_create_once(self) -> None:
        self.assertEqual(entry.TARGET_NEW_RESULT_TOKEN, TARGET_NEW_RESULT_TOKEN)
        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
            root = expected_target_new_result_name(alias)
            args = entry._parser().parse_args(
                [
                    "--model",
                    alias,
                    "--output-root",
                    root,
                    "--source-head",
                    "d" * 40,
                    "--run-token",
                    TARGET_NEW_RESULT_TOKEN,
                ]
            )
            self.assertEqual(args.model, alias)
            self.assertTrue(root.startswith("s05-target-new-nll-routing-"))

    def test_s05_failure_schema_and_unowned_root_collision_are_fail_closed(self) -> None:
        failure_schema = (
            "ode-edit-s05-ode-bf-target-new-nll-routing-failure/v1"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "owned"
            _, payload = write_p1_failure_once(
                root,
                ODEBFContractError("fixture"),
                repo_root=ROOT,
                instruction_id=TARGET_NEW_INSTRUCTION_ID,
                failure_schema=failure_schema,
            )
            self.assertEqual(payload["schema"], failure_schema)
            self.assertEqual(payload["instruction_id"], TARGET_NEW_INSTRUCTION_ID)

            collision = Path(directory) / "foreign"
            collision.mkdir()
            sentinel = collision / "sentinel"
            sentinel.write_bytes(b"unchanged")
            argv = [
                "--model",
                "llama3-8b-inst",
                "--output-root",
                str(collision),
                "--source-head",
                "d" * 40,
                "--run-token",
                TARGET_NEW_RESULT_TOKEN,
            ]
            with mock.patch.object(
                entry,
                "run_p1",
                side_effect=P1OutputRootCollision("create-once"),
            ), mock.patch.object(entry, "write_p1_failure_once") as writer:
                self.assertEqual(entry.main(argv), 1)
            writer.assert_not_called()
            self.assertEqual(sentinel.read_bytes(), b"unchanged")
            self.assertFalse((collision / "failure.json").exists())

    def test_source_handoff_is_independently_reverified(self) -> None:
        observed = submit._verify_handoff()
        self.assertEqual(observed, submit.HANDOFF_SHA256)

    def test_scheduler_accounting_fails_on_ambiguous_project_gpu(self) -> None:
        def completed(stdout: str):
            return type("Completed", (), {"stdout": stdout, "stderr": ""})()

        with mock.patch.dict("os.environ", {"USER": "fixture"}), mock.patch.object(
            submit,
            "_run",
            return_value=completed("1|odebf_fixture|RUNNING|gpu:1\n"),
        ):
            rows = submit._scheduler_snapshot()
        self.assertEqual(rows[0]["gpu"], 1)
        with mock.patch.dict("os.environ", {"USER": "fixture"}), mock.patch.object(
            submit,
            "_run",
            return_value=completed("1|odebf_fixture|PENDING|N/A\n"),
        ):
            with self.assertRaisesRegex(ODEBFContractError, "ambiguous"):
                submit._scheduler_snapshot()

    def test_pair_is_fully_accepted_while_held_before_single_release(self) -> None:
        completed = type(
            "Completed", (), {"stdout": "", "stderr": "", "returncode": 0}
        )()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            submit,
            "_submit_one",
            side_effect=("101", "102"),
        ) as submit_one, mock.patch.object(
            submit,
            "_run",
            return_value=completed,
        ) as run:
            jobs, receipt = submit._submit_held_pair(
                source_head="d" * 40,
                state=Path(directory),
                intent_sha256="e" * 64,
            )
            self.assertEqual(
                jobs,
                {"llama3-8b-inst": "101", "qwen2.5-7b-inst": "102"},
            )
            self.assertEqual(len(receipt), 64)
            self.assertEqual(submit_one.call_count, 2)
            run.assert_called_once_with(["scontrol", "release", "101", "102"])
            payload = json.loads(
                (
                    Path(directory)
                    / "s05-target-new-nll-routing-p1-v1.submission-receipt.json"
                ).read_text()
            )
            self.assertTrue(payload["pair_accepted_while_held"])
            self.assertEqual(payload["release_command_job_count"], 2)

    def test_partial_held_submission_is_cancelled_and_never_released(self) -> None:
        completed = type(
            "Completed", (), {"stdout": "", "stderr": "", "returncode": 0}
        )()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            submit,
            "_submit_one",
            side_effect=("101", OSError("second fixture submit failed")),
        ), mock.patch.object(
            submit,
            "_run",
            return_value=completed,
        ) as run:
            with self.assertRaisesRegex(OSError, "second fixture"):
                submit._submit_held_pair(
                    source_head="d" * 40,
                    state=Path(directory),
                    intent_sha256="e" * 64,
                )
            run.assert_called_once_with(["scancel", "101"], check=False)
            self.assertFalse(
                (
                    Path(directory)
                    / "s05-target-new-nll-routing-p1-v1.submission-receipt.json"
                ).exists()
            )
            failure = json.loads(
                (
                    Path(directory)
                    / "s05-target-new-nll-routing-p1-v1.submission-failure.json"
                ).read_text()
            )
            self.assertEqual(
                failure["accepted_held_jobs"], {"llama3-8b-inst": "101"}
            )
            self.assertEqual(
                failure["cancelled_before_pair_release"],
                {"llama3-8b-inst": True},
            )
            self.assertFalse(failure["pair_released"])


if __name__ == "__main__":
    unittest.main()
