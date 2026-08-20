from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from project.run_scripts import (
    session05_ode_bf_p1r52_target_depth_il5_sequential_b100x10_dry_plan as dry,
)
from project.run_scripts.ode_bf.common_coldcoord_fixed_e8_runtime import (
    _heldout_additive_lookup_geometry,
)
from project.run_scripts.ode_bf.common_cold_coordinate import (
    common_terminal_residual_input,
)
from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.bg_soft_diagnostics import (
    HeldoutRequestResidualActivationOverlay,
)
from project.run_scripts.ode_bf.p1_scalable_batched_experiment import _run_ode_arm
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    expected_p1r52_sequential_result_name,
    run_p1r52_sequential,
)
from project.run_scripts.ode_bf.p1r52_piru_postenergy_warn import (
    classify_h_postsolve_certificate,
)
from project.run_scripts.ode_bf.p1r52_target_depth import (
    P1R52_SEQUENTIAL_TARGET_DEPTH_INNER_COUNTS,
    P1R52_TARGET_DEPTH_INNER_COUNTS,
    P1R52TargetDepth,
)
from project.run_scripts.ode_bf.p1r52_target_depth_inner_telemetry import (
    _pinned_numeric_vectors_sha256,
)
from project.run_scripts.ode_bf.p1r52_target_depth_sequential_il5 import (
    ATTEMPT_SUFFIX,
    POLICY,
    RESULT_NAME,
    ROLE,
    STREAM_ORDER,
    STREAM_ROOT,
    validate_runtime_activation,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


class P1R52TargetDepthSequentialIL5Tests(unittest.TestCase):
    def test_il5_is_sequential_only_and_other_depth_registries_are_unchanged(self) -> None:
        self.assertEqual(P1R52_TARGET_DEPTH_INNER_COUNTS, (1, 3, 8, 10, 15))
        self.assertEqual(P1R52_SEQUENTIAL_TARGET_DEPTH_INNER_COUNTS, (5,))
        self.assertEqual(P1R52TargetDepth.from_inner_count(5), P1R52TargetDepth.IL5_FULL)
        self.assertEqual(POLICY.expected_inner_rows_per_batch, 40)
        self.assertEqual(POLICY.expected_request_inner_rows_per_batch, 4000)
        self.assertTrue(POLICY.postsolve_energy_warn_enabled)

    def test_activation_is_exact_and_fail_closed(self) -> None:
        self.assertIs(
            validate_runtime_activation(
                role=ROLE,
                scale_id="b100x10",
                attempt_suffix=ATTEMPT_SUFFIX,
                depth="IL5-FULL",
                inner_telemetry=True,
                batch_entry_evaluator=False,
            ),
            POLICY,
        )
        for changed in (
            {"role": "r52-soft-sequential-alphacache-on-structuralh-off"},
            {"depth": "IL3-FULL"},
            {"inner_telemetry": False},
            {"batch_entry_evaluator": True},
        ):
            values = {
                "role": ROLE,
                "scale_id": "b100x10",
                "attempt_suffix": ATTEMPT_SUFFIX,
                "depth": "IL5-FULL",
                "inner_telemetry": True,
                "batch_entry_evaluator": False,
                **changed,
            }
            with self.assertRaises(ODEBFContractError):
                validate_runtime_activation(**values)

    def test_result_and_dry_plan_bind_one_b1000_cell(self) -> None:
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst",
                ROLE,
                scale=__import__(
                    "project.run_scripts.ode_bf.p1r52_sequential_scale",
                    fromlist=["P1R52_B100X10_SCALE"],
                ).P1R52_B100X10_SCALE,
                attempt_suffix=ATTEMPT_SUFFIX,
            ),
            RESULT_NAME,
        )
        plan = dry.build_plan("1" * 40)
        self.assertEqual(plan["job_count"], 1)
        self.assertEqual(plan["target_depth"], "IL5-FULL")
        self.assertEqual(plan["inner_count_per_outer"], 5)
        self.assertEqual(plan["inner_rows_per_batch"], 40)
        self.assertEqual(plan["request_inner_rows_per_batch"], 4000)
        self.assertEqual(plan["batch_entry_evaluator_count"], 0)
        self.assertTrue(plan["postsolve_energy_warn_enabled"])
        self.assertEqual(plan["postsolve_energy_decision_influence_count"], 0)
        self.assertEqual(plan["postsolve_energy_tolerance"], 1e-12)

    def test_postsolve_energy_excess_is_record_only_without_tolerance_change(self) -> None:
        receipt = classify_h_postsolve_certificate(
            solver_stage_success={"H": True, "P": True, "CAPACITY": True},
            strength_residual=5.551115123125783e-17,
            energy_residual=5.545341963397732e-12,
            p_residual=7.0013439490423934e-15,
            selected_minimum=0.07453928323853867,
            primal_tolerance=1e-8,
            energy_tolerance=1e-12,
            postsolve_energy_warn_enabled=True,
        )
        self.assertEqual(receipt["status"], "WARN_POSTSOLVE_ENERGY_RESIDUAL")
        self.assertIsNone(receipt["first_false_gate"])
        self.assertEqual(receipt["postsolve_energy_decision_influence_count"], 0)
        self.assertEqual(receipt["energy_tolerance"], 1e-12)
        self.assertEqual(receipt["coefficient_shrink_count"], 0)
        self.assertEqual(receipt["retry_count"], 0)
        self.assertEqual(receipt["tolerance_relaxation_count"], 0)

    def test_warn_does_not_hide_nonenergy_certificate_failure(self) -> None:
        receipt = classify_h_postsolve_certificate(
            solver_stage_success={"H": True, "P": True, "CAPACITY": True},
            strength_residual=2e-8,
            energy_residual=5.545341963397732e-12,
            p_residual=0.0,
            selected_minimum=0.1,
            primal_tolerance=1e-8,
            energy_tolerance=1e-12,
            postsolve_energy_warn_enabled=True,
        )
        self.assertEqual(receipt["status"], "HARD_FAIL")
        self.assertEqual(receipt["first_false_gate"], "STRENGTH_RESIDUAL")

    def test_runtime_uses_shared_scheduler_and_explicit_sequential_gate(self) -> None:
        scalable = inspect.getsource(_run_ode_arm)
        sequential = inspect.getsource(run_p1r52_sequential)
        scheduler = (
            REPO_ROOT / "project/run_scripts/ode_bf/p1r52_target_depth.py"
        ).read_text(encoding="utf-8")
        self.assertIn("run_p1r52_target_depth_scheduler(", scalable)
        self.assertIn("p1r52_sequential_target_depth", scalable)
        self.assertIn("validate_il5_sequential_activation", sequential)
        self.assertIn("p1r52_sequential_target_depth=", sequential)
        self.assertNotIn("writer_materialize(", scheduler)
        self.assertNotIn("evaluate_heldout", scheduler)

    def test_source_ast_keeps_one_outer_writer_and_no_inner_writer(self) -> None:
        source = inspect.getsource(run_p1r52_sequential)
        tree = ast.parse(source)
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        self.assertIn("target_depth_telemetry_observer", names)
        self.assertIn("target_depth_policy", names)
        self.assertIn("commit_sequential_batch", source)
        self.assertIn('"target_depth_controller_heldout_access_count": 0', source)
        self.assertIn('"target_depth_observation_action_influence_count": 0', source)
        self.assertIn('"batch_entry_pre_evaluator_count"', source)

    def test_stream_and_task_contract_are_exact(self) -> None:
        self.assertEqual(
            STREAM_ROOT,
            "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a",
        )
        self.assertEqual(
            STREAM_ORDER,
            "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3",
        )
        self.assertEqual(POLICY.round_count * POLICY.batch_size, 1000)

    def test_lookup_geometry_has_typed_b100_cardinality_extension(self) -> None:
        class Tokenizer:
            padding_side = "left"

            def __call__(self, value, **kwargs):
                if isinstance(value, str):
                    return {"input_ids": [1, 2]}
                return {
                    "attention_mask": torch.ones(
                        (len(value), 4), dtype=torch.long
                    )
                }

        requests = tuple(
            {
                "request_sha256": f"request-{index}",
                "subject": f"subject-{index}",
            }
            for index in range(2)
        )
        cases = tuple(
            SimpleNamespace(
                request_sha256=f"request-{index}",
                rewrite_prompt=f"subject-{index} rewrite",
                paraphrase_prompts=(f"subject-{index} rephrase",),
                neighborhood_prompts=(f"neighbor-{index}",),
                target_new="new",
                target_true="true",
            )
            for index in range(2)
        )
        with patch(
            "easyeditor.models.alphaedit.AlphaEdit_main.find_fact_lookup_idx",
            return_value=-1,
        ):
            positions, patched, receipt = _heldout_additive_lookup_geometry(
                Tokenizer(),
                requests,
                cases,
                fact_token_strategy="subject_last",
                expected_batch_size=2,
            )
        self.assertEqual(len(positions), 2)
        self.assertEqual(patched, (4, 4))
        self.assertEqual(receipt["request_count"], 2)
        with self.assertRaises(ODEBFContractError):
            _heldout_additive_lookup_geometry(
                Tokenizer(),
                requests,
                cases,
                fact_token_strategy="subject_last",
                expected_batch_size=100,
            )

    def test_residual_and_overlay_keep_b10_default_and_allow_typed_b100_axis(self) -> None:
        target = torch.ones((4, 2), dtype=torch.float32)
        current = torch.zeros_like(target)
        residual = common_terminal_residual_input(
            target,
            current,
            "1" * 64,
            expected_request_count=2,
        )
        self.assertEqual(residual.request_count, 2)
        overlay = HeldoutRequestResidualActivationOverlay(
            torch.nn.Module(),
            "layer",
            residual.residual,
            ((0, 0), (0, 0)),
            (2, 2),
        )
        self.assertEqual(overlay.request_count, 2)
        with self.assertRaises(ODEBFContractError):
            common_terminal_residual_input(
                target,
                current,
                "1" * 64,
                expected_request_count=100,
            )

    def test_accuracy_schema_numeric_vector_digest_is_deterministic(self) -> None:
        raw = {
            "legacy_primary": {
                "target_span_sha256": "2" * 64,
                "evaluation_case_identity_sha256": "3" * 64,
            }
        }
        left = _pinned_numeric_vectors_sha256(raw, ((1.0,),), ((2.0, 3.0),))
        right = _pinned_numeric_vectors_sha256(raw, ((1.0,),), ((2.0, 3.0),))
        self.assertEqual(left, right)
        self.assertEqual(len(left), 64)
        self.assertEqual(
            _pinned_numeric_vectors_sha256(
                {"numeric_vectors_sha256": "4" * 64},
                ((9.0,),),
                ((8.0,),),
            ),
            "4" * 64,
        )


if __name__ == "__main__":
    unittest.main()
