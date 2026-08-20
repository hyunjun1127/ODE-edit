from __future__ import annotations

import json
import inspect
import statistics
import unittest
from pathlib import Path

from project.run_scripts.ode_bf.p1r52_piru_postenergy_warn import (
    ATTEMPT_SUFFIX,
    RESULT_NAME,
    actual_update_norm_share_receipt,
    classify_h_postsolve_certificate,
    z_w_realization_receipt,
)
from project.run_scripts.ode_bf.p1r52_piru_sequential_adapter import (
    P1R52_PIRU_SEQUENTIAL_ROLE,
)
from project.run_scripts.ode_bf.p1r52_piru_postenergy_warn_panel import (
    LOCK_FILE,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    expected_p1r52_sequential_result_name,
    run_p1r52_sequential,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import P1R52_B100X10_SCALE
from project.run_scripts import (
    session05_ode_bf_p1r52_piru_postenergy_warn_b100x10 as runner,
    session05_ode_bf_p1r52_piru_postenergy_warn_b100x10_dry_plan as dry,
)
from project.run_scripts.ode_bf.progress_simplex_routing import (
    SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
    SIMPLEX_PRIMAL_TOLERANCE,
)


CAPTURE_RECEIPT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r52-piru-seq-10xb100-hon-v1/local/odebf/private/"
    "p1r52-piru-b4-diag-c2-v1/capsule/capture-receipt.json"
)
TECH_R2_RESULT = Path(
    "/mnt/raid5/janghj/.codex/worktrees/"
    "odeeditsh1-s05-p1r52-piru-seq-10xb100-hon-v1/local/odebf/results/"
    "s05-p1r52-pir-u-soft-sequential-structuralh-on-10xb100-tech-r2-v1"
)


def _stages(*, h: bool = True, p: bool = True, capacity: bool = True):
    return {"H": h, "P": p, "CAPACITY": capacity}


class P1R52PIRUPostEnergyWarnTests(unittest.TestCase):
    def test_numerical_lock_and_runner_bind_only_authorized_delta(self) -> None:
        repo = Path(__file__).resolve().parents[4]
        lock, lock_sha = load_and_validate_lock(
            repo / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(len(lock_sha), 64)
        self.assertEqual(lock["optimizer_energy_constraint"], "UNCHANGED_ACTIVE")
        self.assertEqual(lock["replacement_energy_threshold_count"], 0)
        source = inspect.getsource(runner.main)
        self.assertIn("p1r52_postsolve_energy_warn_enabled=True", source)
        self.assertIn("p1r52_accepted_z_observation=True", source)
        self.assertIn("p1r52_accepted_z_sealed_w_reuse=False", source)

    def test_new_result_and_dry_plan_are_create_once_one_cell(self) -> None:
        self.assertIn("tech-r1", ATTEMPT_SUFFIX)
        self.assertIn("tech-r1", RESULT_NAME)
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst",
                P1R52_PIRU_SEQUENTIAL_ROLE,
                scale=P1R52_B100X10_SCALE,
                attempt_suffix=ATTEMPT_SUFFIX,
            ),
            RESULT_NAME,
        )
        plan = dry.build_plan("a" * 40)
        self.assertEqual(plan["job_count"], 1)
        self.assertEqual(plan["batch_entry_evaluator_count"], 0)
        self.assertEqual(plan["accepted_z_added_backward_generation_action"], [0, 0, 0])
        self.assertEqual(plan["postsolve_energy_residual"], "WARN_CONTINUE")

    def test_inline_accepted_z_does_not_require_sealed_reference(self) -> None:
        source = inspect.getsource(run_p1r52_sequential)
        self.assertIn(
            "if accepted_z_observation_enabled and accepted_z_sealed_w_reuse:\n"
            "                assert reference is not None\n"
            "                terminal_anchor_receipt",
            source,
        )
        self.assertNotIn(
            "if accepted_z_observation_enabled:\n"
            "                assert reference is not None\n"
            "                terminal_anchor_receipt",
            source,
        )

    def test_exact_captured_b4_k2_energy_residual_warns_and_continues(self) -> None:
        capture = json.loads(CAPTURE_RECEIPT.read_text(encoding="utf-8"))
        value = capture["values"]
        self.assertEqual(value["energy_violation"], 1.9768631176475537e-12)
        decision = classify_h_postsolve_certificate(
            solver_stage_success=_stages(),
            strength_residual=value["strength_residual"],
            energy_residual=value["energy_violation"],
            p_residual=value["p_violation"],
            selected_minimum=value["selected_minimum"],
            primal_tolerance=SIMPLEX_PRIMAL_TOLERANCE,
            energy_tolerance=SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
            postsolve_energy_warn_enabled=True,
        )
        self.assertEqual(decision["status"], "WARN_POSTSOLVE_ENERGY_RESIDUAL")
        self.assertIsNone(decision["first_false_gate"])
        self.assertEqual(decision["postsolve_energy_decision_influence_count"], 0)
        self.assertEqual(decision["replacement_energy_threshold_count"], 0)
        self.assertEqual(decision["polish_count"], 0)
        self.assertEqual(decision["coefficient_shrink_count"], 0)
        self.assertEqual(decision["retry_count"], 0)
        self.assertEqual(decision["tolerance_relaxation_count"], 0)

    def test_old_strict_energy_gate_remains_available(self) -> None:
        decision = classify_h_postsolve_certificate(
            solver_stage_success=_stages(),
            strength_residual=0.0,
            energy_residual=1.9768631176475537e-12,
            p_residual=0.0,
            selected_minimum=0.1,
            primal_tolerance=1.0e-8,
            energy_tolerance=1.0e-12,
            postsolve_energy_warn_enabled=False,
        )
        self.assertEqual(decision["status"], "HARD_FAIL")
        self.assertEqual(decision["first_false_gate"], "POSTSOLVE_ENERGY_RESIDUAL")

    def test_solver_strength_p_and_nonnegative_remain_hard(self) -> None:
        cases = (
            (_stages(capacity=False), 0.0, 0.0, 0.0, 0.1, "SOLVER_STAGE_SUCCESS"),
            (_stages(), 2.0e-8, 0.0, 0.0, 0.1, "STRENGTH_RESIDUAL"),
            (_stages(), 0.0, 0.0, 2.0e-8, 0.1, "PRETRAINED_P_RESIDUAL"),
            (_stages(), 0.0, 0.0, 0.0, -2.0e-8, "NONNEGATIVE_SELECTED"),
        )
        for stages, strength, energy, p_value, minimum, expected in cases:
            with self.subTest(expected=expected):
                decision = classify_h_postsolve_certificate(
                    solver_stage_success=stages,
                    strength_residual=strength,
                    energy_residual=energy,
                    p_residual=p_value,
                    selected_minimum=minimum,
                    primal_tolerance=1.0e-8,
                    energy_tolerance=1.0e-12,
                    postsolve_energy_warn_enabled=True,
                )
                self.assertEqual(decision["status"], "HARD_FAIL")
                self.assertEqual(decision["first_false_gate"], expected)

    def test_actual_update_norm_share_and_zero_energy(self) -> None:
        receipt = actual_update_norm_share_receipt({"layer4": 1.0, "layer5": 4.0})
        self.assertAlmostEqual(receipt["layers"][0]["actual_update_norm_share"], 1.0 / 3.0)
        self.assertAlmostEqual(receipt["layers"][1]["actual_update_norm_share"], 2.0 / 3.0)
        self.assertAlmostEqual(receipt["layers"][0]["squared_energy_share"], 0.2)
        self.assertAlmostEqual(receipt["layers"][1]["squared_energy_share"], 0.8)
        zero = actual_update_norm_share_receipt({"layer4": 0.0, "layer5": 0.0})
        self.assertEqual(zero["status"], "ZERO_ENERGY")
        self.assertEqual(
            [row["actual_update_norm_share"] for row in zero["layers"]],
            [0.0, 0.0],
        )

    def test_prior_valid_prefix_actual_norm_share_recomputed(self) -> None:
        files = sorted(TECH_R2_RESULT.glob("raw/batches/b*/raw/ode/*/accepted-k*.json"))
        rows = []
        for path in files:
            source = json.loads(path.read_text(encoding="utf-8"))
            receipt = actual_update_norm_share_receipt(
                source["materialization"]["realized_bf16_step_energy"]
            )
            rows.append(receipt)
        self.assertEqual(len(rows), 25)
        by_layer: dict[int, list[float]] = {layer: [] for layer in range(4, 9)}
        for receipt in rows:
            for layer in receipt["layers"]:
                by_layer[int(layer["layer"])].append(layer["actual_update_norm_share"])
        means = [statistics.fmean(by_layer[layer]) for layer in range(4, 9)]
        expected = [0.0941, 0.1347, 0.1800, 0.2376, 0.3536]
        for observed, rounded in zip(means, expected, strict=True):
            self.assertAlmostEqual(observed, rounded, places=4)
        layer8 = sorted(by_layer[8])
        p90 = layer8[int(0.9 * (len(layer8) - 1))]
        self.assertAlmostEqual(statistics.median(layer8), 0.3490, places=4)
        self.assertAlmostEqual(p90, 0.3825, places=4)
        self.assertAlmostEqual(max(layer8), 0.5001, places=4)

    def test_z_w_pairing_keeps_rewrite_and_rephrase_denominators(self) -> None:
        panel = lambda values: {  # noqa: E731
            "target_new_nll_by_request": values,
            "prompt_denominator": sum(len(row) for row in values),
        }
        z = {
            "rewrite_success": panel([[1.0], [2.0]]),
            "rewrite_acc": {"prompt_denominator": 2},
            "paraphrase_success": panel([[1.0, 2.0], [3.0, 4.0]]),
            "paraphrase_acc": {"prompt_denominator": 4},
        }
        w = {
            "rewrite_success": panel([[2.0], [1.0]]),
            "rewrite_acc": {"prompt_denominator": 2},
            "paraphrase_success": panel([[2.0, 2.0], [2.0, 8.0]]),
            "paraphrase_acc": {"prompt_denominator": 4},
        }
        receipt = z_w_realization_receipt(z, w)
        self.assertEqual(receipt["rewrite"]["request_denominator"], 2)
        self.assertEqual(receipt["rewrite"]["prompt_denominator"], 2)
        self.assertEqual(receipt["rephrase"]["request_denominator"], 2)
        self.assertEqual(receipt["rephrase"]["prompt_denominator"], 4)
        self.assertEqual(receipt["duplicate_physical_w_evaluator_forward_count"], 0)
        self.assertEqual(receipt["action_influence_count"], 0)


if __name__ == "__main__":
    unittest.main()
