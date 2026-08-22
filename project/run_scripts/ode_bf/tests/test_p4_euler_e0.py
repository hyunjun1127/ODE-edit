from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.functional import tensor_sha256
from project.run_scripts.ode_bf.p4_euler_binding import (
    P4EulerObjectiveTerms,
    build_euler_nonsemantic_identity,
    build_p4_euler_objective_callback,
)
from project.run_scripts.ode_bf.p4_euler_cache_transaction import (
    P4EulerCacheTransaction,
    verify_k_target_bridge,
)
from project.run_scripts.ode_bf.p4_euler_integrator import (
    EulerObjectiveEvaluation,
    project_request_columns_to_ball,
    run_raw_projected_euler,
)
from project.run_scripts.ode_bf.p4_euler_native_reference import (
    build_native_source_contract,
    verify_native_execution_boundary,
)
from project.run_scripts.ode_bf.p4_euler_orchestration import (
    build_receding_horizon_write_target,
    build_za_phase_contract,
    verify_euler_outer_refresh,
)
from project.run_scripts.ode_bf.p4_euler_runtime_contracts import (
    FULL_FP32_ROLES,
    tensor_inventory_content_sha256,
    validate_full_fp32_inventory,
)
from project.run_scripts.ode_bf.p4_semantic_barrier import (
    P4TargetArm,
    smooth_semantic_logodds_potential,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
EASYEDIT_ROOT = Path("/data/janghj/EasyEdit")
LOCK_PATH = (
    REPO_ROOT
    / "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p4_euler_projected_semantic_ode_e0.json"
)


def quadratic(target: torch.Tensor):
    def callback(state: torch.Tensor) -> EulerObjectiveEvaluation:
        per_request = 0.5 * torch.sum((state - target).square(), dim=0)
        return EulerObjectiveEvaluation(per_request, {"fixture": "quadratic"})

    return callback


class P4EulerE0FocusedGateTests(unittest.TestCase):
    def test_gate_01_one_step_exact_identity(self) -> None:
        origin = torch.zeros((2, 2), dtype=torch.float32)
        target = torch.tensor([[1.0, -2.0], [0.5, 1.5]], dtype=torch.float32)
        result = run_raw_projected_euler(
            origin,
            origin=origin,
            radius_by_request=torch.full((2,), 100.0, dtype=torch.float32),
            target_horizon=0.25,
            microsteps=1,
            objective_callback=quadratic(target),
        )
        expected, _ = project_request_columns_to_ball(
            origin + 0.25 * target,
            origin=origin,
            radius_by_request=torch.full((2,), 100.0, dtype=torch.float32),
        )
        self.assertTrue(torch.equal(result.final_state, expected))

    def test_gate_02_manual_m_step_byte_and_allclose(self) -> None:
        origin = torch.zeros((2, 2), dtype=torch.float32)
        target = torch.tensor([[1.0, -2.0], [0.5, 1.5]], dtype=torch.float32)
        radius = torch.full((2,), 100.0, dtype=torch.float32)
        result = run_raw_projected_euler(
            origin,
            origin=origin,
            radius_by_request=radius,
            target_horizon=0.5,
            microsteps=5,
            objective_callback=quadratic(target),
        )
        manual = origin.clone()
        for _ in range(5):
            manual, _ = project_request_columns_to_ball(
                manual + torch.tensor(0.1, dtype=torch.float32) * (target - manual),
                origin=origin,
                radius_by_request=radius,
            )
        self.assertTrue(torch.equal(result.final_state, manual))
        self.assertTrue(torch.allclose(result.final_state, manual, atol=0.0, rtol=0.0))

    def test_gate_03_zero_field_runs_all_fixed_steps(self) -> None:
        origin = torch.ones((2, 2), dtype=torch.float32)

        def zero(state: torch.Tensor) -> EulerObjectiveEvaluation:
            return EulerObjectiveEvaluation(
                torch.sum(state * 0.0, dim=0), {"fixture": "zero"}
            )

        result = run_raw_projected_euler(
            origin,
            origin=origin,
            radius_by_request=torch.ones(2, dtype=torch.float32),
            target_horizon=1.0,
            microsteps=5,
            objective_callback=zero,
        )
        self.assertTrue(torch.equal(result.final_state, origin))
        self.assertEqual(len(result.step_receipts), 5)
        self.assertTrue(
            all(all(row["zero_field_by_request"]) for row in result.step_receipts)
        )

    def test_gate_04_inner_w_content_hash_is_frozen(self) -> None:
        origin = torch.zeros((1, 1), dtype=torch.float32)
        weight = torch.tensor([[3.0]], dtype=torch.float32)
        content = lambda: tensor_inventory_content_sha256({"w": weight})
        result = run_raw_projected_euler(
            origin,
            origin=origin,
            radius_by_request=torch.ones(1, dtype=torch.float32),
            target_horizon=0.5,
            microsteps=2,
            objective_callback=quadratic(torch.ones_like(origin)),
            frozen_content_sha256=content,
        )
        self.assertEqual(
            result.receipt["frozen_content_entry_sha256"],
            result.receipt["frozen_content_exit_sha256"],
        )

        calls = 0

        def mutating(state: torch.Tensor) -> EulerObjectiveEvaluation:
            nonlocal calls
            calls += 1
            if calls == 1:
                weight.add_(1.0)
            return quadratic(torch.ones_like(origin))(state)

        with self.assertRaises(ODEBFContractError):
            run_raw_projected_euler(
                origin,
                origin=origin,
                radius_by_request=torch.ones(1, dtype=torch.float32),
                target_horizon=0.5,
                microsteps=2,
                objective_callback=mutating,
                frozen_content_sha256=content,
            )

    def test_gate_05_adam_optimizer_backward_and_state_are_absent(self) -> None:
        source = (
            REPO_ROOT / "project/run_scripts/ode_bf/p4_euler_integrator.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("p4_fixed_target_solver", source)
        self.assertNotIn("p4_target_solver_binding", source)
        self.assertEqual(source.count("torch.autograd.grad("), 1)
        origin = torch.zeros((1, 1), dtype=torch.float32)
        result = run_raw_projected_euler(
            origin,
            origin=origin,
            radius_by_request=torch.ones(1, dtype=torch.float32),
            target_horizon=0.5,
            microsteps=1,
            objective_callback=quadratic(torch.ones_like(origin)),
        )
        for name in (
            "optimizer_object_count",
            "adam_state_count",
            "sgd_state_count",
            "moment_count",
            "bias_correction_count",
            "parameter_gradient_count",
            "loss_backward_count",
        ):
            self.assertEqual(result.receipt[name], 0)

    def test_gate_06_positive_arms_share_nonsemantic_identity(self) -> None:
        kwargs = {
            "request_order_sha256": "1" * 64,
            "context_sha256": "2" * 64,
            "teacher_sha256": "3" * 64,
            "origin_sha256": "4" * 64,
            "radius_sha256": "5" * 64,
            "target_layer_name": "model.layers.8.mlp.down_proj",
            "kl_factor": 0.0625,
            "decay_factor": 0.5,
            "clamp_factor": 0.75,
            "target_horizon": 1.25,
            "microsteps": 5,
        }
        plus = build_euler_nonsemantic_identity(**kwargs)
        plus_minus = build_euler_nonsemantic_identity(**kwargs)
        self.assertEqual(plus, plus_minus)
        self.assertNotIn("arm", plus)

    def test_gate_07_context_softplus_precedes_mean(self) -> None:
        new = torch.tensor([[-8.0, 0.0, 0.0, 0.0, 0.0, 8.0]], dtype=torch.float32)
        old = torch.zeros_like(new)
        result = smooth_semantic_logodds_potential(
            new, old, arm=P4TargetArm.POSITIVE_NEGATIVE
        )
        correct = torch.mean(-new + torch.nn.functional.softplus(old - new), dim=1)
        incorrect = -torch.mean(new, dim=1) + torch.nn.functional.softplus(
            torch.mean(old - new, dim=1)
        )
        self.assertTrue(torch.equal(result.per_request_objective, correct))
        self.assertFalse(torch.allclose(correct, incorrect))

    def test_gate_08_old_gradient_field_sign_is_negative_sigma(self) -> None:
        origin = torch.zeros((1, 1), dtype=torch.float32)

        def terms(state: torch.Tensor) -> P4EulerObjectiveTerms:
            old = state.T.repeat(1, 6)
            new = state.T * 0.0
            new = new.repeat(1, 6)
            zeros = torch.zeros(1, dtype=torch.float32)
            return P4EulerObjectiveTerms(new, old, zeros, zeros, {})

        callback = build_p4_euler_objective_callback(
            terms,
            arm=P4TargetArm.POSITIVE_NEGATIVE,
            kl_factor=0.0,
            decay_factor=0.0,
        )
        result = run_raw_projected_euler(
            origin,
            origin=origin,
            radius_by_request=torch.ones(1, dtype=torch.float32),
            target_horizon=1.0,
            microsteps=1,
            objective_callback=callback,
        )
        self.assertTrue(torch.equal(result.final_state, torch.tensor([[-0.5]])))

    def test_gate_09_clamp_is_idempotent_with_fixed_origin_radius(self) -> None:
        origin = torch.zeros((2, 1), dtype=torch.float32)
        candidate = torch.tensor([[3.0], [4.0]], dtype=torch.float32)
        radius = torch.tensor([2.0], dtype=torch.float32)
        first, _ = project_request_columns_to_ball(
            candidate, origin=origin, radius_by_request=radius
        )
        second, _ = project_request_columns_to_ball(
            first, origin=origin, radius_by_request=radius
        )
        self.assertTrue(torch.equal(first, second))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(first)), 2.0)

    def test_gate_10_same_horizon_and_boxed_calibration_order(self) -> None:
        origin = torch.zeros((1, 1), dtype=torch.float32)

        def constant_field(state: torch.Tensor) -> EulerObjectiveEvaluation:
            return EulerObjectiveEvaluation(-state.sum(dim=0), {})

        common = {
            "origin": origin,
            "radius_by_request": torch.tensor([100.0], dtype=torch.float32),
            "target_horizon": 2.0,
            "objective_callback": constant_field,
        }
        m5 = run_raw_projected_euler(origin, microsteps=5, **common)
        m10 = run_raw_projected_euler(origin, microsteps=10, **common)
        self.assertTrue(torch.allclose(m5.final_state, m10.final_state, atol=5e-7, rtol=0.0))
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(lock["selected_target_horizon"], None)
        self.assertEqual(lock["stage_1"]["microsteps"], [1, 3, 5, 10])
        self.assertEqual(lock["stage_1"]["interpretation"], "STRENGTH_PSEUDO_TIME_CURVE_NOT_REFINEMENT")
        self.assertEqual(lock["stage_2"]["microsteps"], [5, 10])
        self.assertEqual(lock["stage_2"]["interpretation"], "FIXED_TARGET_HORIZON_NUMERICAL_REFINEMENT")

    def test_gate_11_only_final_iterate_is_selected(self) -> None:
        origin = torch.zeros((1, 1), dtype=torch.float32)
        result = run_raw_projected_euler(
            origin,
            origin=origin,
            radius_by_request=torch.ones(1, dtype=torch.float32) * 100.0,
            target_horizon=1.0,
            microsteps=5,
            objective_callback=quadratic(torch.ones_like(origin)),
        )
        self.assertEqual(result.receipt["selected_iterate"], "FINAL_ONLY")
        self.assertEqual(result.receipt["selected_iterate_index"], 5)
        self.assertEqual(result.receipt["best_iterate_decision_influence_count"], 0)

    def test_gate_12_cache_k1_k7_restore_k8_append_and_abort(self) -> None:
        transaction = P4EulerCacheTransaction("a" * 64, 7, "b" * 64, 10)
        for index in range(8):
            bridge = verify_k_target_bridge(
                outer_step_index=index,
                provided_target_sha256=f"{index + 1:x}" * 64,
                consumed_target_sha256=f"{index + 1:x}" * 64,
                cache_template_sha256="c" * 64,
                bridge_namespace=f"fixture/K{index + 1}/accepted-z",
            )
            transaction.observe_writer_step(
                outer_step_index=index,
                writer_entry_cache_sha256="a" * 64,
                writer_exit_candidate_sha256="d" * 64,
                restored_cache_sha256="a" * 64 if index < 7 else None,
                bridge=bridge,
            )
        committed = transaction.commit(
            final_cache_sha256="d" * 64,
            final_cache_width=17,
            final_weight_sha256="e" * 64,
            append_event_count=1,
        )
        self.assertEqual(committed["final_append_event_count"], 1)
        self.assertTrue(
            all(
                row["post_writer_restored_cache_sha256"] == "a" * 64
                for row in committed["outer_rows"][:7]
            )
        )
        aborted = P4EulerCacheTransaction("a" * 64, 7, "b" * 64, 10).abort(
            restored_cache_sha256="a" * 64,
            restored_weight_sha256="b" * 64,
        )
        self.assertEqual(aborted["final_append_event_count"], 0)
        self.assertEqual(aborted["weight_rollback_count"], 1)
        self.assertEqual(aborted["cache_rollback_count"], 1)

    def test_gate_13_k_specific_target_source_sha_must_match(self) -> None:
        receipt = verify_k_target_bridge(
            outer_step_index=2,
            provided_target_sha256="1" * 64,
            consumed_target_sha256="1" * 64,
            cache_template_sha256="2" * 64,
            bridge_namespace="run/K3/accepted-z",
        )
        self.assertEqual(receipt["provided_target_sha256"], receipt["consumed_target_sha256"])
        with self.assertRaises(ODEBFContractError):
            verify_k_target_bridge(
                outer_step_index=2,
                provided_target_sha256="1" * 64,
                consumed_target_sha256="3" * 64,
                cache_template_sha256="2" * 64,
                bridge_namespace="run/K3/accepted-z",
            )

    def test_gate_14_euler_arms_call_native_compute_z_zero_times(self) -> None:
        za = build_za_phase_contract(arm="Z+", local_solve_count=1)
        self.assertEqual(za["native_compute_z_call_count"], 0)
        rows = [
            {
                "outer_step_index": index,
                "current_terminal_refresh_count": 1,
                "current_key_refresh_count": 1,
                "current_residual_refresh_count": 1,
                "kl_teacher_capture_count": 1,
                "target_geometry_refresh_count": 1,
                "alpha_solve_refresh_count": 1,
                "official_writer_apply_count": 1,
                "fresh_target_reset_count": 1,
                "warm_start_count": 0,
                "native_compute_z_call_count": 0,
                "heldout_access_count": 0,
            }
            for index in range(8)
        ]
        verified = verify_euler_outer_refresh(rows, arm="A±")
        self.assertEqual(verified["native_compute_z_call_count"], 0)

    def test_gate_15_native_sources_are_canonical_easyedit_members(self) -> None:
        memit = build_native_source_contract(EASYEDIT_ROOT, method="native-memit")
        alpha = build_native_source_contract(EASYEDIT_ROOT, method="native-alphaedit")
        self.assertEqual(
            memit["canonical_entrypoint"],
            "easyeditor.models.memit.memit_main.apply_memit_to_model",
        )
        self.assertEqual(
            alpha["canonical_entrypoint"],
            "easyeditor.models.alphaedit.AlphaEdit_main.apply_AlphaEdit_to_model",
        )
        self.assertTrue(memit["main_source"]["path"].startswith(str(EASYEDIT_ROOT)))
        self.assertTrue(alpha["compute_z_source"]["path"].startswith(str(EASYEDIT_ROOT)))

    def test_gate_16_native_is_one_shot_without_k8_waypoint_override(self) -> None:
        for method in ("native-memit", "native-alphaedit"):
            receipt = verify_native_execution_boundary(
                build_native_source_contract(EASYEDIT_ROOT, method=method)
            )
            self.assertEqual(receipt["execution_mode"], "CANONICAL_ONE_SHOT")
            for name in (
                "k8_count",
                "waypoint_count",
                "provided_z_override_count",
                "schedule_match_count",
            ):
                self.assertEqual(receipt[name], 0)
        y = torch.zeros((1, 1), dtype=torch.float32)
        _, gain = build_receding_horizon_write_target(y, y + 1.0, outer_step_index=0)
        self.assertEqual(gain["fixed_endpoint_waypoint_claim_count"], 0)

    def test_gate_17_full_fp32_tensor_state_solver(self) -> None:
        parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        tensors = {
            role: (parameter,) if role == "model_parameters" else (torch.ones(1, dtype=torch.float32),)
            for role in FULL_FP32_ROLES
        }
        receipt = validate_full_fp32_inventory(
            tensors, autocast_enabled=False, quantization_enabled=False
        )
        self.assertEqual(receipt["tensor_state_solver"], "FULL_FP32")
        self.assertEqual(receipt["parameter_gradient_count"], 0)
        tensors["cache"] = (torch.ones(1, dtype=torch.bfloat16),)
        with self.assertRaises(ODEBFContractError):
            validate_full_fp32_inventory(
                tensors, autocast_enabled=False, quantization_enabled=False
            )


if __name__ == "__main__":
    unittest.main()
