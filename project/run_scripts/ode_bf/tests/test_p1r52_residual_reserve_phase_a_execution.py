from __future__ import annotations

import inspect
import unittest

import torch

from project.run_scripts.ode_bf.functional import WaypointFactor
from project.run_scripts.ode_bf.p1r52_residual_reserve_phase_a_execution import (
    AcceptedPhysicalStateFP32Materializer,
    PHASE_A_ARMS,
    expected_phase_a_result_name,
)
from project.run_scripts.session05_ode_bf_p1r52_residual_reserve_phase_a_dry_plan import (
    build_plan,
)


class ResidualReservePhaseAExecutionTests(unittest.TestCase):
    def test_result_names_and_dry_plan(self) -> None:
        plan = build_plan("a" * 40, "llama3-8b-inst")
        self.assertEqual([item["arm"] for item in plan["cells"]], list(PHASE_A_ARMS))
        self.assertEqual(plan["array"], "0-3%4")
        self.assertEqual(plan["max_concurrent_gpu"], 4)
        self.assertEqual(
            plan["cells"][0]["result_name"],
            expected_phase_a_result_name("llama3-8b-inst", "j0"),
        )
        self.assertEqual(
            expected_phase_a_result_name(
                "llama3-8b-inst", "j0", attempt_suffix="tech-r1"
            ),
            "s05-p1r52-residual-reserve-phase-a-fp32-llama3-8b-inst-j0-tech-r1-v1",
        )
        self.assertEqual(
            expected_phase_a_result_name(
                "llama3-8b-inst", "j0", attempt_suffix="tech-r2"
            ),
            "s05-p1r52-residual-reserve-phase-a-fp32-llama3-8b-inst-j0-tech-r2-v1",
        )

    def test_fp32_j0_native_materialization_and_restore(self) -> None:
        model = torch.nn.Module()
        model.add_module("linear", torch.nn.Linear(3, 2, bias=False, dtype=torch.float32))
        model.linear.weight.requires_grad_(False)
        with torch.no_grad():
            model.linear.weight.zero_()
        entry = {"linear.weight": model.linear.weight.detach().clone()}
        left = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        right = torch.tensor([[3.0], [4.0], [5.0]], dtype=torch.float32)
        factor = WaypointFactor("linear.weight", 4, 0, 0, 0, 0.125, left, right, True, 1)
        materializer = AcceptedPhysicalStateFP32Materializer(model, entry)
        receipt = materializer.materialize({"linear.weight": (factor,)}, transition_index=1)
        expected = entry["linear.weight"] + torch.tensor(0.125, dtype=torch.float32) * (right @ left.T).T
        self.assertTrue(torch.equal(model.linear.weight, expected))
        self.assertEqual(receipt["numeric_storage_cast_count"], 0)
        self.assertEqual(receipt["bf16_path_call_count"], 0)
        restore = materializer.restore()
        self.assertTrue(restore["byte_restored_exact"])
        self.assertTrue(torch.equal(model.linear.weight, entry["linear.weight"]))

    def test_runtime_source_has_typed_fp32_mode(self) -> None:
        from project.run_scripts.ode_bf import p1_runtime
        from project.run_scripts.ode_bf import p1_scalable_batched_experiment

        runtime = inspect.getsource(p1_runtime.run_p1)
        arm = inspect.getsource(p1_scalable_batched_experiment._run_ode_arm)
        self.assertIn("p1r52_residual_reserve_phase_a_arm", runtime)
        self.assertIn("p1r52_residual_reserve_phase_a_arm is None", runtime)
        self.assertIn("load_phase_a_fp32_model", runtime)
        self.assertIn("p1r52_residual_reserve_writer.execute", arm)
        self.assertIn("if not p1r52_fp32_phase_a", arm)


if __name__ == "__main__":
    unittest.main()
