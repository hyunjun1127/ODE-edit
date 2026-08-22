from __future__ import annotations

import inspect
import unittest

import torch

from project.run_scripts.ode_bf import p1r52_c_writer_phase1_sequential as phase1
from project.run_scripts.ode_bf import p1r52_joint_pc_fp32_runtime as fp32
from project.run_scripts.ode_bf import p1r52_sequential_runtime as sequential
from project.run_scripts.session05_ode_bf_p1r52_c_writer_phase1_dry_plan import build_plan


class Phase1ContractTest(unittest.TestCase):
    def test_cell_map_and_cache_applicability(self) -> None:
        self.assertEqual(len(phase1.ROLES), 5)
        self.assertEqual(tuple(phase1.label_for_role(phase1.role_for_cell(i)) for i in range(5)), phase1.CELL_LABELS)
        plan = build_plan()
        self.assertEqual(plan["array"], "0-4%3")
        self.assertEqual(plan["project_gpu_cap"], 3)
        self.assertEqual(plan["cache_valid_denominator"], "4/4_APPLICABLE_ARMS")
        memit = plan["rows"][1]
        self.assertEqual(memit["alpha_cache_status"], "ALPHA_CACHE_NOT_APPLICABLE_NATIVE_MEMIT")
        self.assertEqual(memit["alpha_cache_influence_count"], 0)
        self.assertEqual(memit["request_history_width"], 0)

    def test_full_fp32_and_native_baseline_boundaries(self) -> None:
        source = inspect.getsource(phase1)
        self.assertIn("easyeditor.models.alphaedit.AlphaEdit_main.apply_AlphaEdit_to_model", source)
        self.assertIn("easyeditor.models.memit.memit_main.apply_memit_to_model", source)
        self.assertIn("native_compute_z_writer_semantics", source)
        self.assertNotIn("AcceptedPhysicalStateMaterializer", source)
        self.assertNotIn("AtomicBatchTransaction", source)
        self.assertNotIn("assemble_effective_bf16", source)
        self.assertNotIn("CachedBF16FunctionalTrial", source)

    def test_history_interface_is_typed_fp32(self) -> None:
        signature = inspect.signature(fp32._run_pc_arm_fp32)
        self.assertIn("solve_history_keys_by_layer", signature.parameters)
        empty = {layer: torch.empty((3, 0), dtype=torch.float32) for layer in (4, 5, 6, 7, 8)}
        self.assertTrue(all(value.dtype is torch.float32 and value.shape[1] == 0 for value in empty.values()))

    def test_result_names_create_once_distinct(self) -> None:
        values = tuple(phase1.expected_result_name(role) for role in phase1.ROLES)
        self.assertEqual(len(values), len(set(values)))
        self.assertTrue(all("full-fp32" in value and "10xb100" in value for value in values))
        self.assertTrue(all("tech-r1" in value for value in values))

    def test_native_baseline_common_receipt_initializes_norm_rows(self) -> None:
        source = inspect.getsource(sequential.run_p1r52_sequential)
        initialized = source.index("batch_norm_rows: list[dict[str, Any]] = []")
        role_branch = source.index("if role in R52_ROLES:", initialized)
        receipt_read = source.index('"actual_update_norm_share": batch_norm_rows', role_branch)
        self.assertLess(initialized, role_branch)
        self.assertLess(role_branch, receipt_read)


if __name__ == "__main__":
    unittest.main()
