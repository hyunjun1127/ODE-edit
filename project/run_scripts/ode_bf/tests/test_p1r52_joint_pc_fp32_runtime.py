from __future__ import annotations

import inspect
import unittest

from project.run_scripts.ode_bf import p1r52_joint_pc_fp32_runtime as runtime
from project.run_scripts.ode_edit_motivation.gpu_runtime import assert_fixed_runtime


class JointPCFullFP32SourceGateTest(unittest.TestCase):
    def test_role_and_matrix_identity(self) -> None:
        self.assertTrue(runtime.is_joint_pc_full_fp32_role(runtime.ROLE))
        self.assertIn("full-fp32", runtime.RESULT_NAME)

    def test_no_bf16_planner_or_materializer_import(self) -> None:
        source = inspect.getsource(runtime)
        for forbidden in (
            "AcceptedPhysicalStateMaterializer",
            "CachedBF16FunctionalTrial",
            "assemble_effective_bf16",
            "plan_c1_writer(",
            "plan_c2_writer(",
        ):
            self.assertNotIn(forbidden, source)

    def test_c3_is_direct_official_apply(self) -> None:
        source = inspect.getsource(runtime.run_joint_pc_full_fp32_b100)
        self.assertIn("run_official_native_apply", source)
        self.assertIn("expected_native_compute_z_call_count=0", source)
        self.assertIn("accepted_z_cache_template", source)

    def test_dtype_counters_fail_closed(self) -> None:
        source = inspect.getsource(runtime.run_joint_pc_full_fp32_b100)
        self.assertIn('"bf16_conversion_count": 0', source)
        self.assertIn('"numeric_storage_cast_count": 0', source)
        self.assertIn('"autocast_count": 0', source)

    def test_current_python_patch_is_explicitly_compatible(self) -> None:
        observed = assert_fixed_runtime(allow_python_patch_compatible=True)
        self.assertEqual(observed["python"].split(".")[:2], ["3", "12"])


if __name__ == "__main__":
    unittest.main()
