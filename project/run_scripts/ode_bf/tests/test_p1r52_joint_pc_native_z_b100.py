from __future__ import annotations

import unittest
from pathlib import Path

from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    JOINT_PC_ALPHAEDIT_NATIVE_Z_ROLE,
    JOINT_PC_MEMIT_NATIVE_Z_ROLE,
    expected_p1r52_sequential_result_name,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import P1R52_B100X10_SCALE


ROOT = Path(__file__).resolve().parents[4]


class JointPCNativeZB100Tests(unittest.TestCase):
    def test_result_names_are_distinct_and_b100(self) -> None:
        names = {
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst", role, scale=P1R52_B100X10_SCALE
            )
            for role in (
                JOINT_PC_ALPHAEDIT_NATIVE_Z_ROLE,
                JOINT_PC_MEMIT_NATIVE_Z_ROLE,
            )
        }
        self.assertEqual(len(names), 2)
        self.assertTrue(all("native-z-b100" in name for name in names))

    def test_target_only_source_lock(self) -> None:
        source = (
            ROOT / "project/run_scripts/ode_bf/p1r52_joint_pc_native_z_b100.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"writer_apply_count": 0', source)
        self.assertIn('"writer_materialization_count": 0', source)
        self.assertIn('"persistent_weight_mutation_count": 0', source)
        self.assertNotIn("run_official_native_apply", source)
        self.assertNotIn("run_official_memit_apply", source)
        self.assertIn("evaluate_accepted_z_batch", source)
        self.assertIn("compute_z_call_count\": 100", source)

    def test_same_sample_and_w0_gates_are_literal(self) -> None:
        source = (
            ROOT / "project/run_scripts/ode_bf/p1r52_joint_pc_native_z_b100.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a",
            source,
        )
        self.assertIn(
            "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3",
            source,
        )
        self.assertIn(
            "bb2e661ad44bf8dc20e0224f21dc90164ed4b3dcf3316d751afb7d8a00bfaed6",
            source,
        )
        self.assertGreaterEqual(source.count("_hashes(touched) != entry_hashes"), 2)


if __name__ == "__main__":
    unittest.main()
