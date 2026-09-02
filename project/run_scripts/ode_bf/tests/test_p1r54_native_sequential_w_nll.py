from __future__ import annotations

from pathlib import Path
import inspect
import unittest

from project.run_scripts.ode_bf.p1r52_c_writer_phase1_sequential import run_phase1
from project.run_scripts.ode_bf.p1r52_sequential_runtime import run_p1r52_sequential
from project.run_scripts.ode_bf.p1_runtime import run_p1
from project.run_scripts.ode_bf.p1r54_native_sequential_w_nll import (
    CELLS,
    STREAM_ORDER,
    STREAM_ROOT,
    cell_for_index,
    source_equivalence_receipt,
)
from project.run_scripts.session05_ode_bf_p1r54_native_sequential_w_nll_dry_plan import (
    build_plan,
)


class NativeSequentialWNLLTests(unittest.TestCase):
    def test_exact_two_original_native_cells(self) -> None:
        self.assertEqual([item.cell for item in CELLS], [0, 1])
        self.assertEqual(
            [item.label for item in CELLS],
            ["OFFICIAL-ALPHAEDIT", "OFFICIAL-MEMIT"],
        )
        self.assertEqual(cell_for_index(0), CELLS[0])
        self.assertEqual(cell_for_index(1), CELLS[1])
        with self.assertRaisesRegex(Exception, "cell differs"):
            cell_for_index(2)

    def test_original_native_entrypoints_and_final_request_table_are_reused(self) -> None:
        phase1_source = inspect.getsource(run_phase1)
        runtime_source = inspect.getsource(run_p1r52_sequential)
        self.assertIn("run_p1r52_sequential", phase1_source)
        self.assertIn("NATIVE_CORRECTED_ROLE", phase1_source)
        self.assertIn("MEMIT_ROLE", phase1_source)
        self.assertIn("accepted_z_observation_enabled=True", phase1_source)
        self.assertIn("immediate-post-final-w10-requests.json", runtime_source)
        self.assertIn("build_post_final_request_rows", runtime_source)

    def test_plan_is_same_stream_full_fp32_and_no_unrelated_reruns(self) -> None:
        plan = build_plan()
        self.assertEqual(plan["array"], "0-1%2")
        self.assertEqual(plan["stream_root"], STREAM_ROOT)
        self.assertEqual(plan["order_root"], STREAM_ORDER)
        self.assertEqual(plan["dtype"], "FULL_FP32")
        self.assertEqual(plan["independent_baseline_rerun_count"], 0)
        self.assertEqual(plan["fz_sequential_rerun_count"], 0)
        self.assertEqual(
            [row["required_final_w10_request_rows"] for row in plan["cells"]],
            [1000, 1000],
        )

    def test_source_equivalence_is_observation_backfill_only(self) -> None:
        receipt = source_equivalence_receipt()
        self.assertEqual(receipt["native_equation_change_count"], 0)
        self.assertEqual(receipt["native_writer_change_count"], 0)
        self.assertEqual(receipt["stream_or_evaluator_change_count"], 0)
        self.assertEqual(receipt["new_native_arm_count"], 0)
        self.assertEqual(receipt["required_request_count_per_cell"], 1000)

    def test_capacity_override_is_scoped_to_existing_phase1_roles(self) -> None:
        source = inspect.getsource(run_p1)
        start = source.index("if runtime_gpu_capacity_validator is not None")
        end = source.index("p3r1_fp32_mode =", start)
        scope = source[start:end]
        self.assertIn("is_phase1_role(p1r52_sequential_role)", scope)
        self.assertIn("runtime GPU capacity override scope differs", scope)

    def test_server4_sbatch_is_exact_two_cell_cap4(self) -> None:
        path = Path(__file__).resolve().parents[2] / "session05_ode_bf_p1r54_native_sequential_w_nll_server4.sbatch"
        source = path.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-1%2", source)
        self.assertIn("export PROJECT_GPU_CAP=4", source)
        self.assertIn("#SBATCH --gres=gpu:1", source)
        self.assertIn("#SBATCH --mem=60416M", source)


if __name__ == "__main__":
    unittest.main()
