from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from project.run_scripts.ordered_response_barrier_ode.round0_analysis import (
    _endpoint_tables,
    _lineage_table,
    _markdown_table,
    _mechanism_tables,
    _method_contrast_table,
    _paired_delta_table,
    _stat,
    _load_cells,
)
from project.run_scripts.ordered_response_barrier_ode.round0_analysis_contracts import (
    AnalysisBoundary,
    PRIMARY_ARMS,
    regular_file,
)


STATE_BASE = Path("/mnt/raid5/janghj/ODE-edit/local/state/ordered-response-barrier-ode-server1-gated-v1")
RAW_ROOT = STATE_BASE / "round0-b1-gated-tech-r1"


class Round0AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RAW_ROOT.is_dir():
            raise unittest.SkipTest("sealed server1 round0 raw root unavailable")
        cls.cells, cls.inputs, cls.load_gates = _load_cells(RAW_ROOT)
        (
            cls.performance,
            cls.nll,
            cls.prompt,
            cls.request,
            cls.indexed,
            cls.endpoint_gates,
        ) = _endpoint_tables(cls.cells)

    def test_canonical_cell_request_endpoint_denominators(self) -> None:
        self.assertEqual(self.load_gates["canonical_cell_count"], 4)
        self.assertEqual(self.load_gates["canonical_request_count"], 400)
        self.assertEqual(self.load_gates["primary_endpoint_count"], 2000)
        self.assertEqual(len(self.inputs), 12)

    def test_pre_edit_identity_and_primary_order(self) -> None:
        self.assertTrue(self.load_gates["same_pre_edit_within_model"])
        self.assertEqual(tuple(self.cells[0]["round"]["primary_arm_order"]), PRIMARY_ARMS)
        self.assertEqual(tuple(self.cells[3]["round"]["primary_arm_order"]), PRIMARY_ARMS)

    def test_endpoint_rows_and_recomputed_preferences(self) -> None:
        self.assertEqual(len(self.performance), 24)
        self.assertEqual(len(self.nll), 120)
        self.assertEqual(len(self.prompt), 38_400)
        self.assertEqual(len(self.request), 2_400)
        self.assertTrue((self.performance[self.performance.stage != "PRE_EDIT"].rewrite_success_denominator == 100).all())

    def test_paired_and_adjacent_contrast_denominators(self) -> None:
        paired = _paired_delta_table(self.indexed, self.request)
        contrast = _method_contrast_table(self.request)
        self.assertEqual(len(paired), 112)
        self.assertEqual(len(contrast), 84)
        self.assertTrue((paired.better_count + paired.exact_equal_count + paired.worse_count == paired.paired_denominator).all())

    def test_mechanism_denominators_and_global_no_hit(self) -> None:
        arm, step, request_step, layer, compute, preamble, gates = _mechanism_tables(self.cells)
        self.assertEqual((len(arm), len(step), len(request_step), len(layer), len(compute), len(preamble)), (20, 320, 32_000, 100, 20, 4))
        self.assertEqual(gates["global_first_hit_count"], 0)
        self.assertEqual(gates["horizon_semantic_miss_dynamic_endpoint_count"], 16)
        self.assertEqual(gates["official_wrapper_direct_fidelity_failure_count"], 0)

    def test_scheduler_lineage_is_separated(self) -> None:
        lineage = _lineage_table(STATE_BASE)
        canonical = lineage[lineage.parent_job_id == "35694"]
        pilot = lineage[lineage.parent_job_id == "35615"]
        self.assertEqual(len(lineage), 21)
        self.assertTrue(canonical.scheduler_state.str.startswith("COMPLETED").all())
        self.assertTrue(canonical.result_inclusion.str.startswith("INCLUDED_CANONICAL").all())
        self.assertTrue(pilot.result_inclusion.str.contains("SEPARATE_B1").all())

    def test_stat_uses_linear_quantiles(self) -> None:
        observed = _stat(range(10))
        self.assertEqual(observed["mean"], 4.5)
        self.assertEqual(observed["median"], 4.5)
        self.assertAlmostEqual(observed["p90"], 8.1)
        self.assertEqual(observed["max"], 9.0)
        with self.assertRaises(AnalysisBoundary):
            _stat([1.0, float("nan")])

    def test_regular_file_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("x", encoding="utf-8")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(AnalysisBoundary):
                regular_file(link)

    def test_markdown_table_has_uniform_pipe_count(self) -> None:
        table = _markdown_table(("a", "b"), ((1, 2), ("x|y", "z")))
        counts = [len(re.findall(r"(?<!\\)\|", line)) for line in table.splitlines()]
        self.assertEqual(len(set(counts)), 1)

    def test_analysis_firewall_has_no_model_or_submit_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = "\n".join(
            (root / name).read_text(encoding="utf-8")
            for name in ("round0_analysis_contracts.py", "round0_analysis.py", "round0_figures.py", "round0_package_verify.py")
        )
        self.assertNotIn("import torch", text)
        self.assertNotIn("srun", text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("transformers", text)


if __name__ == "__main__":
    unittest.main()
