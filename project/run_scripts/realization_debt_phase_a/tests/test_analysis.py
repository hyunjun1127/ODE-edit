from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from project.run_scripts.realization_debt_phase_a.analysis import (
    _derive_layer,
    _fixed_group_summary,
    _request_endpoint_join,
)
from project.run_scripts.realization_debt_phase_a.contracts import (
    AnalysisBoundary,
    debt_components,
    normalized_potential_reduction,
    regular_file,
)


class PhaseAAnalysisTest(unittest.TestCase):
    def test_debt_failure_mode_partition(self) -> None:
        for rho, expected in ((0.25, "debt_under"), (1.5, "debt_over"), (-0.5, "debt_opposite")):
            row = debt_components(rho, 0.4)
            self.assertAlmostEqual(
                row["debt_native"],
                row["debt_under"] + row["debt_over"] + row["debt_opposite"] + row["debt_orthogonal"],
            )
            self.assertGreater(row[expected], 0.0)

    def test_normalized_potential_reduction(self) -> None:
        self.assertEqual(normalized_potential_reduction(1.0, 0.0), 0.5)
        self.assertEqual(normalized_potential_reduction(0.5, 0.5), 0.0)
        with self.assertRaises(AnalysisBoundary):
            normalized_potential_reduction(float("nan"), 0.0)

    def test_rowwise_debt_precedes_summary(self) -> None:
        rows = []
        for index, rho in enumerate((0.0, 2.0)):
            rows.append(
                {
                    "model": "m", "method": "a", "batch_index": 1,
                    "request_sha256": f"r{index}", "layer": 4,
                    "case_identity_sha256": f"c{index}", "q_pre": 1.0,
                    "q_post": 0.0, "rho": rho, "tau": 0.0,
                }
            )
        derived, gate = _derive_layer(pd.DataFrame(rows))
        summary = _fixed_group_summary(derived, ("model", "method", "batch_index", "layer"), ("rho", "debt_native"), 2)
        self.assertEqual(gate["debt_decomposition_identity_failure_count"], 0)
        self.assertEqual(float(summary.loc[0, "debt_native_mean"]), 1.0)
        self.assertNotEqual(float(summary.loc[0, "debt_native_mean"]), (1.0 - float(summary.loc[0, "rho_median"])) ** 2)

    def test_cvar_is_top_decile_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "model": ["m"] * 100,
                "method": ["x"] * 100,
                "batch_index": [1] * 100,
                "layer": [4] * 100,
                "value": np.arange(100, dtype=np.float64),
            }
        )
        summary = _fixed_group_summary(frame, ("model", "method", "batch_index", "layer"), ("value",), 100)
        self.assertEqual(float(summary.loc[0, "value_cvar90"]), 94.5)
        self.assertAlmostEqual(float(summary.loc[0, "value_p90"]), 89.1)

    def test_strict_nll_tie_is_failure(self) -> None:
        rows = []
        for layer in (4, 5, 6, 7, 8):
            rows.append(
                {
                    "model": "m", "method": "x", "batch_index": 1,
                    "request_sha256": "r", "layer": layer,
                    "case_identity_sha256": "c", "q_pre": 1.0,
                    "q_post": 0.0, "rho": 1.0, "tau": 0.0,
                }
            )
        derived, _ = _derive_layer(pd.DataFrame(rows))
        endpoint = pd.DataFrame(
            [{
                "model": "m", "method": "x", "batch_index": 1,
                "request_sha256": "r", "case_identity_sha256": "c",
                "target_new_nll": 1.0, "target_true_nll": 1.0,
                "target_new_margin": 0.0, "target_true_margin": 0.0,
                "target_new_strict": 0, "target_true_strict": 0,
            }]
        )
        joined = _request_endpoint_join(derived, endpoint)
        self.assertEqual(int(joined.loc[0, "rs_current"]), 0)
        self.assertEqual(int(joined.loc[0, "nll_tie"]), 1)
        self.assertEqual(float(joined.loc[0, "nll_advantage"]), 0.0)

    def test_group_size_fails_closed(self) -> None:
        frame = pd.DataFrame({"model": ["m"], "method": ["x"], "batch_index": [1], "layer": [4], "x": [1.0]})
        with self.assertRaises(AnalysisBoundary):
            _fixed_group_summary(frame, ("model", "method", "batch_index", "layer"), ("x",), 100)

    def test_regular_file_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("x", encoding="utf-8")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(AnalysisBoundary):
                regular_file(link)

    def test_analysis_source_has_no_experiment_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = "\n".join((root / name).read_text(encoding="utf-8") for name in ("analysis.py", "contracts.py", "figures.py"))
        self.assertNotIn("import torch", text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("model.load", text)


if __name__ == "__main__":
    unittest.main()
