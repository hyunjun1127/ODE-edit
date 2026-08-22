from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

from project.run_scripts.ode_bf.contracts import canonical_hash


REPO_ROOT = Path(__file__).resolve().parents[4]
REPORT_ROOT = REPO_ROOT / (
    "experiment-reports/servers/server4/"
    "p4-euler-za-llama-h1-b2b10-2026-08-23"
)


class P4EulerZAH1AnalysisTests(unittest.TestCase):
    def test_analysis_identity_and_primary_projection_boundary(self) -> None:
        value = json.loads((REPORT_ROOT / "analysis.json").read_text())
        body = dict(value)
        identity = body.pop("identity_sha256")
        self.assertEqual(identity, canonical_hash(body))
        self.assertEqual(
            value["classification"], "PROJECTION_DOMINATED_EXPLORATORY_RUN"
        )
        clamp = value["primary_limitation"]["aggregate"]
        self.assertEqual(clamp["z_plus"]["hit_numerator"], 286)
        self.assertEqual(clamp["z_plus"]["request_microstep_denominator"], 450)
        self.assertEqual(
            clamp["z_plus"]["all_5_step_saturated_request_count"], 19
        )
        self.assertEqual(clamp["z_plus_minus"]["hit_numerator"], 381)
        self.assertEqual(
            clamp["z_plus_minus"]["all_5_step_saturated_request_count"], 54
        )
        self.assertFalse(value["claim_boundary"]["scientific_promotion"])

    def test_manifest_and_package_receipt_are_rooted(self) -> None:
        manifest = json.loads((REPORT_ROOT / "analysis-manifest.json").read_text())
        body = dict(manifest)
        root = body.pop("root_digest")
        self.assertEqual(root, canonical_hash(body))
        self.assertEqual(manifest["raw_input_count"], 171)
        self.assertEqual(manifest["raw_member_root"], canonical_hash(manifest["raw_inputs"]))

        receipt = json.loads((REPORT_ROOT / "rooted-receipt.json").read_text())
        body = dict(receipt)
        identity = body.pop("identity_sha256")
        self.assertEqual(identity, canonical_hash(body))
        self.assertEqual(
            receipt["package_member_root"], canonical_hash(receipt["package_members"])
        )
        for row in receipt["package_members"]:
            path = REPORT_ROOT / row["path"]
            raw = path.read_bytes()
            self.assertEqual(len(raw), row["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])

    def test_exact_table_denominators(self) -> None:
        expected = {
            "slice-arm-clamp.csv": 18,
            "request-endpoint-train.csv": 180,
            "paired-endpoint-deltas.csv": 90,
            "slice-microstep-summary.csv": 90,
            "request-microsteps.csv": 900,
            "evaluator-request.csv": 540,
            "evaluator-paired-deltas.csv": 180,
            "compute.csv": 27,
            "slurm-accounting.csv": 9,
            "clamp-metric-relations.csv": 6,
            "clamp-strata.csv": 12,
        }
        for name, denominator in expected.items():
            with (REPORT_ROOT / name).open(encoding="utf-8", newline="") as handle:
                self.assertEqual(sum(1 for _ in csv.DictReader(handle)), denominator)

    def test_report_leads_with_required_limit_and_pause(self) -> None:
        report = (REPORT_ROOT / "report-ko.md").read_text(encoding="utf-8")
        self.assertIn("projection boundary saturation", report[:700])
        self.assertIn("Z+는 286/450", report[:700])
        self.assertIn("Z±는 381/450", report[:700])
        self.assertIn("USER_DIRECTED_POST_CALIBRATION_HYPERPARAMETER_REVISION", report)
        self.assertIn("IDLE_AWAITING_GH_CALL", report)


if __name__ == "__main__":
    unittest.main()
