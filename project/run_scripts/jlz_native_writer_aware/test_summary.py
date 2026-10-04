"""Pure stdlib tests for publication; no model, fit, GPU, or evaluator."""
import json
from pathlib import Path
import tempfile
import unittest

from .summarize_results import create_or_reuse, csv_bytes, encoded, paired, stats


class SummaryTests(unittest.TestCase):
    def test_null_is_not_zero(self):
        result = stats([None, 0, 2])
        self.assertEqual((result["count"], result["undefined"], result["mean"]), (2, 1, 1))
        self.assertIsNone(stats([None])["mean"])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            stats([float("nan")])
        with self.assertRaises(ValueError):
            encoded({"x": float("inf")})

    def test_create_once_and_exact_reuse(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "summary.json"
            data = encoded({"나노초": 1791139100293025828})
            create_or_reuse(path, data)
            create_or_reuse(path, data)
            self.assertEqual(json.loads(path.read_bytes())["나노초"], 1791139100293025828)
            with self.assertRaises(ValueError):
                create_or_reuse(path, b"different")

    def test_csv_roundtrip(self):
        import csv
        import io
        data = csv_bytes([{"scope": "KL", "value": None}, {"scope": "R", "value": 0}])
        rows = list(csv.DictReader(io.StringIO(data.decode(), newline="")))
        self.assertEqual(rows[0]["value"], "")
        self.assertEqual(rows[1]["value"], "0")

    def test_paired_ties_failure_and_direction(self):
        before, after = [], []
        for kind in ("R", "P", "N"):
            row = dict(identity=kind, kind=kind, case_id=1, prompt_index=0,
                       new_token_identity="new", true_token_identity="true",
                       new_nll=2., true_nll=1., new_strict=False, true_strict=True)
            before.append(row)
            after.append(row | dict(new_nll=1., new_strict=True, true_strict=False))
        result = paired(before, after)
        self.assertEqual(result["R"]["preference"]["both_failure"], 1)
        self.assertEqual(result["N"]["preference"]["lost"], 1)
        self.assertEqual(result["R"]["strict"]["gained"], 1)

    def test_pair_identity_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            paired([{"identity": "one"}], [{"identity": "two"}])


if __name__ == "__main__":
    unittest.main()
