import copy
import unittest
from official.evaluation.generation.paper_display import paper_cell, paper_generation


class PaperDisplay(unittest.TestCase):
    def test_unrounded_endpoints(self):
        for flu, con, expected in (
            (6.352242334333923, .24636896048599818, ("635.22", "24.64")),
            (6.252105796227186, .2591242773267912, ("625.21", "25.91")),
            (4.71017497777678, .030072135827285053, ("471.02", "3.01")),
        ):
            self.assertEqual(paper_cell(flu, metric="Flu", raw_unit="bits"), expected[0])
            self.assertEqual(paper_cell(con, metric="Con", raw_unit="cosine_0_to_1"), expected[1])

    def test_display_cannot_be_reapplied(self):
        with self.assertRaises(ValueError):
            paper_cell("635.22", metric="Flu", raw_unit="bits")
        with self.assertRaises(ValueError):
            paper_cell(635.22, metric="Flu", raw_unit="paper_x100")
        with self.assertRaises(ValueError):
            paper_cell(24.64, metric="Con", raw_unit="cosine_0_to_1")

    def test_missing_and_deferred(self):
        self.assertEqual(paper_cell("DEFERRED", metric="Flu", raw_unit="bits"), "DEFERRED")
        self.assertIsNone(paper_cell(None, metric="Con", raw_unit="cosine_0_to_1"))
        self.assertEqual(paper_generation({}), {})

    def test_summary_nonmutation(self):
        raw = dict(ngram_entropy=6.352242334333923, reference_score=.24636896048599818,
                   fluency_unit="bits", consistency_unit="cosine_0_to_1",
                   fluency_count=2000, consistency_count=2000)
        before = copy.deepcopy(raw)
        self.assertEqual(paper_generation(raw), {"Flu_paper_x100": "635.22", "Con_paper_x100": "24.64"})
        self.assertEqual(raw, before)

    def test_invalid_and_unmeasured(self):
        for value in (True, float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                paper_cell(value, metric="Flu", raw_unit="bits")
        with self.assertRaises(ValueError):
            paper_generation(dict(ngram_entropy=0, fluency_count=0, fluency_unit="bits"))


if __name__ == "__main__":
    unittest.main()
