"""Presentation-only regression tests; no raw data, model, or GPU access.

Run: uv run --no-project --with markdown-it-py==4.2.0 python -B -m unittest
     project.run_scripts.bg_tw_reference.ep_tw.test_review_markdown
"""
import importlib.util
from pathlib import Path
import re
import unittest

from markdown_it import MarkdownIt


SOURCE = Path(__file__).resolve().parent / "review_nogate" / "build_publication.py"
SPEC = importlib.util.spec_from_file_location("ep_review_publication", SOURCE)
PUBLICATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLICATION)
WORKTREE = Path(__file__).resolve().parents[4]
REPORT = WORKTREE / (
    "experiment-reports/servers/server4/ep-tw1-c4-2026-09-15-v1/"
    "gate-skip-r1/completed-review-v1/diagnostic-report-ko.md"
)


class MarkdownTableTests(unittest.TestCase):
    def setUp(self):
        self.parser = MarkdownIt("commonmark").enable("table")

    def test_literal_pipes_in_headers_render_as_one_cell(self):
        text = PUBLICATION.md(["B", "||gE||", "||gD||"], [[1, 0.02677, 0.00201]])
        tokens = self.parser.parse(text)
        self.assertEqual(sum(t.type == "th_open" for t in tokens), 3)
        html = self.parser.render(text)
        self.assertIn("<th>||gE||</th>", html)
        self.assertIn("<th>||gD||</th>", html)

    def test_literal_pipes_in_data_are_preserved_not_replaced(self):
        text = PUBLICATION.md(["name", "formula"], [["D", "KL(p0||pV)"]])
        self.assertIn("<td>KL(p0||pV)</td>", self.parser.render(text))
        self.assertEqual(sum(t.type == "td_open" for t in self.parser.parse(text)), 2)

    def test_inline_code_pipes_do_not_split_cells(self):
        text = PUBLICATION.md(["formula"], [["`x|y`"]])
        self.assertIn("<td><code>x|y</code></td>", self.parser.render(text))

    def test_numeric_formatting_is_unchanged(self):
        self.assertEqual(
            PUBLICATION.md(["n", "x", "missing", "flag"], [[998, 0.026770689, None, True]]),
            "\n| n | x | missing | flag |\n| --- | --- | --- | --- |\n"
            "| 998 | 0.026770689 | NA | True |\n\n",
        )

    def test_every_published_table_renders_all_rows_and_columns(self):
        text = REPORT.read_text()
        blocks = re.findall(r"(?m)^\|[^\n]*\n(?:\|[^\n]*(?:\n|$))+", text)
        self.assertEqual(len(blocks), 16)
        self.assertEqual(sum(t.type == "table_open" for t in self.parser.parse(text)), 16)
        for index, block in enumerate(blocks, 1):
            with self.subTest(table=index):
                lines = block.strip().splitlines()
                width = len(lines[1].strip("|").split("|"))
                tokens = self.parser.parse(block)
                self.assertEqual(sum(t.type == "table_open" for t in tokens), 1)
                self.assertEqual(sum(t.type == "th_open" for t in tokens), width)
                self.assertEqual(sum(t.type == "tr_open" for t in tokens), len(lines) - 1)
                self.assertEqual(sum(t.type == "td_open" for t in tokens), width * (len(lines) - 2))
                # A renderer may silently discard surplus body cells, so check source width too.
                for line in lines:
                    self.assertEqual(len(re.split(r"(?<!\\)\|", line)[1:-1]), width)

    def test_all_norm_headers_keep_their_visible_math(self):
        html = self.parser.render(REPORT.read_text())
        for label in (
            "||gE||", "||gD||", "||C||", "||CA||", "||Vp−We|| recorded",
            "||Wsel−Vp||", "||Wsel−We|| CPU",
        ):
            with self.subTest(header=label):
                self.assertIn(f"<th>{label}</th>", html)


if __name__ == "__main__":
    unittest.main()
