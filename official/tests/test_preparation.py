import copy
import json
from pathlib import Path
import tempfile
import unittest

from official.experiments.prepare import (build_matrix, load_plan, normalize_records,
                                         prepare_stream, write_new)
from official.evaluation.reduce import counterfact, zsre
from official.tools.verify import verify


class PreparationTests(unittest.TestCase):
    def test_scope_and_grid_reuse(self):
        c, p = load_plan()
        rows = build_matrix(c, p)
        self.assertEqual(sum(r["group"] == "main" for r in rows), 36)
        self.assertEqual(len({r["run_id"] for r in rows}), 41)
        self.assertEqual(sum(r["execution_kind"] == "new_chain" for r in rows), 40)
        self.assertNotIn("PRICE", {r["method"] for r in rows})
        grid = [r for r in rows if r["group"] == "qwen_grid"]
        self.assertEqual([r["hparams"]["L2"] for r in grid], [1, 10, 95])
        self.assertTrue(all(r["selection_dependency"] is None for r in grid))

    def test_upstream_integrity_and_source_boundary(self):
        self.assertEqual(verify()["external_task_imports"], 0)

    def test_zsre_subject_mapping_preserves_middle_position(self):
        raw = dict(subject="Ada", src="Where did Ada live?", answers=["London"],
                   alt="Paris", rephrase="Ada lived where?", loc="nq question: Country", loc_ans="UK")
        row = normalize_records([raw], "zsre", 1)[0]
        self.assertEqual(row["requested_rewrite"]["prompt"], "Where did {} live?")
        self.assertEqual(row["requested_rewrite"]["target_new"]["str"], "London")
        self.assertIsNone(row["specificity_W0_predictions"])
        with self.assertRaisesRegex(ValueError, "AMBIGUOUS"):
            normalize_records([dict(raw, src="Ada knows Ada")], "zsre", 1)

    def test_stream_order_and_hash_refuse_silent_change(self):
        raw = [dict(case_id=x, requested_rewrite=dict(prompt="{} is", subject="A",
                   target_new={"str": "B"}), paraphrase_prompts=["A was"],
                   neighborhood_prompts=["C is"]) for x in [8, 3]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.json"
            source.write_text(json.dumps(raw))
            receipt = prepare_stream(source, "cf", root / "out", count=2, batch_size=1)
            self.assertEqual(receipt["batches"][1]["start_inclusive"], 1)
            self.assertEqual([r["case_id"] for r in json.loads((root / "out/cf-stream.json").read_text())], [8, 3])
            raw.reverse()
            source.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "SOURCE_SHA"):
                prepare_stream(source, "cf", root / "out", receipt["source_sha256"], 2, 1)

    def test_immutable_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            write_new(path, {"x": 1})
            write_new(path, {"x": 1})
            with self.assertRaisesRegex(ValueError, "REFUSE_OVERWRITE"):
                write_new(path, {"x": 2})

    def test_cf_macro_aggregation_ties_and_harmonic(self):
        case = dict(rewrite_prompts_probs=[dict(target_new=1., target_true=2.)],
                    paraphrase_prompts_probs=[dict(target_new=1., target_true=2.)],
                    neighborhood_prompts_probs=[dict(target_new=2., target_true=1.)])
        other = copy.deepcopy(case)
        other["paraphrase_prompts_probs"] = [dict(target_new=2., target_true=2.)] * 3
        out = counterfact([case, other])
        self.assertEqual(out["Generalization"], 50)
        self.assertEqual(out["Score"], 75)
        bad = copy.deepcopy(case)
        bad["rewrite_prompts_probs"][0]["target_new"] = float("nan")
        with self.assertRaises(ValueError):
            counterfact([bad])

    def test_zsre_W0_agreement_is_distinct_from_answer_accuracy(self):
        out = zsre([dict(rewrite_prompts_correct=[True, False], paraphrase_prompts_correct=[True],
                         neighborhood_W0_agreement=[True, True], neighborhood_prompts_correct=[False, False])])
        self.assertEqual(out["Specificity"], 0)
        self.assertEqual(out["W0_prediction_agreement"], 100)
        self.assertEqual(out["Specificity_loc_ans"], 0)


if __name__ == "__main__":
    unittest.main()
