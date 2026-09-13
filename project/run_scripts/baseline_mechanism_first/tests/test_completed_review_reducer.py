import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).parents[1]/"completed_review_reducer.py"
spec = importlib.util.spec_from_file_location("completed_review_reducer", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def row(cid=1, new=1., true=2., metric="RS", identity="sealed"):
    return dict(case_id=cid, prompt_index=0, identity=identity, new_nll=new,
                true_nll=true, margin=true-new,
                success=(new > true if metric == "NS" else new < true),
                new_token_correct=1, new_token_count=1, new_strict=True,
                true_token_correct=0, true_token_count=1, true_strict=False)


class CompletedReducerTests(unittest.TestCase):
    def test_tie_fails_both_directions(self):
        for metric in ("RS", "PS", "NS"):
            self.assertFalse(next(iter(m.index_rows([row(new=1., true=1., metric=metric)], metric).values()))["success"])

    def test_wrong_direction_rejected(self):
        r = row(); r["success"] = False
        with self.assertRaises(m.IntegrityError): m.index_rows([r], "RS")

    def test_nonfinite_rejected(self):
        with self.assertRaises(m.IntegrityError): m.index_rows([row(new=float("nan"))], "RS")

    def test_duplicate_rejected(self):
        with self.assertRaises(m.IntegrityError): m.index_rows([row(), row()], "RS")

    def test_target_identity_mismatch_rejected(self):
        with self.assertRaises(m.IntegrityError): m.index_rows([row()], "RS", {("RS", 1, 0):"other"})

    def test_token_consistency_rejected(self):
        r = row(); r["new_strict"] = False
        with self.assertRaises(m.IntegrityError): m.index_rows([r], "RS")

    def test_keyed_pair_not_positional(self):
        a = m.index_rows([row(1), row(2, new=3.)], "RS")
        b = m.index_rows([row(2), row(1, new=3.)], "RS")
        p, t, _ = m.reduce_pair(a, b, endpoint="x", panel="p", metric="RS")
        self.assertEqual((p["delta_pp"], t["lost"], t["gained"]), (0, 1, 1))

    def test_same_position_other_identity_not_pairable(self):
        a = m.index_rows([row()], "RS"); b = m.index_rows([row(identity="other")], "RS")
        with self.assertRaises(m.IntegrityError): m.reduce_pair(a, b, endpoint="x", panel="p", metric="RS")

    def test_ns_desired_sign(self):
        self.assertEqual(next(iter(m.index_rows([row(metric="NS")], "NS").values()))["desired_margin"], -1.)

    def test_quantile_linear(self):
        self.assertAlmostEqual(m.stats([0, 10])["p95"], 9.5)

    def test_empty_subset_no_fabricated_rate(self):
        p, t, d = m.reduce_pair({}, {}, endpoint="x", panel="p", metric="NS")
        self.assertIsNone(p["delta_pp"])
        self.assertEqual(p["status"], "EMPTY_METADATA_SUBSET")

    def test_overwrite_metadata_latest_wins(self):
        meta = {1:dict(ordinal=0, fact=("a", "r"), target="x"),
                2:dict(ordinal=1, fact=("a", "r"), target="y"),
                3:dict(ordinal=2, fact=("a", "r"), target="y")}
        self.assertEqual(m.versions(meta, 3), {1:"superseded_conflicting_target", 2:"superseded_same_target", 3:"active"})


if __name__ == "__main__": unittest.main()
