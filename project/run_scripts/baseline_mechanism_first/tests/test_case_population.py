import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from project.run_scripts.baseline_mechanism_first.case_population import (
    MATCHING, compare_layers, deterministic_gzip_jsonl, interval_status,
    interval_update, metadata, read_rows, sha_file, source_digest, summarize,
)
from project.run_scripts.baseline_mechanism_first.case_analysis import CaseIntegrityError, MatchingSpec
from project.run_scripts.baseline_mechanism_first.evidence import canonical_sha


class PopulationTests(unittest.TestCase):
    def row(self, case=1, margin=1., success=True, relation="P1"):
        return dict(metric="RS", case_id=case, prompt_index=0, identity=f"id{case}",
                    family="AlphaEdit", checkpoint_requests=10000, age_requests=9900,
                    atwrite_success=True, checkpoint_success=success,
                    atwrite_desired_margin=margin, desired_margin=1 if success else -1,
                    desired_margin_delta=0, new_nll=1, true_nll=2,
                    new_nll_delta=0, true_nll_delta=0, new_token_count=1,
                    relation=relation, active_at_checkpoint=True,
                    later_different_target_in_seen_prefix=False,
                    transition="retained" if success else "lost")

    def spec(self):
        return MatchingSpec(tuple(MATCHING["margin_cutpoints"]), tuple(MATCHING["age_cutpoints"]),
                            tuple(MATCHING["token_length_cutpoints"]), canonical_sha(MATCHING))

    def test_bins_do_not_use_final_outcomes(self):
        groups = {f"AlphaEdit_L{layer}_ONLY": [self.row(), self.row(2, 3.)] for layer in range(4,9)}
        _, _, _, before = compare_layers(groups, self.spec())
        for rows in groups.values():
            rows[0]["checkpoint_success"] = False
            rows[0]["desired_margin"] = -1000
            rows[0]["transition"] = "lost"
        _, _, _, after = compare_layers(groups, self.spec())
        self.assertEqual(before, after)

    def test_empty_bin_and_original_denominator(self):
        groups = {f"AlphaEdit_L{layer}_ONLY": [self.row(), self.row(2, 3.)] for layer in range(4,9)}
        groups["AlphaEdit_L8_ONLY"][1]["atwrite_desired_margin"] = -3
        summary, bins, _, support = compare_layers(groups, self.spec())
        self.assertEqual(support["groups"]["AlphaEdit_L8_ONLY"]["common_support_denominator"], 1)
        self.assertTrue(any(r["empty_bin"] and r["denominator"] == 0 for r in bins))
        self.assertTrue(all(r["denominator"] == 2 for r in summary if r["subset"] == "all"))

    def test_identity_mismatch_fails(self):
        groups = {f"AlphaEdit_L{layer}_ONLY": [self.row()] for layer in range(4,9)}
        groups["AlphaEdit_L8_ONLY"][0]["identity"] = "wrong"
        with self.assertRaises(CaseIntegrityError):
            compare_layers(groups, self.spec())

    def test_interval_censoring_not_hidden_failure(self):
        state = interval_update(None, dict(checkpoint_requests=100, success=True))
        state = interval_update(state, dict(checkpoint_requests=1000, success=False))
        self.assertEqual(state["first_observed_failure"], 1000)
        self.assertEqual(state["lower_success"], 100)
        self.assertEqual(interval_status(state), "INTERVAL_CENSORED_FAILURE")
        state = interval_update(state, dict(checkpoint_requests=5000, success=True))
        self.assertEqual(state["observed_recoveries"], 1)
        self.assertEqual(state["observations"], 3)

    def test_same_checkpoint_conflict_fails(self):
        state = interval_update(None, dict(checkpoint_requests=100, success=True))
        with self.assertRaises(CaseIntegrityError):
            interval_update(state, dict(checkpoint_requests=100, success=False))

    def test_metadata_exact_unicode_identity(self):
        record = dict(case_id=1, requested_rewrite=dict(subject="é", relation_id="P1", prompt="{} is",
                        target_new=dict(str="Paris"), target_true=dict(str="Rome")),
                      paraphrase_prompts=["one", "two"], neighborhood_prompts=[str(i) for i in range(10)])
        meta, ids = metadata([record])
        self.assertNotIn("subject", meta[1])
        self.assertEqual(ids["RS",1,0], source_digest([1,0,"é is","Paris","Rome"]))
        self.assertNotEqual(source_digest(["é"]), canonical_sha(["é"]))

    def test_reproduction_and_empty_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b = Path(directory)/"a.gz",Path(directory)/"b.gz"
            deterministic_gzip_jsonl(a, [dict(a=2)])
            deterministic_gzip_jsonl(b, [dict(a=2)])
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(json.loads(gzip.decompress(a.read_bytes())), dict(a=2))
        self.assertIsNone(summarize([])["final_rate"])

    def test_original_current_and_seen_wrappers(self):
        metrics, identities = {}, {}
        for metric,n in (("RS",1),("PS",2),("NS",10)):
            rows = []
            for i in range(n):
                identity = f"{metric}{i}"
                identities[metric,1,i] = identity
                rows.append(dict(case_id=1,prompt_index=i,identity=identity,new_nll=1.,true_nll=2.,
                                 margin=1.,success=metric != "NS"))
            metrics[metric] = dict(rows=rows)
        scope = dict(family="AlphaEdit",arm="AlphaEdit_L4_ONLY",layer=4,
                     checkpoint_requests=100,source_identity="source")
        with tempfile.TemporaryDirectory() as directory:
            for filename in ("current.json", "seen-full.json"):
                raw = dict(requests=1,metrics=metrics)
                if filename == "current.json":
                    raw.update(request_order=source_digest([1]),before_after_exact=True,evaluator_controller_influence=0)
                else:
                    raw.update(evaluation_type="CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS",current_rows_reused=True,
                               state=dict(weights=dict(weight="sha"),cache="sha"))
                path = Path(directory)/filename
                path.write_text(json.dumps(raw))
                member = dict(destination=str(path),source_path="/source/"+filename,
                              bytes=path.stat().st_size,sha256=sha_file(path))
                result = read_rows(member,[1],scope,identities,[])
                self.assertEqual(len(result),13)
                wrong = dict(identities)
                wrong["RS",1,0] = "wrong"
                with self.assertRaises(CaseIntegrityError):
                    read_rows(member,[1],scope,wrong,[])


if __name__ == "__main__":
    unittest.main()
