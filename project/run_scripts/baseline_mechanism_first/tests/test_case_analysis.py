from copy import deepcopy
import unittest

from project.run_scripts.baseline_mechanism_first.case_analysis import (
    CaseIntegrityError, MatchingSpec, common_support, exact_pair, fact_versions,
    first_observed_failure, normalize_row, rows_from_evaluation, attach_case_metadata,
)


def row(case=1, metric="RS", margin=2., checkpoint=100, arm="L4", identity=None):
    original = dict(case_id=case, prompt_index=0, identity=identity or f"prompt-{case}-{metric}",
                    new_nll=3., true_nll=3.+margin, margin=margin,
                    success=(-margin if metric == "NS" else margin)>0)
    return normalize_row(original, metric=metric, metadata=dict(
        family="AlphaEdit", arm=arm, layer=int(arm[1:]),
        checkpoint_requests=checkpoint, source_identity="sealed-source"))


class CaseTests(unittest.TestCase):
    def test_raw_ns_margin_preserved_and_ties_fail(self):
        out = row(metric="NS", margin=-2)
        self.assertEqual(out["margin"], -2)
        self.assertEqual(out["raw_margin"], -2)
        self.assertEqual(out["desired_margin"], 2)
        self.assertTrue(out["success"])
        for metric in ("RS", "PS", "NS"):
            self.assertFalse(row(metric=metric, margin=0)["success"])

    def test_nonfinite_bad_margin_and_aggregate_fail(self):
        valid = row(); metadata = {k:valid[k] for k in ("family", "arm", "layer", "checkpoint_requests", "source_identity")}
        for patch in (dict(margin=float("nan")), dict(margin=99), dict(success=False)):
            bad = dict(valid, **patch)
            with self.assertRaises(CaseIntegrityError): normalize_row(bad,metric="RS",metadata=metadata)
        with self.assertRaises(CaseIntegrityError): normalize_row(dict(numerator=1, denominator=1),metric="RS",metadata=metadata)

    def test_exact_pair_rejects_identity_and_source_mismatch(self):
        a = row(); b = row(checkpoint=1000)
        for key in ("identity", "source_identity"):
            bad = dict(b, **{key:"other"})
            with self.assertRaises(CaseIntegrityError): exact_pair([a],[bad])
        with self.assertRaises(CaseIntegrityError): exact_pair([a,a],[b])

    def test_pair_missing_inventory_not_imputed(self):
        pairs, receipt = exact_pair([row(),row(2)], [row(checkpoint=1000,margin=-1)])
        self.assertEqual(receipt["joined"],1)
        self.assertEqual(len(receipt["missing_checkpoint"]),1)
        self.assertEqual(pairs[0]["transition"],"lost")
        self.assertEqual(pairs[0]["age_requests"],900)
        self.assertEqual(receipt["atwrite_denominator"],2)
        self.assertEqual(receipt["imputation"],0)

    def test_prefix_versions_do_not_use_future_overwrites(self):
        records=[dict(case_id=i+1,ordinal=i,subject="  Cafe\u0301  ",relation="r",target_new_sha256=t)
                 for i,t in enumerate(["a","a","b"])]
        first=fact_versions(records,checkpoint_requests=1)
        self.assertTrue(first[0]["active_at_checkpoint"])
        all_seen=fact_versions(records,checkpoint_requests=3)
        self.assertEqual([r["active_at_checkpoint"] for r in all_seen],[False,False,True])
        self.assertTrue(all_seen[0]["later_different_target_in_seen_prefix"])
        self.assertEqual(len({r["fact_identity"] for r in all_seen}),1)

    def test_case_metadata_join_is_exact_and_raw_free(self):
        metadata=[dict(case_id=1,ordinal=0,relation="r",fact_identity="sealed-fact",prompt="not published")]
        out=attach_case_metadata([row()],case_metadata=metadata,sample_identity="sealed-sample")
        self.assertEqual(out[0]["ordinal"],0)
        self.assertNotIn("prompt",out[0])
        with self.assertRaises(CaseIntegrityError):
            attach_case_metadata([row(2)],case_metadata=metadata,sample_identity="sealed-sample")
        with self.assertRaises(CaseIntegrityError):
            attach_case_metadata([row()],case_metadata=[dict(metadata[0],ordinal=100)],sample_identity="sealed-sample")

    def test_failure_is_observed_interval_not_inferred_event(self):
        result=first_observed_failure([row(checkpoint=100),row(checkpoint=1000),row(checkpoint=5000,margin=-1),row(checkpoint=9000)])
        self.assertEqual(result["lower_last_observed_success"],1000)
        self.assertEqual(result["upper_first_observed_failure"],5000)
        self.assertEqual(result["observed_recovery_checkpoints"],[9000])
        self.assertEqual(result["unobserved_transitions_imputed"],0)
        with self.assertRaises(CaseIntegrityError): first_observed_failure([row(),row(arm="L8",checkpoint=1000)])

    def test_empty_and_initial_failure_not_forgetting(self):
        self.assertEqual(first_observed_failure([])["status"],"NO_OBSERVATIONS")
        self.assertEqual(first_observed_failure([row(margin=-1)])["status"],"FIRST_OBSERVATION_FAILED")

    def test_common_support_separate_primary_and_common_success(self):
        groups={}
        for arm, margins in (("L4",[1.,-1.,10.]),("L8",[1.,1.,2.])):
            rows=[]
            for case, margin in enumerate(margins,1):
                pairs,_=exact_pair([row(case,margin=margin,arm=arm)], [row(case,margin=-3.,arm=arm,checkpoint=1000)])
                pairs[0].update(relation="r",new_token_count=1);rows+=pairs
            groups[arm]=rows
        original=deepcopy(groups)
        spec=MatchingSpec((0.,5.),(1000,),(2,),"predeclared-lock")
        out=common_support(groups,spec=spec)
        self.assertEqual(out["groups"]["L4"]["all_population_denominator"],3)
        self.assertEqual(out["groups"]["L4"]["common_atwrite_success_denominator"],2)
        self.assertEqual(out["groups"]["L4"]["common_support_denominator"],1)
        self.assertEqual(out["groups"]["L8"]["common_support_denominator"],3)
        self.assertFalse(out["matching_uses_final_outcomes"])
        self.assertEqual(groups,original)
        for rows in groups.values():
            for r in rows:r["checkpoint_success"]=True;r["desired_margin"]=999
        self.assertEqual(common_support(groups,spec=spec),out)

    def test_matching_requires_predeclared_sorted_cutpoints(self):
        with self.assertRaises(CaseIntegrityError): MatchingSpec((1.,0.),(),(),"sealed")
        with self.assertRaises(CaseIntegrityError): MatchingSpec((),(),(),"")


if __name__ == "__main__": unittest.main()
