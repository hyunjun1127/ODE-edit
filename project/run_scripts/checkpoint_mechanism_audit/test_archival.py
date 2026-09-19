"""Model-free fixtures for independent archival arithmetic and selection."""
import unittest
import pandas as pd
from . import archival as a


def row(cid=1, metric="NS", batch=1, success=True, margin=1.0):
    return dict(arm=a.ARM,case_id=cid,metric_tag=metric,identity=f"{metric}-{cid}",checkpoint_batch=batch,arrival_batch=1,subject_relation_group=f"g{cid}",success=success,target_strict=success,safety_margin=margin,target_token_count=2,target_token_correct=2 if success else 1)


class ArchiveTests(unittest.TestCase):
    def test_registry_prompt_identity_and_raw_hash(self):
        raw={"case_id":7,"requested_rewrite":{"subject":"Subject","relation_id":"P1","prompt":"{} lives in","target_new":{"str":"New","id":"Q1"},"target_true":{"str":"True","id":"Q2"}},"paraphrase_prompts":["p0","p1"],"neighborhood_prompts":[f"n{i}" for i in range(10)]}
        req=raw["requested_rewrite"]
        record={"case_id":7,"ordinal":0,"batch_index":1,"batch_ordinal":0,"raw_record_sha256":a.digest(raw),"request_sha256":a.digest(req),"target_new_sha256":a.digest(req["target_new"]),"target_true_sha256":a.digest(req["target_true"]),"subject_relation_group":"g"}
        reg,ids=a.registry_from_sample({"records":[record]},[raw])
        ident=a.digest([7,0,"Subject lives in","New","True"])
        self.assertIn(("RS",ident),ids);self.assertEqual(len(ids),13)
        self.assertEqual(reg.iloc[0].arrival_batch,1)
        raw["neighborhood_prompts"][0]="corrupted"
        with self.assertRaises(ValueError):a.registry_from_sample({"records":[record]},[raw])

    def test_ns_sign_and_tie(self):
        r=dict(case_id=1,prompt_index=0,identity="x",new_nll=3.,true_nll=2.,margin=-1.,success=True,new_strict=False,true_strict=True,new_token_correct=1,new_token_count=2,true_token_correct=2,true_token_count=2)
        meta={("NS","x"):dict(case_id=1,prompt_index=0)}
        def metric(r):
            return dict(rows=[r],numerator=int(r["success"]),denominator=1,rate=float(r["success"]),bit_order_sha256=a.digest([("x",r["success"])]))
        out=a.validate_rows(metric(r),"NS",meta)[0]
        self.assertEqual(out["safety_margin"],1.)
        self.assertTrue(out["target_strict"])
        r.update(new_nll=2.,margin=0.,success=False)
        self.assertFalse(a.validate_rows(metric(r),"NS",meta)[0]["success"])
        r["success"]=True
        with self.assertRaises(ValueError):a.validate_rows(metric(r),"NS",meta)

    def test_metric_tag_prevents_identity_collision(self):
        left=pd.DataFrame([row(metric="RS"),row(metric="NS")]);left["identity"]="same"
        self.assertEqual(len(a.join_pair(left,left)),2)

    def test_dedup_checks_every_scalar(self):
        x={k:0 for k in a.ROW_SCALARS};y=x.copy()
        self.assertTrue(a.same_observation(x,y));y["true_token_correct"]=1
        self.assertFalse(a.same_observation(x,y))

    def test_loss_recovery_denominators(self):
        left=pd.DataFrame([row(1,success=True),row(2,success=True),row(3,success=False),row(4,success=False)])
        right=pd.DataFrame([row(1,success=True),row(2,success=False),row(3,success=True),row(4,success=False)])
        r=a.summarize_pair(a.join_pair(left,right))
        self.assertEqual([r[x] for x in ("n11","n10","n01","n00")],[1,1,1,1])
        self.assertEqual(r["loss_denominator"],2);self.assertEqual(r["gain_denominator"],2)

    def test_overwrite_and_same_batch_conflict_no_last_winner(self):
        reg=pd.DataFrame([dict(case_id=1,subject_relation_group="g",arrival_batch=1,target_new_sha256="a",batch_internal_conflict=False,group_versions=3),dict(case_id=2,subject_relation_group="g",arrival_batch=2,target_new_sha256="b",batch_internal_conflict=True,group_versions=3),dict(case_id=3,subject_relation_group="g",arrival_batch=2,target_new_sha256="c",batch_internal_conflict=True,group_versions=3)])
        free,active=a.conflict_masks(reg,1,2)
        self.assertFalse(any(free.values()));self.assertFalse(any(active.values()))
        free,active=a.conflict_masks(reg,1,1)
        self.assertTrue(free[1]);self.assertTrue(active[1]);self.assertFalse(active[2])

    def test_same_target_repeat_active_versions_together(self):
        reg=pd.DataFrame([dict(case_id=i,subject_relation_group="g",arrival_batch=b,target_new_sha256="a",batch_internal_conflict=False,group_versions=3) for i,b in ((1,1),(2,2),(3,2))])
        free,active=a.conflict_masks(reg,1,2)
        self.assertTrue(all(free.values()));self.assertFalse(active[1]);self.assertTrue(active[2]);self.assertTrue(active[3])

    def test_first_failure_uses_arrival_not_first_sparse(self):
        anchor=pd.DataFrame([row(batch=2)])
        anchor["arrival_batch"]=2
        obs=pd.DataFrame([dict(row(batch=b,success=s),arrival_batch=2) for b,s in ((2,True),(5,True),(10,False),(20,True))])
        r=a.first_failures(obs,anchor).iloc[0]
        self.assertEqual(r.failure_after_batch,5);self.assertEqual(r.failure_observed_by_batch,10)
        self.assertEqual(r.first_failure_status,"INTERVAL_CENSORED_FIRST_OBSERVED_FAILURE")

    def test_joint_requires_exact_three(self):
        rows=[row(metric="RS"),row(metric="PS"),dict(row(metric="PS"),identity="p1",success=False)]
        j=a.joint_rows(pd.DataFrame(rows));self.assertFalse(j.iloc[0].success)
        with self.assertRaises(ValueError):a.joint_rows(pd.DataFrame(rows[:2]))

    def test_bootstrap_reproducible_shared_case_draws(self):
        left=pd.DataFrame([row(i,success=True) for i in range(1,5)])
        right=pd.DataFrame([row(i,success=i%2==0) for i in range(1,5)])
        pair=a.join_pair(left,right)
        x=a.bootstrap_pair(pair,"case_id",200)
        self.assertEqual(x,a.bootstrap_pair(pair,"case_id",200))
        self.assertEqual(x["loss_rate_finite_replicates"],200)
        self.assertEqual(x["gain_rate_finite_replicates"],0)

    def test_same_identity_changed_token_count_blocks(self):
        left=pd.DataFrame([row()]);right=left.copy();right["target_token_count"]=3
        with self.assertRaises(ValueError):a.join_pair(left,right)

    def test_panel_fixed_first100_and_without_replacement(self):
        reg=pd.DataFrame([dict(case_id=i,subject_relation_group=f"g{i}",arrival_batch=1,target_new_sha256="a",batch_internal_conflict=False,group_versions=1) for i in range(100)])
        rows=[]
        for batch in (1,10,50,100):
            for cid in range(100):
                for pi in range(10):
                    r=row(cid,batch=batch,success=not(batch in (10,100) and cid<4),margin=(-1 if batch in (10,100) and cid<4 else 1))
                    r.update(identity=f"i-{cid}-{pi}",prompt_index=pi,relation="P1",new_token_count=2,true_token_count=2)
                    rows.append(r)
        panel,excluded=a.mechanism_panel(pd.DataFrame(rows),reg)
        self.assertEqual(len(panel),64)
        for _,part in panel.groupby(["interval_start","interval_end"]):
            self.assertEqual(part.identity.nunique(),32)
            self.assertEqual(set(part.population),{"FIRST100_NS1000"})
            self.assertEqual(part[part.selection_role=="matched_retained"].target_length_mismatch.sum(),0)
        self.assertEqual(len(excluded[excluded.reason=="PANEL_COUNTS"]),2)


if __name__=="__main__":unittest.main()
