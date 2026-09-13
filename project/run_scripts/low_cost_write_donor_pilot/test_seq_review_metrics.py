"""Essential CPU contracts for independent seq10 reduction."""
import unittest
from .seq_review_metrics import (annotation, validate, digest, panel_digest, expected_rows,
    aggregate, pairs, distribution, outcome, subset)

def record(i=1,target='new'):
    return dict(case_id=i,requested_rewrite=dict(subject='subject',relation_id='P1',prompt='{} is',target_new={'str':target},target_true={'str':'old'}),paraphrase_prompts=['p1','p2'],neighborhood_prompts=['n'+str(k) for k in range(10)])

def row(k,new=1.,true=2.,metric='RS'):
    return dict(case_id=k[0],prompt_index=k[1],identity=k[2],new_nll=new,true_nll=true,margin=true-new,desired_margin=new-true if metric=='NS' else true-new,success=true<new if metric=='NS' else new<true,new_strict=True,true_strict=False,new_token_correct=1,new_token_count=1,true_token_correct=0,true_token_count=1)

class Tests(unittest.TestCase):
    def test_directions_and_ties(self):
        for metric in ('RS','PS','NS'):
            self.assertFalse(outcome(row((1,0,'x'),1.,1.,metric),metric))
        self.assertTrue(outcome(row((1,0,'x')),'RS'))
        self.assertFalse(outcome(row((1,0,'x')),'NS'))
    def test_identity_cardinality_and_bit_hash(self):
        rec=[record()];doc=dict(requests=1,request_order=digest([1]),endpoint_state_sha256='end',metrics={})
        for m in ('RS','PS','NS'):
            rr=[row(k,metric=m) for k in expected_rows(rec,m)];a=aggregate(rr,m)
            doc['metrics'][m]=dict(rows=rr,numerator=a['numerator'],denominator=len(rr),rate=a['rate'],bit_order_sha256=a['bit_order_sha256'])
        validate(doc,rec,'end')
        doc['metrics']['NS']['rows'][0]['identity']='wrong'
        with self.assertRaises(ValueError):validate(doc,rec,'end')
    def test_prompt_not_cluster_success(self):
        rr=[row((1,0,'a'),0.,1.,'PS'),row((1,1,'b'),100.,2.,'PS')]
        a=aggregate(rr,'PS');self.assertEqual(a['numerator'],1);self.assertEqual(a['prompt_denominator'],2);self.assertEqual(a['request_all_success_numerator'],0)
    def test_paired_counts_and_order_fail_close(self):
        a=[row((1,0,'a')),row((2,0,'b'),3.,2.)];b=[row((1,0,'a'),4.,2.),row((2,0,'b'))]
        p=pairs(a,b,'RS');self.assertEqual((p['lost'],p['gained'],p['delta_numerator']),(1,1,0))
        with self.assertRaises(ValueError):pairs(a,list(reversed(b)),'RS')
    def test_annotation_exact_target_not_all_repeat(self):
        r=[record(1,'a'),record(2,'a'),record(3,'b')]
        self.assertEqual(annotation(r,2)[1],'ACTIVE_TARGET')
        self.assertEqual(annotation(r,3)[1],'SUPERSEDED')
    def test_quantiles_and_empty(self):
        self.assertEqual(distribution([0.,10.])['p95'],9.5)
        self.assertIsNone(distribution([])['mean'])
        with self.assertRaises(ValueError):distribution([float('nan')])
    def test_subset_not_recounting(self):
        a=[row((i,0,str(i))) for i in range(6)]
        self.assertEqual(len(subset(a,[1,2])),2)

if __name__=='__main__':unittest.main()
