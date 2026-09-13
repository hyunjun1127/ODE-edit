"""Small synthetic CPU tests; never import model/evaluator/runtime modules."""
import copy
import tempfile
import unittest
from pathlib import Path

from .review_metrics import (aggregate, check_rows, desired, digest, distribution,
                             expected_rows, group_rows, mmlu_prediction, outcome,
                             pairs, panel_digest, write_csv)


def row(case=1, index=0, new=1., true=2., metric='RS'):
    r=dict(case_id=case,prompt_index=index,identity=digest([case,index]),
           new_nll=new,true_nll=true,margin=true-new,
           new_strict=True,true_strict=False,new_token_correct=2,
           new_token_count=2,true_token_correct=1,true_token_count=2)
    r['desired_margin']=desired(r,metric)
    r['success']=outcome(r,metric)
    return r


class ReviewMetricsTests(unittest.TestCase):
    def test_inequality_and_tie_are_not_accuracy(self):
        self.assertTrue(outcome(row(), 'RS'))
        self.assertTrue(outcome(row(), 'PS'))
        self.assertFalse(outcome(row(), 'NS'))
        for tag in ('RS','PS','NS'):
            self.assertFalse(outcome(row(new=1,true=1), tag))
        self.assertEqual(desired(row(), 'NS'), -1.)

    def test_no_imputation_empty_group(self):
        a=aggregate([], 'PS')
        self.assertEqual(a['prompt_denominator'],0)
        self.assertIsNone(a['rate'])
        self.assertEqual(distribution([])['n'],0)
        self.assertIsNone(distribution([])['p99'])

    def test_pairing_full_identity_and_order(self):
        a=[row(1),row(2,new=3)]
        b=[row(1,new=3),row(2)]
        p=pairs(a,b,'RS')
        self.assertEqual((p['lost'],p['gained'],p['delta_numerator']),(1,1,0))
        self.assertEqual(p['conditional_loss_rate'],1.)
        with self.assertRaisesRegex(ValueError,'IDENTITY'):
            pairs(a,list(reversed(b)),'RS')
        c=copy.deepcopy(b);c[0]['identity']='different-target'
        with self.assertRaisesRegex(ValueError,'IDENTITY'):
            pairs(a,c,'RS')

    def test_duplicate_and_nonfinite_fail_close(self):
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):
            check_rows([row(),row()],'RS')
        x=row();x['new_nll']=float('nan')
        with self.assertRaisesRegex(ValueError,'NONFINITE'):
            check_rows([x],'RS')

    def test_strict_and_token_units(self):
        a=[row(1,0),row(1,1,new=3)]
        a[1]['new_strict']=False;a[1]['new_token_correct']=0
        check_rows(a,'PS')
        s=aggregate(a,'PS')
        self.assertEqual((s['prompt_denominator'],s['request_denominator']),(2,1))
        self.assertEqual(s['new_strict_numerator'],1)
        self.assertEqual(s['new_request_all_strict_numerator'],0)
        self.assertEqual(s['new_token_accuracy'],.5)
        self.assertEqual(s['request_all_success_numerator'],0)

    def test_percentiles_are_explicit_linear(self):
        d=distribution([0.,10.])
        self.assertEqual((d['median'],d['p90'],d['p99']),(5.,9.,9.9))
        self.assertEqual(d['iqr'],5.)

    def test_superseded_and_unicode_hash_conventions(self):
        x=row();x['historical_status']={'status':'SUPERSEDED'}
        self.assertEqual(len(group_rows([x],'SUPERSEDED')),1)
        self.assertNotEqual(digest('한글'),panel_digest('한글'))
        record={'case_id':3,'requested_rewrite':{'prompt':'{} x','subject':'한글','target_new':{'str':'new'},'target_true':{'str':'old'}},'paraphrase_prompts':['p','q'],'neighborhood_prompts':['n']*10}
        ids=expected_rows([record],'RS')
        self.assertEqual(ids[0],(3,0,digest([3,0,'한글 x','new','old'])))

    def test_mmlu_unique_max_tie_and_underflow(self):
        self.assertEqual(mmlu_prediction([.1,.2,.8,.3]),2)
        self.assertEqual(mmlu_prediction([.8,.8,.2,.3]),-1)
        self.assertEqual(mmlu_prediction([0.,0.,0.,0.]),-1)
        with self.assertRaises(ValueError):mmlu_prediction([1.,2.,3.,4.])

    def test_csv_is_deterministic_and_does_not_write_raw(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'aggregate.csv';data=[{'state':'N4','n':2,'d':3,'rate':2/3}]
            write_csv(p,data);first=p.read_bytes();write_csv(p,data)
            self.assertEqual(first,p.read_bytes())
            self.assertNotIn(b'prompt',first)


if __name__=='__main__':unittest.main()
