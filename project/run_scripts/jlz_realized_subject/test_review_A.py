"""Regression checks for the independent CPU reviewer; no model/GPU imports."""
import copy
import unittest
from .review_A import reduce_rows,validate_identity,pair,active_ids,distribution


def row(cid,kind,i=0,n=1,c=1,new=1.,true=2.):
    return dict(case_id=cid,kind=kind,prompt_index=i,identity=f'{cid}-{kind}-{i}',
        new_nll=new,true_nll=true,margin_true_minus_new=true-new,
        new_token_correct=c,new_token_count=n,new_strict=c==n,new_token_identity=f'new-{cid}-{kind}-{i}',
        true_token_correct=0,true_token_count=n,true_strict=False,true_token_identity=f'true-{cid}-{kind}-{i}')


def complete(cid):
    return [row(cid,'R')]+[row(cid,'P',i) for i in range(2)]+[row(cid,'N',i) for i in range(10)]

class ReviewerTests(unittest.TestCase):
    def test_tie_failure_and_true_desired(self):
        r=reduce_rows([row(1,'R',new=2.,true=2.),row(1,'N')])
        self.assertEqual(r['R']['ties'],1);self.assertEqual(r['R']['numerator'],0)
        self.assertEqual(r['N']['numerator'],0);self.assertEqual(r['N']['desired_token_correct'],0)

    def test_micro_macro_strict_are_distinct(self):
        r=reduce_rows([row(1,'R'),row(2,'R',n=3,c=0)])['R']
        self.assertEqual(r['token_micro'],.25);self.assertEqual(r['prompt_macro'],.5);self.assertEqual(r['strict_numerator'],1)

    def test_reject_nonfinite_duplicate_bad_strict(self):
        r=row(1,'R')
        for rows in [[r,r],[dict(r,new_nll=float('nan'))],[dict(r,new_strict=False)],[dict(r,new_token_correct=2)]]:
            with self.assertRaises(ValueError):reduce_rows(rows)

    def test_reference_family_major_only_is_canonicalized(self):
        actual=complete(9)+complete(3)
        reference=sorted(actual,key=lambda r:(r['kind'],r['case_id'],r['prompt_index']))
        validate_identity(actual,reference,[9,3])
        for corrupted in [actual[::-1],actual[:-1]]:
            with self.assertRaises(ValueError):validate_identity(corrupted,reference,[9,3])
        wrong=copy.deepcopy(actual);wrong[0]['new_token_identity']='wrong'
        with self.assertRaises(ValueError):validate_identity(wrong,reference,[9,3])

    def test_equal_total_different_ids_preserved(self):
        b=[row(1,'R'),row(2,'R',new=3.)];a=[row(1,'R',new=3.),row(2,'R')]
        summary,ids=pair(b,a,'x');p=summary[0]
        self.assertEqual((p['before'],p['after'],p['lost'],p['gained']),(1,1,1,1))
        self.assertEqual(ids['x/R/PREFERENCE']['lost'][0]['case_id'],1)
        with self.assertRaises(ValueError):pair(b,a[:1],'x')

    def test_active_uses_w5_not_future_field(self):
        def rec(cid,target):return dict(case_id=cid,requested_rewrite=dict(subject='Subject',relation_id='r',target_new=dict(id=target,str=target)))
        self.assertEqual(active_ids([rec(1,'A'),rec(2,'A'),rec(3,'B')]),{3})
        self.assertEqual(active_ids([rec(1,'A'),rec(2,'A')]),{1,2})

    def test_quantiles_linear_and_nonfinite_rejected(self):
        self.assertEqual(distribution([0,10])['p95'],9.5)
        with self.assertRaises(ValueError):distribution([float('inf')])

if __name__=='__main__':unittest.main()
