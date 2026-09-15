"""Reducer fixtures only: no native method/model/GPU numerical validation."""
import unittest
from .review_nogate import summary, success, transition, quantile

def row(case=1, prompt=0, identity='a', new=1., true=2., ns=True, ts=False):
    return dict(case_id=case,prompt_index=prompt,identity=identity,new_nll=new,true_nll=true,
                new_strict=ns,true_strict=ts,new_token_correct=int(ns),new_token_count=1,
                true_token_correct=int(ts),true_token_count=1)

class ReducerTests(unittest.TestCase):
    def test_ties_fail_all(self):
        for m in ('RS','PS','NS'):self.assertFalse(success(row(new=1,true=1),m))
    def test_NS_opposite_inequality(self):
        self.assertTrue(success(row(new=3,true=2),'NS'));self.assertFalse(success(row(new=3,true=2),'RS'))
    def test_strict_is_not_NLL_preference(self):
        s=summary([row(ns=False)],'RS');self.assertEqual(s['numerator'],1);self.assertEqual(s['desired_strict_n'],0)
    def test_two_rephrases_cluster_request(self):
        s=summary([row(),row(prompt=1,identity='b',ns=False)],'PS')
        self.assertEqual((s['denominator'],s['request_denominator'],s['all_prompt_strict_requests']),(2,1,0))
    def test_same_count_not_same_success_ID(self):
        a=[row(),row(case=2,new=3)];b=[row(new=3),row(case=2)]
        t=transition(a,b,'RS');self.assertEqual((t['before_num'],t['after_num'],t['lost'],t['gained']),(1,1,1,1))
    def test_pair_identity_guard(self):
        with self.assertRaises(AssertionError):transition([row()],[row(identity='different')],'RS')
    def test_duplicate_guard(self):
        with self.assertRaises(AssertionError):summary([row(),row()],'RS')
    def test_nonfinite_guard(self):
        with self.assertRaises(AssertionError):summary([row(new=float('nan'))],'RS')
    def test_quantile_linear(self):self.assertEqual(quantile([0,10],.95),9.5)
    def test_conditional_denominators(self):
        t=transition([row(),row(case=2,new=3)],[row(new=3),row(case=2)],'RS')
        self.assertEqual((t['conditional_loss_den'],t['conditional_recovery_den']),(1,1))

if __name__=='__main__':unittest.main()
