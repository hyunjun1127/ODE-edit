"""CPU-only fixtures; never imports execution code or opens model weights."""
import math
import unittest
from .reduce import pair, preferred, summarize, quantile, digest


class Metrics(unittest.TestCase):
    def raw(self,new=.2,true=.8):
        def r(side,nll):
            return dict(case_id=1,prompt_index=0,prompt='fixture',target=side,kind='rewrite_target_'+side,
                        nll=nll,target_token_ids=[1,2],token_predictions=[1,3],token_correct=[True,False],all_tokens_correct=False)
        return r('new',new),r('true',true),(1,0,'fixture','new','true')

    def test_rewrite_rephrase_direction(self):
        self.assertTrue(preferred(.2,.8,'RS')); self.assertTrue(preferred(.2,.8,'PS'))
    def test_neighborhood_reverse_direction(self):
        self.assertFalse(preferred(.2,.8,'NS')); self.assertTrue(preferred(.8,.2,'NS'))
    def test_ties_fail_all(self):
        for k in ('RS','PS','NS'):self.assertFalse(preferred(1.,1.,k))
    def test_nonfinite_rejected(self):
        for v in (math.nan,math.inf,-math.inf):
            with self.assertRaises(ValueError):preferred(v,1.,'RS')
    def test_raw_identity(self):
        n,t,e=self.raw();r=pair(n,t,e,'RS');self.assertEqual(r['identity'],digest(list(e)))
    def test_prompt_mismatch(self):
        n,t,e=self.raw();t['prompt']='wrong'
        with self.assertRaises(ValueError):pair(n,t,e,'RS')
    def test_token_secondary(self):
        n,t,e=self.raw();r=pair(n,t,e,'RS');self.assertEqual(r['new_token_correct'],1);self.assertFalse(r['new_strict'])
    def test_false_token_bit_rejected(self):
        n,t,e=self.raw();t['token_correct']=[True,True]
        with self.assertRaises(ValueError):pair(n,t,e,'RS')
    def test_empty_tokens_rejected(self):
        n,t,e=self.raw();t.update(target_token_ids=[],token_predictions=[],token_correct=[],all_tokens_correct=True)
        with self.assertRaises(ValueError):pair(n,t,e,'RS')
    def test_negative_nll_rejected(self):
        n,t,e=self.raw(new=-.1)
        with self.assertRaises(ValueError):pair(n,t,e,'RS')
    def test_summary_counts(self):
        n,t,e=self.raw();s=summarize([pair(n,t,e,'RS')],'RS')
        self.assertEqual((s['numerator'],s['denominator'],s['new_token_den']),(1,1,2))
    def test_duplicate_rejected(self):
        n,t,e=self.raw();r=pair(n,t,e,'RS')
        with self.assertRaises(ValueError):summarize([r,r],'RS')
    def test_cluster_denominator_rejected(self):
        n,t,e=self.raw();r=pair(n,t,e,'RS')
        with self.assertRaises(ValueError):summarize([r],'PS')
    def test_linear_quantile(self):
        self.assertEqual(quantile([0,10],.9),9);self.assertEqual(quantile([2],.9),2)
    def test_incorrect_saved_success_rejected(self):
        n,t,e=self.raw();r=pair(n,t,e,'RS');r['success']=False
        with self.assertRaises(ValueError):summarize([r],'RS')


if __name__=='__main__':unittest.main()
