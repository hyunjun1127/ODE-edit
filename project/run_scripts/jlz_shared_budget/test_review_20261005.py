"""Regression tests for independent review: known-count fixtures and corruption rejection."""
import copy,math,unittest
from review_20261005 import reduce,paired
from review_20261005_tables import score

def row(i,kind='R',new=1.,true=2.,nc=1,nn=1,tc=0,tn=1):
    return dict(identity=str(i),kind=kind,new_nll=new,true_nll=true,new_token_correct=nc,new_token_count=nn,
                true_token_correct=tc,true_token_count=tn,new_strict=nc==nn,true_strict=tc==tn)

class ReviewTest(unittest.TestCase):
    def test_preference_ties_fail_and_N_true(self):
        a=[row(1),row(2,new=2.,true=2.),row(3,'N',new=3.,true=1.,tc=1)]
        s=reduce(a);self.assertEqual(s['R']['success'],1);self.assertEqual(s['R']['ties'],1)
        self.assertEqual(s['N']['success'],1);self.assertEqual(s['N']['desired_nll'],1.);self.assertEqual(s['N']['strict'],1.)
    def test_micro_macro_strict_are_distinct(self):
        s=reduce([row(1,nc=1,nn=1),row(2,nc=1,nn=3)])['R']
        self.assertEqual(s['token_micro'],.5);self.assertAlmostEqual(s['prompt_macro'],2/3);self.assertEqual(s['strict'],.5)
    def test_reject_corrupt(self):
        for field,value in [('new_nll',float('nan')),('new_nll',float('inf')),('new_token_correct',2),('new_token_count',0),('new_strict',False)]:
            r=row(1);r[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):reduce([r])
        with self.assertRaises(ValueError):reduce([row(1),row(1)])
    def test_paired_lost_gained_retained(self):
        a=[row(1),row(2),row(3,new=3.)];b=[row(1,new=3.),row(2),row(3)]
        r=paired(a,b)['R'];self.assertEqual((r['before'],r['after'],r['lost'],r['gained'],r['retained']),(2,2,1,1,1))
    def test_paired_identity_corruption(self):
        for b in [[row(2)],[row(1),row(1)],[row(1,'N')]]:
            with self.assertRaises(ValueError):paired([row(1)],b)
    def test_harmonic_known_exact_and_zero(self):
        self.assertAlmostEqual(score([.5,1,1]),.75);self.assertEqual(score([0,1,1]),0)
        self.assertAlmostEqual(100*score([1970/2000,3612/4000,15212/20000]),87.27527506508585)
    def test_harmonic_missing_nonfinite_invalid(self):
        for values in [[1,1],[1,1,float('nan')],[1,1,1.01],[-.1,1,1]]:
            with self.assertRaises(ValueError):score(values)

if __name__=='__main__':unittest.main()
