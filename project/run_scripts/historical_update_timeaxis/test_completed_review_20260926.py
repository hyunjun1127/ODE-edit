"""CPU review arithmetic/censoring regression; no production runtime imports."""
import unittest
from .completed_review_20260926 import active,contribution_terms,pair_terms,rate,finite,digest,same


class ReviewTests(unittest.TestCase):
    def test_contribution_signs(self):
        self.assertEqual(contribution_terms(3.,1.,4.,0.),(2.,4.,-1.,1.,-2.))
    def test_negative_contribution_retained(self):
        ct,cb,dm,db,dc=contribution_terms(.2,3.,4.,0.)
        self.assertLess(ct,0);self.assertAlmostEqual(dm,db+dc)
    def test_lost_while_contribution_increased(self):
        ct,cb,dm,db,dc=contribution_terms(-1.,-4.,1.,0.)
        self.assertGreater(dc,0);self.assertLess(dm,0)
    def test_pair_interaction(self):
        self.assertEqual(pair_terms(4.,2.,3.,2.),(1.,0.,1.))
    def test_censor_first_conflict_permanent(self):
        f=dict(same_batch_conflict='False',first_later_conflict_batch='25')
        self.assertTrue(active(f,24));self.assertFalse(active(f,25));self.assertFalse(active(f,100))
    def test_same_batch_conflict(self):
        self.assertFalse(active(dict(same_batch_conflict='True',first_later_conflict_batch=''),1))
    def test_redundancy_not_censored(self):
        self.assertTrue(active(dict(same_batch_conflict='False',first_later_conflict_batch='',repeated_group='True'),100))
    def test_zero_denominator(self):
        self.assertIsNone(rate(0,0));self.assertEqual(rate(0,4),0.)
    def test_nonfinite_not_diagnostic_warning(self):
        for x in (float('inf'),float('nan')):
            with self.assertRaises(AssertionError):contribution_terms(x,0.,0.,0.)
    def test_identity_not_approximate(self):
        self.assertNotEqual(digest([1,2]),digest([2,1]))
        with self.assertRaises(AssertionError):same(1.,1.00000001,'IDENTITY')


if __name__=='__main__':unittest.main()
