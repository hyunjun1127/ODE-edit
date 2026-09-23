import unittest
from project.run_scripts.alpha_key_concentration_causal.reduce import independent_pairs

def row(nll):return dict(case_id=1,prompt_index=0,prompt='x',target='y',nll=nll)
class IndependentReducer(unittest.TestCase):
    def test_rewrite_tie_failure(self):
        self.assertEqual(independent_pairs({'rewrite_target_new':[row(1.)],'rewrite_target_true':[row(1.)]})['RS']['numerator'],0)
    def test_neighborhood_reverse(self):
        self.assertEqual(independent_pairs({'locality_target_new':[row(2.)],'locality_target_true':[row(1.)]})['NS']['numerator'],1)
    def test_nonfinite_is_error(self):
        with self.assertRaises(AssertionError):independent_pairs({'rewrite_target_new':[row(float('nan'))],'rewrite_target_true':[row(1.)]})
    def test_denominator_not_filled(self):
        with self.assertRaises(AssertionError):independent_pairs({'rewrite_target_new':[row(1.)],'rewrite_target_true':[]})
