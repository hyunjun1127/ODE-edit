import unittest
from .review_b010 import pairs, summarize, transitions

def rows(panel='continuation',kind='R',new=1.,true=2.):
    return [dict(pair_id='x',row_id=label,case_id=1,role=panel,kind=kind,prompt_index=0,checkpoint='B010',label=label,
        nll=nll,token_nll=[nll],token_predictions=[2],target_count=1,token_correct=1,strict=True,
        input_sha=label,target_sha=label,position_sha='p') for label,nll in [('true',true),('new',new)]]

class Review(unittest.TestCase):
    def test_tie_failure(self):self.assertEqual(summarize(pairs(rows(new=2.)))[0]['numerator'],0)
    def test_direction(self):
        self.assertTrue(next(iter(pairs(rows()).values()))['success'])
        self.assertFalse(next(iter(pairs(rows('base_observer')).values()))['success'])
        self.assertFalse(next(iter(pairs(rows(kind='N')).values()))['success'])
    def test_transitions(self):
        self.assertEqual(transitions(pairs(rows()),pairs(rows(new=3.)))[0]['lost'],1)
    def test_identity(self):
        b=rows();b[0]['input_sha']='wrong'
        with self.assertRaises(AssertionError):transitions(pairs(rows()),pairs(b))
    def test_duplicate(self):
        r=rows()
        with self.assertRaises(AssertionError):pairs(r+[r[0]])
    def test_nonfinite(self):
        with self.assertRaises(AssertionError):pairs(rows(new=float('nan')))

if __name__=='__main__':unittest.main()
