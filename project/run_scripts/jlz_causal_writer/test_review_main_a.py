"""CPU-only reducer regressions; no model imports."""
import copy
import unittest
from .review_main_a import reduce, paired, success


class ReducerTest(unittest.TestCase):
    def row(self, kind='R'):
        return dict(identity='x',case_id=1,kind=kind,prompt_index=0,
            true_nll=1.0,new_nll=1.0,true_token_count=2,new_token_count=2,
            true_token_correct=1,new_token_correct=2,true_strict=False,new_strict=True,
            true_token_identity='t',new_token_identity='n')

    def test_ties_fail_and_tf_is_separate(self):
        for kind in 'RPN':
            r=self.row(kind);self.assertFalse(success(r))
            s=reduce([r])[kind]
            self.assertEqual(s['numerator'],0)
            self.assertEqual(s['token_micro'],.5 if kind=='N' else 1)

    def test_invalid_rows_block(self):
        r=self.row()
        with self.assertRaises(ValueError):reduce([r,r])
        for key,value in [('new_nll',float('nan')),('new_nll',float('inf')),
                          ('new_token_count',0),('new_strict',False)]:
            bad=copy.deepcopy(r);bad[key]=value
            with self.assertRaises(ValueError):reduce([bad])

    def test_paired_token_identity_block(self):
        before=[self.row(k) for k in 'RPN']
        for r in before:r['identity']=r['kind']
        after=copy.deepcopy(before);after[0]['new_token_identity']='changed'
        with self.assertRaises(ValueError):paired(before,after)

    def test_paired_lost_gained(self):
        before=[self.row(k) for k in 'RPN']
        for r in before:r['identity']=r['kind'];r['new_nll']=0.0
        after=copy.deepcopy(before)
        for r in after:r['new_nll']=2.0
        rows=paired(before,after)
        self.assertEqual([(r['lost'],r['gained']) for r in rows],[(1,0),(1,0),(0,1)])


if __name__=='__main__':unittest.main()
