"""Synthetic CPU error-injection tests for the recall-only reducer."""
import copy, unittest
from project.run_scripts.memit_hj.review_1k_completed import summary, compare_stored, transitions, panel, digest

def row(i, new, true, c=1, n=2):
    return dict(identity=str(i), case_id=i,prompt_index=0,new_nll=new,true_nll=true,margin=true-new,success=new<true,
                new_token_correct=c,new_token_count=n,new_strict=c==n,true_token_correct=0,true_token_count=1,true_strict=False)

class ReviewTests(unittest.TestCase):
    def test_micro_macro_strict_distinct(self):
        s=summary([row(1,1,2,1,1),row(2,1,2,1,3)],'RS')
        self.assertEqual(s['tf_token_micro'],.5);self.assertAlmostEqual(s['tf_prompt_macro'],2/3);self.assertEqual(s['tf_strict'],.5)
    def test_tie_failure(self):
        self.assertEqual(summary([row(1,1,1)],'PS')['ties'],1)
        self.assertEqual(summary([row(1,1,1)],'PS')['numerator'],0)
    def test_ns_desired_true(self):
        r=row(1,2,1);r['success']=True;r['true_token_correct']=1;r['true_strict']=True
        s=summary([r],'NS');self.assertEqual(s['desired'],'true');self.assertEqual(s['tf_strict'],1)
    def test_duplicate_rejected(self):
        with self.assertRaises(AssertionError):summary([row(1,1,2),row(1,1,2)],'RS')
    def test_nonfinite_rejected(self):
        with self.assertRaises(AssertionError):summary([row(1,float('nan'),2)],'RS')
    def test_wrong_direction_rejected(self):
        r=row(1,1,2);r['success']=False
        with self.assertRaises(AssertionError):summary([r],'RS')
    def test_strict_inconsistency_rejected(self):
        r=row(1,1,2);r['new_strict']=True
        with self.assertRaises(AssertionError):summary([r],'RS')
    def test_zero_token_rejected(self):
        with self.assertRaises(AssertionError):summary([row(1,1,2,0,0)],'RS')
    def test_stored_corruption(self):
        s=summary([row(1,1,2)],'RS')
        with self.assertRaises(AssertionError):compare_stored(s,dict(numerator=0))
    def test_equal_total_different_ids(self):
        a=[row(1,1,2),row(2,2,1)];b=[row(1,2,1),row(2,1,2)]
        s,ids=transitions(a,b,'RS');self.assertEqual((s['lost'],s['gained'],s['delta_pp']),(1,1,0));self.assertEqual(ids['lost'],['1'])
    def test_unmatched_ids_rejected(self):
        with self.assertRaises(AssertionError):transitions([row(1,1,2)],[row(2,1,2)],'RS')
    def test_strict_transition_separate(self):
        a=[row(1,1,2,1,1)];b=[row(1,1,2,0,1)]
        self.assertEqual(transitions(a,b,'RS')[0]['lost'],0);self.assertEqual(transitions(a,b,'RS','TF_strict')[0]['lost'],1)
    def test_panel_quartiles(self):
        rows=[dict(requested_rewrite=dict(subject=str(i))) for i in range(1000)]
        p=panel(rows,1000);self.assertEqual(len(set(p)),400)
        self.assertEqual([sum(q*250<=i<(q+1)*250 for i in p) for q in range(4)],[100]*4)
        self.assertEqual(panel(rows,0),[])
    def test_unicode_identity_modes(self):
        self.assertNotEqual(digest(['한글']),digest(['한글'],False))
if __name__=='__main__':unittest.main()
