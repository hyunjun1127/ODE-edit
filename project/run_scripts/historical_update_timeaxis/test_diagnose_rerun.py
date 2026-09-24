"""Synthetic CPU tests of the failure-review reducer; no model/scheduler access."""
import copy
import unittest
from .diagnose_rerun import compare_rows


def fixture():
    return {category:[dict(case_id=1,kind=category,prompt_index=0,prompt='synthetic',target='synthetic',
                           target_token_ids=[1,2],token_predictions=[1,2],token_correct=[True,True],
                           all_tokens_correct=True,nll=2. if category.endswith('_new') else 3.)]
            for category in ('rewrite_target_new','rewrite_target_true','rephrase_target_new','rephrase_target_true')}


class DiagnosisReducer(unittest.TestCase):
    def test_exact_rows(self):
        a=fixture();d=compare_rows(a,copy.deepcopy(a))
        self.assertEqual(d,dict(answer_rows=4,max_nll=0.,max_margin=0.,boundary_flips=[]))

    def test_difference_is_reported_not_clipped_to_tolerance(self):
        a=fixture();b=copy.deepcopy(a);b['rewrite_target_new'][0]['nll']+=.001
        d=compare_rows(a,b);self.assertAlmostEqual(d['max_nll'],.001)
        self.assertAlmostEqual(d['max_margin'],.001)

    def test_identity_mismatch_rejected(self):
        a=fixture();b=copy.deepcopy(a);b['rewrite_target_true'][0]['case_id']=2
        with self.assertRaisesRegex(AssertionError,'ROW_IDENTITY'):compare_rows(a,b)

    def test_nonfinite_rejected(self):
        a=fixture();b=copy.deepcopy(a);b['rewrite_target_new'][0]['nll']=float('nan')
        with self.assertRaisesRegex(AssertionError,'NONFINITE_NLL'):compare_rows(a,b)


if __name__=='__main__':unittest.main()
