import copy
import unittest
from .common import digest
from .review_completed import validate_rows, independent_counts, general_contrast


def fixture(kind='N',values=(1.,2.)):
    panel=[];rows=[]
    for label,nll in zip(('true','new'),values):
        p=dict(row_id=label,pair_id='pair',panel='fixture',kind=kind,label=label,
            case_id=4,subject='subject',prompt_cluster='prompt',input_ids=[1,2],positions=[0],target_ids=[3])
        r={k:p[k] for k in ('row_id','pair_id','panel','kind','label','case_id','subject','prompt_cluster')}
        r.update(input_sha=digest([[1,2],[0],[3]]),target_count=1,token_predictions=[3],token_nll=[nll],
                 nll=nll,strict=True,token_correct=1)
        panel.append(p);rows.append(r)
    return panel,rows


class ReviewTests(unittest.TestCase):
    def test_independent_preferences_and_ties(self):
        for kind,count in [('N',1),('BASE',1),('R',0),('P',0)]:
            p,r=fixture(kind);self.assertEqual(independent_counts(r)[0]['success'],count)
        p,r=fixture(values=(1.,1.));self.assertEqual(independent_counts(r)[0]['success'],0)
        self.assertEqual(independent_counts(r)[0]['ties'],1)

    def test_actual_tokens_recount_not_hash_assumption(self):
        p,r=fixture();self.assertTrue(validate_rows(r,p)['token_predictions_recount_exact'])
        r[0]['token_predictions']=[9]
        with self.assertRaises(AssertionError):validate_rows(r,p)

    def test_token_identity_order_and_missing_rows(self):
        p,r=fixture()
        for bad in [r[::-1],r[:1],r+r[:1]]:
            with self.assertRaises(AssertionError):validate_rows(bad,p)
        p[0]['target_ids']=[9]
        with self.assertRaises(AssertionError):validate_rows(r,p)

    def test_finite_and_mean_route_disclosed(self):
        p,r=fixture();r[0]['nll']+=1e-7
        self.assertGreater(validate_rows(r,p)['nll_mean_route_max_abs'],0)
        r[0]['token_nll']=[float('nan')]
        with self.assertRaises(AssertionError):validate_rows(r,p)

    def test_general_paired_count_and_sign(self):
        x=[dict(kind='GENERAL',row_id=str(i),case_id=i,subject=str(i),prompt_cluster=str(i),nll=1.,
                strict=True,token_correct=2,w0_top1_agree=2,w0_forward_kl=.1) for i in range(128)]
        y=copy.deepcopy(x)
        for r in y:r.update(nll=1.25,strict=False,token_correct=1,w0_top1_agree=1,w0_forward_kl=.2)
        z=general_contrast('fixed',x,y)
        self.assertEqual(z['NLL_delta'],.25);self.assertEqual(z['strict_lost'],128)
        self.assertEqual(z['token_correct_delta'],-128);self.assertEqual(z['cluster_count'],128)
        with self.assertRaises(AssertionError):general_contrast('bad',x,y[:-1])


if __name__=='__main__':unittest.main()
