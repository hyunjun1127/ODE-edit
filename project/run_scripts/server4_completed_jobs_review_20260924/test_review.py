import copy
import tempfile
import unittest
from pathlib import Path
from . import review as r

def row(target=1,pred=1,nll=1.0):
    return dict(case_id=10,prompt_index=0,prompt='fixture',target=str(target),endpoint_id='fixture-endpoint',
        target_token_ids=[target],token_predictions=[pred],token_correct=[target==pred],all_tokens_correct=target==pred,nll=nll,
        full_vocab=dict(prompt_token_ids_sha256='fixture-token-hash',prompt_token_count=3,vocab_size=128256,
            full_vocab_checked=True,true_token_ids=[target],argmax_token_ids=[pred],scoring_positions_unpadded=[2],
            argmax_tie_counts=[1],all_unique_argmax_correct=target==pred))

class ReviewTest(unittest.TestCase):
    def reduce(self,a,b):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.json';r.write_json(p,dict(compact={},raw_local_only=dict(rewrite_target_new=[a],rewrite_target_true=[b])))
            return r.reduce_raw(p)['RS']
    def test_tie_failure(self):
        self.assertFalse(self.reduce(row(),row(2,nll=1))[0]['success'])
    def test_strict_inequality(self):
        self.assertTrue(self.reduce(row(nll=.5),row(2,nll=1))[0]['success'])
    def test_endpoint_mismatch(self):
        a,b=row(),row(2);b['endpoint_id']='other'
        with self.assertRaises(AssertionError):self.reduce(a,b)
    def test_tokenization_mismatch(self):
        a,b=row(),row(2);b['full_vocab']['prompt_token_ids_sha256']='other'
        with self.assertRaises(AssertionError):self.reduce(a,b)
    def test_finite_required(self):
        a,b=row(),row(2);a['nll']=float('inf')
        with self.assertRaises(ValueError):self.reduce(a,b)
    def test_token_flags_not_trusted(self):
        a=row();a['token_correct']=[False]
        with self.assertRaises(AssertionError):r.token_stats(a)
    def test_unique_tie_distinct(self):
        a=row();a['full_vocab']['argmax_tie_counts']=[2];a['full_vocab']['all_unique_argmax_correct']=False
        self.assertEqual(r.token_stats(a),(1,1,True,False))
    def test_paired_zero_net_not_same_ids(self):
        one=self.reduce(row(nll=.5),row(2,nll=1))[0];a=[copy.deepcopy(one),copy.deepcopy(one)]
        a[1]['identity']='second';a[1]['case_id']=11;a[1]['success']=False
        b=copy.deepcopy(a);b[0]['success']=False;b[1]['success']=True
        p=r.paired(a,b,True);self.assertEqual((p['lost'],p['gained'],p['delta_pp']),(1,1,0))
        self.assertEqual(p['cluster_cases'],2)
    def test_pair_reordering_rejected(self):
        a=self.reduce(row(nll=.5),row(2,nll=1));b=copy.deepcopy(a);b[0]['identity']='other'
        with self.assertRaises(AssertionError):r.paired(a,b)

if __name__=='__main__':unittest.main()
