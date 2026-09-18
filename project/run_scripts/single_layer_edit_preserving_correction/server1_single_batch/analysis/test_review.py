import copy
import unittest
from review import pair_rows, quantiles


def raw(new=1., true=2.):
    def row(nll, target):
        return dict(case_id=1, prompt_index=0, prompt='synthetic fixture', target=target,
                    target_token_ids=[1 if target=='new' else 2], token_correct=[True],
                    all_tokens_correct=True, nll=nll)
    return {p+'_target_'+kind:[row(n,kind)] for p in ['rewrite','rephrase','locality']
            for kind,n in [('new',new),('true',true)]}


class ReviewTests(unittest.TestCase):
    def test_success_direction(self):
        self.assertTrue(next(iter(pair_rows(raw(),'RS').values()))['success'])
        self.assertFalse(next(iter(pair_rows(raw(),'NS').values()))['success'])
    def test_ties_fail_all(self):
        for m in ['RS','PS','NS']:
            self.assertFalse(next(iter(pair_rows(raw(1,1),m).values()))['success'])
    def test_duplicate_rejected(self):
        r=raw();r['rewrite_target_new']*=2
        with self.assertRaises(AssertionError):pair_rows(r,'RS')
    def test_prompt_mismatch_rejected(self):
        r=raw();r['rewrite_target_true'][0]['prompt']='different'
        with self.assertRaises(AssertionError):pair_rows(r,'RS')
    def test_nonfinite_rejected(self):
        with self.assertRaises(AssertionError):pair_rows(raw(float('nan')),'RS')
    def test_token_inventory_in_pair_identity(self):
        a=raw();b=copy.deepcopy(a);b['rewrite_target_new'][0]['target_token_ids']=[3]
        self.assertNotEqual(pair_rows(a,'RS').keys(),pair_rows(b,'RS').keys())
    def test_raw_not_mutated(self):
        r=raw();before=copy.deepcopy(r);pair_rows(r,'RS');self.assertEqual(r,before)
    def test_tail_definition(self):
        q=quantiles([0,1,2]);self.assertEqual(q['median'],1);self.assertAlmostEqual(q['p95'],1.9)


if __name__=='__main__':unittest.main()
