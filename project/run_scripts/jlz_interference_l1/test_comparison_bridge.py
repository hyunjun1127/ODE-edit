"""Real recorded endpoint replay, not an experimental numerical/toy gate."""
import copy
import unittest
from .comparison_bridge import read,metric_row,Target

BINDINGS='audits/servers/server4/wandb-model-views/comparison-bindings.json'
class ComparisonBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b=next(b for b in read(BINDINGS)['targets'] if b['cell']=='LLAMA_CAP075')
        cls.c=read(cls.b['attempt']+'/LLAMA_CAP075/batch-01/commit.json')

    def test_native_percent_and_harmonic(self):
        row=metric_row('current/post',self.c['post_current'],100)
        self.assertAlmostEqual(row['current/post/R/success_pct'],100*self.c['post_current']['R']['rate'])
        rates=[row[f'current/post/{k}/success_pct'] for k in 'RPN']
        self.assertAlmostEqual(row['current/post/success_harmonic_pct'],3/sum(1/x for x in rates))

    def test_cannot_relabel_current_as_cumulative(self):
        with self.assertRaisesRegex(ValueError,'ENDPOINT_DENOMINATOR'):
            metric_row('all_seen/post',self.c['post_current'],500)

    def test_nonfinite_rejected(self):
        groups=copy.deepcopy(self.c['post_current']);groups['N']['new_nll_mean']=float('nan')
        with self.assertRaises(ValueError):metric_row('current/post',groups,100)

    def test_sealed_raw_identity_reduction(self):
        t=Target(self.b);self.assertTrue(t.collect());self.assertGreaterEqual(t.last,4)
        self.assertFalse(t.collect())
        for n,row in t.rows.items():
            self.assertEqual('all_seen/post/R/count' in row,n in (5,10,15,20))
            self.assertEqual(row['current/post/R/count'],100)

if __name__=='__main__':unittest.main()
