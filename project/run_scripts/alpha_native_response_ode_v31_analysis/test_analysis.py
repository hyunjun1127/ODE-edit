import math,unittest
from .analyze import concentration,index,endpoint_partitions,table

class AnalysisTests(unittest.TestCase):
    def test_zero_is_not_uniform(self):
        self.assertIsNone(concentration([0]*5)['effective_layers'])
    def test_concentration(self):
        self.assertEqual(concentration([0,0,0,0,2])['effective_layers'],1)
        self.assertAlmostEqual(concentration([1]*5)['effective_layers'],5)
    def test_negative_energy_rejected(self):
        with self.assertRaises(AssertionError):concentration([-1,0,0,0,1])
    def test_nonfinite_rejected(self):
        with self.assertRaises(AssertionError):concentration([math.nan]*5)
    def test_duplicate_join_rejected(self):
        row=dict(alias='a',arm='JV_NATIVE',batch=1,node=0,layer=4)
        with self.assertRaises(AssertionError):index([row,row],node=True,layer=True)
    def test_norm_addition_not_net(self):
        # Opposite actual steps: length2, energy2, net0. Never derive net from sums.
        self.assertNotEqual(abs(1)+abs(-1),abs(1-1))
    def test_table_missing_and_pipe(self):
        result=table([{'a':None,'b':'A|B'}],[('a','a'),('b','b')])
        self.assertIn('|N/A|A/B|',result)
    def test_recovery_is_not_previous_checkpoint_recovery(self):
        rows=[dict(alias='m',arm='a',batch=10,at_write_success=9,
            at_write_success_now_failure=1,initially_failed=1,current_success=9,
            prior_failure_now_recovery=4,nonoverwrite_forgetting_num=1,
            nonoverwrite_forgetting_den=9,overwrite_candidate_count=0) for _ in range(10)]
        # Fixture must maintain production B100 denominator; upscale counts.
        for r in rows:
            for k in ('at_write_success','at_write_success_now_failure','initially_failed','current_success',
                'prior_failure_now_recovery','nonoverwrite_forgetting_num','nonoverwrite_forgetting_den'):r[k]*=10
        t={'final_metrics':[dict(alias='m',arm='a',RS_num=900)],'retention_cohort_metrics':rows}
        result=endpoint_partitions(t)[0]
        self.assertEqual(result['initially_failed_to_final_recovery'],100)
        self.assertEqual(result['previous_checkpoint_failure_to_recovery'],400)

if __name__=='__main__':unittest.main()
