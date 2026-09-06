import unittest
from .analysis import distribution,old_loss,paired_deltas,matched_progress


class AnalysisTests(unittest.TestCase):
    def test_tail_and_nonfinite(self):
        values=distribution([0.,1.,2.,3.])
        self.assertEqual(values['n'],4);self.assertEqual(values['mean'],1.5)
        self.assertAlmostEqual(values['p90'],2.7)
        with self.assertRaises(ValueError):distribution([float('nan')])

    def test_old_failure_tie_and_initial_failure(self):
        def panel(pairs):
            return {'rewrite_target_new':[dict(case_id=i,nll=a) for i,(a,b) in enumerate(pairs)],
                    'rewrite_target_true':[dict(case_id=i,nll=b) for i,(a,b) in enumerate(pairs)]}
        result,rows=old_loss(panel([(1.,2.),(2.,2.),(3.,2.)]),panel([(2.,2.),(1.,2.),(1.,2.)]))
        self.assertEqual(result['entry_success_d'],1)
        self.assertEqual(result['new_failure_n'],1)
        self.assertEqual(result['initially_failed_n'],2)
        self.assertEqual(result['old_after_RS_n'],2)

    def test_exact_paired_join_and_ties(self):
        base=dict(cell=0,case_id=12,metric='RS',prompt_index=0,target_new_nll=1.,target_true_nll=2.,
                  nll_advantage=1.,success=1,strict_teacher_forced=True)
        rows=[dict(base,arm='O_NATIVE'),dict(base,arm='JV_NATIVE',target_new_nll=2.,nll_advantage=0.,success=0)]
        result=paired_deltas(rows)
        advantage=next(r for r in result if r['field']=='nll_advantage')
        self.assertEqual(advantage['mean'],-1.);self.assertEqual(advantage['unfavorable'],1)
        with self.assertRaises(ValueError):paired_deltas(rows[1:])
        with self.assertRaises(ValueError):paired_deltas(rows+rows[:1])

    def test_matched_progress_no_interpolation_or_endpoint_invention(self):
        node=dict(cell=0,arm='JV_NATIVE',V0=1.,V_exit=.5,node=1,N=4)
        result=matched_progress([], [node])
        row=next(r for r in result if r['cell']==0 and r['arm']=='JV_NATIVE' and r['requested_V_ratio']==.5)
        self.assertEqual(row['status'],'MATCHED_SAVED_NODE')
        self.assertEqual(row['endpoint_evaluation'],'NOT_EVALUATED_INTERMEDIATE_NODE')
        other=next(r for r in result if r['cell']==0 and r['arm']=='JV_NATIVE' and r['requested_V_ratio']==.75)
        self.assertEqual(other['status'],'NOT_REACHED')


if __name__=='__main__':unittest.main()
