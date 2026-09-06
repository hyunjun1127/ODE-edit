import unittest
import hashlib
from pathlib import Path
import tempfile
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
        locality=paired_deltas([dict(r,metric='NS') for r in rows])
        new_nll=next(r for r in locality if r['field']=='target_new_nll')
        self.assertEqual(new_nll['favorable_direction'],'higher')
        self.assertEqual(new_nll['favorable'],1)

    def test_matched_progress_no_interpolation_or_endpoint_invention(self):
        node=dict(cell=0,arm='JV_NATIVE',V0=1.,V_exit=.5,node=1,N=4)
        result=matched_progress([], [node])
        row=next(r for r in result if r['cell']==0 and r['arm']=='JV_NATIVE' and r['requested_V_ratio']==.5)
        self.assertEqual(row['status'],'MATCHED_SAVED_NODE')
        self.assertEqual(row['endpoint_evaluation'],'NOT_EVALUATED_INTERMEDIATE_NODE')
        other=next(r for r in result if r['cell']==0 and r['arm']=='JV_NATIVE' and r['requested_V_ratio']==.75)
        self.assertEqual(other['status'],'NOT_REACHED')

    def test_followup_warm_seal_rejects_wrong_bytes_and_link(self):
        from .followup import verify_warm_member
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'warm';p.write_bytes(b'CPU fixture only')
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            verify_warm_member(p,digest)
            with self.assertRaises(RuntimeError):verify_warm_member(p,'0'*64)
            link=Path(d)/'link';link.symlink_to(p)
            with self.assertRaises(RuntimeError):verify_warm_member(link,digest)

    def test_refinement_uses_dense_delta_distance_not_hash(self):
        import torch
        from .diagnostics_analysis import endpoint_distances
        a={'delta':{'w':torch.tensor([1.,2.])},'activation':torch.tensor([1.,2.])}
        b={'delta':{'w':torch.tensor([2.,2.])},'activation':torch.tensor([1.,3.])}
        result=endpoint_distances(a,b)
        self.assertEqual(result['delta_W_distance'],1.)
        self.assertEqual(result['activation_distance'],1.)
        self.assertAlmostEqual(result['relative_to_second_update'],1/(8**.5))
        self.assertFalse(result['hash_only_convergence_claim'])


if __name__=='__main__':unittest.main()
