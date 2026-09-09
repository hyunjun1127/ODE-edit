import unittest
from .analysis import paired_rows,aggregate,bootstrap,csv_bytes

class AnalysisTests(unittest.TestCase):
    def test_pairs_denominators_and_tie(self):
        def row(i,new,true):
            return dict(panel='Past100',metric='NS',case_id=i,prompt_index=0,identity=str(i),
                 new_nll=new,true_nll=true,margin=true-new,success=true<new,new_strict=False,true_strict=True)
        base=[row(1,2,1),row(2,1,2)];post=[row(1,1,1),row(2,3,2)]
        paired=paired_rows(post,base,base);a=aggregate(paired)
        self.assertEqual((a['numerator'],a['denominator'],a['loss'],a['recovery']),(1,2,1,1))
        self.assertEqual(a['entry_success_denominator'],1)
        self.assertEqual(a['conditional_loss_rate'],1)
    def test_bootstrap_request_cluster_and_determinism(self):
        rows=[dict(case_id=i,success=True,entry_success=False,new_nll_delta=-.5) for i in range(4) for _ in range(10)]
        a=bootstrap(rows);self.assertEqual(a['cluster_count'],4)
        self.assertEqual(a,bootstrap(rows));self.assertEqual(a['success_delta_ci95'],[1.,1.])
        self.assertEqual(csv_bytes([a]),csv_bytes([a]))

if __name__=='__main__':unittest.main()
