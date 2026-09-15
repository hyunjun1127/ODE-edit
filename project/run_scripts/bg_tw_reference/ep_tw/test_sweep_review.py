"""CPU reducer API/cluster/source-isolation fixtures, no new scientific rows."""
import ast
from pathlib import Path
import unittest
from .sweep_review import cluster_interval
from .test_review_nogate import row
from .build_sweep_report import scheduler_rows

class SweepReviewTests(unittest.TestCase):
    def test_cluster_unit_and_reproducibility(self):
        before=[row(case=1,prompt=0,identity='a'),row(case=1,prompt=1,identity='b'),
                row(case=2,prompt=0,identity='c'),row(case=2,prompt=1,identity='d')]
        after=[dict(x,new_nll=3.) for x in before]
        got=cluster_interval(before,after,'PS',draws=20)
        self.assertEqual((got['CI_low'],got['CI_high']),(-100.,-100.))
        self.assertEqual(got,cluster_interval(before,after,'PS',draws=20))
        self.assertIn('REQUEST_CLUSTER',got['bootstrap_unit'])

    def test_cluster_identity_not_row_index(self):
        with self.assertRaises(AssertionError):cluster_interval([row()],[row(identity='different')],'RS')
        a=[row(),row(case=2,new=3)];b=[row(case=2),row(new=3)]
        got=cluster_interval(a,b,'RS',draws=20)
        self.assertLessEqual(got['CI_low'],0);self.assertGreaterEqual(got['CI_high'],0)

    def test_analysis_modules_syntax_and_no_model_execution(self):
        root=Path(__file__).parent
        for name in ('sweep_review.py','plot_sweep.py','build_sweep_report.py'):
            source=(root/name).read_text();ast.parse(source)
            for forbidden in ('from_pretrained(','.cuda(','.backward(',"['sbatch'",'sbatch --'):
                self.assertNotIn(forbidden,source)
        source=(root/'sweep_review.py').read_text()
        self.assertIn('ACTUAL_ALLOCATION_REQUIRED_NOT_PRIOR_ESTIMATE',source)
        self.assertIn('CAP1_CP_current',source)

    def test_scheduler_completion_not_artifact_presence(self):
        record='123|toy|janghj|COMPLETED|0:0|2026-09-15T00:00:00|2026-09-15T00:01:00|2026-09-15T00:03:00|120|cpu=8,gres/gpu=1||||server4\n'
        x=dict(states=[dict(arm='FIXTURE',job_id='123')],queries={'sacct':{'stdout':record}})
        got=scheduler_rows(x)[0]
        self.assertEqual((got['allocated_GPU_seconds'],got['queued_seconds']),(120,60))
        x['queries']['sacct']['stdout']=record.replace('COMPLETED','RUNNING')
        with self.assertRaises(AssertionError):scheduler_rows(x)

if __name__=='__main__':unittest.main()
