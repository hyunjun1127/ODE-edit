import pathlib
import sys
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).parent))
import aos_review as a
import aos_report as report


class TestReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h=a.helper(pathlib.Path(__file__).resolve().parents[4])

    def row(self,metric='RS',success=True,new=1.,true=2.,identity='x'):
        return dict(panel='Current100',metric=metric,identity=identity,case_id=1,prompt_index=0,
                    new_nll=new,true_nll=true,margin=true-new,success=success)

    def test_pair_directions(self):
        for metric,sign in [('RS',1),('PS',1),('NS',-1)]:
            rows=a.pair_rows(self.h,[self.row(metric)],[self.row(metric,new=1.5)],'test')
            self.assertEqual(rows[0]['desired_margin_delta'],sign*-.5)

    def test_transition_denominator(self):
        rows=a.pair_rows(self.h,[self.row()],[self.row(success=False,new=3.)],'test')
        out=a.pair_summary(self.h,rows)
        self.assertTrue(all(r['denominator']==1 and r['loss_n']==1 and r['delta_pp']==-100 for r in out))

    def test_support_fail_close(self):
        with self.assertRaises(self.h.AnalysisBoundary):
            a.pair_rows(self.h,[self.row()],[self.row(identity='other')],'test')

    def test_duplicate_fail_close(self):
        with self.assertRaises(self.h.AnalysisBoundary):
            a.pair_rows(self.h,[self.row(),self.row()],[self.row()],'test')

    def test_symmetric_signed_no_clipping(self):
        r=self.h.symmetric_effects(-2,3,1)
        self.assertEqual(r['signed_L4'],-2);self.assertEqual(r['signed_L8'],3)
        self.assertEqual(r['signed_L4']+r['signed_L8'],1)

    def test_table_commonmark(self):
        text=report.table(['a','b'],[['1','2']])
        self.assertEqual(text,'\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\n')

    def test_no_runtime_or_scheduler(self):
        source=pathlib.Path(a.__file__).read_text()
        for forbidden in ('from transformers','import transformers','sbatch','squeue','torch.cuda.synchronize'):
            self.assertNotIn(forbidden,source)


if __name__=='__main__':unittest.main()
