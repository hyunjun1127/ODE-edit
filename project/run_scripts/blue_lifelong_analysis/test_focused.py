import tempfile,unittest
from pathlib import Path
from .metrics import preferred,reduce_rows,reduce_eval
from .aggregate import transition
from .common import digest,table,read,SCHEDULE,MULT
from .plots import update_figure,PANEL_ARMS,setup
import matplotlib.pyplot as plt

def row(case,p,new,true):
    return dict(case_id=case,prompt_index=p,identity=digest([case,p]),new_nll=new,true_nll=true,margin=true-new,success=new<true,
                new_strict=False,true_strict=False,new_token_correct=0,true_token_correct=0,new_token_count=2,true_token_count=1)

class Focused(unittest.TestCase):
    def test_inequality_direction(self):
        self.assertTrue(preferred(1,2,'RS'));self.assertTrue(preferred(1,2,'PS'));self.assertFalse(preferred(1,2,'NS'))
    def test_tie_failure(self):
        self.assertTrue(all(not preferred(2,2,k) for k in MULT))
    def test_nonfinite_closed(self):
        with self.assertRaises(AssertionError):preferred(float('nan'),2,'RS')
    def test_prompt_not_mean_preference(self):
        rr=[row(1,0,0,1),row(1,1,20,1)]
        r=reduce_rows(rr,'PS');self.assertEqual((r['numerator'],r['denominator'],r['pair_strict_num']),(1,2,0));self.assertEqual(r['new_nll_request_mean'],10)
    def test_transition_partition(self):
        a=[row(1,0,1,2),row(2,0,2,1)];b=[row(1,0,2,1),row(2,0,1,2)]
        t=transition(a,b);self.assertEqual((t['lost'],t['gained'],t['after_num']),(1,1,1))
    def test_identity_mismatch_closed(self):
        with self.assertRaises(AssertionError):transition([row(1,0,1,2)],[row(2,0,1,2)])
    def test_job_id_format_preserved(self):self.assertIn('39283_1',table([dict(job='39283_1')],['job']))
    def test_schedule_counts(self):self.assertEqual((len(SCHEDULE),sum(SCHEDULE)*100*6),(12,333600))
    def test_panel_order(self):self.assertEqual(PANEL_ARMS[:3],['MEMIT_ORIGINAL','MEMIT_L4_ONLY','MEMIT_L8_ONLY'])
    def test_weight_plot_no_reference(self):
        setup();rr=[]
        for arm in PANEL_ARMS:
            layers=[4,8] if arm.endswith('ORIGINAL') else [4] if arm.endswith('L4_ONLY') else [8]
            for l in layers:rr.append(dict(arm=arm,layer=l,update_norm=2))
        fig=update_figure(rr);self.assertEqual(fig._suptitle.get_text(),'Layer-wise Update Magnitude');self.assertTrue(all(len(a.lines)==0 for a in fig.axes))
        self.assertFalse(any('bars:' in x.get_text().lower() for x in fig.texts))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.png';fig.savefig(p);self.assertGreater(p.stat().st_size,1000)
        plt.close(fig)

if __name__=='__main__':unittest.main()
