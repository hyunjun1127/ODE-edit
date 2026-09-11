"""CPU synthetic reducer/order/denominator and plot semantics tests."""
import unittest,tempfile
from pathlib import Path
from .common import *
from .plots import ordered,variant,weight_panel
from project.run_scripts.blue_lifelong_analysis.metrics import preferred

def row(case=1,index=0,new=1.,true=2.,tag='RS'):
    return dict(case_id=case,prompt_index=index,identity=digest([case,index]),new_nll=new,true_nll=true,margin=true-new,success=preferred(new,true,tag),new_token_correct=1,new_token_count=2,new_strict=False,true_token_correct=0,true_token_count=2,true_strict=False)

class Focused(unittest.TestCase):
    def test_directions_ties(self):
        self.assertTrue(preferred(1,2,'RS'));self.assertTrue(preferred(1,2,'PS'));self.assertFalse(preferred(1,2,'NS'))
        for tag in MULT:self.assertFalse(preferred(2,2,tag))
    def test_nonfinite_reject(self):
        with self.assertRaises(AssertionError):preferred(float('nan'),1,'RS')
    def test_prompt_not_request_average(self):
        r=[row(index=0,new=1,true=2,tag='PS'),row(index=1,new=9,true=2,tag='PS')]
        x=reduce_rows(r,'PS');self.assertEqual((x['numerator'],x['denominator'],x['pair_strict_num']),(1,2,0))
    def test_pair_identity_mismatch(self):
        a=[row()];b=[row(case=2)]
        with self.assertRaises(AssertionError):transition(a,b)
    def test_transition_arithmetic(self):
        a=[row(case=1),row(case=2,new=3),row(case=3),row(case=4,new=3)]
        b=[row(case=1,new=3),row(case=2),row(case=3),row(case=4,new=3)]
        q=transition(a,b)
        self.assertEqual([q[k] for k in ['retained','lost','gained','both_failed']],[1,1,1,1])
    def test_cardinality_order(self):
        ev=dict(requests=1,metrics={tag:dict(rows=[row(index=i,tag=tag) for i in range(n)]) for tag,n in MULT.items()})
        for tag,m in ev['metrics'].items():
            z=reduce_rows(m['rows'],tag);m.update({k:z[k] for k in ['numerator','denominator','rate','bit_order_sha256']})
        self.assertEqual(len(reductions(ev,[1],'x',1,'fixture')),3)
        ev['metrics']['NS']['rows'].pop()
        with self.assertRaises(AssertionError):reductions(ev,[1],'x',1,'fixture')
    def test_labels_order(self):
        for f in ['MEMIT','AlphaEdit']:
            self.assertEqual([variant(a) for a in ordered(f)],['Native','BLUE','L4','L5','L6','L7','L8'])
            self.assertTrue(all('ORIGINAL' not in label(a) for a in ordered(f)))
    def test_weight_plot_no_reference(self):
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots();weight_panel(ax,[dict(arm=a,layer=l,update_norm_mean=1) for a in ordered('MEMIT') for l in [4]],'MEMIT')
        fig.suptitle('Layer-wise Update Magnitude')
        self.assertEqual(len(ax.lines),0);self.assertEqual(fig._suptitle.get_text(),'Layer-wise Update Magnitude')
        self.assertFalse(any('bars:' in x.get_text().lower() for x in ax.texts))
        with tempfile.TemporaryDirectory() as d:
            fig.savefig(Path(d)/'test.png');self.assertGreater((Path(d)/'test.png').stat().st_size,0)
        plt.close(fig)
    def test_hash_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.csv';csvwrite(p,[dict(job='39283_1',rate=.2),dict(job='39307',rate=.3)])
            self.assertEqual(len(csvread(p)),2);self.assertEqual(csvread(p)[0]['job'],'39283_1');self.assertEqual(sha(p),sha(p))
    def test_checkpoint_count_scope(self):
        self.assertEqual(len(SCHEDULE)*len(ARMS),168);self.assertEqual(sum(SCHEDULE),556)

if __name__=='__main__':unittest.main()
