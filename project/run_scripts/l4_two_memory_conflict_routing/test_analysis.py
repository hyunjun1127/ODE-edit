import tempfile
import unittest
from pathlib import Path
from .analysis import summarize,csv_save
from .plotting import render
from .report import table

class AnalysisTest(unittest.TestCase):
    def rows(self):
        return [dict(panel='Current100',metric='RS',case_id=i,prompt_index=0,identity=str(i),new_nll=float(i),true_nll=.5,
            margin=.5-i,success=i==0,new_strict=i==0,true_strict=False,new_token_correct=1-i,new_token_count=1,true_token_correct=0,true_token_count=1) for i in (0,1)]
    def test_identity_and_pairing(self):
        rr=self.rows();s=summarize(rr,rr)[0]
        self.assertEqual((s['numerator'],s['denominator'],s['loss'],s['recovery']),(1,2,0,0))
        self.assertEqual((s['paired_ci_low'],s['paired_ci_high']),(0,0))
        with self.assertRaises(ValueError):summarize(rr,rr[:1])
    def test_table_escape(self):
        x=table(['arm','value'],[['a|b',2]])
        self.assertIn('a\\|b',x);self.assertEqual(len(x.splitlines()),3)
    def test_preservation_recovery_separation(self):
        from .diagnostics import transitions
        z=self.rows();e=[dict(r) for r in z];w=[dict(r) for r in z]
        e[0]['success']=False;w[0]['success']=True
        rr=transitions(w,e,z)
        self.assertTrue(rr[0]['W0_success_inherited_loss_recovered'])
        self.assertEqual(rr[0]['inherited_success_change']+rr[0]['additional_success_change'],0)
        self.assertFalse(rr[0]['entry_success_to_failure'])
    def test_context_response_physical_projection(self):
        import torch
        from .observations import response_components
        delta=torch.tensor([[1.,2.],[3.,4.]])
        keys=torch.tensor([[1.,0.],[0.,1.]])
        goal=torch.tensor([[1.,0.],[0.,0.]])
        rr=response_components(delta,keys,goal,2,1)
        for r in rr:self.assertAlmostEqual(r['goal_component_sq']+r['perpendicular_sq'],r['response_change_sq'])
        self.assertTrue(rr[1]['zero_goal']);self.assertEqual(rr[1]['goal_component_sq'],0)
    def test_png_deterministic(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);summary=[]
            for m in ('RS','PS','NS'):
                for arm in ('N','OS','BF1','BF8','Frozen-BF8'):
                    summary.append(dict(entry='Middle',batch_raw=100,reference='N',panel='Current100',metric=m,arm=arm,rate=50,delta_pp=0,paired_new_nll_delta_mean=0))
            csv_save(root/'endpoint-summary.csv',summary)
            csv_save(root/'trajectory.csv',[dict(entry='Middle',batch_raw=100,arm='BF8',node=0,s_next=.125,normalized_actual=[.1,.2])])
            csv_save(root/'base-preservation-versus-recovery.csv',[dict(entry='Middle',batch_raw=100,arm=a,bank=b,controller_value=.1) for a in ('N','OS','BF1','BF8','Frozen-BF8') for b in ('Past','Base')])
            csv_save(root/'compute-ledger.csv',[dict(entry='Middle',batch_raw=100,category='counts',component='functional_backward',value=192)])
            for kind in ('endpoint','paired','harm','trajectory','compute'):
                a=render(root,kind);b=render(root,kind)
                self.assertEqual(a,b);self.assertTrue(a.startswith(b'\x89PNG'))

if __name__=='__main__':unittest.main()
