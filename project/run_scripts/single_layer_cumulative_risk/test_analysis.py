import unittest
import tempfile
import json
from pathlib import Path
from .analysis import paired_rows,aggregate,bootstrap,csv_bytes,observed_resolutions
from .auxiliary_analysis import ledger_delta
from .direction_analysis import central

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
    def test_cumulative_child_compute_is_differenced_not_summed(self):
        one={'seconds':{'objective':10},'counts':{'forward':100}}
        two={'seconds':{'objective':25},'counts':{'forward':250}}
        self.assertEqual(ledger_delta(two,one),{'seconds.objective':15,'counts.forward':150})
        with self.assertRaisesRegex(ValueError,'NONMONOTONE'):ledger_delta(one,two)
    def test_sign_pair_derivative_axis_and_curvature(self):
        # f(x)=x^2 at x=2; GFminus points toward smaller x.
        r=central(1.5**2,2.5**2,4.,.5)
        self.assertEqual(r,{'derivative_along_GFminus':-4.,'central_curvature':2.})
    def test_full_curve_exact_reuse_not_new_measurement(self):
        rows=[];neighbors={}
        for p,panel in enumerate(['Current100','Fixed100','Past100']):
            for i in range(100):
                case=p*100+i;neighbors[str(case)]=[2,7]
                for metric,count in [('RS',1),('PS',2),('NS',10)]:
                    rows.extend(dict(panel=panel,metric=metric,case_id=case,prompt_index=j) for j in range(count))
        full,curve=list(observed_resolutions({'resolution':'full','rows':rows},{'neighbors':neighbors}))
        self.assertEqual((len(full[1]),len(curve[1]),curve[2]),(3900,1100,True))
        self.assertTrue(all(r['panel']=='Current100' for r in curve[1] if r['metric']=='PS'))
        from .evaluation import reuse_native_observation
        from .records import Ledger,save,digest
        panel={'neighbors':neighbors}
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'native.json';target=Path(tmp)/'curve.json'
            save(source,dict(resolution='full',rows=rows,panel_identity=digest(panel)))
            ledger=Ledger()
            with self.assertRaisesRegex(ValueError,'NONIDENTICAL_WEIGHT'):
                reuse_native_observation(source,target,panel,False,'different','native',ledger)
            self.assertFalse(target.exists())
            reuse_native_observation(source,target,panel,False,'native','native',ledger)
            result=json.loads(target.read_text())
            self.assertEqual(result['pairs'],1100)
            self.assertEqual(result['new_model_forwards'],0)
            self.assertEqual(ledger.counts,{'evaluation_pairs_reused':1100})

if __name__=='__main__':unittest.main()
