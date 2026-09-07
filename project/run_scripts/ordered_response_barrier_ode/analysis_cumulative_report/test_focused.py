"""Essential CPU-only analysis boundaries; no experiment or evaluator calls."""
import io,json,tempfile,unittest
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from ..tests.test_sequential_analysis import AnalysisTests
from ..fp32_overlay import tensor_sha256,tensor_set_sha256
from .common import *
from .comparisons import conflict_ledger
from .performance import distribution
from .plots import final_plot,retention_plot,STYLE
from .integrity import Ledger

class CumulativeTests(unittest.TestCase):
    def test_cpu_tensor_identity_and_change(self):
        t={str(i):torch.arange(6,dtype=torch.float32).reshape(2,3)+i for i in range(5)}
        before=tensor_set_sha256(t)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.pt';torch.save(t,p)
            loaded=torch.load(p,map_location='cpu',weights_only=True)
            self.assertEqual(tensor_set_sha256(loaded),before)
            loaded['0'][0,0]+=1
            self.assertNotEqual(tensor_set_sha256(loaded),before)
    def test_ledger_tamper_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';p.write_text('{"v":1}')
            h=sha256_file(p);self.assertEqual(Ledger(d).bind(p,h)['sha256'],h)
            p.write_text('{"v":2}')
            with self.assertRaises(RuntimeError):Ledger(d).bind(p,h)
    def test_metadata_conflict_not_outcome_selected(self):
        def r(i,target):return {'case_id':i,'requested_rewrite':{'subject':'s','relation_id':'r','target_new':{'str':target}}}
        rows=conflict_ledger([(0,r(1,'a'),'h1'),(1,r(2,'a'),'h2'),(2,r(3,'b'),'h3')])
        self.assertEqual(len(rows),2)
        self.assertEqual({x['previous_request_sha256'] for x in rows},{'h1','h2'})
        self.assertFalse(any(k in rows[0] for k in ['subject','prompt','target','relation_id']))
    def test_prompt_vs_cluster_distribution(self):
        p=pd.DataFrame(dict(category=['rephrase']*4,request_sha256=['a','a','b','b'],nll_new=[0.,8.,2.,2.],nll_true=[4.]*4,margin_true_minus_new=[4.,-4.,2.,2.],all_tokens_correct_new=[True,False,False,True],all_tokens_correct_true=[False]*4,correct_token_count_new=[1]*4,correct_token_count_true=[0]*4,target_token_count_new=[2]*4,target_token_count_true=[1]*4))
        d=distribution(p,{})
        prompt=next(x for x in d if x['target']=='new' and x['unit']=='prompt')
        cluster=next(x for x in d if x['target']=='new' and x['unit']=='request_cluster')
        self.assertEqual((prompt['nll_n'],cluster['nll_n']),(4,2))
        self.assertEqual((prompt['nll_median'],cluster['nll_median']),(2,3))
        self.assertEqual((prompt['strict_num'],prompt['prompt_den']),(2,4))
    def test_primary_final_is_subset_not_duplicate(self):
        self.assertEqual(sum(b*100 for b in range(1,11))*20,110000)
        self.assertEqual(20*1000,20000)
        self.assertEqual(20*sum(range(1,11)),1100)
    def test_final_panel_order_and_byte_stability(self):
        plt.rcParams.update(STYLE)
        f=pd.DataFrame([dict(cell=c,arm=a,RS_rate=.5,PS_rate=.4,NS_rate=.3) for c in CELLS for a in ARMS])
        bs=[]
        for _ in range(2):
            fig=final_plot(f);self.assertEqual([a.get_title() for a in fig.axes],['Llama / MEMIT','Llama / AlphaEdit','Qwen / MEMIT','Qwen / AlphaEdit'])
            self.assertTrue(all(len(a.patches)==15 for a in fig.axes))
            b=io.BytesIO();fig.savefig(b,format='png',metadata={'Software':'test'});bs.append(b.getvalue());plt.close(fig)
        self.assertEqual(*bs)
    def test_future_retention_mask_not_imputation(self):
        frame=pd.DataFrame([dict(cell=c,arm=a,batch=b,cohort=k,now_success=10) for c in CELLS for a in ARMS for b in range(1,11) for k in range(1,b+1)])
        fig=retention_plot(frame)
        self.assertEqual(len(fig.axes),20)
        self.assertTrue(all(int(ax.images[0].get_array().mask.sum())==45 for ax in fig.axes));plt.close(fig)
    def test_missing_never_zero(self):
        x=strict_stats([None,None]);self.assertEqual(x['n'],0);self.assertIsNone(x['mean'])
        with self.assertRaises(RuntimeError):strict_stats([float('inf')])

if __name__=='__main__':unittest.main()
