import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from . import ROOT,LAYERS,write
from .adapter import Adapter,solve_write,stop_reason,replace,hidden
from .collect import metrics,paired
from .submit import argv,launcher

class Block(nn.Module):
    def __init__(self):
        super().__init__();self.proj=nn.Linear(4,4,bias=False);self.mlp=SimpleNamespace(down_proj=self.proj)
    def forward(self,x):return x+.02*self.proj(x)
class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module();self.model.layers=nn.ModuleList([Block() for _ in range(32)]);self.model.norm=nn.LayerNorm(4)
        self.embedding=nn.Embedding(8,4);self.lm_head=nn.Linear(4,8,bias=False);self.config=SimpleNamespace(hidden_size=4)
    def forward(self,input_ids,attention_mask,use_cache=False):
        x=self.embedding(input_ids)
        for b in self.model.layers:x=b(x)
        return SimpleNamespace(logits=self.lm_head(self.model.norm(x)))

class Tests(unittest.TestCase):
    def test_budget_and_stop_before_update(self):
        self.assertIsNone(stop_reason(.05,0));self.assertEqual(stop_reason(.049,0),'TOTAL_BELOW_005')
        self.assertEqual(stop_reason(1.,34),'BUDGET_EXHAUSTED');self.assertEqual(stop_reason(.01,34),'TOTAL_BELOW_005')
        with self.assertRaises(RuntimeError):stop_reason(float('nan'),0)
    def test_tensor_tuple(self):
        x=torch.ones(1,3,4);y=x+2
        self.assertIs(replace(x,y),y);self.assertIs(hidden(replace((x,None),y)),y)
    def test_writer_exact_rounding_and_prewrite_H(self):
        torch.manual_seed(2);W=torch.randn(3,4);H=torch.randn(4,4);H=H@H.T;C=torch.eye(4);K=torch.randn(4,2);R=torch.randn(3,2)
        originalW=W.clone();originalH=H.clone();kd=K.double();gram=kd@kd.T
        delta=torch.linalg.solve(gram+H.double()+15000*C.double(),kd@R.double().T).T
        evidence=solve_write(W,H,C,K,R)
        self.assertTrue(torch.equal(H,(originalH.double()+gram).float()))
        self.assertTrue(torch.equal(W,(originalW.double()+delta).float()))
        self.assertEqual(evidence['history_appends'],1);self.assertEqual(evidence['divisor'],1)
    def test_checkpoint_MB1_matches_full7_gradient_step(self):
        torch.manual_seed(3);m=Tiny().eval().requires_grad_(False)
        a=object.__new__(Adapter);a.model=m;a.device=torch.device('cpu');a.tok=None;a.contexts=None;a.lookup=None
        a.calls={'fit_logical':0,'fit_physical':0,'recompute':0};a.weights={l:m.model.layers[l].proj.weight for l in LAYERS}
        tok=dict(input_ids=torch.randint(0,8,(7,3)),attention_mask=torch.ones(7,3,dtype=torch.long))
        lab=torch.full((7,3),-100);lab[:6,2]=2;p=dict(tokens=tok,labels=lab,lookup=[1]*7,identity='fixture')
        d=torch.zeros(4,requires_grad=True);opt=torch.optim.Adam([d],lr=.1);initial=[]
        def hook(mod,args,out):
            if not initial:initial.append(out[0,1].detach().clone())
            x=out.clone();x[:,1]+=d;return x
        h=m.model.layers[4].register_forward_hook(hook)
        logits=m(**tok).logits;teacher=logits[6,1].log_softmax(-1).detach()
        loss=-logits[:6,2].log_softmax(-1)[:,2].mean()+.0625*torch.nn.functional.kl_div(teacher[None],logits[6:7,1].log_softmax(-1),log_target=True,reduction='batchmean')+.5*d.norm()/initial[0].norm().square()
        loss.backward();opt.step()
        with torch.no_grad():
            if d.norm()>.75*initial[0].norm():d.mul_(.75*initial[0].norm()/d.norm())
        expected=initial[0]+d.detach();h.remove()
        with tempfile.TemporaryDirectory() as t,patch('project.run_scripts.fe_baseline.adapter.pack',return_value=p),patch('project.run_scripts.fe_baseline.adapter.stop_reason',side_effect=lambda total,i:'TEST_END' if i==1 else None):
            value=a.fit({'case_id':9},0,Path(t)/'fit.json','fixture');r=json.loads((Path(t)/'fit.json').read_text())
        torch.testing.assert_close(value,expected,atol=2e-6,rtol=2e-6)
        self.assertEqual(r['evaluations'],2);self.assertEqual(r['Adam_updates'],1);self.assertGreater(r['checkpoint_recompute'],0)
    def test_submit_caps_and_no_resume(self):
        a=Path('/attempt');c={'runtime':{'python':'/python'}}
        for role in ('main','collector'):
            args=argv(role,a,'afterany:123');self.assertIn('--cpus-per-task=6',args);self.assertIn('--export=NONE',args);self.assertIn('--no-requeue',args)
            self.assertEqual('--gres=gpu:1' in args,role=='main');self.assertIn('--dependency=afterany:123',args)
            self.assertIn('OMP_NUM_THREADS=6',launcher(role,a,'source',c))
    def test_source_noCP_no_online_fit(self):
        path=ROOT/'project/run_scripts/fe_baseline/run.py';tree=ast.parse(path.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ('save','save_pretrained') for n in calls))
        self.assertEqual(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='fit' for n in calls),1)
        self.assertIn("range(1,horizon['batches']+1)",path.read_text());self.assertIn('table[j,:,(batch-1)*100:batch*100]',path.read_text())
    def test_tracking_required_before_submit(self):
        source=(ROOT/'project/run_scripts/fe_baseline/submit.py').read_text()
        self.assertLess(source.index('LOGGING_BLOCKED_SHARED_HELPER_NOT_BOUND'),source.index('before=inventory()'))
        self.assertIn('READY_ONLINE_VERIFIED',source)

if __name__=='__main__':unittest.main()
