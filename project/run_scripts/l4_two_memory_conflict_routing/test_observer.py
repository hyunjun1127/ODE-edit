import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from .observer import View,psi,pack
from .identity import Ledger,WEIGHT

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embed=torch.nn.Embedding(11,6)
        self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(5)])
        self.model.layers[4].mlp=torch.nn.Module();self.model.layers[4].mlp.down_proj=torch.nn.Linear(6,4,bias=False)
        self.head=torch.nn.Linear(4,11,bias=False)
    def forward(self,input_ids,attention_mask,use_cache=False):
        x=self.embed(input_ids);x=torch.tanh(self.model.layers[4].mlp.down_proj(x))
        return SimpleNamespace(logits=self.head(x))

class ObserverTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2);self.model=Toy().eval().requires_grad_(False);self.w=dict(self.model.named_parameters())[WEIGHT]
        self.rows=[dict(input_ids=[1,2,3,4][:i+2],positions=list(range(1,i+2)),target_ids=[5]*(i+1),
            identity=str(i),ordinal=i,case_id=i,context_index=0,token_weight=1/(3*(i+1)),context_weight=1/3) for i in range(3)]
    def test_packing_shift_and_uneven_weights(self):
        inp,pos=pack(self.rows,0,'cpu')
        self.assertEqual([p.tolist() for p in pos],[[3],[2,3],[1,2,3]])
        self.assertEqual(inp['attention_mask'].sum().item(),9)
        self.assertAlmostEqual(sum(r['token_weight']*len(r['target_ids']) for r in self.rows),1.)
    def test_kl_teacher_direction_microbatch_gradient_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            view=View(self.model,0,Ledger());before=self.w.clone();version=self.w._version
            teacher,factor=view.prepare(self.w,self.rows,Path(tmp)/'teacher.pt',micro=2)
            self.assertEqual(factor.shape,(6,6))
            state=self.w+.04*torch.randn_like(self.w)
            outputs=[]
            for micro in (1,2,8):
                obs,g=view.harm(state,teacher,'Base',gradient=True,micro=micro);outputs.append((obs,g))
            self.assertLess(abs(outputs[0][0]['value']-outputs[2][0]['value']),1e-7)
            self.assertTrue(torch.allclose(outputs[0][1],outputs[2][1],atol=2e-7,rtol=1e-5))
            d=torch.randn_like(state);analytic=float((outputs[0][1]*d).sum())
            # FP32 forward central FD: binary neighboring scales expose rounding
            # and truncation separately; this is a CPU fixture, not a run knob.
            for eps in (2**-6,2**-7,2**-8):
                hi,_=view.harm(state+eps*d,teacher,'Base');lo,_=view.harm(state-eps*d,teacher,'Base')
                numeric=(hi['value']-lo['value'])/(2*eps)
                self.assertAlmostEqual(numeric,analytic,delta=2e-5)
            self.assertTrue(torch.equal(before,self.w));self.assertEqual(version,self.w._version)
            zero,_=view.harm(self.w,teacher,'Base');self.assertLess(zero['value'],1e-7)
            with torch.no_grad():self.model.head.weight.add_(.1)
            with self.assertRaisesRegex(RuntimeError,'LIVE_PARAMETER_MUTATION'):view.forward(self.w,self.rows[:1])
    def test_context_psi_before_request_mean(self):
        d=torch.tensor([-.1,.02,.005],dtype=torch.float64)
        self.assertTrue(torch.allclose(psi(d),torch.tensor([0.,.015,.00125],dtype=torch.float64)))
        self.assertGreater(float(psi(d).mean()),float(psi(d.mean())))

if __name__=='__main__':unittest.main()
