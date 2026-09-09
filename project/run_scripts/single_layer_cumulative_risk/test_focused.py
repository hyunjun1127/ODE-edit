import unittest
import torch
from .algebra import action,reduced_metric,calibrated_eta,momentum_update,orthobasis
from .objective import essence_kl,AffineForward,WEIGHT
from .records import Ledger
from .panels import writer_plan,choose_alpha,select
from .microbatch import bounded_reader

class CoreTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(20260910)
    def test_orthobasis_action(self):
        u,_=orthobasis(torch.randn(9,3,dtype=torch.float64));a=torch.randn(4,3,dtype=torch.float64)
        m=torch.randn(9,9,dtype=torch.float64);m=m@m.T
        torch.testing.assert_close((a@u.T).norm(),a.norm())
        torch.testing.assert_close(action(a@u.T,m),((a@reduced_metric(u,m))*a).sum())
    def test_momentum_lr_once(self):
        a=torch.zeros(4,3);g=torch.randn_like(a)
        eta,_=calibrated_eta(.06,torch.tensor(2.),g)
        new,v=momentum_update(a,g,None,eta)
        torch.testing.assert_close(new.norm(),torch.tensor(.12))
        new2,v2=momentum_update(new,2*g,v,eta)
        torch.testing.assert_close(v2,2.9*g)
    def test_zero_gradient_not_rescue(self):
        self.assertEqual(calibrated_eta(.02,torch.tensor(1.),torch.zeros(2))[0],0.)
    def test_native_physical_batch_preserves_order_and_both(self):
        def reader(model,tok,contexts,idxs,layer,module,track):
            x=torch.tensor([c+i[0] for c,i in zip(contexts,idxs)])[:,None]
            return (x,x+1) if track=='both' else x
        ledger=Ledger();args=(None,None,list(range(7)),[[j] for j in range(7)],4,'layer.{}','both')
        actual=bounded_reader(reader,ledger,2)(*args)
        for a,b in zip(actual,reader(*args)):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(ledger.counts['native_representation_physical_batches'],4)
    def test_essence_direction_coefficient(self):
        t=torch.tensor([[.8,.2]],dtype=torch.float64).log()
        s=torch.tensor([[.3,.7]],dtype=torch.float64).log().requires_grad_()
        expected=(s.exp()*(s-t)).sum()
        torch.testing.assert_close(essence_kl(t,s),expected)
        self.assertGreater(abs(float(essence_kl(t,s)-essence_kl(s,t))),.01)
        torch.testing.assert_close(.0625*essence_kl(t,s),expected/16)
    def test_train_only_selection(self):
        inputs={a:[dict(step=s,objective=a,NS=100-a) for s in range(29,33)] for a in [.02,.06,.2]}
        endpoints={a:dict(completed_steps=32,finite=True,endpoint_sha256='fixture',endpoint_exists_verified=True) for a in inputs}
        self.assertEqual(choose_alpha(inputs,endpoints)[0],.02)
        endpoints[.02]['finite']=False
        self.assertEqual(choose_alpha(inputs,endpoints)[0],.06)
        inputs[.06][0]['objective']=float('nan')
        self.assertEqual(choose_alpha(inputs,endpoints)[0],.2)
        endpoints[.2]['endpoint_exists_verified']=False
        with self.assertRaises(RuntimeError):choose_alpha(inputs,endpoints)
    def test_plan_denominator(self):
        plan=writer_plan();self.assertEqual(len(plan),13)
        self.assertEqual(sum(r['arm']!='N' for r in plan)*32,320)
    def test_panels_outcome_blind(self):
        records=[dict(case_id=i) for i in range(10000)]
        for e,start in [('Early',1000),('Middle',5000),('Late',9000)]:
            p=select(records,e)
            self.assertEqual(p['panels']['Current100'],list(range(start,start+100)))
            self.assertEqual(len(p['generation']),20)
            self.assertTrue(all(len(v)==2 for v in p['neighbors'].values()))
            self.assertEqual(p,select(records,e))
    def test_functional_weight_all_tokens(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,4,bias=False),torch.nn.Tanh(),torch.nn.Linear(4,2,bias=False)).double()
        x=torch.randn(2,5,3,dtype=torch.float64);w=model[0].weight.detach().clone()
        u,_=torch.linalg.qr(torch.randn(3,2,dtype=torch.float64));a=torch.randn(4,2,dtype=torch.float64,requires_grad=True)
        ptr=model[0].weight.data_ptr();version=model[0].weight._version
        out=torch.func.functional_call(model,{'0.weight':w+a@u.T},(x,))
        explicit=torch.nn.functional.linear(torch.tanh(torch.nn.functional.linear(x,w+a@u.T)),model[2].weight)
        torch.testing.assert_close(out,explicit)
        out.sum().backward();self.assertGreater(float(a.grad.norm()),0)
        self.assertEqual(model[0].weight.data_ptr(),ptr);self.assertEqual(model[0].weight._version,version)
        torch.testing.assert_close(model[0].weight,w,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
