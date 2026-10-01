import copy
import unittest
import torch
from project.run_scripts.jlz_pilot.test_integration import DriverIntegrationTest
from project.run_scripts.jlz_pilot.solver import solve as original_solve
from project.run_scripts.jlz_sequential.oracle import Oracle
from .core import effective_linear,crop_tokens,RouteOracle,DirectLinear,early_keys
from .budget import BudgetAccountant,PanelBudget
from .solver import solve

class BudgetTest(unittest.TestCase):
    def test_shared_final_caps_and_original_math(self):
        torch.set_num_threads(1)
        for cap in (2,3,12,120):
            x=torch.zeros(2,3);c=torch.zeros(2);rho=torch.ones(2)*5;mask=torch.tensor([True,False])
            fun=lambda x:(float(((x-2)*torch.tensor([1.,2.,4.])).square().sum()),2*(x-2)*torch.tensor([1.,4.,16.]),{})
            ref=original_solve(fun,x,c,rho,mask,cap=cap,tol=1e-20);a=BudgetAccountant(cap)
            out=solve(fun,x,c,rho,mask,cap=cap,tol=1e-20,account=a)
            self.assertEqual(out['calls'],a.used);self.assertEqual(a.events[-1],'final');self.assertTrue(torch.equal(ref['x'],out['x']))
            self.assertEqual((ref['status'],ref['backtracks'],ref['accepted_steps']),(out['status'],out['backtracks'],out['accepted_steps']))
    def test_rechecks_reserve_and_final_failure(self):
        a=BudgetAccountant(3);a.charge('initial');a.charge('history_reference')
        with self.assertRaises(RuntimeError):a.charge('trial')
        a.charge('final');self.assertEqual(a.used,3)
        with self.assertRaises(RuntimeError):a.charge('final')
        calls=[]
        def f(x):
            calls.append(1)
            if len(calls)==2:raise FloatingPointError('FINAL_FAIL')
            return 1.,torch.ones_like(x),{}
        out=solve(f,torch.zeros(1,2),torch.zeros(1),torch.ones(1),torch.ones(1,dtype=torch.bool),cap=2)
        self.assertFalse(out['final_recomputed']);self.assertEqual(out['calls'],2)
    def test_partition_budgets_no_reserve_transfer(self):
        b=PanelBudget()
        for _ in range(12):b.charge('short','A')
        with self.assertRaises(RuntimeError):b.charge('short','A')
        self.assertEqual(b.counts['reserve'],0)
        for _ in range(8):b.charge('B100')
        with self.assertRaises(RuntimeError):b.charge('B100')

class Integration(DriverIntegrationTest):
    def test_all_dense_routes_actual_production_reference(self):
        ref=Oracle(self.model,self.spec,self.teacher,self.keys,self.adj,self.active,2)(self.point)
        for route in ('E1_MB2','E12_MB2','E123_MB4','E123_MB8','E123_SYNC_MB2','E123_DIRECT_R_MB2'):
            o=RouteOracle(self.model,self.spec,self.teacher,self.adj,self.active,route)
            value,gradient,payload=o(self.point)
            self.assertLessEqual(abs(value-ref[0]),1e-3)
            self.assertLessEqual(float((gradient-ref[1]).norm()/ref[1].norm().clamp_min(1)),1e-5)
            for l in self.keys:self.assertTrue(torch.equal(payload['weights'][l],ref[2]['weights'][l]))
    def test_override_restores_after_forward_backward_exceptions(self):
        original={l:self.model.model.layers[l].mlp.down_proj.forward for l in self.keys}
        eff={l:self.model.model.layers[l].mlp.down_proj.weight for l in self.keys}
        for stage in ('forward','loss','backward'):
            with self.assertRaisesRegex(RuntimeError,stage):
                with effective_linear(self.model,eff):raise RuntimeError(stage)
            for l,fn in original.items():self.assertEqual(fn,self.model.model.layers[l].mlp.down_proj.forward)
    def test_crop_coordinates_and_key_stop(self):
        for start in range(0,len(self.spec['lookup']),2):
            rows=list(range(start,min(start+2,len(self.spec['lookup']))))
            t,w=crop_tokens(self.spec,rows,True)
            self.assertTrue(bool((self.spec['targets'][rows,w:]==-100).all()))
        out=early_keys(self.model,self.spec,self.contexts,2)
        for l in out:self.assertTrue(torch.equal(out[l],self.keys[l]))

class DirectTest(unittest.TestCase):
    def test_dense_overflow_not_hidden_and_input_gradient(self):
        x=torch.tensor([[1e20,-1e20]],requires_grad=True);q=torch.ones(2,1,dtype=torch.float64);w=torch.zeros(1,2);r=torch.zeros(1,1,dtype=torch.float64,requires_grad=True)
        ledger={'bound':torch.zeros((),dtype=torch.float64),'finite':torch.ones((),dtype=torch.bool)}
        y=DirectLinear.apply(x,r,w,q,None,ledger);dx,dr=torch.autograd.grad(y,(x,r),torch.tensor([[1e20]]))
        self.assertTrue(torch.isfinite(dr).all());self.assertGreater(float(ledger['bound']),torch.finfo(torch.float32).max/2)
        self.assertTrue(torch.equal(dx,torch.zeros_like(x)))
        torch.manual_seed(3);x=torch.randn(3,5,requires_grad=True);w=torch.randn(7,5);q=torch.randn(5,2,dtype=torch.float64);r=torch.zeros(7,2,dtype=torch.float64,requires_grad=True)
        ledger={'bound':torch.zeros((),dtype=torch.float64),'finite':torch.ones((),dtype=torch.bool)}
        y=DirectLinear.apply(x,r,w,q,None,ledger);g=torch.randn_like(y);dx,dr=torch.autograd.grad(y,(x,r),g)
        self.assertTrue(torch.equal(dx,g@w));torch.testing.assert_close(dr,g.double().T@(x.double()@q))

if __name__=='__main__':unittest.main()
