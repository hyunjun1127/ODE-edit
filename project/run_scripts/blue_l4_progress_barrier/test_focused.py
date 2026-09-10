import unittest
import torch
from .algebra import spd_roots,correction,calibrate,risk,dot,groups

class FormulaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260911);torch.set_num_threads(2)
        self.dtype=torch.float64
    def rand(self,*shape):return torch.randn(*shape,dtype=self.dtype)
    def fixture(self):
        m=self.rand(5,5);b=m@m.T+torch.eye(5,dtype=self.dtype)
        root,inv,_=spd_roots(b)
        return b,root,inv,self.rand(3,5),self.rand(3,5),self.rand(2,3,5)
    def test_ep_kkt_progress(self):
        for _ in range(12):
            b,r,ir,v,a,j=self.fixture();ep,receipt=correction(v,a,j,r,ir,.13,-2.,'EP')
            self.assertLess(float((j*(ep-v)).flatten(1).sum(1).abs().max()),1e-10)
            self.assertLessEqual(float(dot(a,ep)+4-receipt['slack']),1e-10)
            if receipt['factor']>0:self.assertAlmostEqual(receipt['slack'],.13*receipt['factor'],places=10)
            # Stationarity in the feasible null space.
            residual=(ep-v)@b+receipt['factor']*a
            white=(residual@ir).flatten();rows=(j@ir).flatten(1)
            coeff=torch.linalg.lstsq(rows.T,white).solution
            self.assertLess(float((white-rows.T@coeff).norm()),1e-10)
    def test_r_and_empty_observer(self):
        b,r,ir,v,a,j=self.fixture();j.zero_()
        ep,_=correction(v,a,j,r,ir,.1,-1,'EP');rr,_=correction(v,a,j,r,ir,.1,-1,'R')
        torch.testing.assert_close(ep,rr,atol=1e-12,rtol=1e-12)
    def test_rank_deficient(self):
        b,r,ir,v,a,j=self.fixture();j=torch.stack([j[0],j[0]*1e-20,j[0]*2,torch.zeros_like(j[0])])
        _,rc=correction(v,a,j,r,ir,.1,-1,'EP')
        self.assertEqual(rc['rank'],1);self.assertEqual(rc['zero_rows'],1)
    def test_zero_risk_and_full_observer(self):
        b,r,ir,v,a,j=self.fixture()
        ep,rc=correction(v,torch.zeros_like(a),j,r,ir,0.,0.,'EP')
        torch.testing.assert_close(ep,v);self.assertEqual(rc['q'],0)
        j=torch.eye(v.numel(),dtype=self.dtype).reshape(v.numel(),*v.shape)
        ep,rc=correction(v,a,j,r,ir,.1,-1,'EP')
        torch.testing.assert_close(ep,v,atol=1e-12,rtol=1e-12)
    def test_risk_expansion(self):
        u=self.rand(7,5);c=self.rand(7,7);c=c@c.T;d=self.rand(3,7);x=self.rand(3,5);dx=self.rand(3,5)
        hc=u.T@c@u;cc=d@c@u;sf=2.3
        old,a=risk(x,cc,hc,sf);new,_=risk(x+dx,cc,hc,sf)
        self.assertAlmostEqual(float(new-old),float(dot(a,dx)+.5*dot(dx@hc,dx)/sf),places=10)
        direct=(dot((d+x@u.T)@c,d+x@u.T)-dot(d@c,d))/2/sf
        self.assertAlmostEqual(float(old),float(direct),places=10)
    def test_units_and_n16(self):
        b,r,ir,g,a,j=self.fixture();u=self.rand(7,5);hc=torch.eye(5,dtype=self.dtype)
        nu,r,ir,eps,rc=calibrate(g,b,u,2.,a,hc)
        v=-nu*torch.linalg.solve(b,g.T).T
        self.assertAlmostEqual(float((v@u.T).norm())/8,.02,places=11)
        ep,_=correction(v,a,j,r,ir,eps,-.7,'EP')
        ep16,_=correction(v,a,j,r,ir,eps,-.7,'EP-N16')
        torch.testing.assert_close(ep,ep16)
        scaled,_=correction(v,13*a,j,r,ir,169*eps,-.7*13,'EP')
        torch.testing.assert_close(ep,scaled,atol=1e-10,rtol=1e-10)
    def test_group_mapping(self):
        ids=list(range(100));g=groups(ids)
        self.assertEqual(sorted(sum(g,[])),ids);self.assertEqual([len(v) for v in g],[25]*4)

    def test_existing_backward_request_observer(self):
        from types import SimpleNamespace
        from .objective import ProgressObjective
        from project.run_scripts.single_layer_cumulative_risk.objective import AffineForward
        from project.run_scripts.single_layer_cumulative_risk.records import Ledger
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(5)])
                self.model.layers[4].mlp=torch.nn.Module();self.model.layers[4].mlp.down_proj=torch.nn.Linear(3,2,bias=False)
                self.emb=torch.nn.Embedding(10,3);self.head=torch.nn.Linear(2,10)
            def forward(self,input_ids,attention_mask,use_cache=False):
                return SimpleNamespace(logits=self.head(self.model.layers[4].mlp.down_proj(self.emb(input_ids))))
        m=Tiny().requires_grad_(False);u=torch.randn(3,2);entry=m.model.layers[4].mlp.down_proj.weight.detach().clone()
        obj=ProgressObjective.__new__(ProgressObjective);obj.forward=AffineForward(m,entry,u,Ledger());obj.ledger=obj.forward.ledger
        obj.requests=list(range(100));obj.contexts=['{}']*6;obj.microbatch=2;obj.tok=SimpleNamespace(pad_token_id=0)
        obj.training=[[([i%8+1,2,3],[4])] * 6 for i in range(100)];obj.essence=[([1,2],1)]*100
        obj.teacher=[torch.zeros(10).log_softmax(0)]*100;obj.metric=torch.eye(2,dtype=torch.float64)
        obj.penalty_cross=torch.zeros(2,2,dtype=torch.float64);obj.penalty_constant=2.;obj.j_native=2.
        terms,g,j=obj.grouped(torch.zeros(2,2),groups(list(range(100))))
        self.assertEqual(obj.last_request_gradients.shape,(100,2,2))
        self.assertLess(max(terms['request_gradient_group_pullback_difference']),2e-6)
        self.assertEqual(obj.ledger.counts['edit_backward'],312)
        self.assertEqual(obj.ledger.counts['essence_backward'],50)
        self.assertEqual(obj.ledger.counts['request_gradient_observer_bmm'],312)

if __name__=='__main__':unittest.main()
