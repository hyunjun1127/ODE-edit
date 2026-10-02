"""Production-module CPU regression; not Llama qualification."""
import unittest
import torch
from .dynamic_solve import solve
from .allocation import loss as allocation, Root
from .physical_linear import materialize,linear
from .optimize import partitions

class Core(unittest.TestCase):
    def setUp(self):torch.manual_seed(20261002);torch.set_num_threads(2)
    def test_solve_value_gradient_explicit_K_T(self):
        A=torch.randn(7,7,dtype=torch.float64);A=A@A.T+torch.eye(7,dtype=torch.float64)
        K=torch.randn(7,6,dtype=torch.float64,requires_grad=True)
        alpha=torch.tensor([.5,.5,0.,.5,.1,.4],dtype=torch.float64);owners=torch.tensor([0,0,0,1,1,1])
        out=solve(K,torch.linalg.cholesky(A),alpha,owners,2)
        z=torch.nn.functional.one_hot(owners,2).T.double()
        p=torch.linalg.solve(A+(K*alpha)@K.T,(K*alpha)@z.T)
        r=p.T@K-z;G=p.T@A@p;E=(r*alpha)@r.T
        torch.testing.assert_close(out['P'],p,atol=1e-9,rtol=1e-8)
        D=torch.randn(3,2,dtype=torch.float64,requires_grad=True)
        f=(D.T@D*(out['G']+out['E']).T).sum()+out['P'].square().sum()
        ref=(D.T@D*(G+E).T).sum()+p.square().sum()
        g=torch.autograd.grad(f,(K,D),retain_graph=True);gr=torch.autograd.grad(ref,(K,D))
        for x,y in zip(g,gr):torch.testing.assert_close(x,y,atol=1e-9,rtol=1e-8)
    def test_direct_D_P_input(self):
        for dtype in (torch.float32,torch.float64):
            x=torch.randn(2,3,7,dtype=dtype,requires_grad=True)
            D=torch.randn(5,2,dtype=dtype,requires_grad=True);P=torch.randn(7,2,dtype=torch.float64,requires_grad=True)
            entry=torch.randn(5,7,dtype=dtype)
            W=materialize(entry,D,P);g=torch.randn(2,3,5,dtype=dtype)
            direct=linear(x,D,P,W.detach())
            ref=torch.nn.functional.linear(x,W)
            torch.testing.assert_close(direct,ref,atol=0,rtol=0)
            gd=torch.autograd.grad((direct*g).sum(),(x,D,P),retain_graph=True)
            gr=torch.autograd.grad((ref*g).sum(),(x,D,P))
            for v,r in zip(gd,gr):torch.testing.assert_close(v,r,atol=2e-6,rtol=2e-6)
    def test_causal_total_gradient_and_stopP(self):
        ds=[torch.randn(3,2,dtype=torch.float64,requires_grad=True)*.1 for _ in range(3)]
        x0=torch.randn(4,3,dtype=torch.float64);entries=[torch.randn(3,3,dtype=torch.float64)*.2 for _ in ds]
        L=torch.eye(3,dtype=torch.float64);alpha=torch.ones(4,dtype=torch.float64)*.5;owner=torch.tensor([0,0,1,1])
        def f(values,stop=False):
            x=x0;geo={}
            for l,d in enumerate(values):
                k=x.tanh().T
                o=solve(k,L,alpha,owner,2)
                p=o['P'].detach() if stop else o['P']
                if stop:
                    z=torch.nn.functional.one_hot(owner,2).T.double();r=p.T@k-z
                    o=dict(G=p.T@p,E=(r*alpha)@r.T)
                geo[l]=o;x=x+torch.nn.functional.linear(x.tanh(),entries[l]+d@p.T)
            policy,_=allocation(dict(enumerate(values)),geo,{l:torch.ones(2) for l in range(3)},'B')
            return x.square().mean()+policy
        self.assertTrue(torch.autograd.gradcheck(lambda *v:f(v),tuple(ds),eps=1e-6,atol=2e-7,rtol=5e-5))
        gd=torch.autograd.grad(f(ds),ds);gs=torch.autograd.grad(f(ds,True),ds)
        self.assertGreater(float((gd[0]-gs[0]).abs().max()),1e-7)
    def test_zero_and_partitions(self):
        q=torch.tensor(0.,requires_grad=True);Root.apply(q).backward();self.assertEqual(float(q.grad),0.)
        for n in (0,1,2,3,100):
            p=partitions(n,20261002,'fixed',2,'current')
            self.assertEqual(sorted(sum(p,[])),list(range(n)))
            self.assertEqual(p,partitions(n,20261002,'fixed',2,'current'))

if __name__=='__main__':unittest.main()
