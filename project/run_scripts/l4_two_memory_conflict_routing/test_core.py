import unittest
import torch
from .geometry import Projector,Metric,static_solution,dot
from .controller import joint_dual,step
from .banks import latest,candidate_inventory,select

class GeometryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260911);torch.set_num_threads(2)
        self.d=7; self.o=3
        q,_=torch.linalg.qr(torch.randn(self.d,self.d,dtype=torch.float64))
        self.p=Projector.from_raw(q[:,:5]@q[:,:5].T)
    def test_full_projector(self):
        p=self.p.right(torch.eye(self.d,dtype=torch.float64))
        self.assertLess(float((p@p-p).norm()),1e-12)
        self.assertEqual(self.p.receipt['rank'],5)
    def test_arbitrary_batch_static_stationarity(self):
        for b in (1,7,64,100,257,1000):
            with self.subTest(batch=b):
                k=torch.randn(self.d,b,dtype=torch.float64);r=torch.randn(self.o,b,dtype=torch.float64)
                if b==7:k[:,1]=k[:,0]
                delta=torch.randn(self.o,self.d,dtype=torch.float64)
                factors=[torch.randn(self.d,n,dtype=torch.float64)/max(n,1)**.5 for n in (6*b,0,4)]
                z,m,rc=static_solution(delta,k,r,*factors,self.p)
                self.assertLess(rc['algebraic_equality_relative'],1e-10)
                self.assertLess(rc['span_relative'],1e-10)
                self.assertLess(rc['stationarity_allowed_norm'],1e-9)
                rhs=torch.randn(self.o,self.d,dtype=torch.float64)
                inv=m.inverse(rhs)
                a=m.ridge*torch.eye(self.d,dtype=torch.float64)+m.factors@m.factors.T
                self.assertLess(float((self.p.right(inv@a-rhs)).norm()),1e-10)
    def test_zero_progress_and_batch_zero(self):
        k=torch.randn(self.d,1,dtype=torch.float64);empty=k[:,:0]
        delta=torch.randn(self.o,self.d,dtype=torch.float64)
        z,_,rc=static_solution(delta,empty,delta[:,:0],empty,empty,empty,self.p)
        self.assertEqual(rc['status'],'B0_NOOP');self.assertTrue((z==0).all())
        z,m,rc=static_solution(delta,k,torch.zeros(self.o,1,dtype=torch.float64),empty,empty,empty,self.p)
        self.assertTrue(rc['zero_allowed_progress'])
    def test_we_not_w0_and_base_cross_term(self):
        delta=torch.randn(self.o,self.d,dtype=torch.float64);k=torch.randn(self.d,7,dtype=torch.float64)
        r=torch.randn(self.o,7,dtype=torch.float64);f=torch.randn(self.d,4,dtype=torch.float64)
        z,m,_=static_solution(delta,k,r,f,f,f,self.p)
        # W0 cannot be passed to static_solution; change audit reference only.
        for w0 in (torch.zeros_like(delta),torch.randn_like(delta)):
            zz,_,_=static_solution(delta,k,r,f,f,f,self.p)
            self.assertTrue(torch.equal(z,zz))
        ll=delta@(k@k.T/7+2*f@f.T+m.ridge*torch.eye(self.d,dtype=torch.float64))-r@k.T/7
        self.assertLess(float((z+m.allowed_inverse(ll)).norm()),1e-11)
    def test_joint_step_anchor_time_and_primal_stationarity(self):
        f=torch.randn(self.d,5,dtype=torch.float64);m=Metric(self.p,f,.2)
        m.bind_progress(torch.randn(self.o,self.d,dtype=torch.float64))
        z=m.allowed_inverse(torch.randn_like(m.t));g=[torch.randn_like(z),torch.randn_like(z)]
        for h in (1.,.125):
            v=torch.tensor([.2,.1],dtype=torch.float64);eps=torch.tensor([.01,.02],dtype=torch.float64)
            zn,c,rc=step(m,z,g,v,v+2,v*.5,0.,h,eps,1.)
            self.assertLess(abs(float(dot(m.t,c))),1e-10)
            self.assertTrue(all(x<1e-10 for x in rc['constraint_residual']))
            lam=torch.tensor(rc['kkt']['dual'],dtype=torch.float64)
            a=m.ridge*torch.eye(self.d,dtype=torch.float64)+f@f.T
            grad=c@a/h+zn@a+sum(l*gi for l,gi in zip(lam,g))
            self.assertLess(float(m.allowed_white(grad).norm()),1e-9)
            self.assertTrue(torch.allclose(torch.tensor(rc['xi'],dtype=torch.float64),h*eps*lam))
    def test_joint_not_independent_clip(self):
        a=torch.tensor([[2.,1.],[1.,2.]],dtype=torch.float64);e=torch.ones(2,dtype=torch.float64)
        lam,rc=joint_dual(a,e)
        self.assertTrue(torch.allclose(lam,e/3,atol=1e-12,rtol=0))
        self.assertLess(rc['complementarity'],1e-12)
    def test_weighted_duplication(self):
        k=torch.randn(self.d,7,dtype=torch.float64);r=torch.randn(self.o,7,dtype=torch.float64)
        self.assertTrue(torch.allclose(k@k.T/7,k.repeat_interleave(2,1)@k.repeat_interleave(2,1).T/14))
        self.assertTrue(torch.allclose(r@k.T/7,r.repeat_interleave(2,1)@k.repeat_interleave(2,1).T/14))

class BankTest(unittest.TestCase):
    def test_latest_and_hash_selection(self):
        rows=[dict(case_id=i,requested_rewrite=dict(subject=str(i),relation_id='r')) for i in range(1000)]
        rows[7]['requested_rewrite']['subject']='  6 '
        self.assertNotIn(6,latest(rows,range(10)));self.assertIn(7,latest(rows,range(10)))
        panel=dict(panels={'Fixed100':list(range(20)),'Past100':list(range(100,120))})
        inv=candidate_inventory(rows,300,list(range(300,307)),panel,'Middle')
        self.assertFalse(set(inv['past_candidates'])&set(range(20)))
        self.assertFalse(set(inv['base_candidates'])&set(inv['base_audit']))
        self.assertEqual(select(list(range(512)),list(range(512)))[:64],list(range(64)))
        self.assertEqual(len(select(list(range(7)),list(range(7)))),7)

if __name__=='__main__':unittest.main()
