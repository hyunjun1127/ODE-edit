"""Remaining small contract edge fixtures; no model/GPU."""
import unittest
import torch
from .controller import joint_dual
from .geometry import Projector,static_solution

class EdgeTest(unittest.TestCase):
    def test_full_declared_batch_topology_past_matrix(self):
        torch.manual_seed(20260911);torch.set_num_threads(2)
        p=Projector.from_raw(torch.diag(torch.tensor([1.,1.,1.,0.],dtype=torch.float64)))
        for b in (1,7,64,100,257,1000):
            for mode in ('ordinary','duplicate','collinear'):
                for past_n in (0,3):
                    with self.subTest(b=b,mode=mode,past=past_n):
                        k=torch.randn(4,b,dtype=torch.float64)
                        if mode=='duplicate' and b>1:k[:,1]=k[:,0]
                        if mode=='collinear':k=k[:,:1].expand(-1,b).clone()
                        r=torch.randn(2,b,dtype=torch.float64);delta=torch.randn(2,4,dtype=torch.float64)
                        factors=[torch.randn(4,n,dtype=torch.float64)/max(n,1)**.5 for n in (6*b,past_n,3)]
                        z,m,rc=static_solution(delta,k,r,*factors,p)
                        self.assertTrue(bool(torch.isfinite(z).all()))
                        self.assertLess(rc['algebraic_equality_relative'],1e-10)
                        self.assertLess(rc['span_relative'],1e-10)
                        self.assertLess(rc['stationarity_allowed_norm'],1e-9)
    def test_all_four_joint_active_sets(self):
        matrix=torch.tensor([[2.,.3],[.3,1.]],dtype=torch.float64)
        for active in ((False,False),(True,False),(False,True),(True,True)):
            exact=torch.tensor([.2 if a else 0 for a in active],dtype=torch.float64)
            dual_gradient=torch.tensor([0 if a else .1 for a in active],dtype=torch.float64)
            rhs=matrix@exact-dual_gradient
            result,rc=joint_dual(matrix,rhs)
            self.assertTrue(torch.allclose(result,exact,rtol=0,atol=1e-12))
            self.assertEqual(tuple(rc['active']),active)
    def test_hard_null_exhaustion_does_not_remove_soft_method(self):
        p=Projector.from_raw(torch.eye(3,dtype=torch.float64))
        k=torch.eye(3,dtype=torch.float64);r=torch.ones(2,3,dtype=torch.float64)
        empty=k[:,:0];delta=torch.zeros(2,3,dtype=torch.float64)
        self.assertEqual(torch.linalg.matrix_rank(k).item(),3)
        z,metric,rc=static_solution(delta,k,r,empty,empty,empty,p)
        # Here the exact optimum is Z=0. Relative residual divided by ||Z||
        # is undefined at that solution; use the absolute unit-scale error.
        self.assertTrue(torch.isfinite(z).all());self.assertLess(float(z.norm()),1e-10)
        self.assertLess(abs(rc['algebraic_equality_residual']),1e-10)
    def test_bank_weighted_duplication(self):
        torch.manual_seed(3)
        keys=torch.randn(7,9,dtype=torch.float64);weights=torch.rand(9,dtype=torch.float64);weights/=weights.sum()
        gram=(keys*weights)@keys.T
        duplicate=keys.repeat_interleave(2,1);ww=weights.repeat_interleave(2)/2
        self.assertTrue(torch.allclose(gram,(duplicate*ww)@duplicate.T,rtol=1e-12,atol=1e-12))

if __name__=='__main__':unittest.main()
