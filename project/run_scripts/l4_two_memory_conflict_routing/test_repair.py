import unittest
import torch
from .controller_repair import joint_dual
from .controller import joint_dual as old

class RepairTest(unittest.TestCase):
    def test_large_gram_negative_rhs_is_inactive(self):
        a=torch.tensor([[8e9,8e4],[8e4,1.5e7]],dtype=torch.float64)
        b=torch.tensor([-.03,-.09],dtype=torch.float64)
        self.assertTrue(bool((old(a,b)[0]<0).any()))
        x,r=joint_dual(a,b)
        self.assertTrue(torch.equal(x,torch.zeros_like(x)))
        self.assertEqual(r['active'],[False,False])
    def test_all_active_sets_across_physical_scales(self):
        for scale in (1e-9,1.,1e9):
            a=torch.tensor([[2.,.3],[.3,1.]],dtype=torch.float64)*scale
            for active in ((False,False),(True,False),(False,True),(True,True)):
                exact=torch.tensor([.2 if k else 0 for k in active],dtype=torch.float64)/scale
                grad=torch.tensor([0 if k else .1 for k in active],dtype=torch.float64)
                x,r=joint_dual(a,a@exact-grad)
                self.assertTrue(torch.allclose(x,exact,rtol=1e-12,atol=0))
                self.assertTrue(bool((x>=0).all()))
                self.assertEqual(tuple(r['active']),active)
    def test_valid_positive_original_solution_exact(self):
        a=torch.tensor([[8e9,8e4],[8e4,1.5e7]],dtype=torch.float64)
        b=torch.tensor([.03,.09],dtype=torch.float64)
        self.assertTrue(torch.equal(old(a,b)[0],joint_dual(a,b)[0]))

if __name__=='__main__':unittest.main()
