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
    def test_private_step_reuse_under_budget_is_no_correction(self):
        from .controller_repair import step
        from . import controller
        from .geometry import Projector,Metric
        metric=Metric(Projector.from_raw(torch.eye(3,dtype=torch.float64)),torch.eye(3,dtype=torch.float64),.1)
        metric.bind_progress(torch.ones(2,3,dtype=torch.float64))
        z=torch.zeros(2,3,dtype=torch.float64);g=[torch.ones_like(z),torch.eye(2,3,dtype=torch.float64)]
        v=torch.zeros(2,dtype=torch.float64)
        zn,c,r=step(metric,z,g,v,v,torch.ones_like(v),0.,.125,torch.ones_like(v),1.)
        self.assertTrue(torch.equal(zn,z));self.assertTrue(torch.equal(c,z))
        self.assertEqual(r['xi'],[0.,0.]);self.assertIs(controller.joint_dual,old)

if __name__=='__main__':unittest.main()
