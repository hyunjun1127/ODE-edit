import unittest,torch
from .algebra import capacity,joint_residual,psd
from .spg import solve
from .plan import past_panel
class Core(unittest.TestCase):
 def test_rank_deficient_and_noncommuting(self):
  torch.manual_seed(3);a=torch.randn(8,8,dtype=torch.float64);a=a@a.T+torch.eye(8);k=torch.randn(8,3,dtype=torch.float64);k[:,2]=k[:,0]
  adj=torch.linalg.solve(a+k@k.T,k);g,rec=capacity(k,adj,lambda k:torch.linalg.solve(a,k),True);self.assertLess(rec['adj_identity_error'],1e-8)
  h=torch.diag(torch.tensor([1.,2.,3.],dtype=torch.float64));r=torch.randn(4,3,dtype=torch.float64)
  self.assertTrue(torch.equal(joint_residual(r,[g]),r));torch.testing.assert_close(joint_residual(r,[g,h]),r@torch.linalg.solve(torch.eye(3)+g+h,torch.eye(3)+g))
 def test_zero_and_nonzero_solver(self):
  def f(x):return float(((x-2)**2).sum()),2*(x-2),{}
  x,r=solve(f,3,1.,1e-6,100,device='cpu');self.assertEqual(r['status'],'CONVERGED');self.assertLessEqual(r['calls'],100);self.assertTrue(r['final_recomputed'])
  x,r=solve(lambda x:(.01,torch.zeros_like(x),{}),2,1.,1e-6,25,device='cpu');self.assertEqual(r['status'],'POLICY_ZERO_STEP');self.assertEqual(r['calls'],2)
 def test_no_forced_armijo_or_stale_pg(self):
  n=[0]
  def f(x):n[0]+=1;return (1. if n[0]==1 or torch.count_nonzero(x)==0 else 100.),torch.ones_like(x),{}
  x,r=solve(f,2,1.,1e-6,100,device='cpu');self.assertEqual(r['status'],'LINESEARCH_FAILED');self.assertEqual(n[0],r['calls']);self.assertTrue(torch.equal(x,torch.zeros(2)))
 def test_panel(self):
  rows=[dict(requested_rewrite=dict(v=i)) for i in range(1000)];p=past_panel(rows,1000);self.assertEqual(len(p),400);self.assertEqual([sum(i//250==q for i in p) for q in range(4)],[100]*4)
 def test_condition_fallback_without_jitter(self):
  a=torch.eye(2,dtype=torch.float64)*1e-8;k=torch.eye(2,dtype=torch.float64)
  adj=torch.linalg.solve(a+k@k.T,k)
  g,r=capacity(k,adj,lambda k:torch.linalg.solve(a,k))
  self.assertTrue(r['direct_fallback']);torch.testing.assert_close(g,torch.eye(2,dtype=torch.float64)*1e8)
if __name__=='__main__':unittest.main()
