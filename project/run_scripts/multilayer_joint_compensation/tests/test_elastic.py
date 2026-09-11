import itertools
import unittest
import torch
from project.run_scripts.multilayer_joint_compensation.elastic_qp import joint_dual, solve_constrained


def dense_primal_oracle(k,u,a,t,g,risks,budgets,h,rho):
    """Independent augmented primal equality solve for each active inequality.

    This tiny CPU test intentionally forms a dense fixture matrix; production
    kernels never build the full weight-space Hessian/inverse.
    """
    d=u.numel();q=torch.block_diag(k,torch.diag(rho/h));linear=torch.cat([u,torch.zeros(2,dtype=u.dtype)])
    solutions=[]
    # Nonnegative slack rows are included as independent primal constraints.
    eq=torch.cat([a,torch.zeros(2,dtype=u.dtype)])[None,:]
    ine=torch.cat([torch.cat([g.T,-torch.eye(2,dtype=u.dtype)],1),
                   torch.cat([torch.zeros(2,d,dtype=u.dtype),-torch.eye(2,dtype=u.dtype)],1)],0)
    bound=torch.cat([budgets-risks,torch.zeros(2,dtype=u.dtype)])
    for mask in itertools.product([False,True],repeat=4):
        ids=torch.tensor(mask,dtype=torch.bool)
        constraints=torch.cat([eq,ine[ids]],0);b=torch.cat([t.reshape(1),bound[ids]])
        saddle=torch.cat([torch.cat([q,constraints.T],1),torch.cat([constraints,torch.zeros(len(b),len(b),dtype=u.dtype)],1)],0)
        try:answer=torch.linalg.solve(saddle,torch.cat([-linear,b]))
        except torch.linalg.LinAlgError:continue
        x=answer[:d+2];mult=answer[d+2:]
        if (ine@x-bound).max()>1e-9 or (mult[1:] < -1e-9).any():continue
        solutions.append((float(.5*x@q@x+linear@x),x))
    if not solutions:raise AssertionError('DENSE_PRIMAL_ORACLE_EMPTY')
    return min(solutions,key=lambda row:row[0])[1]


class TestElastic(unittest.TestCase):
    def test_joint_against_independent_primal(self):
        gen=torch.Generator().manual_seed(9611);seen=set()
        for _ in range(32):
            d=5;m=torch.randn(d,d,generator=gen,dtype=torch.float64)
            k=m.T@m+torch.eye(d,dtype=torch.float64)
            u=torch.randn(d,generator=gen,dtype=torch.float64);a=torch.randn(d,generator=gen,dtype=torch.float64)
            g=torch.randn(d,2,generator=gen,dtype=torch.float64)
            risks=torch.randn(2,generator=gen,dtype=torch.float64);budgets=torch.zeros_like(risks)
            t=torch.randn((),generator=gen,dtype=torch.float64)*.1;h=.25;rho=torch.tensor([2.,1.],dtype=torch.float64)
            r=solve_constrained(lambda x:(k@x[0],),(u,),(a,),float(t),
                gradients=((g[:,0],),(g[:,1],)),risks=risks,budgets=budgets,h=h,rtol=1e-12)
            oracle=dense_primal_oracle(k,u,a,t,g,risks,budgets,h,rho)
            torch.testing.assert_close(torch.cat([r.correction[0],r.slack]),oracle,rtol=1e-9,atol=1e-10)
            self.assertTrue(bool((r.dual>=0).all()));self.assertLess(abs(r.diagnostics['equality_residual']),1e-10)
            self.assertLess(r.diagnostics['stationarity_norm'],1e-9)
            self.assertLess(max(r.diagnostics['primal_residual']),1e-10)
            seen.add(tuple(r.diagnostics['dual_diagnostics']['active']))
        self.assertEqual(len(seen),4)

    def test_negative_dual_never_admitted_or_clipped(self):
        nu,receipt=joint_dual(torch.eye(2,dtype=torch.float64),torch.tensor([-1e-18,1.],dtype=torch.float64))
        self.assertEqual(nu.tolist(),[0.,1.]);self.assertEqual(receipt['active'],[False,True])

    def test_os_and_zero_current(self):
        u=(torch.tensor([2.,-3.],dtype=torch.float64),);zero=(torch.zeros(2,dtype=torch.float64),)
        r=solve_constrained(lambda x:x,u,zero,0.)
        torch.testing.assert_close(r.correction[0],-u[0]);self.assertEqual(r.diagnostics['independent_rhs_count'],1)
        r=solve_constrained(lambda x:x,u,zero,-1.)
        self.assertEqual(r.diagnostics['status'],'CURRENT_DEFICIT_FIRST_ORDER_UNRECOVERABLE')


if __name__=='__main__':unittest.main()
