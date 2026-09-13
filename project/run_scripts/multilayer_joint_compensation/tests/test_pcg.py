import unittest
import torch
from project.run_scripts.multilayer_joint_compensation.linear_solve import pcg, dot


class TestPCG(unittest.TestCase):
    def test_spd_tuple_and_true_residual(self):
        gen=torch.Generator().manual_seed(771)
        m=torch.randn(7,7,generator=gen,dtype=torch.float64)
        k=m.T@m+torch.eye(7,dtype=torch.float64)
        rhs=torch.randn(7,generator=gen,dtype=torch.float64)
        split=lambda v:(v[:3],v[3:].reshape(2,2))
        flatten=lambda x:torch.cat([t.reshape(-1) for t in x])
        calls=[]
        def op(x):calls.append(1);return split(k@flatten(x))
        result=pcg(op,split(rhs),rtol=1e-11,maxiter=20)
        torch.testing.assert_close(flatten(result.solution),torch.linalg.solve(k,rhs),rtol=1e-10,atol=1e-11)
        self.assertEqual(result.status,'CONVERGED')
        self.assertEqual(result.matvec_calls,len(calls))
        self.assertEqual(result.matvec_calls,result.iterations+1)
        self.assertLess(result.relative_residual,1e-11)

    def test_empty_zero_rhs(self):
        def forbidden(x):raise AssertionError('ZERO_RHS_OPERATOR_CALL')
        for rhs in [(),(torch.zeros(3),)]:
            r=pcg(forbidden,rhs)
            self.assertEqual(r.status,'ZERO_RHS');self.assertEqual(r.matvec_calls,0)

    def test_approximate_is_not_capacity_failure(self):
        k=torch.diag(torch.tensor([1.,3.,7.],dtype=torch.float64))
        r=pcg(lambda x:(k@x[0],),(torch.ones(3,dtype=torch.float64),),maxiter=1)
        self.assertEqual(r.status,'APPROXIMATE_PCG_NONCONVERGENCE')
        self.assertGreater(r.relative_residual,1e-4)

    def test_bad_curvature_and_shape_fail(self):
        with self.assertRaisesRegex(FloatingPointError,'CURVATURE'):
            pcg(lambda x:(-x[0],),(torch.ones(3),))
        with self.assertRaisesRegex(ValueError,'SHAPE'):
            dot((torch.ones(2),),(torch.ones(3),))

    def test_exact_preconditioner(self):
        diagonal=torch.tensor([.1,1,1000.],dtype=torch.float64)
        r=pcg(lambda x:(diagonal*x[0],),(torch.ones(3,dtype=torch.float64),),lambda x:(x[0]/diagonal,))
        self.assertEqual(r.iterations,1)


if __name__=='__main__':unittest.main()
