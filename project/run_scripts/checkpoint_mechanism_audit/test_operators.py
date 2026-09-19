"""Small full-FP64 algebra tests; not actual-model gate evidence."""
import unittest

import torch

from .operators import (HistoryOperator, counterfactual_summary, elementwise_parity,
                        factor_from_response, fitting_transfer, fixed_probe_scores,
                        native_dense, native_mode_decomposition, norm_discrepancy,
                        reconstruction_metrics, solve_residual)


class OperatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.generator=torch.Generator().manual_seed(932)
        self.d,self.n,self.o=9,4,6
        q,_=torch.linalg.qr(self.rand(self.d,self.d))
        self.p=q[:,:6]@q[:,:6].T
        a=self.rand(self.d,self.d)
        self.m=a@a.T
        self.k=self.rand(self.d,self.n)
        self.r=self.rand(self.o,self.n)

    def rand(self,*shape):
        return torch.randn(shape,generator=self.generator,dtype=torch.float64)

    def test_direct_factor_nonzero_nonsymmetric_history(self):
        op=HistoryOperator(self.p,self.m)
        self.assertGreater(float((op.matrix-op.matrix.T).norm()),1.)
        y,receipt=op.solve_bank(self.k)
        s,b=factor_from_response(self.k,y)
        dense,dr=native_dense(self.p,self.m,self.k,self.r)
        self.assertTrue(receipt['residual']['passed'])
        self.assertTrue(dr['residual']['passed'])
        torch.testing.assert_close(self.r@b.T,dense,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(fitting_transfer(self.r,s),dense@self.k,rtol=1e-12,atol=1e-12)

    def test_zero_history_and_identity_reductions(self):
        for p in (self.p,torch.eye(self.d,dtype=torch.float64)):
            m=torch.zeros_like(p)
            y,_=HistoryOperator(p,m,2.).solve_bank(self.k)
            torch.testing.assert_close(y,(p@self.k)/2.)
            _,b=factor_from_response(self.k,y)
            direct,_=native_dense(p,m,self.k,self.r,2.)
            torch.testing.assert_close(self.r@b.T,direct,rtol=1e-12,atol=1e-12)

    def test_raw_s_is_not_symmetrized_and_right_solve_orientation(self):
        p=self.p+0.03*self.rand(self.d,self.d)
        y,_=HistoryOperator(p,self.m).solve_bank(self.k)
        s,b=factor_from_response(self.k,y)
        self.assertGreater(float((s-s.T).norm()),1e-5)
        torch.testing.assert_close(b@(torch.eye(self.n)+s),y,rtol=1e-12,atol=1e-12)
        wrong=torch.linalg.solve(torch.eye(self.n)+s,y.T).T
        self.assertGreater(float((wrong-b).norm()),1e-4)
        direct,_=native_dense(p,self.m,self.k,self.r)
        torch.testing.assert_close(self.r@b.T,direct,rtol=1e-11,atol=1e-11)

    def test_rhs_bank_factor_once_native_blocks_separate(self):
        op=HistoryOperator(self.p,self.m)
        bank=torch.cat((self.k,self.k*3),dim=1)
        y,_=op.solve_bank(bank)
        _,b1=factor_from_response(self.k,y[:,:self.n])
        _,b2=factor_from_response(self.k*3,y[:,self.n:])
        _,bad=factor_from_response(bank,y)
        self.assertEqual(op.factorization_count,1)
        self.assertGreater(float((bad[:,:self.n]-b1).norm()),1e-3)
        for k,b in ((self.k,b1),(self.k*3,b2)):
            direct,_=native_dense(self.p,self.m,k,self.r)
            torch.testing.assert_close(self.r@b.T,direct,rtol=1e-12,atol=1e-12)

    def test_per_rhs_column_denominator_not_global_norm(self):
        a=torch.eye(2,dtype=torch.float64)
        rhs=torch.tensor([[1e9,0.],[0.,1.]],dtype=torch.float64)
        x=rhs.clone();x[1,1]+=.001
        result=solve_residual(a,x,rhs)
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['max_relative'],.001)
        self.assertEqual(result['failed_indices'],[1])

    def test_zero_denominator_absolute_rules(self):
        ref=torch.zeros(2,2,dtype=torch.float64)
        e=torch.tensor([[1e-8,2e-7],[0.,0.]],dtype=torch.float64)
        result=norm_discrepancy(e,ref,1e-3,columnwise=True)
        self.assertFalse(result['passed']);self.assertEqual(result['zero_reference_count'],2)
        self.assertEqual(result['relative_errors'],[None,None])
        self.assertEqual(result['failed_indices'],[1])

    def test_elementwise_not_aggregate_gate(self):
        ref=torch.ones(100,100,dtype=torch.float32);new=ref.clone();new[0,0]+=1e-3
        result=elementwise_parity(new,ref)
        self.assertFalse(result['passed']);self.assertEqual(result['failed_elements'],1)

    def test_actual_delta_response_denominator(self):
        actual=torch.eye(2,dtype=torch.float64)
        recon=actual.clone();recon[1,1]+=0.01
        k=torch.tensor([[1e9,0.],[0.,1.]],dtype=torch.float64)
        result=reconstruction_metrics(actual,recon,k)
        self.assertAlmostEqual(result['response']['max_relative'],.01)
        self.assertFalse(result['response']['passed'])

    def test_projector_range_and_monotonic_ideal_probe(self):
        y0,_=HistoryOperator(self.p,torch.zeros_like(self.m)).solve_bank(self.k)
        y1,_=HistoryOperator(self.p,self.m).solve_bank(self.k)
        r0=fixed_probe_scores(self.p,self.k,y0)
        r1=fixed_probe_scores(self.p,self.k,y1)
        for a,b in zip(r0,r1):
            self.assertLessEqual(b['nu'],a['nu']+1e-12)
            self.assertGreater(b['nu'],0)
        _,b=factor_from_response(self.k,y1)
        self.assertLess(float((b-self.p@b).norm()),1e-12)

    def test_raw_negative_no_clipping_and_zero_pk(self):
        p=-torch.eye(2,dtype=torch.float64);k=torch.tensor([[1.,0.],[0.,0.]],dtype=torch.float64)
        y,_=HistoryOperator(p,torch.zeros_like(p)).solve_bank(k)
        rows=fixed_probe_scores(p,k,y)
        self.assertEqual(rows[0]['nu'],-1.)
        self.assertTrue(rows[0]['negative_score'])
        self.assertTrue(rows[1]['pk_zero']);self.assertIsNone(rows[1]['nu'])

    def test_svd_energy_residual_and_near_degenerate_group(self):
        b=torch.diag(torch.tensor([2.,1.,.9999,.2],dtype=torch.float64))
        r=self.rand(3,4);actual=r@b.T+.001*self.rand(3,4)
        result=native_mode_decomposition(b,r,actual)
        self.assertLess(result['energy_identity_absolute_error'],1e-12)
        self.assertLess(abs(result['reference_residual']['energy_closure_error']),1e-12)
        self.assertEqual(result['groups'][1]['modes'],2)
        self.assertEqual(result['svd_input'],'direct_B')

    def test_zero_factor_svd(self):
        result=native_mode_decomposition(torch.zeros(5,3,dtype=torch.float64),self.rand(2,3))
        self.assertTrue(result['zero_factor']);self.assertEqual(result['factor_energy'],0.)

    def test_permutations_seed_and_frobenius_preserved(self):
        y,_=HistoryOperator(self.p,self.m).solve_bank(self.k)
        _,b=factor_from_response(self.k,y)
        first=counterfactual_summary(self.k,self.r,b,2.)
        second=counterfactual_summary(self.k,self.r,b,2.)
        self.assertEqual(first,second);self.assertEqual(len(first),21)
        self.assertEqual(first[0]['kind'],'ORIGINAL_ALIGNMENT')
        for row in first:
            self.assertAlmostEqual(row['residual_frobenius'],float(self.r.norm()))
            permutation=torch.tensor(row['permutation'])
            self.assertAlmostEqual(row['write_energy'],float((self.r[:,permutation]@b.T).square().sum()))

    def test_fail_closed_inputs(self):
        with self.assertRaises(ValueError):HistoryOperator(self.p,self.m,-1)
        with self.assertRaises(ValueError):HistoryOperator(self.p,self.m.float())
        with self.assertRaises(RuntimeError):HistoryOperator(-torch.eye(2),torch.eye(2))
        with self.assertRaises(ValueError):norm_discrepancy(torch.tensor([float('nan')]),torch.ones(1),1e-3)


if __name__=='__main__':unittest.main()
