"""Synthetic FP64 geometry checks, not large-matrix/model validation."""
import unittest

import torch

from .geometry import (activation_bookkeeping, actual_difference, angle,
                       history_summary, shared_rademacher, stored_action_sketch,
                       tensor_sha256)


class GeometryTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_subtract_after_double(self):
        a=torch.tensor([1.],dtype=torch.float32);b=torch.tensor([1e10],dtype=torch.float32)
        self.assertEqual(float(actual_difference(b,a)),1e10-1)
        self.assertNotEqual(float((b-a).double()),float(actual_difference(b,a)))

    def test_angles_zero_and_opposite(self):
        self.assertEqual(angle(torch.zeros(3),torch.ones(3))['status'],'ZERO_NORM')
        self.assertAlmostEqual(angle(torch.ones(3),-torch.ones(3))['degrees'],180.)

    def test_shared_fixed_256_vectors(self):
        z=shared_rademacher(7)
        self.assertEqual(z.shape,(7,256))
        self.assertEqual(set(z.unique().tolist()),{-1.,1.})
        self.assertEqual(tensor_sha256(z),tensor_sha256(shared_rademacher(7)))
        with self.assertRaises(ValueError):shared_rademacher(7,128)

    def test_diagonal_stored_action_exact(self):
        d=torch.diag(torch.tensor([1.,2.,3.],dtype=torch.float64))
        m=torch.diag(torch.tensor([4.,5.,6.],dtype=torch.float64))
        result=stored_action_sketch(d,m,shared_rademacher(3),requests=100,chunk_rows=2)
        exact=float(torch.trace(d@m@d.T))
        self.assertEqual(result['mean'],exact);self.assertEqual(result['mc_se'],0.)
        self.assertEqual(result['J_per_request'],exact/100)
        self.assertEqual(result['J_per_trace'],exact/15)
        self.assertAlmostEqual(result['alignment_dimensionless'],3*exact/(14*15))
        self.assertEqual([x['probes'] for x in result['diagnostics']],[64,128,256])

    def test_full_sketch_matches_each_scalar(self):
        g=torch.Generator().manual_seed(777)
        d=torch.randn((5,8),generator=g,dtype=torch.float64)
        a=torch.randn((8,8),generator=g,dtype=torch.float64);m=a@a.T
        z=shared_rademacher(5)
        values=torch.tensor([float(z[:,j]@d@m@d.T@z[:,j]) for j in range(256)],dtype=torch.float64)
        result=stored_action_sketch(d,m,z,requests=4,chunk_rows=3)
        self.assertAlmostEqual(result['mean'],float(values.mean()))
        self.assertAlmostEqual(result['mc_se'],float(values.std()/16))

    def test_negative_and_zero_are_flagged_not_clipped(self):
        d=torch.eye(2,dtype=torch.float64);z=shared_rademacher(2)
        negative=stored_action_sketch(d,-torch.eye(2),z,requests=100)
        zero=stored_action_sketch(d,torch.zeros(2,2),z,requests=100)
        self.assertTrue(negative['negative_mean']);self.assertEqual(negative['mean'],-2.)
        self.assertIsNone(negative['relative_halfwidth'])
        self.assertTrue(zero['exact_zero_mean']);self.assertIsNone(zero['J_per_trace'])

    def test_history_symmetry_and_quadratics(self):
        m=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
        result=history_summary(m,requests=100,chunk_rows=1)
        self.assertEqual(result['trace'],5.)
        self.assertAlmostEqual(result['frobenius'],float(m.norm()))
        self.assertAlmostEqual(result['symmetry_absolute'],float((m-m.T).norm()))
        self.assertEqual(result['trace_per_request'],.05)
        self.assertFalse(result['probes_are_psd_or_rank_certificate'])

    def test_activation_cross_identity_not_norm_only(self):
        e=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
        d=-e;k=torch.tensor([[2.,3.],[4.,1.]],dtype=torch.float64)
        result=activation_bookkeeping(e,d,k)
        self.assertLess(result['maximum_closure_error'],1e-12)
        for a,c,v in zip(result['entry_response_energy'],result['cross_term'],result['interval_response_energy']):
            self.assertEqual(c,-2*a);self.assertEqual(v,a)


if __name__=='__main__':unittest.main()
