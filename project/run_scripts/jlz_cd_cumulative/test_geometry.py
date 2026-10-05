"""Narrow torch production-path regression; existing CPU11 is source evidence.

Run with the existing torch runtime and CUDA_VISIBLE_DEVICES=''.  This does
not load a model/dataset and makes no actual-model or GPU qualification claim.
"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from project.run_scripts.jlz_realized_writer.geometry import prepare_metric, solve_realized, owner_weights
from .geometry import build_geometry, cost_gradient_u, projected_action, requested_action, writer_cost
from .calibration import CalibrationLock, CalibrationError


def fixture(*, cold=False, rank_deficient=False, counts=(1, 3, 2)):
    generator = torch.Generator(device='cpu').manual_seed(20261005)
    owners = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    n, q, d, B = len(owners) + 2, len(owners), 4, len(counts)
    z = torch.randn((n, n), generator=generator, dtype=torch.float64)
    C0 = z @ z.T + torch.eye(n, dtype=torch.float64)
    h = torch.randn((n, 2), generator=generator, dtype=torch.float64)
    A = 1.7 * C0 + h @ h.T
    K = torch.randn((n, q), generator=generator, dtype=torch.float64)
    if rank_deficient:
        K[:] = K[:, :1]
    W0 = torch.randn((d, n), generator=generator, dtype=torch.float32)
    Wentry = W0.clone() if cold else W0 + .2 * torch.randn((d, n), generator=generator)
    geometry = build_geometry(K, owners, A, C0, Wentry, W0, lambda_cov=1.7, request_count=B)
    u = .07 * torch.randn((d, B), generator=generator, dtype=torch.float32)
    anchors = 1. + torch.rand(B, generator=generator, dtype=torch.float32)
    return geometry, u, anchors, K, A, C0, Wentry, W0


class GeometryProductionTests(unittest.TestCase):
    def test_retained_map_dense_cd_and_actual_alignment(self):
        g, u, anchors, K, rawA, cov, wentry, w0 = fixture()
        D = requested_action(u, anchors)
        A, L, _ = prepare_metric(rawA)
        U, receipt = solve_realized(A, L, K, D.double()[:, g.owners], owner_weights(g.owners, D.shape[1]), 'equality')
        self.assertTrue(receipt['numerical_projection_verified'])
        torch.testing.assert_close(D.double() @ g.S, U @ K, atol=1e-12, rtol=1e-10)
        result = writer_cost(D, g, alpha=1)
        self.assertAlmostEqual(result['Q'], float((U @ L).square().sum()), places=10)
        cross = float(((wentry.double() - w0.double()) @ ((cov + cov.T) * .5) * U).sum())
        self.assertAlmostEqual(result['c'], cross, places=10)
        self.assertTrue(g.full_operator_compatible)
        self.assertFalse(g.receipt['dense_factor_retained'])
        self.assertEqual(g.receipt['owner_counts'], [1, 3, 2])

    def test_fp32_cast_chain_and_cross_owner_native_gradients(self):
        g, stored, anchors, *_ = fixture(rank_deficient=True)
        self.assertFalse(g.full_operator_compatible)
        u = stored.clone().requires_grad_(True)
        D = requested_action(u, anchors)
        D64 = D.double(); coefficient = .37
        value = coefficient * ((D64 @ g.C * D64).sum()
                               + 2. * g.lambda_c * (D64 * g.J).sum().abs())
        value.backward()
        analytic, _ = cost_gradient_u(stored, anchors, g, alpha=1, allocation_price=coefficient)
        torch.testing.assert_close(analytic, u.grad, rtol=2e-6, atol=2e-7)
        # A single owner's physical row affects every requested owner through S.
        u = stored.clone().requires_grad_(True)
        action = projected_action(requested_action(u, anchors), g, row_indices=[0])
        action.sum().backward()
        self.assertTrue(bool((u.grad.abs().sum(dim=0) > 0).all()))
        # Equal current values do not certify the whole owner-target operator.
        equal = torch.ones_like(stored)
        torch.testing.assert_close(projected_action(equal, g), equal[:, g.owners], atol=1e-6, rtol=1e-6)
        self.assertFalse(g.full_operator_compatible)

    def test_calibration_zero_origin_shared_immutable_receipt(self):
        g, u, a, *_ = fixture(cold=True)
        identity = dict(source='synthetic-existing-runtime', input='toy-pack', W0='toy-cold', plan='declared-method')
        prices = torch.tensor([.2, .3, .4], dtype=torch.float64)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'lambda.json'
            producer = CalibrationLock(path, role='producer', identity=identity)
            kwargs = dict(geometries={1: g}, anchors={1: a}, prices=prices, delta_zero=True)
            self.assertIsNone(producer.resolve(candidate=0, blocks={1: torch.zeros_like(u)}, **kwargs))
            self.assertFalse(path.exists())
            value = producer.resolve(candidate=1, blocks={1: u}, **kwargs)
            self.assertGreater(value, 0)
            before = path.read_bytes()
            consumer = CalibrationLock(path, role='consumer', identity=identity)
            self.assertEqual(consumer.resolve(candidate=1, blocks={1: u}, **kwargs), value)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(producer.resolve(candidate=2, blocks={1: u * 2}, **{**kwargs, 'delta_zero': False}), value)
            mismatch = CalibrationLock(path, role='consumer', identity=identity)
            with self.assertRaises(CalibrationError):
                mismatch.resolve(candidate=1, blocks={1: u * 2}, **kwargs)

    def test_calibration_missing_and_degenerate_are_failures_not_defaults(self):
        g, u, a, *_ = fixture(cold=True)
        kwargs = dict(candidate=1, geometries={1: g}, blocks={1: u}, anchors={1: a},
                      prices=torch.ones(u.shape[1], dtype=torch.float64), delta_zero=True)
        with tempfile.TemporaryDirectory() as folder:
            missing = CalibrationLock(Path(folder) / 'missing.json', role='consumer', identity={})
            with self.assertRaisesRegex(CalibrationError, 'NO_WAIT'):
                missing.resolve(**kwargs)
            bad = replace(g, C=torch.zeros_like(g.C), N=torch.zeros_like(g.N))
            producer = CalibrationLock(Path(folder) / 'bad.json', role='producer', identity={})
            with self.assertRaisesRegex(CalibrationError, 'ZERO_OR_NONFINITE'):
                producer.resolve(**{**kwargs, 'geometries': {1: bad}})
            self.assertFalse((Path(folder) / 'bad.json').exists())
            with self.assertRaisesRegex(CalibrationError, 'ZERO_DISPLACEMENT'):
                producer.resolve(**{**kwargs, 'delta_zero': False})

    def test_zero_rank_and_B1_shape_remain_valid_geometry(self):
        g, u, a, K, A, cov, wentry, w0 = fixture(cold=True, counts=(3,))
        zero = build_geometry(torch.zeros_like(K), g.owners, A, cov, wentry, w0, lambda_cov=1.7, request_count=1)
        self.assertEqual(zero.retained_rank, 0)
        self.assertEqual(zero.C.shape, (1, 1))
        self.assertEqual(writer_cost(requested_action(u, a), zero, alpha=1)['Pi'], 0.)
        self.assertEqual(float(projected_action(requested_action(u, a), zero).norm()), 0.)
        with tempfile.TemporaryDirectory() as folder:
            pending = CalibrationLock(Path(folder) / 'lambda.json', role='producer', identity={})
            value = pending.resolve(candidate=1, geometries={1: zero}, blocks={1: u},
                                    anchors={1: a}, prices=torch.ones(1, dtype=torch.float64), delta_zero=True)
            self.assertIsNone(value)


if __name__ == '__main__':
    unittest.main()
