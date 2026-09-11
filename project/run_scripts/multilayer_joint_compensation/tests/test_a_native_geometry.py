"""Full-P* native geometry identities on compact CPU fixtures."""
import unittest
import io
from unittest.mock import patch

import torch

from project.run_scripts.l4_two_memory_conflict_routing.geometry import Projector
from project.run_scripts.multilayer_joint_compensation.track_a.native_geometry import (
    NativeGeometryBoundary, NativeWriterMetric,
)


def fixture(dimension=9, batch=7, rank=7):
    generator = torch.Generator().manual_seed(20260911 + batch + rank)
    orth, _ = torch.linalg.qr(torch.randn(dimension, dimension, dtype=torch.float64, generator=generator))
    q = orth[:, :rank]
    complement = rank > dimension // 2
    p = Projector(orth[:, rank:] if complement else q, complement, dimension)
    keys = torch.randn(dimension, batch, dtype=torch.float64, generator=generator)
    history_factor = torch.randn(dimension, dimension + 2, dtype=torch.float64, generator=generator)
    history = history_factor @ history_factor.T
    return keys, history, p, q


class NativeGeometryTests(unittest.TestCase):
    def test_writer_and_inverse_equal_explicit_full_q(self):
        for rank in (1, 4, 7, 9):
            with self.subTest(rank=rank):
                k, m, p, q = fixture(rank=rank)
                geometry = NativeWriterMetric(k, m, p)
                s = k @ k.T + m + torch.eye(9, dtype=torch.float64)
                exact = q @ torch.linalg.solve(q.T @ s @ q, q.T)
                torch.testing.assert_close(geometry.writer_factor_raw, k.T @ exact, rtol=1e-12, atol=1e-12)
                x = torch.arange(27, dtype=torch.float64).reshape(3, 9)
                torch.testing.assert_close(geometry.inverse(x), x @ exact, rtol=1e-12, atol=1e-12)
                self.assertEqual(geometry.receipt["factorized_dimension"], 9)
                self.assertEqual(geometry.receipt["projector_rank"], rank)
                self.assertEqual(geometry.receipt["rank_truncation"], 0)

    def test_raw_sum_to_mean_requires_all_terms_scaled(self):
        k, m, p, _ = fixture(batch=7)
        raw = NativeWriterMetric(k, m, p)
        mean = NativeWriterMetric(k / 7**.5, m / 7, p, ridge=1 / 7)
        rhs = torch.arange(14, dtype=torch.float64).reshape(2, 7)
        torch.testing.assert_close(raw.write(rhs, raw_fp64=True),
                                   mean.write(rhs / 7**.5, raw_fp64=True), rtol=1e-12, atol=1e-12)
        same_physical = raw.write(rhs, raw_fp64=True)
        torch.testing.assert_close(raw.energy(same_physical, normalized=True),
                                   mean.energy(same_physical, normalized=True), rtol=1e-12, atol=1e-12)
        wrong = NativeWriterMetric(k / 7**.5, m, p)
        self.assertGreater(float((wrong.write(rhs / 7**.5, raw_fp64=True)
                                  - raw.write(rhs, raw_fp64=True)).norm()), .01)

    def test_metric_and_image_metric_use_native_s(self):
        k, m, p, q = fixture()
        geometry = NativeWriterMetric(k, m, p)
        s = k @ k.T + m + torch.eye(9, dtype=torch.float64)
        x, y = torch.randn(3, 9, dtype=torch.float64), torch.randn(3, 9, dtype=torch.float64)
        torch.testing.assert_close(geometry.s_action(x), x @ s)
        torch.testing.assert_close(geometry.dot(x, y), ((x @ s) * y).sum())
        torch.testing.assert_close(geometry.s_action(x, projected=True), x @ q @ q.T @ s @ q @ q.T)
        j = geometry.writer_factor.double()
        torch.testing.assert_close(geometry.image_metric, j @ s @ j.T)
        torch.testing.assert_close(geometry.raw_image_metric,
                                   geometry.writer_factor_raw @ s @ geometry.writer_factor_raw.T)

    def test_normalization_is_trace_s_over_d_not_projected_average(self):
        k, m, p, q = fixture(rank=3)
        geometry = NativeWriterMetric(k, m, p)
        s = k @ k.T + m + torch.eye(9, dtype=torch.float64)
        mean = float(s.trace() / 9)
        self.assertAlmostEqual(geometry.mean_eigenvalue, mean)
        self.assertAlmostEqual(geometry.receipt["projected_mean_eigenvalue"], float((q.T @ s @ q).trace() / 3))
        x = geometry.project(torch.randn(2, 9, dtype=torch.float64))
        torch.testing.assert_close(geometry.s_action(x, projected=True, normalized=True),
                                   geometry.s_action(x, projected=True) / mean)
        torch.testing.assert_close(geometry.inverse(geometry.s_action(x, projected=True, normalized=True),
                                                   normalized=True), x, rtol=1e-12, atol=1e-12)

    def test_physical_writer_projection_and_autograd(self):
        k, m, p, _ = fixture()
        geometry = NativeWriterMetric(k, m, p)
        rhs = torch.randn(3, 7, requires_grad=True)
        delta = geometry.write(rhs)
        self.assertEqual(delta.dtype, torch.float32)
        torch.testing.assert_close(delta, rhs @ geometry.writer_factor, rtol=0, atol=0)
        delta.sum().backward()
        torch.testing.assert_close(rhs.grad, torch.ones_like(delta) @ geometry.writer_factor.T)
        self.assertLess(geometry.receipt["raw_factor_projection_relative"], 1e-12)
        self.assertLess(geometry.receipt["fp32_factor_projection_relative"], 1e-6)
        self.assertGreater(geometry.receipt["factor_cast_relative"], 0.)
        self.assertLess(geometry.receipt["raw_factor_projected_stationarity_relative"], 1e-12)

    def test_empty_batch_and_zero_projector(self):
        k, m, p, _ = fixture(batch=0)
        geometry = NativeWriterMetric(k, m, p)
        self.assertEqual(tuple(geometry.writer_factor.shape), (0, 9))
        self.assertEqual(torch.count_nonzero(geometry.write(torch.empty(2, 0))), 0)
        pzero = Projector(torch.empty(9, 0, dtype=torch.float64), False, 9)
        empty = NativeWriterMetric(torch.randn(9, 7), m, pzero)
        self.assertEqual(torch.count_nonzero(empty.writer_factor), 0)
        self.assertEqual(torch.count_nonzero(empty.inverse(torch.ones(2, 9))), 0)
        self.assertIsNone(empty.receipt["projected_mean_eigenvalue"])

    def test_duplicate_keys_and_batch_over_256_are_not_truncated(self):
        k, m, p, _ = fixture(dimension=7, batch=1, rank=5)
        repeated = k.repeat(1, 257)
        geometry = NativeWriterMetric(repeated, m, p)
        self.assertEqual(tuple(geometry.writer_factor.shape), (257, 7))
        self.assertEqual(tuple(geometry.image_metric.shape), (257, 257))
        torch.testing.assert_close(geometry.writer_factor_raw[0], geometry.writer_factor_raw[-1])

    def test_input_capture_and_cpu_default(self):
        k, m, p, _ = fixture()
        before = k.clone(), m.clone(), p.vectors.clone()
        geometry = NativeWriterMetric(k, m, p)
        k.add_(3)
        m.add_(4)
        p.vectors.zero_()
        torch.testing.assert_close(geometry.keys, before[0], rtol=0, atol=0)
        torch.testing.assert_close(geometry.history, before[1], rtol=0, atol=0)
        torch.testing.assert_close(geometry.projector.vectors, before[2], rtol=0, atol=0)
        self.assertEqual(str(geometry.cholesky.device), "cpu")

    def test_invalid_inputs_do_not_repair_ridge_or_projector(self):
        k, m, p, _ = fixture()
        with self.assertRaises(NativeGeometryBoundary):
            NativeWriterMetric(k, m, p, ridge=0)
        bad = Projector(torch.ones(9, 1, dtype=torch.float64), False, 9)
        with self.assertRaisesRegex(NativeGeometryBoundary, "orthonormal"):
            NativeWriterMetric(k, m, bad)
        asymmetric = m.clone()
        asymmetric[0, 1] += 1.
        with self.assertRaisesRegex(NativeGeometryBoundary, "nonsymmetric"):
            NativeWriterMetric(k, asymmetric, p)
        pfull = Projector(torch.empty(9, 0, dtype=torch.float64), True, 9)
        indefinite = torch.eye(9, dtype=torch.float64) * 100
        indefinite[0, 0] = -10
        with self.assertRaisesRegex(NativeGeometryBoundary, "not SPD"):
            NativeWriterMetric(torch.zeros(9, 1), indefinite, pfull)

    def test_weights_only_roundtrip_without_refactorization(self):
        k, m, p, _ = fixture()
        original = NativeWriterMetric(k, m, p)
        buffer = io.BytesIO()
        torch.save(original.to_state(), buffer)
        buffer.seek(0)
        state = torch.load(buffer, weights_only=True, map_location="cpu")
        with patch("torch.linalg.cholesky", side_effect=AssertionError("refactorization forbidden")):
            restored = NativeWriterMetric.from_state(state)
        rhs = torch.randn(3, 7)
        x = torch.randn(3, 9, dtype=torch.float64)
        self.assertTrue(torch.equal(original.write(rhs), restored.write(rhs)))
        self.assertTrue(torch.equal(original.s_action(x), restored.s_action(x)))
        self.assertTrue(torch.equal(original.inverse(x), restored.inverse(x)))
        self.assertTrue(torch.equal(original.inverse(x, normalized=True), restored.inverse(x, normalized=True)))
        self.assertTrue(torch.equal(original.image_metric, restored.image_metric))
        self.assertEqual(original.receipt, restored.receipt)
        self.assertEqual(restored.reuse_receipt["full_refactorization_count"], 0)
        state["history"].zero_()
        self.assertTrue(torch.equal(original.history, restored.history))

    def test_state_roundtrip_empty_batch_and_zero_projector(self):
        k, m, p, _ = fixture(batch=0)
        original = NativeWriterMetric(k, m, p)
        restored = NativeWriterMetric.from_state(original.to_state())
        self.assertEqual(tuple(restored.writer_factor.shape), (0, 9))
        pzero = Projector(torch.empty(9, 0, dtype=torch.float64), False, 9)
        zero = NativeWriterMetric(k, m, pzero)
        roundtrip = NativeWriterMetric.from_state(zero.to_state())
        self.assertEqual(roundtrip.rank, 0)
        self.assertEqual(torch.count_nonzero(roundtrip.inverse(torch.ones(2, 9))), 0)

    def test_serialized_state_rejects_shape_dtype_finite_and_factor_corruption(self):
        k, m, p, _ = fixture()
        geometry = NativeWriterMetric(k, m, p)
        for modification, message in (
            (lambda state: state.update(schema="other"), "schema"),
            (lambda state: state.update(keys=state["keys"].float()), "shape/dtype"),
            (lambda state: state.update(image_metric=state["image_metric"][:1]), "shape/dtype"),
            (lambda state: state["history"].fill_(float("nan")), "nonfinite"),
            (lambda state: state["cholesky"][0, 0].add_(.5), "factor-action"),
            (lambda state: state.update(mean_eigenvalue=state["mean_eigenvalue"] * 2), "normalization"),
            (lambda state: state["writer_factor"].add_(1), "raw FP32 cast"),
            (lambda state: state["receipt"].update(batch=99), "receipt identity"),
            (lambda state: state["receipt"].update(unsafe=object()), "primitive"),
        ):
            with self.subTest(message=message):
                state = geometry.to_state()
                modification(state)
                with self.assertRaisesRegex(NativeGeometryBoundary, message):
                    NativeWriterMetric.from_state(state)


if __name__ == "__main__":
    unittest.main()
