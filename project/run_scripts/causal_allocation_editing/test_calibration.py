"""Strict actual-radial scalar calibration/lock CPU tests only."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from .calibration import CalibrationError, PriceLock, calibrate


class CalibrationTests(unittest.TestCase):
    def fixture(self, t_gradient=-3., q_gradient=2., u_value=1., Q=1.):
        u = {4: torch.zeros(1, 1, dtype=torch.float64),
             8: torch.tensor([[u_value]], dtype=torch.float64)}
        anchors = {4: torch.ones(1), 8: torch.ones(1)}
        task = {4: torch.zeros(1, 1, dtype=torch.float64),
                8: torch.tensor([[t_gradient]], dtype=torch.float64)}
        q = {4: torch.zeros(1, 1, dtype=torch.float64),
             8: torch.tensor([[q_gradient]], dtype=torch.float64)}
        return u, anchors, task, q, Q, dict(source="fixture", entry="W0H0", input="cohort")

    def test_positive_native_endpoint_price_and_lock_fixed20(self):
        result = calibrate(*self.fixture())
        self.assertEqual(result.price, 1.25)
        self.assertAlmostEqual(result.receipt["aggregate_radial_residual"], 0.)
        self.assertEqual(result.receipt["d_minus_two_Q"], 0.)
        with tempfile.TemporaryDirectory(prefix="causal-price-test-") as directory:
            path = Path(directory) / "lambda.json"; lock = PriceLock(path)
            lock.write(result.receipt); original = path.read_bytes()
            for _ in range(20):
                read = lock.read(result.receipt["identity"])
                self.assertEqual(read["lambda_Q"], 1.25)
            self.assertEqual(path.read_bytes(), original)
            lock.write(result.receipt) # exact byte reuse, never replace.
            different = calibrate(*self.fixture(t_gradient=-5.)).receipt
            with self.assertRaises(CalibrationError) as ctx:lock.write(different)
            self.assertEqual(ctx.exception.status, "CALIBRATION_LOCK_IMMUTABLE")
            with self.assertRaises(CalibrationError):lock.read(dict(source="different"))

    def test_negative_or_zero_price_typed_no_abs(self):
        for task in (-.1, -.5):
            with self.assertRaises(CalibrationError) as ctx:calibrate(*self.fixture(t_gradient=task))
            self.assertEqual(ctx.exception.status, "CALIBRATION_NO_POSITIVE_PRICE")
            self.assertIsNone(ctx.exception.receipt["lambda_Q"])
            self.assertFalse(ctx.exception.receipt["fallback_used"])
            json.dumps(ctx.exception.receipt, allow_nan=False)

    def test_energy_nonpositive_no_epsilon(self):
        for derivative in (0., -2.):
            with self.assertRaises(CalibrationError) as ctx:
                calibrate(*self.fixture(q_gradient=derivative, Q=derivative / 2))
            self.assertEqual(ctx.exception.status, "CALIBRATION_NO_VALID_ENERGY")

    def test_d_2Q_is_checked_not_repaired(self):
        with self.assertRaises(CalibrationError) as ctx:calibrate(*self.fixture(Q=2.))
        self.assertEqual(ctx.exception.status, "CALIBRATION_RADIAL_Q_IDENTITY")
        self.assertEqual(ctx.exception.receipt["d_minus_two_Q"], -2.)

    def test_nonfinite_channel_and_arithmetic_are_typed_jsonsafe(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(CalibrationError) as ctx:calibrate(*self.fixture(t_gradient=bad))
            self.assertEqual(ctx.exception.status, "CALIBRATION_NONFINITE")
            json.dumps(ctx.exception.receipt, allow_nan=False)
        with self.assertRaises(CalibrationError) as ctx:
            calibrate(*self.fixture(t_gradient=-1e308, u_value=2., q_gradient=1e308, Q=1.))
        self.assertEqual(ctx.exception.status, "CALIBRATION_NONFINITE")
        json.dumps(ctx.exception.receipt, allow_nan=False)

    def test_finite_positive_lambda_underflow_is_not_zero_fallback(self):
        args = self.fixture(t_gradient=-1e-300, q_gradient=1e308, Q=5e307)
        with self.assertRaises(CalibrationError) as ctx:
            calibrate(*args, norm_coefficient=0.)
        self.assertEqual(ctx.exception.status, "CALIBRATION_NO_POSITIVE_PRICE")
        self.assertEqual(ctx.exception.receipt["failure_stage"], "positive_price_underflow")

    def test_zero_reference_only_proven_genuine_native_noop(self):
        args = self.fixture(t_gradient=0., q_gradient=0., u_value=0., Q=0.)
        with self.assertRaises(CalibrationError):calibrate(*args)
        with self.assertRaises(CalibrationError):calibrate(*args, genuine_noop=True)
        result = calibrate(*args, genuine_noop=True, native_noop_requests=[True])
        self.assertIsNone(result.price)
        self.assertEqual(result.receipt["status"], "PENDING_GENUINE_NATIVE_NOOP")
        with tempfile.TemporaryDirectory(prefix="causal-price-test-") as directory:
            with self.assertRaises(CalibrationError):PriceLock(Path(directory) / "lambda.json").write(result.receipt)

    def test_one_reference_layer_not_new_multi_layer_fit(self):
        args = list(self.fixture()); args[0][4].fill_(.1)
        with self.assertRaises(CalibrationError) as ctx:calibrate(*args)
        self.assertEqual(ctx.exception.status, "CALIBRATION_IDENTITY_ERROR")

    def test_aggregate_radial_balance_is_not_each_request_KKT(self):
        u = {8: torch.ones(1, 2, dtype=torch.float64)}
        anchors = {8: torch.ones(2)}
        task = {8: torch.tensor([[-3., -1.]], dtype=torch.float64)}
        q = {8: 2 * u[8]}
        result = calibrate(u, anchors, task, q, 2., dict(source="fixture"))
        self.assertEqual(result.price, .75)
        self.assertEqual(result.receipt["radial_residual_per_request"], [-1., 1.])
        self.assertEqual(result.receipt["aggregate_radial_residual"], 0.)

    def test_anchor_norm_coordinate_native_own_layer(self):
        args = list(self.fixture()); args[1][8].fill_(2.)
        result = calibrate(*args)
        self.assertEqual(result.receipt["n"], .25)
        self.assertEqual(result.price, 1.375)
        self.assertNotEqual(result.receipt["n"], .125) # not /a**2 in u-space.


if __name__ == "__main__":
    unittest.main(verbosity=2)
