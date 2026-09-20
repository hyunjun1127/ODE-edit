"""Narrow user-override routing regression; no model/GPU calls."""
import inspect
import unittest
from unittest.mock import patch
import torch
from . import activation_lane as activation, operator_lane as operator, control
from .test_operator_lane import LaneTests


class RemovalRouting(unittest.TestCase):
    def test_failed_historical_gate_does_not_block(self):
        activation.validate_dependencies({'status':'FAILED','C01':'FAILED'}, {'status':'PASS'}, {'status':'PASS'})
        self.assertEqual(operator.validate_dependencies([90],{'C01':'FAILED'},[])['historical_C01'],'FAILED')

    def test_no_validation_forward_or_dense_solve(self):
        self.assertNotIn('kernel.evaluate_pairs(',inspect.getsource(activation.run_interval))
        self.assertNotIn('native_dense(',inspect.getsource(operator.compute_history))
        with self.assertRaises(AssertionError):control.freeze('/not-used',mode='gate')

    def test_large_finite_residual_is_observation(self):
        f=LaneTests();f.setUp()
        original=operator.HistoryOperator.solve_bank
        def solve(obj,*a,**kw):
            y,receipt=original(obj,*a,**kw)
            receipt['residual'].update(passed=False,max_relative=42.)
            return y,receipt
        with patch.object(operator.HistoryOperator,'solve_bank',solve):
            result=operator.compute_history(f.p.double(),torch.zeros_like(f.p).double(),f.bank,f.residuals,90)
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['numerical_validation'],'NOT_ESTABLISHED')
        self.assertEqual(result['solve']['residual']['max_relative'],42.)

    def test_identity_and_nonfinite_still_block(self):
        with self.assertRaises(ValueError):operator.validate_dependencies([90,90],{},[])
        with self.assertRaises(ValueError):
            operator.HistoryOperator(torch.eye(2,dtype=torch.float64),torch.full((2,2),float('nan'),dtype=torch.float64))


if __name__=='__main__':unittest.main()
