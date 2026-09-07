"""Focused CPU scalar/schema tests; no experiment runtime imports."""
import unittest
from .mechanism import NA, controller_quantities, decode, l8_node, safe_ratio, sum_recorded


class MechanismTests(unittest.TestCase):
    def test_missing_is_not_zero(self):
        self.assertEqual(decode(''), NA)
        self.assertEqual(decode('0'), 0)
        self.assertEqual(sum_recorded([0, NA]), NA)
        self.assertEqual(safe_ratio(0, 0), NA)

    def test_continuous_identity_is_not_finite_barrier(self):
        # e=1, psi=1, lambda=1, optimum c=.5, h=.5.
        n = dict(c=[.5], g=[1.], full_H=[[1.]], G=[[1.]], h=.5,
            lambda_value=1., V_before=.5, V_after=.28125)
        q = controller_quantities(n)
        self.assertEqual(q['continuous_dissipation_residual'], 0)
        self.assertEqual(q['finite_step_defect_reconstructed'], 0)
        self.assertEqual(q['barrier_increment'], -.09375)
        self.assertEqual(q['normalized_native_work'], .125)

    def test_positive_finite_defect_preserved(self):
        n = dict(c=[.5], g=[1.], full_H=[[1.]], G=[[1.]], h=.5,
            lambda_value=1., V_before=.5, V_after=.8)
        q = controller_quantities(n)
        self.assertGreater(q['barrier_increment'], 0)
        self.assertGreater(q['finite_step_defect_reconstructed'], 0)
        self.assertEqual(q['continuous_dissipation_residual'], 0)

    def test_l8_support_fails_closed(self):
        n = dict(layer_actions=[], actual_physical=[dict(layer=l,
            actual_step_DeltaW_squared=1. if l == 4 else 0.,
            actual_net_DeltaW_squared=0., actual_nonzero=0) for l in (4,5,6,7,8)])
        with self.assertRaisesRegex(ValueError, 'SUPPORT'):
            l8_node({}, n)


if __name__ == '__main__':
    unittest.main()
