import unittest
from .review_completed_b1 import paired, quantile, independent_gate


class Tests(unittest.TestCase):
    def test_pair_lost_gained_and_identity(self):
        a=[dict(pair_id='a',success=True,desired_margin=1.),dict(pair_id='b',success=False,desired_margin=-1.)]
        b=[dict(pair_id='a',success=False,desired_margin=-2.),dict(pair_id='b',success=True,desired_margin=2.)]
        r=paired(a,b)
        self.assertEqual((r['lost'],r['gained'],r['net']),(1,1,0))
        self.assertEqual(r['lost_ids'],['a']);self.assertEqual(r['gained_ids'],['b'])
        self.assertEqual(r['margin_mean_change'],0.)
        with self.assertRaises(ValueError):paired(a,b[::-1])

    def test_quantile_linear_and_single(self):
        self.assertEqual(quantile([1,3],.5),2.)
        self.assertEqual(quantile([7],.99),7.)
        self.assertIsNone(quantile([],0))

    def test_gate_no_fallback_or_strict_loss_promotion(self):
        strict=dict.fromkeys(('rewrite_strict','two_P_strict','R_two_P_strict','R_two_P_NLL_joint'),50)
        native=dict(PS=dict(count=100),strict=strict)
        selection=dict(accepted=True,actual_delta_norm=1.,current_pass=True,full512=True,
            repaired_choices=0,phi_reference_native=10.,phi_reference_gain=.5,tau_risk=1e-5)
        self.assertTrue(independent_gate(native,native,selection)['pass_'])
        selection['accepted']=False
        self.assertFalse(independent_gate(native,native,selection)['pass_'])
        selection.update(accepted=True,phi_reference_native=None,phi_reference_gain=None)
        self.assertFalse(independent_gate(native,native,selection)['pass_'])
        selection.update(repaired_choices=1)
        lower=dict(PS=dict(count=100),strict=dict(strict,R_two_P_strict=49))
        self.assertFalse(independent_gate(native,lower,selection)['pass_'])
        self.assertFalse(independent_gate(native,native,selection)['sequential_authorized'])


if __name__=='__main__':unittest.main()
