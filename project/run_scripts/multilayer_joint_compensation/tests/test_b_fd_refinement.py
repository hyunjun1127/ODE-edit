import unittest
from project.run_scripts.multilayer_joint_compensation.track_b.runtime_gate import fd_observation,adjacent_fd_pass,FD_STEPS,FD_REFINEMENT_STEPS

P=[dict(nonzero_fraction=1.,actual_norm=1.,rounding_norm=1e-6)]*2

class TestRefinement(unittest.TestCase):
    def test_allowance_unchanged(self):
        import torch
        r=fd_observation(1.003,1.,.09,.015625,P)
        expected=.02*max(abs(r['ad']),abs(r['fd']))+64*torch.finfo(torch.float32).eps*max(1.003,1.,1.)/.015625
        self.assertEqual(r['numerical_bound'],expected)
    def test_cubic_truncation_raw_refinement(self):
        # f(x)=1+x+10x^3, true derivative1; no extrapolated value is used.
        rr=[fd_observation(1+h+10*h**3,1-h-10*h**3,1.,h,P) for h in FD_STEPS+FD_REFINEMENT_STEPS]
        self.assertFalse(rr[0]['passed']);self.assertTrue(adjacent_fd_pass(rr)[0])
    def test_wrong_ad_rejected(self):
        rr=[fd_observation(1+h,1-h,2.,h,P) for h in FD_STEPS+FD_REFINEMENT_STEPS]
        self.assertFalse(adjacent_fd_pass(rr)[0])
    def test_rounding_signal_rejected(self):
        r=fd_observation(1.,1.,0.,.015625,P)
        self.assertTrue(r['passed']);self.assertFalse(adjacent_fd_pass([r,r])[0])
    def test_nonadjacent_pass_rejected(self):
        rr=[dict(passed=v,sufficient_signal=True) for v in (True,False,True)]
        self.assertFalse(adjacent_fd_pass(rr)[0])

if __name__=='__main__':unittest.main()
