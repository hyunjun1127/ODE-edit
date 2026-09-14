"""CPU-only stored-state descriptive geometry, never model execution."""
import unittest

import torch

from .refresh_state_review import target_geometry


class TargetGeometryTests(unittest.TestCase):
    def evidence(self):
        return dict(u=torch.tensor([3., 4.]), initial_u=torch.zeros(2),
                    a0=torch.tensor([1., 2.]), aj=torch.tensor([1., 4.]),
                    Z=torch.tensor([4., 6.]), m=torch.zeros(2),
                    v=torch.zeros(2), t=2, clamp_hits=1)

    def test_chunk_zero_displacements_and_observed_counters(self):
        result = target_geometry(self.evidence(), dict(actual_adam_updates=2,
                                 clamp_max_norm=8., stop_reason='LOSS_BELOW_0_05'), None, False)
        self.assertEqual(result['u_norm'], 5.)
        self.assertEqual(result['u_chunk_displacement_norm'], 5.)
        self.assertEqual(result['Z_chunk_displacement_norm'], 5.)
        self.assertEqual(result['stored_aj_minus_a0_norm'], 2.)
        self.assertTrue(result['early_stop'])
        self.assertFalse(result['zero_step'])

    def test_frozen_geometry_is_zero_displacement_and_stale_anchor(self):
        e = self.evidence()
        result = target_geometry(e, dict(actual_adam_updates=0), e, True)
        self.assertEqual(result['u_chunk_displacement_norm'], 0.)
        self.assertEqual(result['Z_chunk_displacement_norm'], 0.)
        self.assertEqual(result['aj_scope'], 'STALE_FROZEN_SNAPSHOT_NOT_CURRENT_READOUT')
        self.assertEqual(result['clamp_max_norm'], 'NA_FROZEN_REUSE')
        self.assertTrue(result['zero_step'])
        self.assertFalse(result['early_stop'])


if __name__ == '__main__':
    unittest.main()
