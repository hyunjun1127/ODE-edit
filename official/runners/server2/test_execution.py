"""Metadata-only resource arithmetic; not model/GPU/online certification."""
import unittest

from official.runners.server2.execution import memory_disk_plan


class ExecutionPlanTests(unittest.TestCase):
    def test_native_checkpoint_payload_and_two_lane_disk_reserve(self):
        plan=memory_disk_plan()
        cells=plan['per_method_payload_bytes']
        self.assertEqual(cells['FT'],4096*16384*4+4096*4)
        self.assertEqual(cells['MEMIT'],6*4096*16384*4)
        self.assertEqual(cells['ALPHAEDIT'],6*(4096*16384*4+16384*16384*4))
        self.assertEqual(cells['ALPHAEDIT_BLUE'],2*(4096*16384*4+16384*16384*4))
        self.assertEqual(plan['twelve_final_checkpoints_bytes'],2*sum(cells.values()))
        self.assertGreaterEqual(plan['planned_reserve_bytes'],sum(plan[key] for key in (
            'twelve_final_checkpoints_bytes','qualification_latest_checkpoints_bytes',
            'two_lane_atomic_extra_bytes','raw_error_metadata_reserve_bytes')))

    def test_new_checkpoint_exception_not_old_asset_delete_or_eta(self):
        plan=memory_disk_plan()
        self.assertFalse(plan['old_asset_delete_authorized'])
        self.assertFalse(plan['measured_peak_RAM_VRAM'])
        self.assertTrue(plan['requested_wall_is_not_ETA'])
        self.assertTrue(plan['latest1_reclamation_only_this_new_task_CP_folders'])


if __name__=='__main__':
    unittest.main()
