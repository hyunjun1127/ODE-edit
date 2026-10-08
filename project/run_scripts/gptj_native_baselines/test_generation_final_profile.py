"""Final schedule binding/count tests only; native inputs are read-only."""
import copy
import unittest
from .generation_common import read
from .generation_final_common import PREPARATION, authority, counts, ready


class FinalProfileTests(unittest.TestCase):
    def test_direct_user_override_exact_scope(self):
        value = authority()
        self.assertEqual(value['server'], 'server2')
        self.assertEqual(value['evaluation_schedule'], 'FINAL_W20_ONLY')
        self.assertEqual(value['project_gpu_cap'], 2)
        self.assertTrue(value['noCP'])

    def test_bound_production_plan_is_not_actual_qualification(self):
        config = read(PREPARATION / 'config.json')
        self.assertTrue(ready(config))
        self.assertEqual(config['generation']['repair']['actual_qualification_status'], 'NOT_RUN')
        self.assertNotIn('old_complete_case_inventory', config['generation']['repair'])
        for key in ('W0_generation_enabled', 'intermediate_generation_enabled'):
            bad = copy.deepcopy(config)
            bad['generation'][key] = True
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                ready(bad)

    def test_only_observation_schedule_counts_change(self):
        from .generation_plan import counts as old_counts
        before, after = old_counts(), counts()
        self.assertEqual(before['native_per_arm'], after['native_per_arm'])
        self.assertEqual(after['edit_applications'], 12000)
        self.assertEqual(after['generation_endpoint_calls_per_arm'], 1)
        self.assertEqual(after['generation_edit_state_case_observations_per_arm'], 2000)
        self.assertEqual(after['planned_generation_case_observations'], 12000)
        self.assertEqual(after['cold_W0_generation_case_observations'], 0)
        self.assertEqual(after['qualification_max_prompt_route_evaluations'], 24)
        self.assertEqual(after['additional_generation_per_metric'], 0)
        self.assertEqual(after['checkpoint_saves'], 0)
        self.assertEqual(before['generation_edit_state_case_observations_per_arm'], 8600)


if __name__ == '__main__':
    unittest.main()
