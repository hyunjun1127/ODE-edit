"""New native W20 profile: CPU/fake SDK only; no online or Slurm run."""
import unittest
from . import schema
from .test_w20_generation import config as old_config
from .test_tracking import SDK
from .worker import session

NONCE = 'USER-DIRECT-NATIVE-FLUCON-REPAIR-20261008-R1'
TASK = 'gpt2xl-baselines-native-generation-repair'


def config(arm):
    value = old_config()
    value.update(task_id=TASK, generation_repair_instruction=NONCE,
                 arm=arm, baseline=arm, attempt='w20-native-three-r1',
                 generation_profile='cf-cake-native-casebatch-kv-total100-globalrng-v1',
                 writer={'BASE_MEMIT': 'memit', 'BASE_ALPHAEDIT': 'alphaedit', 'ALPHAEDIT_BLUE': 'alphaedit_blue'}[arm])
    return value


class NativeW20Generation(unittest.TestCase):
    def test_three_real_writers_typed_identity_and_axes(self):
        for arm in ('BASE_MEMIT', 'BASE_ALPHAEDIT', 'ALPHAEDIT_BLUE'):
            c = schema.bind_job_identity(config(arm),
                {'SLURM_JOB_ID': '123', 'SLURM_ARRAY_JOB_ID': '122',
                 'SLURM_ARRAY_TASK_ID': '0', 'SLURM_STEP_ID': '-1'})
            class Fake(SDK):
                def __init__(self):
                    super().__init__()
                    self.axes = []
                def define_metric(self, key, **kwargs):
                    self.axes.append((key, kwargs))
            sdk = Fake()
            output = []
            progress = {'phase': 'W20_generation', 'generation_progress/step': 1,
                        'generation_progress/completed_cases': 1,
                        'generation_progress/total_cases': 2000}
            session(sdk, dict(config=c, run_id='fixtureNativeW20'+arm,
                spool='/tmp/fake-only-no-write', smoke=False, base_url='https://api.wandb.ai'),
                [dict(op='log', values=progress, step=None)], output.append)
            self.assertIn('job122_0', sdk.name)
            self.assertEqual(sdk.config['job_id'], '123')
            self.assertEqual(sdk.config['step_id'], '-1')
            self.assertEqual(sdk.config['task_id'], TASK)
            self.assertEqual(sdk.config['generation_repair_instruction'], NONCE)
            self.assertEqual(sdk.points, [progress])
            self.assertIn(('generation_progress/*',
                dict(step_metric='generation_progress/step', step_sync=False)), sdk.axes)
            self.assertEqual(next(x for x in output if x['status']=='LOGGING_ACCEPTED')['delivery'],
                             'SDK_ASYNC_NOT_REMOTE_ACK')

    def test_old_profile_still_allowed_and_invalid_inputs_rejected(self):
        self.assertEqual(schema.config(old_config())['generation_schedule'], 'W20_ONLY_FIRST2000')
        for key, value in (('generation_schedule', 'W0'),
                           ('generation_repair_instruction', 'UNAPPROVED'),
                           ('raw_prompt', 'PRIVATE')):
            c = config('BASE_MEMIT')
            c[key] = value
            with self.assertRaises(ValueError):
                schema.config(c)

    def test_truthful_progress_separate_from_endpoint_and_fit(self):
        p = {'phase': 'W20_generation', 'generation_progress/step': 1}
        self.assertEqual(schema.metrics(p, scientific=True), p)
        for key, value in (('edits', 2000), ('pre_state_edits', 1900),
                           ('fit/global_candidate', 1), ('raw_text', 'PRIVATE')):
            with self.assertRaises(ValueError):
                schema.metrics(dict(p, **{key: value}), scientific=True)


if __name__ == '__main__':
    unittest.main()

