"""Narrow repair identity/launcher regression, no model/GPU/Slurm execution."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from . import generation_cache_common as repair
from .generation_common import ARMS, TASK as PARENT_TASK
from .generation_run import arm_configuration
from .generation_submit import launcher, sbatch_argv
from .test_generation_run import config as parent_config


class RepairProfileTests(unittest.TestCase):
    def test_published_authority_is_exact_owner_and_scope(self):
        envelope = repair.authority()
        self.assertEqual(envelope['owners']['server2']['task_id'], repair.TASK)
        self.assertEqual(envelope['resources']['project_task_cap_server2'], 2)

    def test_task_projection_changes_metadata_not_native_math(self):
        original = parent_config()
        current = copy.deepcopy(original)
        current.update(task_id=repair.TASK, instruction_id=repair.NONCE,
                       parent_task_id=PARENT_TASK)
        current['generation'] = dict(repair={'status': 'PLAN_BOUND_NOT_ACTUAL_PASS'})
        for arm in ARMS:
            old = arm_configuration(original, arm)
            new = arm_configuration(current, arm)
            self.assertEqual(new['task_id'], repair.TASK)
            self.assertEqual(new['instruction_id'], repair.NONCE)
            for key in set(old) - {'task_id', 'instruction_id'}:
                self.assertEqual(new[key], old[key])
        self.assertEqual(repair.identity(original)[0], PARENT_TASK)

    def test_mixed_nonce_is_rejected(self):
        current = parent_config()
        current['generation'] = dict(repair={})
        with self.assertRaisesRegex(RuntimeError, 'CACHE_REPAIR_PROFILE_IDENTITY'):
            repair.identity(current)

    def test_new_job_names_resources_without_dummy_online_ids(self):
        attempt = Path('/fixture/cache-repair/attempt-r1')
        config = dict(task_id=repair.TASK, runtime={'python': '/fixture/python'},
            resources=dict(cpu=6, collector_cpu=6, host_mib=59392,
                           collector_host_mib=24576, wall='2-00:00:00', collector_wall='04:00:00'))
        for arm in (*ARMS, 'collector'):
            argv = sbatch_argv(attempt, arm, config, [])
            self.assertIn('--job-name=' + repair.TASK + '-' + arm, argv)
            self.assertIn('--hold', argv)
            self.assertIn('--export=NONE', argv)
            self.assertEqual('--gres=gpu:1' in argv, arm != 'collector')
            script = launcher(attempt, arm, config, 'a' * 40)
            self.assertIn('OMP_NUM_THREADS=6', script)
            self.assertNotIn('WANDB_API_KEY', script)
            self.assertNotIn('SLURM_JOB_ID=', script)

    def test_admission_only_detailed_local_server_jobs(self):
        from . import generation_cache_submit as submit
        calls = []

        def command(argv):
            calls.append(argv)
            if argv[0] == 'squeue':
                return ('21|fixture-local|RUNNING|gpu:1|server2|None\n'
                        '22|fixture-protected-other-node|RUNNING|gpu:1|server4|None')
            self.assertEqual(argv[:4], ['scontrol', 'show', 'job', '21'])
            return ('UserId=fixture(1) NodeList=server2 ReqNodeList=server2 '
                    'ReqTRES=cpu=6,gres/gpu=1 AllocTRES=cpu=6,gres/gpu=1 '
                    'Command=/mnt/raid5/janghj/ODE-edit/local/fixture/run.sh '
                    'WorkDir=/mnt/raid5/janghj/ODE-edit/local/fixture/source')

        with patch.object(submit, 'command', command), patch.object(submit.getpass, 'getuser', return_value='fixture'):
            observed = submit.inventory()
        self.assertEqual([row['job'] for row in observed['project']], ['21'])
        self.assertEqual(observed['other_server_detailed_queries'], 0)
        self.assertFalse(any('22' in argv for argv in calls if argv[0] == 'scontrol'))


if __name__ == '__main__':
    unittest.main()
