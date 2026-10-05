"""Read-only Slurm mock checks: no scheduler writes or model execution."""
import getpass
import json
from pathlib import Path
import shlex
import sys
import unittest
from unittest.mock import patch

from . import submit


RESOURCES = dict(wall='2-00:00:00', qualification_wall='04:00:00',
                 collector_wall='04:00:00', host_mib=59392, hard_host_mib=60416,
                 collector_host_mib=24576, task_cap=1)


class SubmitMockTests(unittest.TestCase):
    def detail(self, role, dependency, argv, attempt, job='123'):
        r = RESOURCES
        cpu = role == 'collector'
        wall = r['collector_wall'] if cpu else r['qualification_wall'] if role == 'qualification' else r['wall']
        mem = r['collector_host_mib'] if cpu else r['host_mib']
        tres = f'cpu=8,mem={mem}M,node=1' + ('' if cpu else ',gres/gpu=1')
        return (f'JobId={job} JobName={submit.TASK} UserId={getpass.getuser()}(1025) '
                f'JobState=PENDING Reason=JobHeldUser Requeue=0 CPUs/Task=8 '
                f'NumCPUs=8 ReqTRES={tres} ReqNodeList=server4 Partition=gpu QOS=lab_gpu_s4 '
                f'TimeLimit={wall} Dependency={dependency} '
                + ('' if cpu else 'TresPerNode=gres/gpu:1 ')
                + f'Command={attempt / (role + ".sh")} WorkDir={attempt / "source"} '
                + 'SubmitLine=' + shlex.join(argv))

    def inspect_mock(self, role, dep, observed, changed_detail=None, observed_script=None):
        attempt = submit.LOCAL / 'mock-run-instance'
        argv = submit.arguments(role, dep, attempt, RESOURCES)
        detail = self.detail(role, observed, argv, attempt)
        if changed_detail is not None:detail = changed_detail(detail)
        script = submit.launcher(attempt / 'source', 'frozen-test-source', role, attempt)
        supplied = script if observed_script is None else observed_script
        def call(args, cwd=None):
            if args[:3] == ['scontrol', 'show', 'job']:return detail
            if args[:3] == ['scontrol', 'write', 'batch_script']:return supplied
            raise AssertionError('Unexpected mock command: ' + repr(args))
        with patch.object(submit, 'command', side_effect=call), \
             patch.object(submit.Path, 'read_text', return_value=script), \
             patch.object(submit, 'member', return_value=dict(path='mock', sha256='mock')):
            return submit.inspect('123', role, dep, argv, attempt, RESOURCES)

    def test_two_parent_dependencies_bare_or_state_annotated(self):
        for observed in ('afterany:111:222', 'afterany:111(unfulfilled):222(unfulfilled)',
                         'afterany:111(fulfilled):222(unfulfilled)'):
            value = self.inspect_mock('collector', 'afterany:111:222', observed)
            self.assertEqual(value['role'], 'collector')
        for observed in ('afterany:111:222:333',
                         'afterany:111(unfulfilled):222(fulfilled):333(unfulfilled)'):
            self.inspect_mock('collector', 'afterany:111:222:333', observed)

    def test_dependency_type_identity_not_only_ids(self):
        with self.assertRaises(RuntimeError):self.inspect_mock('main', 'afterany:111', 'afterok:111(unfulfilled)')
        with self.assertRaises(RuntimeError):self.inspect_mock('collector', 'afterany:111:222', 'afterany:111(unfulfilled)')
        with self.assertRaises(RuntimeError):self.inspect_mock('main', 'afterany:111', '(null)')
        self.inspect_mock('qualification', None, '(null)')

    def test_role_argv_resources_name_export_requeue_and_dependency(self):
        attempt = submit.LOCAL / 'mock-run-instance'
        for role in submit.ROLES:
            args = submit.arguments(role, 'afterany:111', attempt, RESOURCES)
            self.assertIn('--hold', args)
            self.assertIn('--job-name=causal-allocation-editing', args)
            self.assertIn('--partition=gpu', args); self.assertIn('--nodelist=server4', args)
            self.assertIn('--cpus-per-task=8', args); self.assertIn('--export=NONE', args)
            self.assertIn('--no-requeue', args); self.assertIn('--dependency=afterany:111', args)
            self.assertNotIn('odeedit', args[args.index('--job-name=causal-allocation-editing')])
            self.assertEqual(args[-1], str(attempt / (role + '.sh')))
            if role == 'collector':
                self.assertNotIn('--gres=gpu:1', args); self.assertIn('--mem=24576M', args)
            else:
                self.assertIn('--gres=gpu:1', args); self.assertIn('--mem=59392M', args)

    def test_held_cpu_gpu_memory_wall_owner_node_checks(self):
        mutations = (
            lambda s:s.replace('mem=59392M', 'mem=60417M'),
            lambda s:s.replace('gres/gpu:1 ', 'gres/gpu:2 '),
            lambda s:s.replace('Requeue=0 ', 'Requeue=1 '),
            lambda s:s.replace('CPUs/Task=8 ', 'CPUs/Task=16 '),
            lambda s:s.replace('ReqNodeList=server4 ', 'ReqNodeList=server3 '),
            lambda s:s.replace('TimeLimit=2-00:00:00 ', 'TimeLimit=3-00:00:00 '),
            lambda s:s.replace('UserId=' + getpass.getuser() + '(', 'UserId=other-owner('),
            lambda s:s.replace('JobName=causal-allocation-editing ', 'JobName=other-task '),
        )
        for mutation in mutations:
            with self.assertRaises(RuntimeError):
                self.inspect_mock('main', 'afterany:111', 'afterany:111(unfulfilled)', mutation)

    def test_exact_source_full_argv_and_batch_script_bytes(self):
        with self.assertRaises(RuntimeError):
            self.inspect_mock('main', 'afterany:111', 'afterany:111(unfulfilled)',
                              lambda s:s.replace('Command=', 'Command=/other-source'))
        with self.assertRaises(RuntimeError):
            self.inspect_mock('main', 'afterany:111', 'afterany:111(unfulfilled)',
                              lambda s:s.replace('--export=NONE', '--export=ALL'))
        with self.assertRaises(RuntimeError):
            self.inspect_mock('main', 'afterany:111', 'afterany:111(unfulfilled)',
                              observed_script='#!/bin/bash\nexec changed-program\n')

    def mock_dag(self, external=None, local_cap=3, tracked_cap=3):
        # Freeze, scontrol/sbatch, file publication, helper and queue are mocked.
        # Qualification failure must eventually terminalize main and collector,
        # not leave main forever DependencyNeverSatisfied behind afterok.
        attempt = submit.LOCAL / 'mock-run-instance'
        config = dict(resources=dict(RESOURCES))
        lock = dict(source_commit='source')
        calls = []; written = []; released = []
        sequence = iter(('201', '202', '203'))
        def call(args, cwd=None):
            calls.append(args)
            if args[0] == 'squeue':return ''
            if args[:3] == ['scontrol', 'show', 'node']:return 'NodeName=server4'
            if args[:3] == ['scontrol', 'show', 'partition']:return 'MaxTime=30-00:00:00'
            if args[0] == 'sbatch':return next(sequence)
            if args[:2] == ['scontrol', 'release']:released.append(args[2]);return ''
            raise AssertionError('Unexpected mock command: ' + repr(args))
        def text(path, *args, **kwargs):
            if path.name == 'gpu-caps.tsv':return f'server4\tignored\t{local_cap}\n'
            if path.name == 'gpu-concurrency-policy.tsv':return f'server4\t{tracked_cap}\n'
            raise AssertionError('Unexpected read: ' + str(path))
        helper = type('Result', (), dict(returncode=0, stdout='ALLOW_RESOURCE', stderr=''))()
        with patch.object(sys, 'argv', ['submit', '--config', '/mock/config.json', '--attempt', str(attempt)]), \
             patch.object(submit, 'command', side_effect=call), \
             patch.object(submit, 'resource_inventory', side_effect=[dict(jobs=external or []), dict(jobs=external or [])]), \
             patch.object(submit, 'freeze', return_value=(lock, config)), \
             patch.object(submit, 'verify_frozen', return_value=(lock, config)), \
             patch.object(submit.Path, 'glob', return_value=iter([])), \
             patch.object(submit.Path, 'read_text', text), \
             patch.object(submit.subprocess, 'run', return_value=helper), \
             patch('builtins.print'), \
             patch.object(submit, 'inspect', side_effect=lambda job, role, dep, argv, attempt, r:dict(job=job, role=role)), \
             patch.object(submit, 'write', side_effect=lambda p, row:written.append((p.name, row))), \
             patch.object(submit, 'member', return_value=dict(path='mock', sha256='mock')):
            submit.main()
        return calls, written, released

    def test_main_is_resource_afterany_then_exact_ready_gate_not_afterok_deadlock(self):
        calls, written, released = self.mock_dag()
        main_args = next(x for x in calls if x[0] == 'sbatch' and x[-1].endswith('/main.sh'))
        collector_args = next(x for x in calls if x[0] == 'sbatch' and x[-1].endswith('/collector.sh'))
        self.assertIn('--dependency=afterany:201', main_args)
        self.assertIn('--dependency=afterany:201:202', collector_args)
        self.assertEqual(released, ['203', '202', '201'])
        submission = next(row for name, row in written if name == 'submission.json')
        self.assertFalse(submission['automatic_retry']); self.assertFalse(submission['monitoring_active'])
        self.assertEqual(submission['GPU_qualification'], 'NOT_OBSERVED')

    def test_stricter_project_cap_resource_barrier_without_otherjob_mutation(self):
        external = [dict(job='111', gpus=1, state='RUNNING'),
                    dict(job='112', gpus=1, state='PENDING')]
        calls, written, released = self.mock_dag(external, local_cap=2, tracked_cap=3)
        qualification_args = next(x for x in calls if x[0] == 'sbatch' and x[-1].endswith('/qualification.sh'))
        self.assertIn('--dependency=afterany:111:112', qualification_args)
        held = next(row for name, row in written if name == 'held-inspection.json')
        self.assertEqual(held['effective_project_cap'], 2)
        self.assertEqual(held['task_cap'], 1)
        self.assertEqual(held['maximum_new_GPU_concurrency'], 1)
        self.assertEqual(held['other_job_mutations'], 0)
        self.assertEqual(released, ['203', '202', '201'])
        self.assertFalse(any(x[0] == 'scancel' for x in calls))

    def test_failed_qualification_ready_absent_blocks_before_model_load(self):
        from . import run
        attempt = submit.LOCAL / 'mock-run-instance'
        with patch.object(sys, 'argv', ['run', '--attempt', str(attempt), '--main-only']), \
             patch.object(run, 'locked', return_value=({}, dict(source_commit='source', config_sha256='config'))), \
             patch.object(run, 'setup') as load_model, \
             patch.object(run.Path, 'mkdir'), \
             patch.object(run.Path, 'is_file', return_value=False), \
             patch.object(run.Path, 'glob', return_value=iter([])), \
             patch.object(run, 'write'), \
             patch.object(run.torch.cuda, 'is_initialized', return_value=False):
            with self.assertRaises(RuntimeError) as ctx:run.main()
            self.assertIn('TECHNICAL_READY_NOT_AVAILABLE', str(ctx.exception))
            load_model.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
