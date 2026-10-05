"""Two-batch RAM state links, rollback, whole horizon and scalar DAG fixtures."""
import getpass
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from project.run_scripts.jlz_realization.writer import Transaction
from .common import batches, selected_for_post, state, member, write, NONCE, TASK
from . import submit, prepare, preflight
from .submit import admission, arguments, jobname, registration_guard, terminal_scheduler_rows, validate_repair_receipt

class FakeAdapter:
    def __init__(self):
        self.weights = {4: torch.zeros(2, 3)}
    def guard(self):
        return 'UNCHANGED_NONSELECTED'

class ControllerTests(unittest.TestCase):
    def test_two_successful_batches_and_failed_third_preserve_prefix(self):
        a = FakeAdapter(); H = {4: torch.zeros(3, 3)}; joined = []
        for i in (1, 2):
            before = state(a, H)
            with Transaction(a, H) as tx:
                a.weights[4].add_(1); H[4].add_(torch.eye(3)); tx.finish()
            joined.append((before, state(a, H)))
        self.assertEqual(joined[0][1], joined[1][0])
        final = state(a, H)
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with Transaction(a, H) as tx:
                a.weights[4].add_(123); H[4].add_(456)
                raise RuntimeError('fixture')
        self.assertEqual(state(a, H), final); self.assertTrue(tx.rollback_verified)
    def test_horizon_and_seen_prefix(self):
        records = list(range(2000)); parts = list(batches(records, 100))
        self.assertEqual(len(parts), 20); self.assertEqual(parts[-1][0], 20)
        self.assertEqual(sum(len(cur) for _, cur, _ in parts), 2000)
        for b, cur, seen in parts:
            self.assertEqual(len(seen), b * 100)
            self.assertEqual(selected_for_post(cur, seen, b), seen if b in (5, 10, 15, 20) else cur)
    def test_single_slot_producer_first_no_deadlock(self):
        inventory = {'jobs': []}
        barrier, serial = admission(inventory, 3, 'CfgTRES=cpu=128,gres/gpu=8\nAllocTRES=cpu=34,gres/gpu=7')
        self.assertEqual(barrier, []); self.assertTrue(serial)
        barrier, serial = admission({'jobs': [{'job':'123', 'gpus':3}]}, 3,
                                    'CfgTRES=gres/gpu=8\nAllocTRES=gres/gpu=7')
        self.assertEqual(barrier, ['123']); self.assertTrue(serial)
    def test_explicit_resources(self):
        from pathlib import Path
        r = dict(host_mib=59392, collector_host_mib=24576, wall='2-00:00:00', collector_wall='04:00:00')
        argv = arguments('CD_C', 'afterany:123', Path('/exact/task/attempt-r1'), r)
        self.assertIn('--mem=59392M', argv); self.assertIn('--export=NONE', argv)
        self.assertIn('--no-requeue', argv); self.assertIn('--dependency=afterany:123', argv)
        self.assertIn('--gres=gpu:1', argv)

class RepairRegistrationTests(unittest.TestCase):
    """Tiny local fixtures; no scheduler commands, model load, or GPU calls."""
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.local = Path(tmp.name); self.prior = self.local / 'attempt-r1'
        self.attempt = self.local / 'attempt-r2'; self.out = self.local / 'preparation-r2'
        for module in (submit, prepare, preflight):
            mocked = patch.object(module, 'LOCAL', self.local); mocked.start(); self.addCleanup(mocked.stop)
        self.jobs = {'CD_Q': '58880', 'CD_C': '58881', 'collector': '58882'}
        self.source = submit.REPAIR_PREDECESSOR_SOURCE
        write(self.prior / 'config.json', dict(instruction_id=NONCE, task_id=TASK))
        write(self.prior / 'source/witness.json', dict(frozen=True))
        lock = dict(instruction_id=NONCE, task_id=TASK, source_commit=self.source,
            config_sha256=member(self.prior / 'config.json')['sha256'], owner=getpass.getuser(), host='server4',
            source_members=[member(self.prior / 'source/witness.json')])
        write(self.prior / 'execution.lock.json', lock)
        mapping = {}
        resources = dict(host_mib=59392, collector_host_mib=24576, wall='2-00:00:00', collector_wall='04:00:00')
        for stage, job in self.jobs.items():
            mapping[stage] = dict(job=job, dependency=None, argv=arguments(stage, None, self.prior, resources))
            write(self.prior / ('submitted-' + stage + '.json'), dict(nonce=NONCE, status='HELD', **mapping[stage]))
        write(self.prior / 'submission.json', dict(nonce=NONCE, status='RELEASED', jobs=self.jobs,
            mapping=mapping, source=self.source, lock=member(self.prior / 'execution.lock.json')))
        terminals = {}; qualifications = {}
        for stage, job in self.jobs.items():
            if stage in submit.ARMS:
                path = self.prior / ('main-' + stage + '/terminal.json')
                write(path, dict(source=self.source, arm=stage, job=job, commits=0,
                    status='TECHNICAL_BLOCKED', checkpoint_saved=False))
                qpath = self.prior / ('main-' + stage + '/qualification/qualification.json')
                write(qpath, dict(pass_=False, GPU_qualified=False, common_candidate_no_fit=True,
                    sealed_budget=dict(fit_calls=0)))
                qualifications[stage] = member(qpath)
            else:
                path = self.prior / 'cpu-report/terminal.json'
                write(path, dict(source=self.source, status='PARTIAL_OR_TECHNICAL_BLOCKED'))
            terminals[stage] = member(path)
        self.receipt = dict(schema=submit.REPAIR_SCHEMA, instruction_id=NONCE, task_id=TASK,
            user_quote='fail되었으니 repair올려', original_KEEP=True, repair_kind='COLD_ZERO_COMMIT',
            predecessor=dict(attempt=str(self.prior), source_commit=self.source, owner=getpass.getuser(), jobs=self.jobs,
                submission=member(self.prior / 'submission.json'), config=member(self.prior / 'config.json'),
                lock=member(self.prior / 'execution.lock.json'), terminals=terminals),
            successor=dict(attempt=str(self.attempt), preparation=str(self.out)), zero_commits={'CD_Q': 0, 'CD_C': 0},
            failure_qualification=qualifications, repair_policy=dict(production_physical_grouping='ORIGINAL_COMPLETE_OWNER_GROUPS_1',
                rejected_physical_regrouping='NOT_QUALIFIED_NOT_USED', new_GPU_READY_required=True,
                gradient_tolerance=dict(atol=1e-6, rtol=2e-4, reduction='elementwise'), tolerance_widening=False,
                method_coefficients_changed=False, extra_fullB_fit=0, automatic_retry=False))
        self.receipt_path = self.local / 'receipt.json'; write(self.receipt_path, self.receipt)
        self.raw = '\n'.join('|'.join((job, getpass.getuser(), jobname(stage),
            'FAILED' if stage in submit.ARMS else 'COMPLETED', '1:0' if stage in submit.ARMS else '0:0',
            str(self.prior / 'source'), '')) for stage, job in self.jobs.items())

    def test_exact_bound_predecessor_and_terminal_snapshot(self):
        receipt = validate_repair_receipt(self.receipt_path, self.attempt, self.out)
        self.assertEqual(receipt, self.receipt)
        registration_guard(self.attempt, receipt)
        self.assertEqual(len(terminal_scheduler_rows(self.raw, receipt)), 3)

    def test_scoped_fresh_scheduler_command_only(self):
        with patch.object(submit, 'command', return_value=self.raw) as command:
            rows = submit.fresh_repair_terminal(self.receipt)
        self.assertEqual(len(rows), 3)
        argv = command.call_args.args[0]
        self.assertEqual(argv[:5], ['sacct', '-n', '-P', '-X', '-j'])
        self.assertEqual(argv[5], '58880,58881,58882')
        self.assertIn('WorkDir%1024', argv[6])

    def test_active_wrong_owner_name_source_and_exit_fail_closed(self):
        replacements = (('FAILED', 'RUNNING'), (getpass.getuser(), 'another-owner'),
            (jobname('CD_Q'), 'another-task'), (str(self.prior / 'source'), '/another/source'), ('1:0', '0:0'))
        for before, after in replacements:
            with self.subTest(after=after), self.assertRaises(RuntimeError):
                terminal_scheduler_rows(self.raw.replace(before, after, 1), self.receipt)

    def test_missing_extra_duplicate_and_step_jobs_fail_closed(self):
        lines = self.raw.splitlines()
        variants = ('\n'.join(lines[:-1]), self.raw + '\n' + lines[0],
                    '\n'.join([lines[0], lines[0], lines[2]]), self.raw.replace('58880|', '58880.batch|', 1))
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaisesRegex(RuntimeError, 'REPAIR_SACCT_EXACT_JOBS'):
                terminal_scheduler_rows(raw, self.receipt)

    def test_other_registration_and_second_submission_are_blocked(self):
        with self.assertRaisesRegex(RuntimeError, 'NO_DUPLICATE_REGISTRATION'):
            registration_guard(self.attempt)
        write(self.local / 'attempt-unexpected/submitted-CD_Q.json', dict(job='99999'))
        with self.assertRaisesRegex(RuntimeError, 'NO_DUPLICATE_REGISTRATION'):
            registration_guard(self.attempt, self.receipt)
        self.attempt.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'CREATE_ONCE_ATTEMPT'):
            registration_guard(self.attempt, self.receipt)

    def test_prior_commit_files_block_cold_repair(self):
        write(self.prior / 'main-CD_Q/batch-01/commit.json', dict(committed=True))
        with self.assertRaisesRegex(RuntimeError, 'REPAIR_PRIOR_COMMIT_PRESENT'):
            validate_repair_receipt(self.receipt_path, self.attempt, self.out)

    def test_changed_predecessor_member_is_rejected(self):
        with (self.prior / 'config.json').open('a') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(RuntimeError, 'CHANGED:'):
            validate_repair_receipt(self.receipt_path, self.attempt, self.out)

    def test_wrong_attempt_and_preparation_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'REPAIR_ATTEMPT_PATH'):
            validate_repair_receipt(self.receipt_path, self.local / 'attempt-r3', self.out)
        with self.assertRaisesRegex(RuntimeError, 'REPAIR_PREPARATION_PATH'):
            validate_repair_receipt(self.receipt_path, self.attempt, self.local / 'preparation-r1')

    def test_prepare_uses_separate_create_once_output_and_explicit_repair(self):
        out, attempt, receipt = prepare.preparation_paths(self.out, self.attempt, self.receipt_path)
        self.assertEqual((out, attempt, receipt), (self.out, self.attempt, self.receipt))
        with self.assertRaisesRegex(RuntimeError, 'EXPLICIT_REPAIR_RECEIPT_REQUIRED'):
            prepare.preparation_paths(self.out, self.attempt)
        write(self.out / 'configuration.json', dict(existing=True))
        with self.assertRaisesRegex(RuntimeError, 'CREATE_ONCE_PREPARATION'):
            prepare.preparation_paths(self.out, self.attempt, self.receipt_path)

    def test_cpu_preflight_filename_and_output_binding(self):
        self.assertEqual(prepare.cpu_preflight_path(self.out), self.out / preflight.CPU_RECEIPT)
        with self.assertRaisesRegex(RuntimeError, 'CPU_PREFLIGHT_OUTPUT_BINDING'):
            prepare.cpu_preflight_path(self.out, self.local / ('preparation-r1/' + preflight.CPU_RECEIPT))
        with self.assertRaisesRegex(RuntimeError, 'SAFE_PREPARATION_OUTPUT'):
            preflight.output_directory(self.local / 'attempt-r1')
        self.assertEqual(preflight.output_directory(), self.local / 'preparation-r1')

    def test_cpu_preflight_never_rewrites_existing_receipt(self):
        write(self.out / preflight.CPU_RECEIPT, dict(existing=True))
        with patch.object(preflight.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'CREATE_ONCE_CPU_PREFLIGHT'):
                preflight.preflight(self.out)
        run.assert_not_called()

if __name__ == '__main__':
    unittest.main()
