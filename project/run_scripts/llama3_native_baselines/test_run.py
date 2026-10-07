"""Small CPU controller/receipt software regressions; no editing toy or model."""
import copy
import unittest
from unittest import mock

from .generation import model_identity, normalize_summary, runtime_identity
from .run import seal_commit, sequence, storage_guard, terminal_before_observation, verified_prefix


def summary():
    return dict(planned_count=100, fluency_count=100, consistency_count=99,
        fluency_sum=300., consistency_sum=49.5, generation_prompt_count=1000,
        generated_token_count=9000, missing_reason_counts={reason: 0 for reason in
            ('missing_generation_prompts', 'missing_reference', 'zero_generated_vector',
             'zero_reference_vector', 'nonfinite_score', 'length_cap_no_continuation',
             'asset_not_available', 'tokenizer_not_available')})


class ControllerSoftwareTests(unittest.TestCase):
    def test_twenty_batch_exact_horizon(self):
        rows = [sequence(number) for number in range(1, 21)]
        self.assertEqual([r['number'] for r in rows if r['milestone']], [5, 10, 15, 20])
        self.assertEqual(rows[-1]['edits'], 2000)
        self.assertEqual(sum(r['requests'] for r in rows), 2000)
        for invalid in (0, 21, 20., True):
            with self.assertRaises(RuntimeError): sequence(invalid)

    def test_prune_only_terminal_before_observation(self):
        engine = mock.Mock(method='PRUNE')
        engine.terminal_prune.return_value = {'prune': 'terminal'}
        self.assertIsNone(terminal_before_observation(engine, 19))
        engine.terminal_prune.assert_not_called()
        self.assertEqual(terminal_before_observation(engine, 20), {'prune': 'terminal'})
        engine.terminal_prune.assert_called_once_with()
        other = mock.Mock(method='MEMIT')
        self.assertIsNone(terminal_before_observation(other, 20))
        other.terminal_prune.assert_not_called()

    def test_shared_summary_preserves_missing_and_units(self):
        value = summary()
        value['missing_reason_counts']['missing_reference'] = 1
        before = copy.deepcopy(value)
        normalized = normalize_summary(value)
        self.assertEqual(normalized['consistency_count'], 99)
        self.assertEqual(normalized['consistency_sum'], 49.5)
        self.assertEqual(normalized['fluency_sum'], 300.)
        self.assertEqual(normalized['reason_counts']['missing_reference'], 1)
        self.assertEqual(value, before)

    def test_asset_runtime_failure_is_not_zero_score(self):
        for reason in ('asset_not_available', 'tokenizer_not_available'):
            value = summary(); value['missing_reason_counts'][reason] = 1
            with self.assertRaisesRegex(RuntimeError, 'READY_ASSET_RUNTIME_LOST'):
                normalize_summary(value)

    def test_generation_model_seed_no_method_job_attempt(self):
        c = dict(model_revision='revision', observation_identity='observation',
            method='MEMIT', job_id='123', attempt='one',
            generation=dict(source_sha='source', assets_sha256='assets'))
        changed = dict(c, method='ALPHAEDIT', job_id='999', attempt='two')
        self.assertEqual(model_identity(c), model_identity(changed))
        self.assertEqual(runtime_identity(c), runtime_identity(changed))
        self.assertEqual(set(model_identity(c)), {'model', 'revision', 'observation_identity'})

    def test_generation_reference_or_source_changes_runtime_identity(self):
        c = dict(model_revision='revision', observation_identity='observation',
                 generation=dict(source_sha='source', assets_sha256='assets'))
        for key in ('source_sha', 'assets_sha256'):
            other = copy.deepcopy(c); other['generation'][key] = 'different'
            self.assertNotEqual(runtime_identity(c), runtime_identity(other))

    def test_storage_enforced_without_deleting_or_reducing(self):
        with mock.patch('shutil.disk_usage', return_value=mock.Mock(free=1024)), \
             mock.patch('os.statvfs', return_value=mock.Mock(f_favail=2000)):
            value = storage_guard('/unused-test-path', 1024)
            self.assertTrue(value['no_deletion'])
            self.assertTrue(value['no_observation_reduction'])
            with self.assertRaisesRegex(RuntimeError, 'RESOURCE_BLOCKED_STORAGE'):
                storage_guard('/unused-test-path', 1025)
        with self.assertRaises(RuntimeError): storage_guard('/unused-test-path', 0)

    def test_commit_hash_failure_invalidates_inside_transaction(self):
        transaction = mock.Mock()
        with mock.patch('project.run_scripts.llama3_native_baselines.run.write'), \
             mock.patch('project.run_scripts.llama3_native_baselines.run.member',
                        side_effect=OSError('hash-read-failed')):
            with self.assertRaisesRegex(OSError, 'hash-read-failed'):
                seal_commit(transaction, '/unused-test-path', {'batch': 1})
        transaction.finish.assert_called_once_with()
        transaction.invalidate.assert_called_once_with()

    def test_commit_member_verified_before_success_return(self):
        transaction = mock.Mock()
        sealed = {'path': '/unused-test-path', 'sha256': 'bound-member'}
        order = []
        transaction.finish.side_effect = lambda: order.append('finish')
        with mock.patch('project.run_scripts.llama3_native_baselines.run.write',
                        side_effect=lambda *args: order.append('write')), \
             mock.patch('project.run_scripts.llama3_native_baselines.run.member',
                        side_effect=lambda *args: order.append('hash') or sealed), \
             mock.patch('project.run_scripts.llama3_native_baselines.run.verify',
                        side_effect=lambda *args: order.append('verify')):
            self.assertEqual(seal_commit(transaction, '/unused-test-path', {'batch': 1}), sealed)
        self.assertEqual(order, ['finish', 'write', 'hash', 'verify'])
        transaction.invalidate.assert_not_called()

    def test_retained_receipt_is_not_verified_prefix(self):
        # A receipt left by failed verification/rollback is retained evidence,
        # not another successful transaction. Prefix reporting is RAM-bound.
        progress = {'commits': 4, 'receipt_files_present': 5}
        self.assertEqual(verified_prefix(progress), 4)
        for invalid in (True, -1, 21):
            with self.assertRaises(RuntimeError): verified_prefix({'commits': invalid})


if __name__ == '__main__': unittest.main()
