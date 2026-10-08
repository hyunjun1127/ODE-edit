"""Final-only reducer fixtures: no GPU/model/SDK/network/scheduler execution."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from . import generation_final_collect as audit


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def metric(entropy=0., cosine=None):
    return dict(ngram_entropy=entropy, reference_score=cosine,
        fluency_valid=entropy is not None, consistency_valid=cosine is not None,
        reasons=['missing_reference'] if cosine is None else [],
        generation_prompt_count=1, generated_token_count=1, length_cap_no_continuation_count=0)


class FinalCollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='generation-final-collector-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(task_id=audit.prior.REPAIR_TASK, instruction_id=audit.prior.REPAIR_NONCE,
            parent_task_id=audit.prior.TASK, registration_profile='final-v1',
            tracking_attempt='final-generation-v1', noCP=True, z_disk_cache=False,
            generation=dict(evaluation_schedule=audit.SCHEDULE, final_generation_requests=2000,
                W0_generation_enabled=False, intermediate_generation_enabled=False,
                schema=audit.prior.SCHEMA, profile=audit.prior.PROFILE, eval_seed=audit.prior.EVAL_SEED,
                source_sha='a' * 40, package_tree='b' * 40, reference_assets_sha256='c' * 64,
                shared_source_members=[]))
        self.lock = dict(task_id=self.config['task_id'], instruction_id=self.config['instruction_id'],
            source_commit='d' * 40, config_sha256='e' * 64,
            shared_generation_source='a' * 40, shared_generation_tree='b' * 40,
            reference_identity='c' * 64)
        self.summary = audit.prior.reduce_generation([{'metrics': metric()} for _ in range(2000)])

    def test_schedule_is_explicit_and_does_not_change_legacy_contract(self):
        audit.schedule(self.config, self.lock)
        for changed in ({'W0_generation_enabled': True}, {'intermediate_generation_enabled': True},
                        {'final_generation_requests': 100}, {'evaluation_schedule': 'EVERY_BATCH'}):
            bad = copy.deepcopy(self.config)
            bad['generation'].update(changed)
            with self.subTest(changed=changed), self.assertRaisesRegex(RuntimeError, 'SCHEDULE_IDENTITY'):
                audit.schedule(bad, self.lock)
        old = dict(self.config, registration_profile='r2')
        with self.assertRaises(RuntimeError):
            audit.schedule(old, self.lock)

    def test_intermediate_generation_unmeasured_not_fake_zero_or_guard_pass(self):
        base = dict(generation_schedule=audit.SCHEDULE, generation_available=False,
            generation_unavailable_reason='FINAL_W20_ONLY_SCHEDULE',
            generation_observer_status='NOT_SCHEDULED_INTERMEDIATE')
        audit.unscheduled_commit(base)
        for key in audit.FORBIDDEN_COMMIT_KEYS:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                audit.unscheduled_commit(dict(base, **{key: None}))
        with self.assertRaisesRegex(RuntimeError, 'UNRUN_GENERATION'):
            audit.unscheduled_commit(dict(base, generation_observer_no_mutation=True))

    def test_missing_consistency_mean_is_omitted_measured_entropy_zero_is_retained(self):
        values = audit.generation_scalars(self.summary)
        self.assertEqual(values['all_seen/post/fluency/ngram_entropy'], 0.)
        self.assertNotIn('all_seen/post/consistency/reference_score', values)
        self.assertEqual(values['all_seen/post/generation/consistency_count'], 0)
        self.assertEqual(values['all_seen/post/generation/planned_count'], 2000)
        summary = audit.prior.reduce_generation([{'metrics': metric(cosine=0.)}])
        self.assertEqual(audit.generation_scalars(summary)['all_seen/post/consistency/reference_score'], 0.)

    def track(self, *, commands=None, method_readback=None, generation_progress_readback=None,
              SDK_finish_status='FINISHED_SDK_FLUSHED'):
        out = self.root / 'BASE_MEMIT'
        cfg = dict(server='server2', task_id=self.config['task_id'], arm='BASE_MEMIT',
            generation_schedule='W20_ONLY_FIRST2000',
            attempt=self.config['tracking_attempt'], source_sha=self.lock['source_commit'],
            config_sha=self.lock['config_sha256'], writer='memit', model='gptj', model_family='gptj',
            role='scientific', execution_backend='slurm', job_id='71300', job_display_id='71300')
        identity = dict(config=cfg, run_id='fixture-run', url='https://example.invalid/fixture-run',
            run_name='server2-BASE_MEMIT-final-generation-v1-job71300',
            startup_remote_identity_verified=True, scientific_completion_claim=False)
        write_json(out / 'tracking' / 'identity.json', identity)
        values = dict(audit.generation_scalars(self.summary), edits=2000, pre_state_edits=2000,
                      post_state_edits=2000)
        commands = commands if commands is not None else [dict(op='log', values=values, step=None)]
        path = out / 'tracking' / 'accepted-scalars.jsonl'
        path.write_text('\n'.join(json.dumps(value) for value in commands) + '\n')
        write_json(out / 'tracking' / 'receipt.json', dict(run_id='fixture-run', status='FINISHED_SDK_FLUSHED',
            result=dict(status=SDK_finish_status,
                method_readback=method_readback if method_readback is not None
                    else dict(status='UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE', rows=None),
                generation_progress_readback=generation_progress_readback if generation_progress_readback is not None
                    else dict(status='UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE', rows=None))))
        return out, values

    def test_only_one_final_all_seen_generation_payload_and_readback_not_assumed(self):
        out, _ = self.track()
        value = audit.tracking_evidence(audit.prior.Reader(), out, self.config, self.lock, 'BASE_MEMIT', self.summary)
        self.assertEqual(value['scalar_payloads'], 1)
        self.assertEqual(value['status'], 'LOCAL_ACCEPTED_SCALAR_JOURNAL_VERIFIED_NOT_REMOTE_ACK')
        self.assertEqual(value['method_readback']['status'], 'UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE')
        self.assertEqual(value['generation_progress_readback']['status'], 'UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE')
        self.assertEqual(value['SDK_finish_status'], 'FINISHED_SDK_FLUSHED')
        self.assertTrue(value['SDK_flush_observed'])
        self.assertTrue(value['scientific_completion_not_implied'])

    def test_production_finish_fields_preserve_distinct_bounded_readbacks(self):
        method = dict(status='REMOTE_BOUNDED_ROWS_VERIFIED', rows=2,
            scope='last evaluation and last fit only', checked=[dict(kind='evaluation', edits=2000)])
        progress = dict(status='REMOTE_BOUNDED_PROGRESS_VERIFIED', rows=1,
            phase='generation_evaluation', generation_step=17)
        out, _ = self.track(method_readback=method, generation_progress_readback=progress)
        value = audit.tracking_evidence(audit.prior.Reader(), out, self.config, self.lock, 'BASE_MEMIT', self.summary)
        self.assertEqual(value['method_readback'], method)
        self.assertEqual(value['generation_progress_readback'], progress)
        self.assertTrue(value['SDK_flush_observed'])
        self.assertTrue(value['scientific_completion_not_implied'])
        out, _ = self.track(method_readback=method, generation_progress_readback=progress,
                            SDK_finish_status='FINISHED_UNVERIFIED')
        value = audit.tracking_evidence(audit.prior.Reader(), out, self.config, self.lock, 'BASE_MEMIT', self.summary)
        self.assertFalse(value['SDK_flush_observed'])
        self.assertEqual(value['method_readback'], method)
        self.assertEqual(value['generation_progress_readback'], progress)

    def test_absent_finish_proofs_are_not_sdk_flush_or_remote_pass(self):
        out, _ = self.track()
        write_json(out / 'tracking' / 'receipt.json', dict(run_id='fixture-run', status='READY_ONLINE',
            result=dict(status='READY_ONLINE', readback={'status': 'UNRELATED_LEGACY_FIELD'})))
        value = audit.tracking_evidence(audit.prior.Reader(), out, self.config, self.lock, 'BASE_MEMIT', self.summary)
        self.assertFalse(value['SDK_flush_observed'])
        self.assertEqual(value['method_readback'], {'status': 'NOT_RECORDED'})
        self.assertEqual(value['generation_progress_readback'], {'status': 'NOT_RECORDED'})
        self.assertTrue(value['scientific_completion_not_implied'])

    def test_logging_duplicate_current_w0_or_wrong_scalar_is_rejected(self):
        _, values = self.track()
        command = dict(op='log', values=values, step=None)
        alternatives = [[command, command], [dict(command, values=dict(values, edits=100))],
            [dict(command, values={key.replace('all_seen/post', 'current/post'): value
                                   for key, value in values.items()})],
            [dict(command, values=dict(values, **{'all_seen/post/generation/planned_count': 1999}))]]
        for commands in alternatives:
            with self.subTest(commands=len(commands)):
                out, _ = self.track(commands=commands)
                with self.assertRaises(RuntimeError):
                    audit.tracking_evidence(audit.prior.Reader(), out, self.config, self.lock, 'BASE_MEMIT', self.summary)

    def final_fixture(self):
        out, _ = self.track()
        state = dict(W={'3': 'actual-final-weight'}, H={})
        path = out / 'generation-raw' / 'endpoints' / 'endpoint.json'
        write_json(path, dict(fixture=True))
        receipt = dict(task=self.config['task_id'], arm='BASE_MEMIT', source=self.lock['source_commit'],
            config=audit.digest(self.config), generation_schedule=audit.SCHEDULE, endpoint='W20',
            edits=2000, actual_model_edits=2000, model_state=state, requests=2000, derived_subset=False,
            checkpoint_saved=False, exact_resume='NOT_AVAILABLE', generation_observer_no_mutation=True,
            raw_endpoint_member=audit.member(path))
        write_json(out / 'generation-final.json', receipt)
        records = [dict(case_id=10000 + index) for index in range(2000)]
        observed = dict(summary=self.summary, rows=[{}] * 2000, identity_sha256='fixture-endpoint',
                        work=dict(new_case_observations=2000, cached_case_observations=0))
        return out, state, receipt, records, observed

    def final_check(self, records, state, observed, qualification=None):
        with patch.object(audit.prior, 'qualification_link') as qlink, \
                patch.object(audit.prior, 'repair_generation_endpoint', return_value=observed) as endpoint:
            result = audit.final_generation(audit.prior.Reader(), self.root, self.config, self.lock,
                'BASE_MEMIT', records, state, qualification or dict(actual_qualification=True))
        return result, qlink, endpoint

    def test_final_endpoint_delegates_full_saved_raw_qualification_order_and_actual_state(self):
        _, state, _, records, observed = self.final_fixture()
        result, qlink, endpoint = self.final_check(records, state, observed)
        self.assertEqual(result[0]['summary']['planned_count'], 2000)
        self.assertEqual(qlink.call_count, 1)
        args = endpoint.call_args.args
        self.assertEqual(args[2], records)
        self.assertEqual(args[3], list(range(1, 2001)))
        self.assertEqual(args[4:7], (state, 'W20', 'ALL_SEEN'))

    def test_old_w0_endpoint_extra_generation_and_cached_state_are_rejected(self):
        out, state, receipt, records, observed = self.final_fixture()
        write_json(out / 'generation-W0-reference.json', {})
        with self.assertRaisesRegex(RuntimeError, 'NO_W0_ENDPOINT'):
            self.final_check(records, state, observed)
        (out / 'generation-W0-reference.json').unlink()
        write_json(out / 'generation-raw' / 'endpoints' / 'second.json', {})
        with self.assertRaisesRegex(RuntimeError, 'EXACTLY_ONE_SAVED_ENDPOINT'):
            self.final_check(records, state, observed)
        (out / 'generation-raw' / 'endpoints' / 'second.json').unlink()
        bad = copy.deepcopy(observed)
        bad['work'].update(new_case_observations=1604, cached_case_observations=396)
        with self.assertRaisesRegex(RuntimeError, 'FRESH_W20'):
            self.final_check(records, state, bad)
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_QUALIFICATION_REQUIRED'):
            self.final_check(records, state, observed, dict(actual_qualification=False))
        receipt['edits'] = 1900
        write_json(out / 'generation-final.json', receipt)
        with self.assertRaisesRegex(RuntimeError, 'PHYSICAL_W20_SOURCE'):
            self.final_check(records, state, observed)

    def test_native_plan_120_vs40_history_and_twenties_links_are_unchanged(self):
        totals = {arm: {key: count * 20 for key, count in audit.expected_counts(arm).items()}
                  for arm in audit.ARMS}
        self.assertEqual(totals['CAKE']['history_appends'], 120)
        self.assertEqual(totals['BASE_ALPHAEDIT']['history_appends'], 120)
        self.assertEqual(totals['ALPHAEDIT_BLUE']['history_appends'], 40)
        self.assertEqual(totals['PRUNE']['history_appends'], 0)
        self.assertEqual(sum(value['solves'] for value in totals.values()), 640)
        self.assertEqual(sum(value['native_z'] for value in totals.values()), 14000)
        self.assertEqual(audit.terminal_status(dict(status='COMPLETED'), 20, True, []),
                         'COMPLETED_VALIDATED_ROWS_COUNTS')
        self.assertEqual(audit.terminal_status(dict(status='COMPLETED'), 19, True, []),
                         'TECHNICAL_BLOCKED_INCOMPLETE_EVIDENCE')
        self.assertEqual(audit.terminal_status(dict(status='FAILED'), 20, True, []), 'FAILED')

    def test_import_contains_no_model_sdk_or_scoring_implementation(self):
        for symbol in ('torch', 'transformers', 'wandb', 'GenerationObserver', 'word_tokenize'):
            self.assertNotIn(symbol, audit.__dict__)

    def test_collector_reports_failure_no_fictitious_completion_and_terminal_last(self):
        records = [dict(case_id=10001 + index) for index in range(2000)]
        stream, identity = self.root / 'stream.json', self.root / 'observer.json'
        write_json(stream, records)
        write_json(identity, [])
        config = dict(self.config, stream=str(stream), assets=[audit.member(stream)],
            observer_identity=audit.member(identity), packs=[dict(ids=[r['case_id'] for r in records[i:i + 100]])
                for i in range(0, 2000, 100)])
        write_json(self.root / 'config.json', config)
        lock = dict(self.lock, config_sha256=audit.member(self.root / 'config.json')['sha256'])
        write_json(self.root / 'execution.lock.json', lock)
        write_json(self.root / 'BASE_MEMIT' / 'terminal.json', dict(status='FAILED', error_type='FixtureNoModel'))
        chunks = [(i + 1, records[i * 100:(i + 1) * 100], records[:(i + 1) * 100]) for i in range(20)]
        ready = Mock(return_value=True)
        module = types.SimpleNamespace(ready=ready)
        writes = []
        original = audit.write
        def recorded(path, value):
            writes.append(Path(path).name)
            return original(path, value)
        with patch.dict('sys.modules', {'project.run_scripts.gptj_native_baselines.generation_final_common': module}), \
                patch.object(audit, 'batches', side_effect=lambda _: iter(chunks)), \
                patch.object(audit.prior, 'qualification_evidence', return_value=dict(actual_qualification=False,
                    status='BLOCKED_QUALIFICATION_PENDING')), patch.object(audit, 'write', side_effect=recorded), \
                patch('project.run_scripts.gptj_native_baselines.generation_cache_common.ready',
                      side_effect=AssertionError('NO_HISTORICAL_W0_CACHE_GATE')), \
                patch.object(audit.prior, 'compatibility_evidence',
                      side_effect=AssertionError('NO_W0_COMPATIBILITY_GATE')), \
                patch.object(audit.prior, 'repair_ready_evidence',
                      side_effect=AssertionError('NO_W0_READY_GATE')), \
                patch.object(audit.prior.subprocess, 'run', side_effect=AssertionError('NO_ACTUAL_SCHEDULER')):
            result = audit.collect(self.root)
        ready.assert_called_once_with(config)
        self.assertFalse(result['scientific_complete'])
        review = json.loads((self.root / 'collector' / 'review.json').read_text())
        self.assertEqual(review['reviews'][0]['scientific_status'], 'FAILED')
        self.assertEqual(review['reviews'][1]['scientific_status'], 'NOT_STARTED_OR_STARTUP_FAILED')
        self.assertEqual(writes[-1], 'terminal.json')
        self.assertLess(writes.index('manifest.json'), len(writes) - 1)
        self.assertNotIn('case_id', json.dumps(review))
        self.assertEqual(review['new_model_forwards'], 0)

    def test_final_ready_identity_failure_stops_before_raw_qualification_or_output(self):
        write_json(self.root / 'config.json', self.config)
        lock = dict(self.lock, config_sha256=audit.member(self.root / 'config.json')['sha256'])
        write_json(self.root / 'execution.lock.json', lock)
        ready = Mock(side_effect=RuntimeError('FINAL_ONLY_USER_AUTHORITY_MEMBER'))
        module = types.SimpleNamespace(ready=ready)
        with patch.dict('sys.modules', {'project.run_scripts.gptj_native_baselines.generation_final_common': module}), \
                patch.object(audit.prior, 'qualification_evidence',
                    side_effect=AssertionError('MUST_CHECK_FINAL_PROFILE_FIRST')) as qualification, \
                patch.object(audit, 'write', side_effect=AssertionError('NO_OUTPUT_BEFORE_READY')) as writing, \
                patch('project.run_scripts.gptj_native_baselines.generation_cache_common.ready',
                    side_effect=AssertionError('NO_HISTORICAL_W0_GATE')):
            with self.assertRaisesRegex(RuntimeError, 'FINAL_ONLY_USER_AUTHORITY_MEMBER'):
                audit.collect(self.root)
        ready.assert_called_once_with(self.config)
        qualification.assert_not_called()
        writing.assert_not_called()
        self.assertFalse((self.root / 'collector').exists())


if __name__ == '__main__':
    unittest.main()
