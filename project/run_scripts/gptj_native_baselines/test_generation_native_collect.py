"""Stored scalar/token receipt fixtures only; no model/GPU/Slurm/SDK calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import generation_native_collect as audit


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


class NativeCollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='native-collector-cpu-fixture-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = dict(task_id=audit.TASK, instruction_id=audit.NONCE,
            registration_profile='native-repo-r1', tracking_attempt='native-repo-repair-r1',
            noCP=True, z_disk_cache=False,
            generation=dict(evaluation_schedule=audit.SCHEDULE, final_generation_requests=2000,
                W0_generation_enabled=False, intermediate_generation_enabled=False,
                schema=audit.prior.SCHEMA, profile=audit.PROFILE, eval_seed=20261007,
                generation_route=audit.ROUTE, model_identity={'model': 'gptj', 'revision': 'fixture'},
                source_sha='a' * 40, package_tree='b' * 40, reference_assets_sha256='c' * 64,
                shared_source_members=[]))
        self.lock = dict(task_id=audit.TASK, instruction_id=audit.NONCE,
            source_commit='d' * 40, config_sha256='e' * 64,
            shared_generation_source='a' * 40, shared_generation_tree='b' * 40,
            reference_identity='c' * 64)
        self.state = dict(W={'3': 'f' * 64}, H={'3': '0' * 64})
        self.records = [dict(case_id=734, requested_rewrite=dict(relation_id='r', target_new={'id': 'n'}),
                             generation_prompts=['synthetic fixture prompt'])]

    def fixture(self):
        runtime = audit.native_runtime(self.config['generation'])
        runtime_sha = audit.digest(runtime)
        record = self.records[0]
        ri = dict(ordered_occurrence=1, case_id=record['case_id'], generation_prompts=record['generation_prompts'],
                  relation_id='r', target_new_id='n')
        stream = audit.digest(dict(runtime=runtime_sha, eval_seed=20261007,
                                   ordered_record_identities=[ri]))
        raw_identity = dict(runtime=runtime_sha, state_identity=self.state,
                            record_identity=ri, sampling_stream_sha256=stream)
        observation = dict(profile=audit.PROFILE, route=audit.ROUTE, seed=20261007,
            sampling_scope='ENDPOINT_GLOBAL_BATCH_STREAM', prompt=record['generation_prompts'][0],
            occurrence=1, prompt_index=0, endpoint_RNG_restore_guard_required=True,
            EOS_stop=False, case_batch_prompt_count=1, initial_batch_width=99,
            model_forwards=1, physical_forward_calls=1,
            sampling=dict(top_k=5, temperature=1, top_p=1, max_total_tokens=100, n_gen_per_prompt=1),
            input_token_ids=[1] * 99, continuation_token_ids=[2], full_token_ids=[1] * 99 + [2],
            padded_input_token_ids=[1] * 99, padded_decode_token_ids=[1] * 99 + [2],
            input_token_count=99, continuation_token_count=1, stop_reason='length_cap',
            text='synthetic fixture text', prefill_query_tokens=99, decode_query_tokens=0,
            full_prefix_token_work=99)
        metrics = dict(ngram_entropy=0., reference_score=None, fluency_valid=True,
            consistency_valid=False, reasons=['missing_reference'], generation_prompt_count=1,
            generated_token_count=1, length_cap_no_continuation_count=0)
        raw = dict(identity=raw_identity, identity_sha256=audit.digest(raw_identity), occurrence=1,
            case_id=record['case_id'], observations=[observation], metrics=metrics,
            raw_local_only=True, checkpoint_saved=False)
        raw['payload_sha256'] = audit.digest(raw)
        out = self.root / 'BASE_MEMIT' / 'generation-raw'
        raw_path = out / 'observations' / (raw['identity_sha256'] + '.json')
        write_json(raw_path, raw)
        row = dict(occurrence=1, case_id=record['case_id'], identity_sha256=raw['identity_sha256'],
            observation_path=str(raw_path), payload_sha256=raw['payload_sha256'], metrics=metrics,
            provenance=dict(origin='NEW_CURRENT_RUNTIME', raw_member=audit.member(raw_path),
                runtime_sha256=runtime_sha, generation_source_sha='a' * 40, route=audit.ROUTE))
        identity = dict(runtime=runtime_sha, state_sha256=audit.digest(self.state), endpoint='W20',
            cohort='ALL_SEEN', ordered_occurrences=[1], observation_identities=[raw['identity_sha256']],
            sampling_stream_sha256=stream)
        key = audit.digest(identity)
        execution = dict(identity=identity, identity_sha256=key, profile=audit.PROFILE, route=audit.ROUTE,
            native_execution_complete=True, qualification_performed=False, no_fallback=True,
            RNG_restored=True, observer_no_mutation=True, raw_local_only=True,
            physical_forward_calls=1, prefill_query_tokens=99, decode_query_tokens=0)
        execution_path = out / 'native-executions' / (key + '.json')
        write_json(execution_path, execution)
        summary = audit.prior.reduce_generation([row])
        saved = dict(identity=identity, identity_sha256=key, rows=[row], summary=summary,
            RNG_restored=True, observer_no_mutation=True, raw_local_only=True,
            native_execution_member=audit.member(execution_path))
        endpoint_path = out / 'endpoints' / (key + '.json')
        write_json(endpoint_path, saved)
        write_json(out / 'observer-identity.json', dict(identity=runtime, identity_sha256=runtime_sha,
            raw_local_only=True, checkpoint_saved=False))
        work = dict(new_case_observations=1, cached_case_observations=0, generation_forwards=1,
            full_prefix_token_work=99, physical_forward_calls=1, prefill_query_tokens=99,
            decode_query_tokens=0, completed_prompts=1, generated_tokens=1, seconds=.125)
        receipt = dict(rows_path=str(endpoint_path), raw_endpoint_member=audit.member(endpoint_path),
            identity=identity, identity_sha256=key, requests=1,
            cohort_identity=audit.digest([record['case_id']]), shared_state_identity=self.state,
            shared_runtime_identity=runtime, summary=summary, shared_summary=summary,
            RNG_restored=True, observer_no_mutation=True, native_execution_member=audit.member(execution_path),
            work=work)
        return receipt, saved, raw, endpoint_path, raw_path

    def test_native_schedule_rejects_legacy_route_plan_and_wrong_horizon(self):
        audit.schedule(self.config, self.lock)
        for key, value in [('repair', {}), ('qualification_owner', 'BASE_MEMIT'),
                           ('generation_microbatch', 8), ('W0_cache', '/fixture/unused')]:
            bad = copy.deepcopy(self.config)
            bad['generation'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'NO_OLD_QUALIFICATION'):
                audit.schedule(bad, self.lock)
        for key, value in [('profile', audit.prior.PROFILE), ('W0_generation_enabled', True),
                           ('intermediate_generation_enabled', True), ('final_generation_requests', 100)]:
            bad = copy.deepcopy(self.config)
            bad['generation'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'SCHEDULE_IDENTITY'):
                audit.schedule(bad, self.lock)

    def test_native_reader_independent_sums_and_physical_work(self):
        receipt, _, _, _, _ = self.fixture()
        result = audit.native_generation_endpoint(audit.prior.Reader(), receipt, self.records,
                                                  self.state, self.config, self.root)
        self.assertEqual(result['summary']['planned_count'], 1)
        self.assertEqual(result['summary']['ngram_entropy'], 0.)
        self.assertNotIn('reference_score', result['summary'])
        self.assertEqual(result['work']['physical_forward_calls'], 1)
        self.assertEqual(result['work']['prefill_query_tokens'], 99)
        self.assertEqual(result['work']['generated_tokens'], 1)

    def test_summary_omits_missing_mean_and_retains_measured_zero(self):
        receipt, _, _, _, _ = self.fixture()
        scalars = audit.generation_scalars(receipt['summary'])
        self.assertEqual(scalars['all_seen/post/fluency/ngram_entropy'], 0.)
        self.assertNotIn('all_seen/post/consistency/reference_score', scalars)
        self.assertEqual(scalars['all_seen/post/generation/consistency_count'], 0)

    def test_wrong_expected_global_stream_is_not_same_cohort(self):
        receipt, _, _, _, _ = self.fixture()
        wrong = copy.deepcopy(self.records)
        wrong[0]['generation_prompts'] = ['different fixture prompt']
        with self.assertRaisesRegex(RuntimeError, 'EXPECTED_RUNTIME_GLOBAL_STREAM_STATE'):
            audit.native_generation_endpoint(audit.prior.Reader(), receipt, wrong,
                                              self.state, self.config, self.root)

    def test_history_part_of_actual_physical_state_not_dropped(self):
        receipt, _, _, _, _ = self.fixture()
        state = dict(W=self.state['W'], H={})
        with self.assertRaisesRegex(RuntimeError, 'EXPECTED_RUNTIME_GLOBAL_STREAM_STATE'):
            audit.native_generation_endpoint(audit.prior.Reader(), receipt, self.records,
                                              state, self.config, self.root)

    def test_stock_alpha_lazy_empty_then_first_commit_six_history_planes(self):
        cold = dict(W={str(layer): 'f' * 64 for layer in range(3, 9)}, H={})
        written = dict(W=cold['W'], H={str(layer): '0' * 64 for layer in range(3, 9)})
        self.assertEqual(audit.history_layout(cold, 'BASE_ALPHAEDIT', initial=True), set())
        self.assertEqual(audit.history_layout(written, 'BASE_ALPHAEDIT'),
                         {str(layer) for layer in range(3, 9)})
        with self.assertRaisesRegex(RuntimeError, 'INITIAL_HISTORY_LAYOUT'):
            audit.history_layout(written, 'BASE_ALPHAEDIT', initial=True)
        with self.assertRaisesRegex(RuntimeError, 'COMMITTED_HISTORY_LAYOUT'):
            audit.history_layout(cold, 'BASE_ALPHAEDIT')
        bad = copy.deepcopy(written)
        del bad['H']['3']
        with self.assertRaisesRegex(RuntimeError, 'COMMITTED_HISTORY_LAYOUT'):
            audit.history_layout(bad, 'BASE_ALPHAEDIT')

    def test_cake_and_blue_preallocated_cold_history_not_lazy_stock_alpha(self):
        six = dict(H={str(layer): '0' * 64 for layer in range(3, 9)})
        two = dict(H={'3': '0' * 64, '8': '0' * 64})
        self.assertEqual(audit.history_layout(six, 'CAKE', initial=True), set(six['H']))
        self.assertEqual(audit.history_layout(two, 'ALPHAEDIT_BLUE', initial=True), set(two['H']))
        with self.assertRaisesRegex(RuntimeError, 'INITIAL_HISTORY_LAYOUT'):
            audit.history_layout(dict(H={}), 'CAKE', initial=True)

    def test_claimed_zero_work_or_reused_case_is_rejected(self):
        receipt, _, _, _, _ = self.fixture()
        for field, value in [('physical_forward_calls', 0), ('prefill_query_tokens', 0),
                             ('generated_tokens', 0), ('cached_case_observations', 1)]:
            wrong = copy.deepcopy(receipt)
            wrong['work'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'LOGICAL_PHYSICAL_EXECUTION'):
                audit.native_generation_endpoint(audit.prior.Reader(), wrong, self.records,
                                                  self.state, self.config, self.root)

    def test_native_raw_EOS_semantics_and_token_order_are_verified(self):
        receipt, _, raw, _, raw_path = self.fixture()
        raw['observations'][0]['EOS_stop'] = True
        raw['payload_sha256'] = audit.digest({k: v for k, v in raw.items() if k != 'payload_sha256'})
        write_json(raw_path, raw)
        # Byte/member mismatch fails closed before unsupported profile can pass.
        with self.assertRaises(RuntimeError):
            audit.native_generation_endpoint(audit.prior.Reader(), receipt, self.records,
                                              self.state, self.config, self.root)

    def test_fake_qualification_member_is_not_native_execution(self):
        receipt, _, _, _, _ = self.fixture()
        receipt['qualification_receipt_member'] = {'path': '/unused'}
        with self.assertRaisesRegex(RuntimeError, 'NOT_QUALIFICATION_OR_SUBSET'):
            audit.native_generation_endpoint(audit.prior.Reader(), receipt, self.records,
                                              self.state, self.config, self.root)

    def test_raw_size_SHA_change_cannot_be_relabelled(self):
        receipt, _, raw, _, raw_path = self.fixture()
        raw['metrics']['ngram_entropy'] = 1.
        write_json(raw_path, raw)
        with self.assertRaises(RuntimeError):
            audit.native_generation_endpoint(audit.prior.Reader(), receipt, self.records,
                                              self.state, self.config, self.root)

    def test_missing_unstarted_and_failed_terminal_are_not_complete(self):
        self.assertEqual(audit.terminal_status({}, 0, False, []), 'NOT_STARTED_OR_STARTUP_FAILED')
        self.assertEqual(audit.terminal_status({'status': 'FAILED'}, 3, False, []), 'FAILED')
        self.assertEqual(audit.terminal_status({'status': 'COMPLETED'}, 20, False, []),
                         'TECHNICAL_BLOCKED_INCOMPLETE_EVIDENCE')
        self.assertEqual(audit.terminal_status({'status': 'COMPLETED'}, 20, True, []),
                         'COMPLETED_VALIDATED_ROWS_COUNTS')

    def test_unmeasured_intermediate_generation_is_not_no_mutation_PASS(self):
        value = dict(generation_schedule=audit.SCHEDULE, generation_available=False,
            generation_unavailable_reason='FINAL_W20_ONLY_SCHEDULE',
            generation_observer_status='NOT_SCHEDULED_INTERMEDIATE')
        audit.unscheduled_commit(value)
        with self.assertRaisesRegex(RuntimeError, 'UNRUN_GENERATION'):
            audit.unscheduled_commit(dict(value, generation_observer_no_mutation=True))

    def test_error_code_does_not_publish_path_or_secret_like_error(self):
        self.assertEqual(audit.compact_error_code(RuntimeError('NATIVE_GEN_TOKEN_ORDER')), 'NATIVE_GEN_TOKEN_ORDER')
        self.assertEqual(audit.compact_error_code(RuntimeError('/fixture/private/path arbitrary error')),
                         'UNCLASSIFIED_REVIEW_ERROR')


if __name__ == '__main__':
    unittest.main()
