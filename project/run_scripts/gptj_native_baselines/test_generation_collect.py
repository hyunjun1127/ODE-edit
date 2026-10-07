"""Small CPU receipt fixtures; no model, generation, fitter or scheduler call."""
import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import generation_collect as audit


def metric(*, entropy=0., cosine=None, reasons=None):
    return dict(ngram_entropy=entropy, reference_score=cosine,
        fluency_valid=entropy is not None, consistency_valid=cosine is not None,
        reasons=sorted(reasons if reasons is not None else ['missing_reference']),
        generation_prompt_count=1, generated_token_count=1,
        length_cap_no_continuation_count=0)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='generation-collect-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = dict(generation=dict(model_identity='fixture-model', source_sha='a' * 40,
            reference_assets_sha256='b' * 64))
        self.records = [dict(case_id=91, generation_prompts=['fixture prompt'], requested_rewrite=dict(
            relation_id='r', target_new=dict(id='t', str=' new'))),
            dict(case_id=12, generation_prompts=['other prompt'], requested_rewrite=dict(
                relation_id='r', target_new=dict(id='t2', str=' new')))]

    def endpoint(self, state=None, *, endpoint='W5', cohort='ALL_SEEN', cached=False):
        state = {'W': {'3': 'edited'}, 'H': {}} if state is None else state
        runtime = audit.runtime_identity(self.config)
        runtime_sha = audit.digest(runtime)
        base = self.root / 'BASE_MEMIT' / 'generation-raw'
        write_json(base / 'observer-identity.json', dict(identity=runtime,
            identity_sha256=runtime_sha, raw_local_only=True, checkpoint_saved=False))
        rows = []
        for occurrence, record in enumerate(self.records, 1):
            prompt = record['generation_prompts'][0]
            seed = int(audit.digest(dict(model_identity='fixture-model', ordered_occurrence=occurrence,
                prompt_index=0, eval_seed=audit.EVAL_SEED))[:16], 16) % (2**63 - 1)
            observation = dict(profile=audit.PROFILE, seed=seed, occurrence=occurrence,
                prompt_index=0, prompt=prompt, input_token_ids=[1, 2], continuation_token_ids=[9],
                full_token_ids=[1, 2, 9], text='fixture generated', input_token_count=2,
                continuation_token_count=1, stop_reason='eos', eos_ids=[9],
                eos_binding={'tokenizer': [9]}, model_forwards=1, full_prefix_token_work=2,
                route='UNPADDED_FULL_PREFIX_NO_CACHE', sampling=dict(top_k=5, temperature=1,
                top_p=1, max_total_tokens=100), RNG_restored=True)
            rewrite = record['requested_rewrite']
            identity = dict(runtime=runtime_sha, state_identity=state, record_identity=dict(
                ordered_occurrence=occurrence, case_id=record['case_id'], generation_prompts=[prompt],
                relation_id=rewrite['relation_id'], target_new_id=rewrite['target_new']['id']))
            raw = dict(identity=identity, identity_sha256=audit.digest(identity), occurrence=occurrence,
                case_id=record['case_id'], observations=[observation], metrics=metric(),
                raw_local_only=True, checkpoint_saved=False)
            raw['payload_sha256'] = audit.digest(raw)
            path = base / 'observations' / (raw['identity_sha256'] + '.json')
            write_json(path, raw)
            rows.append(dict(occurrence=occurrence, case_id=record['case_id'],
                identity_sha256=raw['identity_sha256'], payload_sha256=raw['payload_sha256'],
                observation_path=str(path), metrics=raw['metrics']))
        identity = dict(runtime=runtime_sha, state_sha256=audit.digest(state), endpoint=endpoint,
            cohort=cohort, ordered_occurrences=[1, 2],
            observation_identities=[row['identity_sha256'] for row in rows])
        value = dict(identity=identity, identity_sha256=audit.digest(identity), rows=rows,
            summary=audit.reduce_generation(rows), RNG_restored=True,
            observer_no_mutation=True, raw_local_only=True)
        path = base / 'endpoints' / (value['identity_sha256'] + '.json')
        write_json(path, value)
        receipt = dict(requests=2, cohort_identity=audit.digest([91, 12]),
            shared_state_identity=state, rows_path=str(path), raw_endpoint_member=audit.member(path),
            identity=identity, identity_sha256=value['identity_sha256'], summary=value['summary'],
            shared_summary=value['summary'], RNG_restored=True, observer_no_mutation=True,
            work=dict(new_case_observations=0 if cached else 2, cached_case_observations=2 if cached else 0,
                generation_forwards=0 if cached else 2, full_prefix_token_work=0 if cached else 4, seconds=0.))
        return receipt, value

    def observe(self, receipt, *, state=None, endpoint='W5', cohort='ALL_SEEN', cached=False):
        return audit.generation_endpoint(audit.Reader(), receipt, self.records, [1, 2],
            receipt['shared_state_identity'] if state is None else state, endpoint, cohort,
            self.config, self.root, cached_only=cached)

    def test_independent_reduction_zero_is_measured_missing_mean_omitted(self):
        rows = [dict(metrics=metric()), dict(metrics=metric(entropy=None,
            reasons=['missing_reference', 'tokenizer_not_available']))]
        summary = audit.reduce_generation(rows)
        self.assertEqual(summary['planned_count'], 2)
        self.assertEqual(summary['fluency_count'], 1)
        self.assertEqual(summary['ngram_entropy'], 0.)
        self.assertEqual(summary['consistency_count'], 0)
        self.assertNotIn('reference_score', summary)
        self.assertEqual(summary['missing_reason_counts']['missing_reference'], 2)
        # Native cosine is neither clamped nor a performance threshold.
        self.assertGreater(audit.reduce_generation([dict(metrics=metric(cosine=1.0000000000000002,
            reasons=[]))])['reference_score'], 1.)

    def test_reduction_rejects_missing_zero_nonfinite_and_unknown_reason(self):
        for edit in ({'fluency_valid': False}, {'ngram_entropy': math.nan},
                     {'reasons': ['invented']}, {'generated_token_count': True}):
            value = metric()
            value.update(edit)
            with self.subTest(edit=edit), self.assertRaises(RuntimeError):
                audit.reduce_generation([dict(metrics=value)])

    def test_raw_endpoint_sha_order_state_runtime_and_exact_work(self):
        receipt, _ = self.endpoint()
        result = self.observe(receipt)
        self.assertEqual([row['case_id'] for row in result['rows']], [91, 12])
        self.assertEqual(result['work']['generation_forwards'], 2)
        self.assertEqual(result['summary']['fluency_count'], 2)
        self.assertNotIn('reference_score', result['summary'])
        for edit in ({'identity_sha256': 'bad'}, {'shared_state_identity': {'W': {}}},
                     {'cohort_identity': audit.digest([12, 91])}):
            bad = copy.deepcopy(receipt)
            bad.update(edit)
            with self.subTest(edit=edit), self.assertRaises(RuntimeError):
                self.observe(bad)

    def test_same_summary_does_not_hide_tampered_raw_payload(self):
        receipt, value = self.endpoint()
        path = Path(value['rows'][0]['observation_path'])
        raw = json.loads(path.read_text())
        raw['observations'][0]['text'] = 'tampered'
        write_json(path, raw)
        with self.assertRaisesRegex(RuntimeError, 'GEN_RAW_IDENTITY_PAYLOAD'):
            self.observe(receipt)

    def test_missing_mean_summary_cannot_be_replaced_with_zero(self):
        receipt, _ = self.endpoint()
        receipt['summary'] = dict(receipt['summary'], reference_score=0.)
        with self.assertRaisesRegex(RuntimeError, 'GEN_COMPACT_SUMMARY_KEYS'):
            self.observe(receipt)

    def test_cold_fixed_identity_and_cached_subset_do_not_recharge(self):
        cold = {'full_model_cold_identity': 'fixture-cold'}
        receipt, _ = self.endpoint(cold, endpoint='B1_PRE', cohort='CURRENT', cached=True)
        result = self.observe(receipt, state=cold, endpoint='B1_PRE', cohort='CURRENT', cached=True)
        self.assertEqual(result['work']['new_case_observations'], 0)
        self.assertEqual(result['work']['generation_forwards'], 0)
        bad = copy.deepcopy(receipt)
        bad['work']['generation_forwards'] = 2
        with self.assertRaisesRegex(RuntimeError, 'GEN_CACHE_NO_RECHARGED_WORK'):
            self.observe(bad, state=cold, endpoint='B1_PRE', cohort='CURRENT', cached=True)

    def test_token_and_eos_work_relations_checked_without_generation(self):
        receipt, value = self.endpoint()
        raw = json.loads(Path(value['rows'][0]['observation_path']).read_text())
        row = raw['observations'][0]
        self.assertEqual(audit.validate_prompt(row, 'fixture prompt', 1, 0, 'fixture-model'), (1, 2, False))
        for edit in ({'full_prefix_token_work': 3}, {'stop_reason': 'length_cap'}, {'seed': 0}):
            bad = dict(row, **edit)
            with self.subTest(edit=edit), self.assertRaises(RuntimeError):
                audit.validate_prompt(bad, 'fixture prompt', 1, 0, 'fixture-model')
        long = dict(row, input_token_ids=[1] * 101, continuation_token_ids=[], full_token_ids=[1] * 101,
            input_token_count=101, continuation_token_count=0, stop_reason='length_cap_no_continuation',
            model_forwards=0, full_prefix_token_work=0)
        self.assertEqual(audit.validate_prompt(long, 'fixture prompt', 1, 0, 'fixture-model'), (0, 0, True))

    def test_terminal_completion_is_distinct_from_failed_partial_and_missing(self):
        self.assertEqual(audit.terminal_status({'status': 'FAILED'}, 19, True, []), 'FAILED')
        self.assertEqual(audit.terminal_status({'status': 'BLOCKED'}, 0, False, []), 'BLOCKED')
        self.assertEqual(audit.terminal_status({}, 10, True, []), 'PARTIAL')
        self.assertEqual(audit.terminal_status({'status': 'COMPLETED'}, 19, True, []),
                         'TECHNICAL_BLOCKED_INCOMPLETE_EVIDENCE')
        self.assertEqual(audit.terminal_status({'status': 'COMPLETED'}, 20, True, []),
                         'COMPLETED_VALIDATED_ROWS_COUNTS')
        plan = audit.planned_counts()['native_per_arm']
        self.assertEqual(sum(row['native_z'] for row in plan.values()), 14000)
        self.assertEqual(sum(row['solves'] for row in plan.values()), 640)
        self.assertEqual(sum(row['history_appends'] for row in plan.values()), 280)

    def test_all_six_measured_native_fit_cumulative_receipts(self):
        for arm in audit.ARMS:
            measured = audit.expected_counts(arm)
            prior = {key: 0 for key in audit.COUNT_KEYS}
            counts = dict(measured, fit_trace=[dict(request_index=index + 1, evaluations=2,
                Adam_updates=1, loss=.2, nll_loss=.1, kl_loss=0., weight_decay=.1,
                stop='BUDGET_EXHAUSTED') for index in range(measured['native_z'])])
            if arm not in ('BASE_MEMIT', 'BASE_ALPHAEDIT'):
                counts.update(fit_forwards=2 * measured['native_z'], fit_updates=measured['native_z'])
            native = dict(status='NATIVE_APPLY_RETURNED', arm=arm, writer=audit.writer_identity(arm),
                batch=1, requests=100, same_model_returned=True, native_has_history=arm in audit.HISTORY_ARMS,
                caller_history_appends=0, native_z_disk_cache=False, cache_template=None,
                checkpoint_saved=False, exact_resume='NOT_AVAILABLE', counts=counts,
                delta=measured, cumulative=measured)
            result = audit.native_receipt(native, arm, 1, measured, prior)
            self.assertEqual(result['measured_Adam_updates'], measured['native_z'])
            broken = copy.deepcopy(native)
            broken['cumulative']['solves'] += 1
            with self.subTest(arm=arm), self.assertRaisesRegex(RuntimeError, 'NATIVE_CUMULATIVE_COUNTS'):
                audit.native_receipt(broken, arm, 1, measured, prior)

    def allocation(self, *, stdout=None):
        lock = {'source_commit': 'c' * 40}
        path = self.root / 'execution.lock.json'
        write_json(path, lock)
        jobs = {arm: str(700 + index) for index, arm in enumerate(audit.ARMS)}
        write_json(self.root / 'submission.json', dict(task_id=audit.TASK, instruction_id=audit.NONCE,
            source_commit=lock['source_commit'], jobs=jobs, lock=audit.member(path)))
        if stdout is None:
            stdout = '\n'.join(job + '|COMPLETED|10|cpu=6,gres/gpu:a6000=1' for job in jobs.values())
            stdout += '\n700.batch|COMPLETED|999|gres/gpu=1\n'
        calls = []
        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return SimpleNamespace(returncode=0, stdout=stdout)
        result = audit.allocation_once(audit.Reader(), self.root, lock, runner=runner)
        return result, calls

    def test_allocation_one_exact_six_parent_query_no_step_doublecount(self):
        result, calls = self.allocation()
        self.assertEqual(result['status'], 'RECORDED')
        self.assertEqual(result['allocated_GPU_seconds'], 60)
        self.assertEqual(len(calls), 1)
        self.assertIn('-X', calls[0][0])
        self.assertEqual(calls[0][0][-1], '--format=JobIDRaw,State,ElapsedRaw,AllocTRES')
        self.assertLessEqual(calls[0][1]['timeout'], 45)

    def test_allocation_missing_or_foreign_rows_not_invented_zero(self):
        result, calls = self.allocation(stdout='999|COMPLETED|10|gres/gpu=1\n')
        self.assertEqual(result['status'], 'NOT_RECORDED')
        self.assertNotIn('allocated_GPU_seconds', result)
        self.assertEqual(len(calls), 1)

    def test_import_has_no_model_generation_or_tokenizer_dependency(self):
        self.assertNotIn('torch', audit.__dict__)
        self.assertNotIn('transformers', audit.__dict__)
        self.assertNotIn('SharedObserver', audit.__dict__)

    def test_prune_terminal_exact_saved_cold_base_and_twelve_svd_no_quality_gate(self):
        layers = [dict(layer=layer, native_svd_calls=2, singular_values=2,
            singular_values_compressed=0, **{key: 0. for key in (
                'max_sigma_cold', 'max_sigma_update', 'max_sigma_compressed',
                'dense_delta_norm', 'compressed_delta_norm', 'terminal_change_norm',
                'cold_weight_norm', 'final_weight_norm')}) for layer in audit.ARM_LAYERS['PRUNE']]
        terminal = dict(status='PRUNE_TERMINAL_APPLIED', arm='PRUNE', batch=20,
            state_edits=2000, native_spectral_formula_unchanged=True,
            final_weight_base='saved_cold_W0', W0_RAM_only=True,
            checkpoint_saved=False, exact_resume='NOT_AVAILABLE', layers=layers,
            compressed_singular_values=0, seconds=0., prune_applied=True,
            explicit_repair='PRUNE_TERMINAL_BASE_FIX', native_svd_calls=12)
        native = dict(terminal_prune=terminal, prune_applied=True,
            explicit_repair='PRUNE_TERMINAL_BASE_FIX')
        audit.prune_terminal(native, 20)
        audit.prune_terminal({}, 19)
        broken = copy.deepcopy(native)
        broken['terminal_prune']['final_weight_base'] = 'edited_W20_dense'
        with self.assertRaisesRegex(RuntimeError, 'PRUNE_TERMINAL_PHYSICAL_PROVENANCE'):
            audit.prune_terminal(broken, 20)
        with self.assertRaisesRegex(RuntimeError, 'PRUNE_EXACTLY_ONCE_TERMINAL'):
            audit.prune_terminal(native, 19)

    def test_report_finalization_preserves_failed_and_not_started_science(self):
        records = [dict(case_id=100001 + index) for index in range(2000)]
        stream, identity_path = self.root / 'stream.json', self.root / 'rpn-identity.json'
        write_json(stream, records)
        write_json(identity_path, [])
        config = dict(task_id=audit.TASK, instruction_id=audit.NONCE,
            noCP=True, z_disk_cache=False, stream=str(stream), assets=[audit.member(stream)],
            observer_identity=audit.member(identity_path),
            packs=[dict(ids=[r['case_id'] for r in records[start:start + 100]])
                   for start in range(0, 2000, 100)],
            generation=dict(self.config['generation'], schema=audit.SCHEMA, profile=audit.PROFILE,
                eval_seed=audit.EVAL_SEED, W0_owner='BASE_MEMIT',
                common_source_status='READY_BOUND', reference_status='READY_VERIFIED',
                W0_cache=str(self.root / 'W0-generation'), W0_state_identity={'cold': 'fixture'},
                package_tree='d' * 40, shared_source_members=[]))
        write_json(self.root / 'config.json', config)
        lock = dict(source_commit='c' * 40, instruction_id=audit.NONCE,
            config_sha256=audit.member(self.root / 'config.json')['sha256'],
            shared_generation_source=config['generation']['source_sha'],
            shared_generation_tree=config['generation']['package_tree'],
            reference_identity=config['generation']['reference_assets_sha256'])
        write_json(self.root / 'execution.lock.json', lock)
        write_json(self.root / 'BASE_MEMIT' / 'terminal.json', dict(status='FAILED',
            stage='LOAD_COLD_W0', error_type='FixtureNoModel', commits=0, completed_batches=0))
        chunks = [(index + 1, records[index * 100:(index + 1) * 100], records[:(index + 1) * 100])
                  for index in range(20)]
        with patch.object(audit, 'batches', return_value=iter(chunks)), \
                patch.object(audit.subprocess, 'run', side_effect=AssertionError('NO_ACTUAL_SLURM')):
            result = audit.collect(self.root)
        self.assertFalse(result['scientific_complete'])
        collector = self.root / 'collector'
        terminal = json.loads((collector / 'terminal.json').read_text())
        self.assertEqual(terminal['status'], 'COMPLETED')
        self.assertFalse(terminal['scientific_complete'])
        review = json.loads((collector / 'review.json').read_text())
        self.assertEqual(review['reviews'][0]['scientific_status'], 'FAILED')
        self.assertEqual(review['reviews'][1]['scientific_status'], 'NOT_STARTED_OR_STARTUP_FAILED')
        self.assertNotIn('case_id', json.dumps(review))
        self.assertEqual(json.loads((collector / 'allocation.json').read_text())['status'], 'NOT_RECORDED')
        for name in ('metrics.csv', 'generation.csv', 'counts.csv', 'compute.csv',
                     'report-ko.md', 'manifest.json', 'raw-input-manifest.json'):
            self.assertTrue((collector / name).is_file())


if __name__ == '__main__':
    unittest.main()
