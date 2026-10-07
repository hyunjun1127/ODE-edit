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
            seed = int(audit.digest(dict(model_identity=runtime['model_identity'], ordered_occurrence=occurrence,
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

    def qualification(self, *, actual=True, actual_gpu=True):
        source_files = []
        for index, relative in enumerate(('transformers/models/gptj/modeling_gptj.py', 'transformers/cache_utils.py')):
            path = self.root / ('native-source-fixture-' + str(index) + '.json')
            write_json(path, {'fixture_source': index})
            source_files.append(dict(relative=relative, **audit.member(path)))
        cohort = dict(schema='gptj-generation-cache-qualification-plan-v1', raw_local_only=True,
            prompts=[dict(occurrence=index + 1, case_id=200 + index, prompt_index=0,
                          input_token_count=2, prompt='fixture prompt') for index in range(8)])
        cohort_path = self.root / 'qualification-cohort.json'
        write_json(cohort_path, cohort)
        plan = dict(schema='gptj-generation-cache-qualification-plan-v1',
            status='CPU_PLAN_FROZEN_NOT_GPU_PASS', cohort_sha256=audit.digest(cohort),
            model_identity='fixture-model', shared_source_sha='b' * 40,
            native_source_binding={'versions': {'torch': '2.9.1+cu128', 'transformers': '4.57.1'},
                                   'files': source_files}, model='gptj',
            profile=audit.PROFILE, eval_seed=audit.EVAL_SEED, max_prompts=8, cohort_count=8,
            tolerances=audit.QUALIFICATION_TOLERANCES, routes=list(audit.ROUTES), batch_microbatch=8,
            admission_reason='DEFAULT_MB8', fixture_only=False, coverage={'same_length_width': 8},
            automatic_OOM_retry=False, measure_each_route_once=True, no_fit=True, checkpoint_saved=False)
        path = self.root / 'qualification-plan.json'
        write_json(path, plan)
        actual_path = self.root / 'qualification' / 'receipt.json'
        config = dict(task_id=audit.REPAIR_TASK, instruction_id=audit.REPAIR_NONCE,
            parent_task_id=audit.TASK, generation=dict(model_identity='fixture-model',
                source_sha='b' * 40, repair=dict(qualification_plan=audit.member(path),
                    qualification_plan_sha256=audit.digest(plan), qualification_cohort=audit.member(cohort_path),
                    qualification_receipt_path=str(actual_path))))
        receipt = dict(schema='gptj-generation-cache-qualification-v1',
            status='QUALIFIED_ACTUAL_GPU_ROUTE' if actual_gpu else 'CPU_FIXTURE_NOT_ACTUAL_QUALIFICATION',
            actual_GPU=actual_gpu, plan_sha256=audit.digest(plan), cohort_sha256=plan['cohort_sha256'],
            model_identity='fixture-model', shared_source_sha='b' * 40,
            native_source_binding=plan['native_source_binding'], selected_route='EQUAL_TOKEN_LENGTH_KV_BATCH',
            fixed_microbatch=8, selected_route_passed=True, state_unchanged=True, RNG_restored=True,
            no_fit=True, checkpoint_saved=False, tolerances=audit.QUALIFICATION_TOLERANCES, error=None,
            route_results={route: dict(status='PASS', gates={key: True for key in audit.QUALIFICATION_GATES},
                max_logit_abs_error=0., max_topk_probability_abs_error=0.,
                coverage=dict({key: True for key in audit.QUALIFICATION_COVERAGE},
                              actual_max_microbatch=8, active_row_removal=True)) for route in audit.ROUTES},
            work={route: dict(physical_forward_calls=2, prefill_query_tokens=10, decode_query_tokens=1,
                              logical_row_token_decisions=2) for route in audit.ROUTES},
            cost={route: dict(elapsed_seconds=.1, synchronized_GPU_seconds=.1,
                peak_gpu_allocated_bytes=8, peak_gpu_reserved_bytes=8, peak_host_RSS_bytes=8)
                for route in audit.ROUTES})
        if actual:
            write_json(actual_path, receipt)
        return config, receipt, actual_path

    def test_repair_authority_never_relabels_original_r1(self):
        original = dict(task_id=audit.TASK, instruction_id=audit.NONCE)
        audit.task_authority(original, dict(instruction_id=audit.NONCE))
        config, _, _ = self.qualification(actual=False)
        audit.task_authority(config, dict(task_id=audit.REPAIR_TASK, instruction_id=audit.REPAIR_NONCE))
        with self.assertRaisesRegex(RuntimeError, 'COLLECT_REPAIR_TASK_AUTHORITY'):
            audit.task_authority(dict(config, instruction_id=audit.NONCE), dict(instruction_id=audit.NONCE))

    def test_plan_only_and_cpu_fixtures_never_imply_actual_pass(self):
        config, _, _ = self.qualification(actual=False)
        result = audit.qualification_evidence(audit.Reader(), config, {}, self.root)
        self.assertEqual(result['status'], 'BLOCKED_QUALIFICATION_PENDING')
        self.assertEqual(result['plan_status'], 'PLAN_BOUND_NOT_ACTUAL_PASS')
        self.assertFalse(result['actual_qualification'])
        config, _, _ = self.qualification(actual_gpu=False)
        result = audit.qualification_evidence(audit.Reader(), config, {}, self.root)
        self.assertEqual(result['status'], 'BLOCKED_ACTUAL_QUALIFICATION_NOT_GPU')
        self.assertFalse(result['actual_qualification'])

    def test_actual_qualification_exact_source_plan_route_and_fixed_mb(self):
        config, receipt, path = self.qualification()
        result = audit.qualification_evidence(audit.Reader(), config, {}, self.root)
        self.assertEqual(result['status'], 'QUALIFICATION_ACTUAL_VERIFIED')
        self.assertEqual(result['fixed_microbatch'], 8)
        for edit in ({'plan_sha256': 'bad'}, {'shared_source_sha': 'bad'},
                     {'fixed_microbatch': 3}, {'selected_route_passed': False}, {'RNG_restored': False},
                     {'route_results': {}}, {'work': {}}, {'cost': {}}):
            write_json(path, dict(receipt, **edit))
            with self.subTest(edit=edit), self.assertRaises(RuntimeError):
                audit.qualification_evidence(audit.Reader(), config, {}, self.root)

    def progress(self, completed, step):
        values = dict(completed_cases=completed, total_cases=2000, completed_prompts=completed * 10,
            total_prompts=20000, generated_tokens=completed * 2, new_cases=completed // 2,
            reused_cases=completed - completed // 2, elapsed_sec=float(step), cases_per_sec=1.,
            prompts_per_sec=10., tokens_per_sec=2., physical_forward_calls=completed,
            prefill_query_tokens=completed * 10, decode_query_tokens=completed, step=step)
        return dict(phase='W0_generation', edits=0,
                    **{'generation_progress/' + key: value for key, value in values.items()})

    def test_progress_separate_axis_denominators_partial_and_full_ready(self):
        rows = [self.progress(0, 0), self.progress(396, 1)]
        result = audit.validate_generation_progress(rows, total_cases=2000, total_prompts=20000)
        self.assertEqual(result['status'], 'PROGRESS_PARTIAL')
        self.assertFalse(result['W0_mean_ready'])
        with self.assertRaisesRegex(RuntimeError, 'GEN_PROGRESS_READY_REQUIRES_FULL2000'):
            audit.validate_generation_progress(rows, total_cases=2000, total_prompts=20000, ready=True)
        rows.append(self.progress(2000, 2))
        result = audit.validate_generation_progress(rows, total_cases=2000, total_prompts=20000, ready=True)
        self.assertTrue(result['W0_mean_ready'])
        for edit in ({'generation_progress/total_cases': 396}, {'generation_progress/step': 0},
                     {'edits': 100}, {'text': 'fixture private raw'}):
            bad = [rows[0], dict(rows[1], **edit)]
            with self.subTest(edit=edit), self.assertRaises(RuntimeError):
                audit.validate_generation_progress(bad, total_cases=2000, total_prompts=20000)

    def compatibility_fixture(self):
        qualification_config, actual, actual_path = self.qualification()
        identity_path, asset_path, source_path = (self.root / name for name in
            ('identity.json', 'asset-fixture.json', 'source-fixture.json'))
        for path in (identity_path, asset_path, source_path):
            write_json(path, {'fixture': True})
        config = dict(task_id=audit.REPAIR_TASK, instruction_id=audit.REPAIR_NONCE,
            parent_task_id=audit.TASK, model='fixture-gptj', model_revision='fixture-revision', seed=20261002,
            model_assets=[audit.member(asset_path)], runtime={'torch': 'fixture', 'transformers': 'fixture'},
            observer_identity=audit.member(identity_path),
            arm_configs={arm: {'native': {'closure': 'source-bound-fixture'}} for arm in audit.ARMS},
            generation=dict(schema=audit.SCHEMA, profile=audit.PROFILE, eval_seed=audit.EVAL_SEED,
                source_sha='b' * 40, reference_assets_sha256='b' * 64, scoring_versions={'fixture': '1'},
                repair=qualification_config['generation']['repair']))
        model_identity = audit.digest(dict(model=config['model'], revision=config['model_revision'],
            model_assets=config['model_assets'], runtime=config['runtime'], scorer=config['observer_identity'],
            seed=config['seed'], precision='FP32/eager/TF32off/autocastoff'))
        config['generation'].update(model_identity=model_identity,
            W0_state_identity={'model_identity': model_identity, 'actual_model_edits': 0})
        plan_path = Path(config['generation']['repair']['qualification_plan']['path'])
        plan = json.loads(plan_path.read_text())
        plan['model_identity'] = model_identity
        write_json(plan_path, plan)
        config['generation']['repair'].update(qualification_plan=audit.member(plan_path),
                                             qualification_plan_sha256=audit.digest(plan))
        actual.update(model_identity=model_identity, plan_sha256=audit.digest(plan))
        write_json(actual_path, actual)
        qualifier = audit.qualification_evidence(audit.Reader(), config, {}, self.root)
        old_config = copy.deepcopy(config)
        old_config.update(task_id=audit.TASK, instruction_id=audit.NONCE)
        old_config['generation'].pop('repair')
        old_config['generation']['source_sha'] = '83535c6a47c552cc4e5c6385f3a587d752820150'
        self.config = old_config
        _, endpoint = self.endpoint(config['generation']['W0_state_identity'], endpoint='W0', cohort='FIRST2000')
        observer_path = self.root / 'BASE_MEMIT' / 'generation-raw' / 'observer-identity.json'
        old_config_path, old_lock_path = self.root / 'old-config.json', self.root / 'old-lock.json'
        write_json(old_config_path, old_config)
        old_lock = dict(source_commit='503081fa9bc6efc4dbdd461324fba522b8f36e8b',
            shared_generation_source=old_config['generation']['source_sha'],
            config_sha256=audit.member(old_config_path)['sha256'],
            source_members=[audit.member(source_path)], runtime_sources=[audit.member(source_path)])
        write_json(old_lock_path, old_lock)
        records = self.records + [dict(case_id=1000 + index, generation_prompts=[],
            requested_rewrite={'target_new': {'id': 'fixture'}}) for index in range(1998)]
        ordered = [dict(ordered_occurrence=index, case_id=record['case_id'],
            generation_prompts=record['generation_prompts'], relation_id=record['requested_rewrite'].get('relation_id'),
            target_new_id=record['requested_rewrite']['target_new'].get('id'))
            for index, record in enumerate(records, 1)]
        old_runtime = json.loads(observer_path.read_text())['identity']
        old_identity = dict(schema='gptj-generation-old-w0-inventory-v1', old_config=audit.member(old_config_path),
            old_lock=audit.member(old_lock_path), old_observer=audit.member(observer_path),
            old_runtime_identity=old_runtime, old_runtime_sha256=audit.digest(old_runtime),
            semantic_inputs=audit.semantic_inputs(config),
            source_members_sha256=audit.digest(old_lock['source_members']),
            runtime_source_members_sha256=audit.digest(old_lock['runtime_sources']),
            ordered_records_sha256=audit.digest(ordered))
        entries, proofs = [], []
        for ordinal, record in enumerate(ordered, 1):
            identity = dict(runtime=audit.digest(old_runtime),
                state_identity=config['generation']['W0_state_identity'], record_identity=record)
            entry = dict(occurrence=ordinal, expected_identity_sha256=audit.digest(identity), status='NOT_REUSABLE')
            if ordinal <= 2:
                row = endpoint['rows'][ordinal - 1]
                raw = json.loads(Path(row['observation_path']).read_text())
                proof = dict(raw=audit.member(row['observation_path']),
                    original_identity_sha256=row['identity_sha256'], original_payload_sha256=row['payload_sha256'],
                    original_runtime_sha256=audit.digest(old_runtime),
                    original_source_sha=old_runtime['generation_source_sha'], original_route=audit.ROUTES[0],
                    record_identity_sha256=audit.digest(record),
                    prompt_seed_stream_sha256=audit.digest([v['seed'] for v in raw['observations']]),
                    input_token_bindings_sha256=audit.digest([v['input_token_ids'] for v in raw['observations']]),
                    prompt_count=1, tokenizer_reencoded=False)
                entry.update(status='REUSABLE_COMPLETE_CASE', row=row, provenance=proof)
                proofs.append(proof)
            entries.append(entry)
        inventory = dict(schema=old_identity['schema'], identity=old_identity,
            identity_sha256=audit.digest(old_identity), entries=entries, planned_cases=2000,
            reusable_cases=2, unknown_cases=1998, raw_local_only=True, new_source_relabel=False)
        inventory['payload_sha256'] = audit.digest(inventory)
        inventory_path = self.root / 'inventory.json'
        write_json(inventory_path, inventory)
        new_runtime = dict(old_runtime, generation_source_sha=config['generation']['source_sha'], route=audit.ROUTES[2])
        identity = dict(inventory=audit.member(inventory_path), inventory_identity_sha256=inventory['identity_sha256'],
            qualification=audit.member(actual_path), qualification_expected={key: actual[key] for key in
                ('plan_sha256', 'cohort_sha256', 'shared_source_sha', 'native_source_binding', 'selected_route', 'fixed_microbatch')},
            old_runtime_identity=old_runtime, old_runtime_sha256=audit.digest(old_runtime),
            new_runtime_identity=new_runtime, new_runtime_sha256=audit.digest(new_runtime),
            semantic_inputs=audit.semantic_inputs(config), ordered_records_sha256=audit.digest(ordered),
            provenance_bindings=proofs)
        manifest = dict(schema='gptj-generation-w0-compatibility-v1', status='ACTUAL_QUALIFIED_REUSE_BINDING',
            identity=identity, identity_sha256=audit.digest(identity), raw_local_only=True,
            new_source_relabel=False, original_runtime_checks_disabled=False, full_W0_READY=False,
            planned_cases=2000, reusable_cases=2)
        path = self.root / 'compatibility.json'
        write_json(path, manifest)
        config['generation']['repair']['compatibility_manifest'] = audit.member(path)
        return config, qualifier, records, manifest, path

    def test_partial_old_cases_keep_exact_runtime_source_route_and_do_not_make_ready(self):
        config, qualifier, records, _, _ = self.compatibility_fixture()
        result = audit.compatibility_evidence(audit.Reader(), config, qualifier, records)
        self.assertEqual(result['eligible_reused_cases'], 2)
        self.assertEqual(result['missing_or_unknown_cases'], 1998)
        self.assertFalse(result['full_W0_READY'])
        self.assertEqual(result['newly_generated_physical_work'], 0)
        self.assertNotIn('case_id', json.dumps(result))

    def test_old_row_provenance_and_false_ready_cannot_be_relabelled(self):
        config, qualifier, records, original, path = self.compatibility_fixture()
        for change in ('source', 'ready', 'qualification'):
            manifest = copy.deepcopy(original)
            if change == 'source':
                manifest['identity']['provenance_bindings'][0]['original_source_sha'] = 'new-source'
            elif change == 'qualification':
                manifest['identity']['qualification']['sha256'] = 'bad'
            else:
                manifest['full_W0_READY'] = True
            manifest['identity_sha256'] = audit.digest(manifest['identity'])
            write_json(path, manifest)
            config['generation']['repair']['compatibility_manifest'] = audit.member(path)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                audit.compatibility_evidence(audit.Reader(), config, qualifier, records)

    def test_progress_sha_member_and_explicit_consumer_ready_reuse(self):
        records = [{'generation_prompts': ['fixture'] * 10} for _ in range(2000)]
        path = self.root / 'generation-progress.jsonl'
        path.write_text('\n'.join(json.dumps(self.progress(cases, step)) for step, cases in
                                 enumerate((0, 396, 2000))) + '\n')
        producer = {'generation_progress': audit.member(path)}
        result = audit.progress_member(audit.Reader(), producer, records, self.root, ready=True)
        self.assertTrue(result['W0_mean_ready'])
        absent = audit.progress_member(audit.Reader(), {}, records, self.root, ready=True)
        self.assertEqual(absent['status'], 'BLOCKED_GENERATION_PROGRESS_PENDING')
        self.assertFalse(absent['W0_mean_ready'])
        ready_path = self.root / 'READY.json'
        write_json(ready_path, dict(status='READY', identity={'ordered_occurrences': list(range(1, 2001))}))
        consumer = {'reused_complete_READY': dict(status='REUSED_COMPLETE_READY', completed_cases=2000,
                                                READY=audit.member(ready_path))}
        result = audit.progress_member(audit.Reader(), consumer, records, self.root, consumer=True)
        self.assertEqual(result['status'], 'REUSED_COMPLETE_READY')
        self.assertTrue(result['W0_mean_ready'])
        write_json(ready_path, dict(status='READY', identity={'ordered_occurrences': list(range(1, 397))}))
        consumer['reused_complete_READY']['READY'] = audit.member(ready_path)
        with self.assertRaisesRegex(RuntimeError, 'GEN_CONSUMER_COMPLETE_READY_MEMBER'):
            audit.progress_member(audit.Reader(), consumer, records, self.root, consumer=True)

    def test_plan_only_mixed_endpoint_never_disables_actual_guard(self):
        config, _, _ = self.qualification(actual=False)
        with self.assertRaisesRegex(audit.RepairEvidencePending, 'BLOCKED_QUALIFICATION_PENDING'):
            audit.generation_endpoint(audit.Reader(), {}, self.records, [1, 2], {}, 'W0', 'FIRST2000',
                                      config, self.root)

    @staticmethod
    def shared_member(path):
        # Real SH1 requires exactly these three fields, not a stat-seal alias.
        return {key: audit.member(path)[key] for key in ('path', 'bytes', 'sha256')}

    def mixed_endpoint(self, *, full=False):
        """Synthetic bound schemas only; no live qualification or generation."""
        config, _, records, private_manifest, private_path = self.compatibility_fixture()
        repair = config['generation']['repair']
        plan_path, cohort_path = (Path(repair[key]['path']) for key in ('qualification_plan', 'qualification_cohort'))
        cohort, plan = json.loads(cohort_path.read_text()), json.loads(plan_path.read_text())
        cohort['prompts'] = [cohort['prompts'][7]] + cohort['prompts'][:7]  # SH1 edge-first order.
        shared_plan = dict(schema='gpt2-gptj-kv-fixed8-v1', profile=audit.PROFILE,
            eval_seed=audit.EVAL_SEED, model_identity=config['generation']['model_identity'],
            requests=[dict(occurrence=value['occurrence'], prompt_index=value['prompt_index'])
                      for value in cohort['prompts']], max_prompts=8, microbatch=8,
            actual_qualification=False, fit_calls=0, edit_commits=0, no_oom_retry=True,
            allowed_routes=list(audit.SHARED_ROUTES),
            tolerances={key: audit.QUALIFICATION_TOLERANCES[key]
                        for key in ('logits', 'topk_probabilities', 'metrics')})
        cohort['shared_plan'] = shared_plan
        write_json(cohort_path, cohort)
        shared_plan_path = self.root / 'shared-plan-local.json'
        write_json(shared_plan_path, shared_plan)
        route_map = {key: audit.SHARED_ROUTE[key] for key in audit.ROUTES}
        plan.update(cohort_sha256=audit.digest(cohort), shared_plan_sha256=audit.digest(shared_plan),
                    shared_route_map=route_map)
        write_json(plan_path, plan)
        repair.update(qualification_plan=audit.member(plan_path), qualification_plan_sha256=audit.digest(plan),
            qualification_cohort=audit.member(cohort_path), shared_qualification_plan=audit.member(shared_plan_path),
            shared_qualification_plan_sha256=audit.digest(shared_plan))
        actual_path = Path(repair['qualification_receipt_path'])
        proof = json.loads(actual_path.read_text())
        proof.update(plan_sha256=audit.digest(plan), cohort_sha256=audit.digest(cohort),
                     shared_plan_sha256=audit.digest(shared_plan), shared_route_map=route_map)
        write_json(actual_path, proof)
        results, costs = {}, {}
        for route in audit.ROUTES:
            result, work, cost = proof['route_results'][route], proof['work'][route], proof['cost'][route]
            results[audit.SHARED_ROUTE[route]] = dict(passed=True, executed=True,
                token_sequence_exact=True, topk_mask_position_exact=True, logits_close=True,
                topk_probabilities_close=True, metric_values_close_and_validity_reasons_exact=True,
                max_abs_logit_error=0., max_abs_topk_probability_error=0., coverage=result['coverage'])
            costs[audit.SHARED_ROUTE[route]] = dict(elapsed_sec=cost['elapsed_seconds'],
                GPU_seconds=cost['synchronized_GPU_seconds'], peak_allocated_bytes=cost['peak_gpu_allocated_bytes'],
                peak_reserved_bytes=cost['peak_gpu_reserved_bytes'], host_max_RSS_bytes=cost['peak_host_RSS_bytes'],
                logical_row_forward_decisions=work['logical_row_token_decisions'],
                **{key: work[key] for key in ('physical_forward_calls', 'prefill_query_tokens', 'decode_query_tokens')})
        shared_proof = dict(schema='gpt2-gptj-kv-fixed8-v1', actual_qualification=True,
            qualification_pass=True, pretrained_GPU_PASS=True, profile=audit.PROFILE,
            eval_seed=audit.EVAL_SEED, model_identity=plan['model_identity'], source_identity=plan['shared_source_sha'],
            selected_route=audit.SHARED_ROUTES[2], microbatch=8, fixed_microbatch=8,
            model_no_mutation=True, RNG_restored=True, native_state_no_mutation=True,
            fit_calls=0, edit_commits=0, retry_count=0, caller_proof_member=self.shared_member(actual_path),
            execution_adapter=audit.EXECUTION_ADAPTER, private_plan_sha256=audit.digest(plan),
            plan_sha256=audit.digest(shared_plan), tolerances=shared_plan['tolerances'], route_results=results,
            cost=costs, raw_local_only=True, native_models_scope=['gptj'], coverage=plan['coverage'],
            selection_reason='FIXED_GATE_ORDER_NOT_SCIENTIFIC_QUALITY')
        shared_proof['identity_sha256'] = audit.digest(shared_proof)
        shared_actual_path = self.root / 'shared-qualification-actual.json'
        write_json(shared_actual_path, shared_proof)
        shared_actual_member = self.shared_member(shared_actual_path)
        runtime = audit.shared_compatibility_api().runtime_identity(dict(model_identity=plan['model_identity'],
            generation_source_sha=config['generation']['source_sha'], generation_route=audit.SHARED_ROUTES[2],
            generation_microbatch=8, qualification_receipt_member=shared_actual_member),
            config['generation']['reference_assets_sha256'])
        inventory_path = Path(private_manifest['identity']['inventory']['path'])
        inventory = json.loads(inventory_path.read_text())
        old = inventory['identity']
        # Only the first original case is reusable; the second is newly observed.
        inventory['entries'][1] = dict(occurrence=2, expected_identity_sha256=
            inventory['entries'][1]['expected_identity_sha256'], status='NOT_REUSABLE')
        inventory.update(reusable_cases=1, unknown_cases=1999)
        reference_path = self.root / 'old-reference.json'
        write_json(reference_path, {'identity_sha256': config['generation']['reference_assets_sha256']})
        old_config_path, old_lock_path = Path(old['old_config']['path']), Path(old['old_lock']['path'])
        old_config, old_lock = json.loads(old_config_path.read_text()), json.loads(old_lock_path.read_text())
        old_config['generation']['generation_assets'] = self.shared_member(reference_path)
        write_json(old_config_path, old_config)
        old_lock['config_sha256'] = audit.member(old_config_path)['sha256']
        write_json(old_lock_path, old_lock)
        old.update(old_config=self.shared_member(old_config_path), old_lock=self.shared_member(old_lock_path),
                   old_observer=self.shared_member(old['old_observer']['path']))
        original_entry = inventory['entries'][0]
        original_entry['provenance']['raw'] = self.shared_member(original_entry['provenance']['raw']['path'])
        inventory['identity_sha256'] = audit.digest(old)
        inventory['payload_sha256'] = audit.digest({key: value for key, value in inventory.items()
                                                  if key != 'payload_sha256'})
        write_json(inventory_path, inventory)
        inventory_member = self.shared_member(inventory_path)
        repair['old_complete_case_inventory'] = inventory_member
        private_identity = private_manifest['identity']
        private_identity.update(inventory=inventory_member, inventory_identity_sha256=inventory['identity_sha256'],
            qualification=self.shared_member(actual_path), new_runtime_identity=runtime,
            new_runtime_sha256=audit.digest(runtime), provenance_bindings=[original_entry['provenance']],
            qualification_expected={key: proof[key] for key in private_identity['qualification_expected']})
        private_identity['qualification_route_map'] = route_map
        private_manifest.update(reusable_cases=1, identity_sha256=audit.digest(private_identity))
        write_json(private_path, private_manifest)
        repair['compatibility_manifest'] = self.shared_member(private_path)
        state = config['generation']['W0_state_identity']
        guard_path = self.root / 'old-cold-guard.json'
        guard = dict(source_commit='503081fa9bc6efc4dbdd461324fba522b8f36e8b', phase='W0_generation',
            commits=0, history_appends=0, old_generation_runtime=old['old_runtime_sha256'],
            original_state_identity=state, model_W=state.get('W'), inventory_member=inventory_member,
            whole_endpoint_guard_recorded=False, proof_basis='FROZEN_SOURCE_CONTROL_FLOW_COLD_RUNTIME_AND_RPN',
            partial_rows_authorized=True, measured_final_history_zero=False, final_RAM_history='NOT_RECORDED',
            evidence_members={'original_runtime': self.shared_member(old_lock_path)})
        write_json(guard_path, guard)
        original = original_entry['provenance']
        shared_compatible = dict(schema='generation-cold-W0-compatibility-v1', physical_state=state,
            new_runtime_sha256=audit.digest(runtime), model_identity=plan['model_identity'],
            reference_assets_sha256=config['generation']['reference_assets_sha256'],
            qualification_receipt_member=shared_actual_member,
            actual_qualification_receipt_member=self.shared_member(actual_path),
            qualification_plan_member=self.shared_member(plan_path), inventory_member=inventory_member,
            old_observer_member=old['old_observer'], old_config_member=old['old_config'],
            old_reference_member=self.shared_member(reference_path), old_cold_guard_member=self.shared_member(guard_path),
            ordered_record_identity_sha256=old['ordered_records_sha256'], eligible_cases=1, planned_cases=2000,
            original_entries=[dict(occurrence=1, original_raw_member=original['raw'],
                original_identity_sha256=original['original_identity_sha256'],
                original_payload_sha256=original['original_payload_sha256'],
                original_runtime_sha256=original['original_runtime_sha256'],
                original_generation_source_sha=original['original_source_sha'], original_route=original['original_route'])],
            raw_local_only=True, edited_trajectory_resume=False)
        manifest = dict(identity=shared_compatible, identity_sha256=audit.digest(shared_compatible))
        shared_path = self.root / 'shared-compatibility.json'
        write_json(shared_path, manifest)
        base = self.root / 'BASE_MEMIT' / 'qualified-raw'
        write_json(base / 'observer-identity.json', dict(identity=runtime, identity_sha256=audit.digest(runtime),
            raw_local_only=True, checkpoint_saved=False))
        selected = records if full else records[:2]
        rows = [dict(original_entry['row'], provenance=dict(origin='COMPATIBLE_ORIGINAL_W0',
            raw_member=original['raw'], runtime_sha256=old['old_runtime_sha256'],
            generation_source_sha=old['old_runtime_identity']['generation_source_sha'], route=audit.SHARED_ROUTES[0],
            compatibility_sha256=manifest['identity_sha256']))]
        template = json.loads(Path(original['raw']['path']).read_text())
        for ordinal, record in enumerate(selected[1:], 2):
            rewrite = record['requested_rewrite']
            identity = dict(runtime=audit.digest(runtime), state_identity=state, record_identity=dict(
                ordered_occurrence=ordinal, case_id=record['case_id'], generation_prompts=record['generation_prompts'],
                relation_id=rewrite.get('relation_id'), target_new_id=rewrite['target_new'].get('id')))
            observations = []
            if record['generation_prompts']:
                observation = dict(template['observations'][0], occurrence=ordinal,
                    prompt=record['generation_prompts'][0], route=audit.SHARED_ROUTES[2], full_prefix_token_work=0,
                    physical_forward_calls=1, prefill_query_tokens=2, decode_query_tokens=0)
                observation['seed'] = int(audit.digest(dict(model_identity=plan['model_identity'],
                    ordered_occurrence=ordinal, prompt_index=0, eval_seed=audit.EVAL_SEED))[:16], 16) % (2**63 - 1)
                observations = [observation]
            metrics = metric() if observations else dict(ngram_entropy=None, reference_score=None,
                fluency_valid=False, consistency_valid=False, reasons=['missing_generation_prompts'],
                generation_prompt_count=0, generated_token_count=0, length_cap_no_continuation_count=0)
            raw = dict(identity=identity, identity_sha256=audit.digest(identity), occurrence=ordinal,
                case_id=record['case_id'], observations=observations, metrics=metrics,
                raw_local_only=True, checkpoint_saved=False)
            raw['payload_sha256'] = audit.digest(raw)
            path = base / 'observations' / (raw['identity_sha256'] + '.json')
            write_json(path, raw)
            rows.append(dict(occurrence=ordinal, case_id=record['case_id'], identity_sha256=raw['identity_sha256'],
                payload_sha256=raw['payload_sha256'], observation_path=str(path), metrics=metrics,
                provenance=dict(origin='NEW_CURRENT_RUNTIME', raw_member=self.shared_member(path),
                    runtime_sha256=audit.digest(runtime), generation_source_sha=config['generation']['source_sha'],
                    route=audit.SHARED_ROUTES[2])))
        identity = dict(runtime=audit.digest(runtime), state_sha256=audit.digest(state), endpoint='W0',
            cohort='FIRST2000', ordered_occurrences=list(range(1, len(rows) + 1)),
            observation_identities=[row['identity_sha256'] for row in rows],
            provenance_sha256=audit.digest([row['provenance'] for row in rows]),
            qualification_receipt_sha256=shared_actual_member['sha256'], compatibility_sha256=manifest['identity_sha256'])
        value = dict(identity=identity, identity_sha256=audit.digest(identity), rows=rows,
            summary=audit.reduce_generation(rows), RNG_restored=True, observer_no_mutation=True, raw_local_only=True,
            qualification_receipt_member=shared_actual_member, compatibility_member=self.shared_member(shared_path))
        path = base / 'endpoints' / (value['identity_sha256'] + '.json')
        write_json(path, value)
        receipt = dict(requests=len(rows), cohort_identity=audit.digest([record['case_id'] for record in selected]),
            rows_path=str(path), raw_endpoint_member=self.shared_member(path), shared_state_identity=state,
            shared_runtime_identity=runtime, identity=identity, identity_sha256=value['identity_sha256'],
            summary=value['summary'], shared_summary=value['summary'], RNG_restored=True, observer_no_mutation=True,
            qualification=self.shared_member(actual_path), qualification_plan_sha256=audit.digest(plan),
            shared_qualification_plan_sha256=audit.digest(shared_plan), selected_route=audit.ROUTES[2], fixed_microbatch=8,
            work=dict(new_case_observations=len(rows) - 1, cached_case_observations=1, generation_forwards=1,
                full_prefix_token_work=0, physical_forward_calls=1, prefill_query_tokens=2, decode_query_tokens=0,
                completed_prompts=2, generated_tokens=2, seconds=.1))
        qualifier = audit.qualification_evidence(audit.Reader(), config, {}, self.root)
        return config, qualifier, selected, receipt, value

    def test_actual_shared_mixed_rows_preserve_original_runtime_and_physical_work(self):
        config, _, records, receipt, value = self.mixed_endpoint()
        result = audit.generation_endpoint(audit.Reader(), receipt, records, [1, 2],
            receipt['shared_state_identity'], 'W0', 'FIRST2000', config, self.root)
        self.assertEqual(result['summary']['generated_token_count'], 2)
        self.assertEqual(result['work']['generation_forwards'], 1)
        self.assertEqual(result['work']['physical_forward_calls'], 1)
        self.assertNotEqual(value['rows'][0]['provenance']['runtime_sha256'], receipt['identity']['runtime'])
        cached = copy.deepcopy(receipt)
        cached['work'] = dict(new_case_observations=0, cached_case_observations=2,
            generation_forwards=0, full_prefix_token_work=0, seconds=0.)
        result = audit.generation_endpoint(audit.Reader(), cached, records, [1, 2],
            receipt['shared_state_identity'], 'W0', 'FIRST2000', config, self.root, cached_only=True)
        self.assertEqual(result['work']['generation_forwards'], 0)
        for changed in ({'physical_forward_calls': 2}, {'generated_tokens': 1}, {'generation_forwards': 2}):
            bad = copy.deepcopy(receipt)
            bad['work'].update(changed)
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                audit.generation_endpoint(audit.Reader(), bad, records, [1, 2],
                    receipt['shared_state_identity'], 'W0', 'FIRST2000', config, self.root)

    def test_shared_derived_qualification_cannot_claim_unbound_pass_or_changed_cost(self):
        config, qualifier, _, receipt, value = self.mixed_endpoint()
        path = Path(value['qualification_receipt_member']['path'])
        original = json.loads(path.read_text())
        for key in ('pretrained_GPU_PASS', 'private_plan_sha256', 'cost'):
            body = copy.deepcopy(original)
            if key == 'pretrained_GPU_PASS':
                body[key] = False
            elif key == 'cost':
                body[key][audit.SHARED_ROUTES[2]]['physical_forward_calls'] += 1
            else:
                body[key] = 'unbound'
            body['identity_sha256'] = audit.digest({k: v for k, v in body.items() if k != 'identity_sha256'})
            write_json(path, body)
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                audit.shared_qualification_evidence(audit.Reader(), config, qualifier, self.shared_member(path))

    def test_mixed_old_original_bytes_and_state_remain_strict(self):
        config, _, records, receipt, value = self.mixed_endpoint()
        original = value['rows'][0]['provenance']
        raw_path = Path(original['raw_member']['path'])
        raw = json.loads(raw_path.read_text())
        raw['observations'][0]['route'] = audit.SHARED_ROUTES[2]
        raw['payload_sha256'] = audit.digest({key: val for key, val in raw.items() if key != 'payload_sha256'})
        write_json(raw_path, raw)
        with self.assertRaises((RuntimeError, ValueError)):
            audit.generation_endpoint(audit.Reader(), receipt, records, [1, 2],
                receipt['shared_state_identity'], 'W0', 'FIRST2000', config, self.root)

    def test_repair_ready_requires_all2000_fullrows_and_exact_producer_progress(self):
        config, qualifier, records, receipt, value = self.mixed_endpoint(full=True)
        compatibility = audit.compatibility_evidence(audit.Reader(), config, qualifier, records)
        progress = self.progress(2000, 1)
        for key, count in dict(completed_prompts=2, total_prompts=2, generated_tokens=2,
            new_cases=1999, reused_cases=1, physical_forward_calls=1,
            prefill_query_tokens=2, decode_query_tokens=0).items():
            progress['generation_progress/' + key] = count
        progress_path = self.root / 'generation-progress-W0.jsonl'
        progress_path.write_text(json.dumps(progress) + '\n')
        identity = dict(runtime=receipt['identity']['runtime'],
            cold_state_sha256=audit.digest(receipt['shared_state_identity']), ordered_occurrences=list(range(1, 2001)),
            model_identity=config['generation']['model_identity'], source_sha=config['generation']['source_sha'],
            reference_assets_sha256=config['generation']['reference_assets_sha256'],
            profile=audit.PROFILE, eval_seed=audit.EVAL_SEED,
            cohort_sha256=audit.digest([dict(record, occurrence_index=ordinal) for ordinal, record in enumerate(records, 1)]),
            qualification_sha256=qualifier['actual_member_sha256'],
            shared_qualification_sha256=value['qualification_receipt_member']['sha256'],
            shared_plan_sha256=receipt['shared_qualification_plan_sha256'])
        ready = dict(receipt, status='READY', producer_arm='BASE_MEMIT', identity=identity,
            identity_sha256=audit.digest(identity), endpoint=receipt['raw_endpoint_member'],
            producer_work=receipt['work'], compatibility_manifest=config['generation']['repair']['compatibility_manifest'],
            compatibility_member=value['compatibility_member'],
            shared_qualification_receipt_member=value['qualification_receipt_member'],
            generation_progress=self.shared_member(progress_path), checkpoint_saved=False, raw_local_only=True)
        observed = audit.repair_ready_evidence(audit.Reader(), ready, config, qualifier, compatibility, records, self.root)
        self.assertEqual(observed['status'], 'READY_VERIFIED')
        self.assertTrue(observed['W0_mean_ready'])
        bad = copy.deepcopy(ready)
        bad['producer_work']['physical_forward_calls'] = 2
        with self.assertRaises(RuntimeError):
            audit.repair_ready_evidence(audit.Reader(), bad, config, qualifier, compatibility, records, self.root)
        progress['generation_progress/completed_cases'] = 396
        progress['generation_progress/new_cases'] = 395
        progress_path.write_text(json.dumps(progress) + '\n')
        ready['generation_progress'] = self.shared_member(progress_path)
        with self.assertRaisesRegex(RuntimeError, 'GEN_PROGRESS_READY_REQUIRES_FULL2000'):
            audit.repair_ready_evidence(audit.Reader(), ready, config, qualifier, compatibility, records, self.root)

    def test_actual_runtime_aux_does_not_overwrite_cold_startup(self):
        out = self.root / 'BASE_MEMIT'
        write_json(out / 'runtime.json', {'initial_state': 'cold-startup'})
        write_json(out / 'generation-repair-runtime.json', {'qualification': 'dedicated-proof'})
        self.assertEqual(audit.repair_runtime_holder(audit.Reader(), out), {'qualification': 'dedicated-proof'})
        self.assertEqual(json.loads((out / 'runtime.json').read_text()), {'initial_state': 'cold-startup'})


if __name__ == '__main__':
    unittest.main()
