"""CPU native API/receipt/privacy fixtures, never a model/GPU qualification."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from project.run_scripts.experiment_generation_eval.common import SCHEMA, EVAL_SEED, immutable_write
from project.run_scripts.experiment_generation_eval.native_profile import PROFILE, ROUTE
from project.run_scripts.experiment_generation_eval.progress import FIELDS
from . import generation_native_bridge as bridge
from .generation_common import digest, member, read, expected_counts
from .generation_native_common import TASK, NONCE, SCHEDULE
from .test_generation_run import records
from .test_generation_tracking_shared import raw_summary


def config():
    return dict(task_id=TASK, instruction_id=NONCE, generation=dict(schema=SCHEMA, profile=PROFILE,
        generation_route=ROUTE, eval_seed=EVAL_SEED, evaluation_schedule=SCHEDULE,
        source_sha='a'*40, reference_assets_sha256='b'*64, generation_assets={}, model_identity='CPU_MODEL_FIXTURE'))


class SharedFixture:
    calls = []
    def __init__(self, model, tokenizer, assets, config, out, state_callback=None, progress_callback=None):
        self.config = copy.deepcopy(config)
        self.runtime_identity = dict(CPU_fixture_only=True, profile=config['profile'],
            route=config['generation_route'], generation_source_sha=config['generation_source_sha'])
        self.runtime_sha = digest(self.runtime_identity)
        self.out = Path(out)
        self.state_callback, self.progress_callback = state_callback, progress_callback
        self.calls = []
    def observe(self, rows, endpoint, cohort, state_identity):
        self.calls.append((len(rows), endpoint, cohort, copy.deepcopy(state_identity)))
        identity = dict(runtime=self.runtime_sha, state_sha256=digest(state_identity),
            endpoint=endpoint, cohort=cohort, ordered_occurrences=[r['occurrence_index'] for r in rows],
            sampling_stream_sha256='c'*64)
        execution = dict(identity=identity, profile=PROFILE, route=ROUTE,
            qualification_performed=False, native_execution_complete=True, no_fallback=True)
        path = self.out/'CPU_EXECUTION_ONLY.json'
        immutable_write(path, execution)
        raw = self.out/'CPU_RAW_ENDPOINT_ONLY.json'
        immutable_write(raw, dict(CPU_fixture=True, actual_GPU=False))
        return dict(identity=identity, identity_sha256=digest(identity), summary=raw_summary(len(rows)),
            rows=[{'CPU_fixture':True, 'case_id':row['case_id']} for row in rows],
            rows_path=str(raw), work={'new_case_observations':len(rows)},
            native_execution_member=member(path), RNG_restored=True, observer_no_mutation=True)


def fixture(root):
    current = config()
    view = types.SimpleNamespace(model=object())
    engine = types.SimpleNamespace(next_batch=21, history=lambda:{})
    lock = dict(source_commit='d'*40, config_sha256='e'*64)
    with patch.object(bridge, 'load_assets', return_value=types.SimpleNamespace(sha='b'*64)),\
            patch.object(bridge, 'NativeGenerationObserver', SharedFixture):
        observer = bridge.GenerationObserver(current, lock, view, engine, None, records(),
            root, 'BASE_MEMIT')
    return observer, current, lock


def persist_commits(observer, physical):
    previous = {'W':'CPU_COLD', 'H':{}}
    for number in range(1, 21):
        after = physical if number == 20 else {'W':'CPU_W'+str(number), 'H':{}}
        value = dict(task=TASK, arm=observer.arm, batch=number, seen_requests=number*100,
            source=observer.lock['source_commit'], config=digest(observer.config),
            before=previous, after=after, generation_schedule=SCHEDULE,
            native_counts=expected_counts(observer.arm), checkpoint_saved=False,
            case_ids=list(range((number-1)*100, number*100)))
        immutable_write(observer.out/f'batch-{number:02d}'/'commit.json', value)
        previous = after


class NativeBridgeTests(unittest.TestCase):
    def test_constructor_binds_exact_native_api_without_qualifier_or_w0_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            observer, current, lock = fixture(Path(directory))
            self.assertEqual(observer.shared.config['profile'], PROFILE)
            self.assertEqual(observer.shared.config['generation_route'], ROUTE)
            self.assertEqual(observer.shared.config['generation_source_sha'], 'a'*40)
            self.assertNotIn('generation_microbatch', observer.shared.config)
            self.assertNotIn('qualification_receipt_member', observer.shared.config)
            self.assertNotIn('W0_cache', observer.gen)
            self.assertFalse(observer.shared.calls)
            runtime = read(observer.runtime_aux_member['path'])
            self.assertEqual(runtime['W0_generation'], 'NOT_SCHEDULED')
            self.assertFalse(runtime['qualification_performed'])
            self.assertTrue(runtime['no_fallback'])
            for call in (observer.load_W0, lambda:observer.subset(None, None),
                    lambda:observer.subset_receipt(None, None)):
                with self.assertRaisesRegex(RuntimeError, 'NOT_SCHEDULED'):
                    call()

    def test_final_requires_exact_order_actual_state_and_persisted20_then_one_execution(self):
        physical = {'W':'CPU_ACTUAL_W20', 'H':{}}
        with tempfile.TemporaryDirectory() as directory:
            observer, current, lock = fixture(Path(directory))
            with patch('project.run_scripts.gptj_cake_blue_prune_rect.metrics.state', return_value=physical):
                for endpoint, rows in (('W5', records()[:500]), ('B1_PRE', records()[:100]),
                        ('W20', list(reversed(records())))):
                    with self.assertRaisesRegex(RuntimeError, 'ONE_W20_FULL_FIRST2000'):
                        observer.endpoint(rows, out=None, endpoint=endpoint,
                            model_state=physical, cohort_label='ALL_SEEN')
                observer.engine.next_batch = 20
                with self.assertRaisesRegex(RuntimeError, 'AFTER_TWENTY_NATIVE_COMMITS'):
                    observer.endpoint(records(), out=None, endpoint='W20',
                        model_state=physical, cohort_label='ALL_SEEN')
                observer.engine.next_batch = 21
                with self.assertRaisesRegex(RuntimeError, 'ACTUAL_FINAL_STATE'):
                    observer.endpoint(records(), out=None, endpoint='W20',
                        model_state={'W':'wrong'}, cohort_label='ALL_SEEN')
                with self.assertRaisesRegex(RuntimeError, 'UNSAFE_JSON_MEMBER'):
                    observer.endpoint(records(), out=None, endpoint='W20',
                        model_state=physical, cohort_label='ALL_SEEN')
                self.assertFalse(observer.shared.calls)
                persist_commits(observer, physical)
                result = observer.endpoint(records(), out=Path(directory)/'final', endpoint='W20',
                    model_state=physical, cohort_label='ALL_SEEN')
                self.assertEqual(observer.shared.calls, [(2000, 'W20', 'ALL_SEEN', physical)])
                self.assertEqual(result['summary'], raw_summary(2000))
                self.assertEqual(len(result['native_commit_members']), 20)
                self.assertEqual(result['committed_batch20_member'],
                    member(Path(directory)/'batch-20/commit.json'))
                self.assertEqual(result['generation_profile'], PROFILE)
                self.assertEqual(result['generation_route'], ROUTE)
                self.assertEqual(result['identity']['sampling_stream_sha256'], 'c'*64)
                self.assertFalse(result['qualification_performed'])
                self.assertIn('native_execution_member', read(Path(directory)/'final/receipt.json'))
                with self.assertRaisesRegex(RuntimeError, 'ONE_W20_FULL_FIRST2000'):
                    observer.endpoint(records(), out=None, endpoint='W20',
                        model_state=physical, cohort_label='ALL_SEEN')

    def test_persisted_commit_identity_link_and_counts_fail_closed(self):
        physical = {'W':'CPU_W20', 'H':{}}
        with tempfile.TemporaryDirectory() as directory:
            observer, _, _ = fixture(Path(directory))
            persist_commits(observer, physical)
            self.assertEqual(len(observer._persisted_commits(physical)), 20)
            for field, bad in (('before', {'W':'wrong'}), ('native_counts', {}), ('task', 'wrong')):
                path = observer.out/'batch-10/commit.json'
                original = read(path)
                altered = dict(original, **{field:bad})
                with patch.object(bridge, 'read', side_effect=lambda p:altered if Path(p) == path else read(p)):
                    with self.assertRaisesRegex(RuntimeError, 'PERSISTED_COMMIT'):
                        observer._persisted_commits(physical)

    def test_occurrence_identity_never_sort_dedup_or_drop_input(self):
        with tempfile.TemporaryDirectory() as directory:
            observer, _, _ = fixture(Path(directory))
            rows = records()
            before = copy.deepcopy(rows)
            chosen = observer._selected(rows)
            self.assertEqual([r['occurrence_index'] for r in chosen], list(range(1, 2001)))
            self.assertEqual(rows, before)
            for bad in ([rows[0], rows[0]], [dict(rows[0], occurrence_index=2)],
                    [dict(rows[0], generation_prompts=['CHANGED'])]):
                with self.assertRaises(RuntimeError):
                    observer._selected(bad)

    def test_progress_only_public_scalars_phase_adaptation_and_no_input_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            observer, _, _ = fixture(Path(directory))
            observer.tracker = object()
            payload = {'generation_progress/'+field:0 for field in FIELDS}
            payload.update(phase='generation_evaluation', **{'generation_progress/total_cases':2000,
                'generation_progress/total_prompts':20000})
            before = copy.deepcopy(payload)
            with patch('project.run_scripts.gptj_native_baselines.generation_native_tracking.log_generation_progress') as log:
                observer._progress(payload)
                actual = log.call_args.args[1]
                self.assertEqual(actual, dict(payload, phase='W20_generation'))
                self.assertEqual(payload, before)
                self.assertEqual(set(actual), set(payload))
                self.assertNotIn('job_id', actual)
                self.assertEqual(json.loads((Path(directory)/'generation-progress.jsonl').read_text()), actual)
                with self.assertRaisesRegex(RuntimeError, 'PROGRESS_ENDPOINT'):
                    observer._progress(dict(payload, phase='W0_generation'))


if __name__ == '__main__':
    unittest.main()
