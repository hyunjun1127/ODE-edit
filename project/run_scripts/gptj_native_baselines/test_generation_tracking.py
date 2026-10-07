"""Independent fake-SDK/scalar fixtures only; no network, model, fit or GPU PASS."""
import copy
import io
import json
import os
from pathlib import Path
import queue
import threading
import tempfile
import types
import unittest
from unittest.mock import patch

from project.run_scripts.experiment_tracking import schema as base_schema
from project.run_scripts.experiment_tracking.method import AxisState
from . import generation_tracking as producer
from . import generation_tracking_schema as schema
from . import generation_tracking_client as client_module
from .generation_tracking_client import Tracker
from .generation_tracking_worker import session, settings


CFG = dict(server='server2', task_id='fixture-generation', arm='CAKE', attempt='r1',
    source_sha='a' * 40, config_sha='b' * 64, model='gptj', model_family='gptj',
    writer='cake', role='scientific', metric_schema='price-first2k-scalar-v1',
    baseline='CAKE', generation_metric_schema='counterfact-cake-generation-metrics-v1',
    generation_profile='cf-cake-prompt-inclusive-total100-eos-corrected-v1',
    generation_eval_seed=20261007, reference_assets_sha256='c' * 64,
    generation_source_sha='d' * 64)


def summary(planned=100, *, fluency=100, consistency=100):
    result = {'generation/planned_count': planned, 'generation/fluency_count': fluency,
        'generation/consistency_count': consistency, 'generation/generation_prompt_count': planned,
        'generation/generated_token_count': 0,
        **{'generation/missing_' + reason + '_count': 0 for reason in schema.REASONS}}
    if fluency:
        result['fluency/ngram_entropy'] = 0.
    if consistency:
        result['consistency/reference_score'] = 0.
    return result


def current(edits=100, **kwargs):
    return producer.generation_values('current/post', summary(**kwargs), edits,
                                      edits - 100, edits)


def tracker(capacity=128):
    result = Tracker.__new__(Tracker)
    result.closed = False
    result.done = threading.Event()
    result.queue = queue.Queue(capacity)
    result.log_lock = threading.Lock()
    result.axis = AxisState()
    result.scientific = True
    result.dropped = 0
    result.status = 'READY_ONLINE'
    return result


class FakeSDK:
    """In-memory SDK/readback sentinel. Never imports or invokes the real SDK."""
    __version__ = schema.SDK_VERSION

    def __init__(self, *, login=True, log_error=False, finish_error=False):
        self.auth, self.log_error, self.finish_error = login, log_error, finish_error
        self.calls, self.points, self.axes, self.files_list = [], [], [], []
        self.Settings = lambda **kwargs: kwargs
        self.next_step = 0

    def setup(self, **kwargs):
        self.options = kwargs['settings']

    def login(self, **kwargs):
        assert kwargs['prompt'] is False and kwargs['verify'] is True
        return self.auth

    def init(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        self.id, self.name, self.config = kwargs['id'], kwargs['name'], kwargs['config']
        self.offline = False
        self.url = 'https://wandb.ai/wkdguswns2256/layer%20allocation/runs/' + self.id
        return self

    def define_metric(self, *args, **kwargs):
        self.axes.append((args, kwargs))

    def Api(self, **kwargs):
        return self

    def run(self, path):
        return self

    def log(self, values, step=None):
        if self.log_error:
            raise RuntimeError('PRIVATE_SDK_SENTINEL')
        actual = self.next_step if step is None else step
        self.points.append(dict(copy.deepcopy(values), _step=actual))
        self.next_step = actual + 1

    def finish(self, **kwargs):
        if self.finish_error:
            raise RuntimeError('PRIVATE_SDK_SENTINEL')
        self.finished = kwargs

    def scan_history(self, keys, min_step=0, max_step=None, **kwargs):
        return iter({key: row[key] for key in keys if key in row} for row in self.points
                    if row['_step'] >= min_step and (max_step is None or row['_step'] < max_step))

    def files(self):
        return [types.SimpleNamespace(name=name) for name in self.files_list]


def exercise(sdk=None, commands=None, *, environ=None, smoke=False):
    sdk = sdk or FakeSDK()
    cfg = schema.bind_job_identity(CFG, {} if environ is None else environ)
    request = dict(config=cfg, run_id='distinctUUIDFixture', spool='/fixture/no-write',
                   smoke=smoke, base_url='https://api.wandb.ai')
    emitted = []
    session(sdk, request, commands or [dict(op='finish', exit_code=0)], emitted.append)
    return sdk, emitted


class GenerationSchemaTests(unittest.TestCase):
    def test_all_declared_prefixes_accept_valid_scalar_mapping(self):
        for prefix in schema.PREFIXES:
            with self.subTest(prefix=prefix):
                planned = 2000 if prefix == 'W0_first2000' else 500 if prefix == 'all_seen/post' else 100
                edits = 0 if prefix == 'W0_first2000' else 500 if prefix == 'all_seen/post' else 100
                result = producer.generation_values(prefix, summary(planned, fluency=planned,
                    consistency=planned), edits, max(0, edits - 100), edits)
                self.assertEqual(result[prefix + '/generation/planned_count'], planned)
                self.assertEqual(result[prefix + '/fluency/ngram_entropy'], 0.)
                self.assertEqual(result[prefix + '/consistency/reference_score'], 0.)

    def test_missing_means_are_omitted_not_zero_filled(self):
        values = current(fluency=0, consistency=0)
        self.assertNotIn('current/post/fluency/ngram_entropy', values)
        self.assertNotIn('current/post/consistency/reference_score', values)
        for field in ('fluency/ngram_entropy', 'consistency/reference_score'):
            for value in (0, None):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    schema.metrics(dict(values, **{'current/post/' + field: value}), scientific=True)
        partial = current(fluency=100, consistency=0)
        self.assertEqual(partial['current/post/fluency/ngram_entropy'], 0.)
        self.assertNotIn('current/post/consistency/reference_score', partial)

    def test_positive_valid_count_requires_actual_mean(self):
        for field in ('fluency/ngram_entropy', 'consistency/reference_score'):
            values = current()
            del values['current/post/' + field]
            with self.assertRaisesRegex(ValueError, 'MISSING_NOT_ZERO'):
                schema.metrics(values, scientific=True)

    def test_numeric_generation_means_reject_bool(self):
        for field in ('fluency/ngram_entropy', 'consistency/reference_score'):
            for value in (False, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    schema.metrics(dict(current(), **{'current/post/' + field: value}), scientific=True)

    def test_strict_builtin_finite_scalars_without_conversion(self):
        class NonScalar:
            def __float__(self):
                raise AssertionError('Must not convert or synchronize non-scalar values')
        for value in (NonScalar(), [1], {'x': 1}, 'private text', float('nan'), float('inf')):
            with self.subTest(value=type(value).__name__), self.assertRaises(ValueError):
                schema.metrics(dict(current(), **{'current/post/fluency/ngram_entropy': value}), scientific=True)

    def test_count_types_bounds_and_required_counts(self):
        for field in ('planned_count', 'fluency_count', 'consistency_count',
                      'generation_prompt_count', 'generated_token_count'):
            key = 'current/post/generation/' + field
            for value in (-1, 1.5, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    schema.metrics(dict(current(), **{key: value}), scientific=True)
            values = current()
            del values[key]
            with self.assertRaisesRegex(ValueError, 'COUNTS_REQUIRED'):
                schema.metrics(values, scientific=True)
        for field in ('fluency_count', 'consistency_count'):
            with self.assertRaisesRegex(ValueError, 'VALID_COUNT_RANGE'):
                schema.metrics(dict(current(), **{'current/post/generation/' + field: 101}), scientific=True)

    def test_mean_ranges_and_typed_reason_counts(self):
        for field, value in (('fluency/ngram_entropy', -1.),
                             ('consistency/reference_score', -0.1),
                             ('consistency/reference_score', 1.1)):
            with self.assertRaises(ValueError):
                schema.metrics(dict(current(), **{'current/post/' + field: value}), scientific=True)
        for reason in schema.REASONS:
            key = 'current/post/generation/missing_' + reason + '_count'
            with self.assertRaises(ValueError):
                schema.metrics(dict(current(), **{key: True}), scientific=True)
        with self.assertRaisesRegex(ValueError, 'ALLOWLIST'):
            schema.metrics(dict(current(), **{'current/post/generation/missing_arbitrary_count': 1}), scientific=True)

    def test_current_100_is_distinct_from_milestone_all_seen(self):
        for edits in (500, 1000, 1500, 2000):
            current_values = current(edits)
            all_values = producer.generation_values('all_seen/post', summary(edits,
                fluency=edits, consistency=edits), edits, edits - 100, edits)
            self.assertEqual(current_values['current/post/generation/planned_count'], 100)
            self.assertEqual(all_values['all_seen/post/generation/planned_count'], edits)
        for edits in (100, 600, 2100):
            with self.assertRaisesRegex(ValueError, 'SEEN_STATE_COUNTS'):
                producer.generation_values('all_seen/post', summary(edits, fluency=edits,
                    consistency=edits), edits, edits - 100, edits)
        with self.assertRaisesRegex(ValueError, 'CURRENT_STATE_COUNTS'):
            producer.generation_values('current/post', summary(500, fluency=500,
                consistency=500), 500, 400, 500)

    def test_current_axes_and_w0_first2000_are_exact(self):
        for edits, pre, post in ((100, 1, 100), (100, 0, 99), (100, None, 100)):
            with self.assertRaises(ValueError):
                producer.generation_values('current/pre', summary(), edits, pre, post)
        for edits, planned in ((100, 2000), (0, 100)):
            with self.assertRaisesRegex(ValueError, 'W0_FIRST2K'):
                producer.generation_values('W0_first2000', summary(planned,
                    fluency=planned, consistency=planned), edits)
        values = current()
        with self.assertRaisesRegex(ValueError, 'GEN_AXIS'):
            schema.metrics(values, scientific=False)

    def test_privacy_allowlist_and_shared_rpn_schema_are_unchanged(self):
        raw = dict(summary(), prompt='PRIVATE_PROMPT', raw_tokens=[1, 2], case_id=42,
                   reference_snippet='PRIVATE_REFERENCE', api_key='PRIVATE_KEY')
        values = producer.generation_values('current/post', raw, 100, 0, 100)
        self.assertNotIn('PRIVATE_', json.dumps(values))
        for key in ('prompt', 'raw_tokens', 'case_id', 'activation', 'api_key'):
            with self.assertRaisesRegex(ValueError, 'ALLOWLIST'):
                schema.metrics(dict(values, **{key: 1}), scientific=True)
        self.assertFalse(schema.GEN_KEYS & base_schema.METRICS)
        plain = {'edits': 100, 'pre_state_edits': 0, 'post_state_edits': 100,
            **{f'current/post/{kind}/count': count for kind, count in (('R', 100), ('P', 200), ('N', 1000))},
            **{f'current/post/{kind}/success_count': 0 for kind in 'RPN'},
            **{f'current/post/{kind}/success_pct': 0. for kind in 'RPN'},
            'current/post/success_harmonic_pct': 0.}
        self.assertEqual(schema.metrics(plain, scientific=True), base_schema.metrics(plain, scientific=True))
        self.assertEqual(schema.metrics(dict(plain, **values), scientific=True)['current/post/success_harmonic_pct'], 0.)

    def test_generation_config_requires_exact_public_bindings(self):
        self.assertEqual(schema.config(CFG), CFG)
        for key in schema.EXTRA_CONFIG:
            values = dict(CFG)
            del values[key]
            with self.assertRaisesRegex(ValueError, 'GENERATION_CONFIG_REQUIRED'):
                schema.config(values)
        for key, value in (('generation_eval_seed', True), ('generation_eval_seed', 20261002),
            ('generation_profile', 'legacy'), ('reference_assets_sha256', 'not-a-sha'),
            ('model', 'llama3'), ('model_family', 'GPTJ'), ('api_key', 'PRIVATE_KEY')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                schema.config(dict(CFG, **{key: value}))

    def test_generation_means_keep_raw_bits_cosine_units(self):
        observation = dict(summary(), **{'fluency/ngram_entropy': 3.5,
                                        'consistency/reference_score': 0.25})
        values = producer.generation_values('current/post', observation, 100, 0, 100)
        self.assertEqual(values['current/post/fluency/ngram_entropy'], 3.5)
        self.assertEqual(values['current/post/consistency/reference_score'], 0.25)
        self.assertFalse(any(key.endswith('_pct') for key in values))

    def test_original_rpn_denominators_and_three_way_harmonic_remain_separate(self):
        for prefix, edits, populations in (('current/post', 100, (100, 200, 1000)),
                                          ('W0_first2000', 0, (2000, 4000, 20000))):
            values = dict(edits=edits, pre_state_edits=max(0, edits - 100), post_state_edits=edits)
            for kind, count, percent in zip('RPN', populations, (20, 40, 80)):
                values.update({prefix + '/' + kind + '/count': count,
                    prefix + '/' + kind + '/success_count': count * percent // 100,
                    prefix + '/' + kind + '/success_pct': float(percent)})
            values[prefix + '/success_harmonic_pct'] = 3 / sum(1 / value for value in (20, 40, 80))
            original = base_schema.metrics(values, scientific=True)
            planned = 2000 if prefix == 'W0_first2000' else 100
            generated = producer.generation_values(prefix, summary(planned,
                fluency=planned, consistency=planned), edits, max(0, edits - 100), edits)
            combined = schema.metrics(dict(values, **generated), scientific=True)
            self.assertEqual({key: combined[key] for key in original}, original)
            self.assertGreater(combined[prefix + '/success_harmonic_pct'], 0)
            self.assertEqual(combined[prefix + '/fluency/ngram_entropy'], 0.)
            self.assertEqual(combined[prefix + '/consistency/reference_score'], 0.)


class GenerationIdentityTests(unittest.TestCase):
    def test_local_has_not_applicable_identity_and_no_fabricated_job(self):
        cfg = schema.bind_job_identity(CFG, {})
        self.assertEqual(schema.job_identity(cfg),
                         {'execution_backend': 'local', 'identity_source': 'NOT_APPLICABLE'})
        self.assertTrue(schema.run_name(cfg).endswith('-local'))
        self.assertNotIn('job_id', cfg)
        with self.assertRaisesRegex(ValueError, 'CALLER_ENV_MISMATCH'):
            schema.bind_job_identity(dict(CFG, job_id='70001'), {})

    def test_array_zero_retains_actual_child_job_and_display_identity(self):
        env = dict(SLURM_JOB_ID='70002', SLURM_ARRAY_JOB_ID='70001',
                   SLURM_ARRAY_TASK_ID='0', SLURM_STEP_ID='batch', PRIVATE='ignored')
        cfg = schema.bind_job_identity(CFG, env)
        self.assertEqual(cfg['job_id'], '70002')
        self.assertEqual(cfg['array_task_id'], '0')
        self.assertEqual(cfg['job_display_id'], '70001_0')
        self.assertTrue(schema.run_name(cfg).endswith('-job70001_0'))
        self.assertNotIn('PRIVATE', cfg)
        sdk, out = exercise(environ=env)
        self.assertEqual(sdk.calls[0]['id'], 'distinctUUIDFixture')
        self.assertEqual(out[0]['job_identity']['job_id'], '70002')
        self.assertEqual(out[0]['job_identity']['job_display_id'], '70001_0')

    def test_caller_mismatch_incomplete_array_and_signed_step(self):
        with self.assertRaisesRegex(ValueError, 'CALLER_ENV_MISMATCH'):
            schema.bind_job_identity(dict(CFG, job_id='70001'), {'SLURM_JOB_ID': '70002'})
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE_ARRAY_IDENTITY'):
            schema.bind_job_identity(CFG, {'SLURM_JOB_ID': '70002', 'SLURM_ARRAY_TASK_ID': '0'})
        self.assertEqual(schema.bind_job_identity(CFG,
            {'SLURM_JOB_ID': '70002', 'SLURM_STEP_ID': '-1'})['step_id'], '-1')


class GenerationClientWorkerTests(unittest.TestCase):
    def test_parent_isolates_sidecar_environment_without_starting_a_process(self):
        class IdleThread:
            def __init__(self, *, target, args=(), daemon=False):
                self.target = target
            def start(self):
                if self.target.__name__ == '_read':
                    parent = self.target.__self__
                    parent.status = 'READY_ONLINE'
                    parent.ready.set()
        process = types.SimpleNamespace(stdin=io.StringIO())
        settings_values = dict(ODEEDIT_WANDB_PYTHON='/fixture/python',
            WANDB_ENTITY=schema.ENTITY, WANDB_PROJECT=schema.PROJECT,
            WANDB_BASE_URL='https://api.wandb.ai')
        environment = dict(SLURM_JOB_ID='70002', SLURM_ARRAY_JOB_ID='70001', SLURM_ARRAY_TASK_ID='0',
            UNRELATED_EXPERIMENT_SECRET='PRIVATE_ENV_SENTINEL', PRIVATE_ARGV='PRIVATE_ARGV_SENTINEL')
        with tempfile.TemporaryDirectory(prefix='generation-tracker-fixture-') as temporary, patch.dict(
            os.environ, environment, clear=True), patch.object(client_module, 'load_env',
            return_value=settings_values), patch.object(client_module.subprocess, 'Popen',
            return_value=process) as popen, patch.object(client_module.os, 'pipe',
            return_value=(110, 111)), patch.object(client_module.os, 'close'), patch.object(
            client_module.threading, 'Thread', IdleThread):
            result = Tracker(env_file='/fixture/env', spool=Path(temporary) / 'spool', config_values=CFG)
        child = popen.call_args.kwargs['env']
        self.assertNotIn('UNRELATED_EXPERIMENT_SECRET', child)
        self.assertNotIn('PRIVATE_ARGV', child)
        self.assertNotIn('SLURM_ARRAY_TASK_ID', child)
        self.assertEqual(child['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(child['WANDB_CONSOLE'], 'off')
        self.assertEqual(child['WANDB_DISABLE_CODE'], 'true')
        self.assertEqual(child['WANDB_DISABLE_GIT'], 'true')
        self.assertNotIn('PRIVATE_', json.dumps(json.loads(process.stdin.getvalue())))
        self.assertEqual(result.config_values['job_display_id'], '70001_0')

    def test_client_eval_and_fit_axes_are_independent_and_monotonic(self):
        client = tracker()
        self.assertTrue(client.log(producer.generation_values('W0_first2000',
            summary(2000, fluency=2000, consistency=2000), 0)))
        fit = {'fit/global_candidate': 1, 'batch': 1, 'candidate': 1, 'fit/loss': 0.5}
        self.assertTrue(client.log(fit))
        self.assertTrue(client.log(current(100)))
        self.assertTrue(client.log(dict(fit, **{'fit/global_candidate': 2, 'candidate': 2})))
        self.assertTrue(client.log(current(100)))
        self.assertFalse(client.log(producer.generation_values('W0_first2000',
            summary(2000, fluency=2000, consistency=2000), 0)))
        self.assertFalse(client.log(fit))
        self.assertEqual((client.axis.edits, client.axis.fit), (100, 2))
        self.assertEqual(client.dropped, 2)

    def test_queue_rejection_does_not_advance_axis_and_invalid_points_do_not_throw(self):
        client = tracker(capacity=1)
        self.assertTrue(client.log(current(100)))
        self.assertFalse(client.log(current(200)))
        self.assertEqual(client.axis.edits, 100)
        self.assertFalse(client.log(dict(current(), prompt='PRIVATE_PROMPT')))
        self.assertFalse(client.log(current(), step=True))
        self.assertEqual(client.queue.qsize(), 1)
        self.assertEqual(client.dropped, 3)

    def test_sdk_axes_startup_and_bounded_readback(self):
        commands = [dict(op='log', values=current(), step=None),
            dict(op='log', values={'fit/global_candidate': 1, 'batch': 1,
                                  'candidate': 1, 'fit/loss': .5}, step=None),
            dict(op='finish', exit_code=0)]
        sdk, out = exercise(commands=commands)
        self.assertEqual(out[0]['status'], 'READY_ONLINE')
        self.assertEqual(out[-1]['status'], 'FINISHED_SDK_FLUSHED')
        self.assertEqual(out[-1]['method_readback']['status'], 'REMOTE_BOUNDED_ROWS_VERIFIED')
        self.assertFalse(out[-1]['scientific_completion_claim'])
        for prefix in schema.PREFIXES:
            self.assertIn(((prefix + '/*',), {'step_metric': 'edits', 'step_sync': False}), sdk.axes)
        self.assertIn((('fit/*',), {'step_metric': 'fit/global_candidate', 'step_sync': False}), sdk.axes)
        self.assertTrue(all(type(value) in (int, float, bool) for row in sdk.points for value in row.values()))

    def test_sdk_privacy_settings_no_code_raw_metadata_or_watch(self):
        options = settings(FakeSDK(), 'https://api.wandb.ai')
        self.assertEqual(options['console'], 'off')
        for key in ('save_code', 'x_save_requirements'):
            self.assertFalse(options[key])
        for key in ('disable_code', 'disable_git', 'disable_job_creation', 'x_disable_meta',
                    'x_disable_stats', 'x_disable_machine_info'):
            self.assertTrue(options[key])
        self.assertIn('code/**', options['ignore_globs'])
        sdk, _ = exercise()
        self.assertFalse(sdk.calls[0]['save_code'])
        self.assertEqual(sdk.calls[0]['resume'], 'never')
        self.assertEqual(sdk.calls[0]['project'], 'layer allocation')
        self.assertFalse(hasattr(sdk, 'watch'))

    def test_worker_rejects_remote_job_or_configuration_mismatch(self):
        for field in ('name', 'config'):
            sdk = FakeSDK()
            original = sdk.run
            def mismatched(path, field=field):
                remote = original(path)
                if field == 'name':
                    remote.name = 'wrong-name'
                else:
                    remote.config = dict(remote.config, generation_eval_seed=20261002)
                return remote
            sdk.run = mismatched
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'REMOTE_'):
                exercise(sdk)

    def test_unavailable_readback_and_sdk_failures_are_not_verified(self):
        sdk = FakeSDK()
        sdk.scan_history = lambda **kwargs: iter([])
        _, out = exercise(sdk, [dict(op='log', values=current(), step=None),
                               dict(op='finish', exit_code=0)])
        self.assertEqual(out[-1]['method_readback']['status'], 'UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE')
        self.assertFalse(out[-1]['scientific_completion_claim'])
        for sdk, expected in ((FakeSDK(log_error=True), 'FINISHED_UNVERIFIED'),
                               (FakeSDK(finish_error=True), 'LOGGING_DEGRADED_FINISH')):
            _, out = exercise(sdk, [dict(op='log', values=current(), step=None),
                                   dict(op='finish', exit_code=0)])
            self.assertEqual(out[-1]['status'], expected)
            self.assertNotIn('PRIVATE_SDK_SENTINEL', json.dumps(out))

    def test_worker_rejects_decreasing_transport_or_evaluation_axes(self):
        for commands, code in (([dict(op='log', values=current(100), step=3),
                                 dict(op='log', values=current(200), step=2)], 'TRANSPORT_STEP_DECREASE'),
                                ([dict(op='log', values=current(200), step=None),
                                  dict(op='log', values=current(100), step=None)], 'AXIS_DECREASE')):
            with self.assertRaisesRegex(ValueError, code):
                exercise(commands=commands)

    def test_no_auth_no_run_and_unknown_operation_is_rejected(self):
        sdk, out = exercise(FakeSDK(login=False))
        self.assertFalse(sdk.calls)
        self.assertEqual(out[-1]['status'], 'SETUP_READY_NEEDS_USER_LOGIN')
        with self.assertRaisesRegex(ValueError, 'UNKNOWN_OPERATION'):
            exercise(commands=[{'op': 'upload_raw'}])

    def test_producer_binds_array_zero_without_exposing_experiment_payload(self):
        environment = dict(SLURM_JOB_ID='70002', SLURM_ARRAY_JOB_ID='70001', SLURM_ARRAY_TASK_ID='0')
        cfg = dict(generation=dict(schema=CFG['generation_metric_schema'], profile=CFG['generation_profile'],
            eval_seed=20261007, reference_assets_sha256=CFG['reference_assets_sha256'],
            source_sha=CFG['generation_source_sha']), tracking={'env_file': '/fixture/env'},
            raw_prompts=['PRIVATE_PROMPT'], model_weights='PRIVATE_WEIGHTS')
        calls, written = [], []
        def init(**kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(run_id='uuidfixture', config_values=schema.bind_job_identity(
                kwargs['config']), startup={'url': 'https://wandb.ai/fixture/runs/uuidfixture'})
        with patch.dict(os.environ, environment, clear=True), patch(
            'project.run_scripts.gptj_native_baselines.generation_tracking_client.init', side_effect=init), patch.object(
            producer, 'write', side_effect=lambda path, value: written.append(value)):
            result = producer.start_tracking(cfg, {'source_commit': 'a' * 40,
                'config_sha256': 'b' * 64}, Path('/fixture/no-write'), 'CAKE')
        self.assertEqual(result.config_values['job_display_id'], '70001_0')
        self.assertEqual(result.config_values['job_id'], '70002')
        self.assertNotIn('PRIVATE_', json.dumps(calls[0]['config']))
        self.assertFalse(written[0]['scientific_complete'])


if __name__ == '__main__':
    unittest.main()
