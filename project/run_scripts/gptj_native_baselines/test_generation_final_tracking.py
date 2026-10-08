"""Final-only identity and realtime scalar seams: fake SDK/CPU, never online."""
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from project.run_scripts.experiment_tracking import schema as shared
from . import generation_tracking as producer
from . import generation_tracking_schema as schema
from . import generation_tracking_client as client
from .generation_tracking_worker import session
from .test_generation_tracking import CFG, FakeSDK, current, summary, tracker
from .test_generation_tracking_reader import ENV, read_startup, startup
from .test_generation_cache_tracking import progress


def config():
    return schema.bind_job_identity(dict(CFG, task_id=schema.REPAIR_TASK,
        attempt='final-generation-v1', generation_qualification_plan_sha256='e' * 64,
        generation_schedule='W20_ONLY_FIRST2000'), ENV)


def final_values():
    return producer.generation_values('all_seen/post',
        summary(2000, fluency=2000, consistency=2000), 2000, 1900, 2000)


class FinalTrackingTests(unittest.TestCase):
    def test_final_config_accepts_shared_identity_no_private_keys(self):
        cfg = config()
        self.assertEqual(schema.config(cfg), cfg)
        self.assertEqual(shared.config({k:v for k,v in cfg.items() if k!='generation_schedule'}),
            {k:v for k,v in cfg.items() if k!='generation_schedule'})
        # The source-bound task-private identity extension handles the new
        # canonical schedule key without modifying or bypassing SH1's helper.
        with self.assertRaisesRegex(ValueError,'CONFIG_NOT_ALLOWLISTED'):
            shared.config(cfg)
        self.assertEqual(cfg['generation_schedule'],'W20_ONLY_FIRST2000')
        self.assertNotIn('evaluation_schedule', cfg)
        self.assertNotIn('qualification_plan_sha256', cfg)
        self.assertEqual(cfg['array_task_id'], '0')
        self.assertEqual(cfg['step_id'], '-5')
        self.assertIn('job71001_0', schema.run_name(cfg))

    def test_actual_parent_reader_creates_immutable_final_identity(self):
        cfg = config()
        with tempfile.TemporaryDirectory(prefix='final-identity-fixture-') as out:
            result = read_startup(out, cfg, startup(cfg))
            self.assertEqual(result.status, 'READY_ONLINE')
            self.assertEqual(result.identity['config'], cfg)
            self.assertIn('final-generation-v1-job71001_0', result.identity['run_name'])
            self.assertFalse(result.identity['scientific_completion_claim'])

    def test_final_identity_immutable_privacy_and_exact_schedule_echo(self):
        cfg=config()
        for field, value in (('generation_schedule','UNREGISTERED'),('prompt','PRIVATE_SENTINEL'),
                             ('api_key','PRIVATE_SENTINEL')):
            with self.subTest(field=field), tempfile.TemporaryDirectory(prefix='final-reject-') as out:
                message=startup(cfg)
                message['config'][field]=value
                result=read_startup(out,cfg,message)
                self.assertEqual(result.status,'LOGGING_DEGRADED_CONTROL')
                self.assertFalse((Path(out)/'identity.json').exists())
        with tempfile.TemporaryDirectory(prefix='final-no-overwrite-') as out:
            first=read_startup(out,cfg,startup(cfg))
            original=(Path(out)/'identity.json').read_bytes()
            second=read_startup(out,cfg,startup(cfg,'secondUniqueFixture'))
            self.assertEqual(first.status,'READY_ONLINE')
            self.assertEqual(second.status,'LOGGING_DEGRADED_CONTROL')
            self.assertEqual((Path(out)/'identity.json').read_bytes(),original)

    def test_real_producer_final_attempt_keeps_exact_plan_key(self):
        c = dict(tracking_attempt='final-generation-v1', generation=dict(
            evaluation_schedule='FINAL_W20_ONLY',
            schema='counterfact-cake-generation-metrics-v1',
            profile='cf-cake-prompt-inclusive-total100-eos-corrected-v1', eval_seed=20261007,
            reference_assets_sha256='c' * 64, source_sha='d' * 40,
            repair={'qualification_plan_sha256': 'e' * 64}), tracking={'env_file': '/fixture'})
        captured = {}
        def fake_init(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(run_id='fixture', startup={'status': 'READY_ONLINE'},
                config_values=schema.bind_job_identity(kwargs['config'], ENV))
        with patch.dict('os.environ', ENV, clear=True), patch.object(client, 'init', fake_init), \
                patch.object(producer, 'write'):
            producer.start_tracking(c, dict(source_commit='a' * 40, config_sha256='b' * 64),
                Path('/fixture'), 'BASE_MEMIT')
        self.assertEqual(captured['config']['attempt'], 'final-generation-v1')
        self.assertEqual(captured['config']['generation_qualification_plan_sha256'], 'e' * 64)
        self.assertEqual(captured['config']['generation_schedule'],'W20_ONLY_FIRST2000')
        self.assertEqual(schema.config(schema.bind_job_identity(captured['config'],ENV)),
            schema.bind_job_identity(captured['config'],ENV))

    def test_final_metrics_only_final_full2k_generation_not_rpn_restriction(self):
        values = final_values()
        self.assertEqual(schema.metrics(values, scientific=True, final_only=True), values)
        for prefix, edits, count in (('W0_first2000', 0, 2000), ('current/post', 100, 100),
                                     ('all_seen/post', 500, 500)):
            other = producer.generation_values(prefix, summary(count, fluency=count,
                consistency=count), edits, max(0, edits - 100), edits)
            with self.subTest(prefix=prefix), self.assertRaisesRegex(ValueError, 'FINAL_W20_ONLY'):
                schema.metrics(other, scientific=True, final_only=True)
        # Native observer rows/axes remain legal at every original endpoint.
        rpn = dict(edits=100, pre_state_edits=0, post_state_edits=100,
            **{f'current/post/{kind}/count': n for kind, n in zip('RPN', (100, 200, 1000))},
            **{f'current/post/{kind}/success_count': 0 for kind in 'RPN'},
            **{f'current/post/{kind}/success_pct': 0. for kind in 'RPN'},
            **{'current/post/success_harmonic_pct': 0.})
        self.assertEqual(schema.metrics(rpn, scientific=True, canonical=True, final_only=True), rpn)

    def test_progress_only_final_phase_and_separate_axis(self):
        values = dict(progress(), phase='W20_generation')
        self.assertEqual(schema.metrics(values, scientific=True, final_only=True), values)
        with self.assertRaisesRegex(ValueError, 'FINAL_PROGRESS'):
            schema.metrics(progress(), scientific=True, final_only=True)
        values['generation_progress/total_cases'] = 100
        with self.assertRaisesRegex(ValueError, 'FINAL_PROGRESS'):
            schema.metrics(values, scientific=True, final_only=True)

    def test_parent_one_final_payload_and_rejected_queue_does_not_consume_slot(self):
        t = tracker(capacity=1)
        t.config_values = config()
        self.assertTrue(t.log(dict(progress(), phase='W20_generation')))
        self.assertFalse(t.log(final_values()))
        self.assertFalse(getattr(t, 'final_generation_logged', False))
        t.queue.get_nowait()
        self.assertTrue(t.log(final_values()))
        self.assertTrue(t.final_generation_logged)
        t.queue.get_nowait()
        self.assertFalse(t.log(final_values()))
        self.assertEqual(t.queue.qsize(), 0)

    def test_worker_single_final_point_remote_fake_readback_separate_axes(self):
        sdk, emitted = FakeSDK(), []
        values = [dict(progress(), phase='W20_generation'), final_values()]
        request = dict(config=config(), run_id='finalFixtureUnique', spool='/fixture',
            smoke=False, base_url='https://api.wandb.ai')
        commands = [dict(op='log', values=row, step=None) for row in values]
        session(sdk, request, commands + [dict(op='finish', exit_code=0)], emitted.append)
        self.assertEqual(len(sdk.points), 2)
        self.assertEqual(sdk.points[-1]['edits'], 2000)
        self.assertEqual(sdk.config['attempt'], 'final-generation-v1')
        self.assertEqual(sdk.config['generation_schedule'],'W20_ONLY_FIRST2000')
        self.assertIn('job71001_0', sdk.name)
        self.assertEqual(emitted[0]['status'], 'READY_ONLINE')
        self.assertEqual(emitted[-1]['status'], 'FINISHED_SDK_FLUSHED')
        self.assertFalse(emitted[-1]['scientific_completion_claim'])
        self.assertIn((('generation_progress/*',), dict(
            step_metric='generation_progress/step', step_sync=False)), sdk.axes)
        # Repeating the final observation is a schema violation, not a retry.
        sdk2 = FakeSDK()
        with self.assertRaisesRegex(ValueError, 'FINAL_SINGLE'):
            session(sdk2, request, [commands[-1], copy.deepcopy(commands[-1])], lambda _: None)
        self.assertEqual(len(sdk2.points), 1)


if __name__ == '__main__':
    unittest.main()
