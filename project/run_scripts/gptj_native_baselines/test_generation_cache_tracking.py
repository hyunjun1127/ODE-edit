"""Private repair tracking CPU/fake-SDK proof, never online or GPU proof."""
import copy
import math
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from . import generation_tracking as producer
from . import generation_tracking_client as client
from . import generation_tracking_schema as schema
from .generation_tracking_worker import session, verify_last_progress
from .test_generation_tracking import CFG, FakeSDK, current, tracker
from .test_generation_tracking_shared import raw_summary


ENV=dict(SLURM_JOB_ID='71002',SLURM_ARRAY_JOB_ID='71001',SLURM_ARRAY_TASK_ID='0',SLURM_STEP_ID='batch')


def config():
    return schema.bind_job_identity(dict(CFG,task_id=schema.REPAIR_TASK,
        attempt=schema.REPAIR_ATTEMPT,generation_qualification_plan_sha256='e'*64),ENV)


def progress(step=0, completed=0, reused=0, route=None):
    result={'generation_progress/'+key:0 for key in schema.PROGRESS_FIELDS}
    result.update(phase='W0_generation',route=route or schema.PROGRESS_ROUTES[1],
        model='gptj',job_id=ENV['SLURM_JOB_ID'])
    result.update({'generation_progress/step':step,'generation_progress/completed_cases':completed,
        'generation_progress/new_cases':completed-reused,'generation_progress/reused_cases':reused,
        'generation_progress/total_cases':2000,'generation_progress/total_prompts':4000,
        'generation_progress/completed_prompts':completed*2})
    return result


def exercise(values, sdk=None):
    sdk=sdk or FakeSDK();emitted=[]
    request=dict(config=config(),run_id='cacheRepairFixture',spool='/fixture/no-write',
                 smoke=False,base_url='https://api.wandb.ai')
    commands=[dict(op='log',values=row,step=None) for row in values]+[dict(op='finish',exit_code=0)]
    session(sdk,request,commands,emitted.append)
    return sdk,emitted


class CacheRepairTrackingTests(unittest.TestCase):
    def test_exact_fifteen_scalars_and_closed_metadata_accept_axis_zero(self):
        for route in schema.PROGRESS_ROUTES:
            values=progress(route=route)
            self.assertEqual(len(schema.PROGRESS_FIELDS),15)
            self.assertEqual(schema.metrics(values,scientific=True),values)
            self.assertNotIn('edits',values)
            self.assertEqual(values['generation_progress/step'],0)
        for key in ('edits','pre_state_edits','post_state_edits','fit/global_candidate',
                    'prompt','case_id','raw_tokens','api_key'):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'EXACT_KEYS'):
                schema.metrics(dict(progress(),**{key:1}),scientific=True)
        missing=progress();missing.pop('generation_progress/new_cases')
        with self.assertRaisesRegex(ValueError,'EXACT_KEYS'):
            schema.metrics(missing,scientific=True)

    def test_scalar_privacy_has_no_conversion_or_nonfinite_values(self):
        class NonScalar:
            def __float__(self):
                raise AssertionError('No scalar conversion or synchronization')
        for value in (True,-1,float('nan'),float('inf'),'PRIVATE_TEXT',[1],{'secret':1},NonScalar()):
            with self.subTest(type=type(value).__name__),self.assertRaises(ValueError):
                schema.metrics(dict(progress(),**{'generation_progress/step':value}),scientific=True)
        with self.assertRaisesRegex(ValueError,'INTEGER_COUNTS'):
            schema.metrics(dict(progress(),**{'generation_progress/step':.5}),scientific=True)

    def test_public_metadata_and_count_arithmetic_are_closed(self):
        for key,value in (('phase','current_generation'),('phase',123),('route','PENDING'),
                          ('route','PRIVATE_PROMPT'),('model','gpt2xl'),('job_id','0'),('job_id','71_0')):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'PUBLIC_METADATA'):
                schema.metrics(dict(progress(),**{key:value}),scientific=True)

    def test_shared_route_names_and_edit_generation_phase_are_explicit(self):
        for route in ('UNPADDED_KV_SINGLETON','EQUAL_LENGTH_KV_BATCH'):
            values=progress(route=route)
            self.assertEqual(schema.metrics(values,scientific=True),values)
            for total in (100,500,1000,1500,2000):
                changed=dict(values,phase='generation_evaluation')
                changed['generation_progress/total_cases']=total
                changed['generation_progress/total_prompts']=total*10
                self.assertEqual(schema.metrics(changed,scientific=True),changed)
        invalid=dict(progress(),phase='generation_evaluation')
        invalid['generation_progress/total_cases']=1604
        with self.assertRaisesRegex(ValueError,'COUNT_ARITHMETIC'):
            schema.metrics(invalid,scientific=True)
        for key,value in (('generation_progress/total_cases',100),
                          ('generation_progress/completed_cases',1),
                          ('generation_progress/completed_prompts',4001)):
            with self.assertRaisesRegex(ValueError,'COUNT_ARITHMETIC'):
                schema.metrics(dict(progress(),**{key:value}),scientific=True)

    def test_repair_identity_plan_sha_and_array_index_zero_are_exact(self):
        cfg=config()
        self.assertEqual(cfg['array_task_id'],'0')
        self.assertEqual(cfg['job_display_id'],'71001_0')
        self.assertEqual(schema.config(CFG),CFG)
        for key,value in (('attempt','attempt-r1'),('generation_qualification_plan_sha256','f'*40),
                          ('generation_qualification_plan_sha256','F'*64),('private_path','/PRIVATE')):
            with self.assertRaises(ValueError):schema.config(dict(cfg,**{key:value}))

    def test_repair_startup_uses_private_client_and_original_keeps_shared(self):
        c=dict(generation=dict(schema='counterfact-cake-generation-metrics-v1',
            profile='cf-cake-prompt-inclusive-total100-eos-corrected-v1',eval_seed=20261007,
            reference_assets_sha256='c'*64,source_sha='d'*40,
            repair={'qualification_plan_sha256':'e'*64}),tracking={'env_file':'/fixture/not-read'})
        captured={}
        def fake_init(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(run_id='fixture',config_values=schema.bind_job_identity(
                kwargs['config'],ENV),startup={'status':'READY_ONLINE'})
        with patch.dict(os.environ,dict(ENV,PRIVATE='ignored'),clear=True),\
                patch.object(client,'init',side_effect=fake_init) as private,\
                patch.object(producer,'init') as shared,patch.object(producer,'write'):
            producer.start_tracking(c,dict(source_commit='a'*40,config_sha256='b'*64),
                                    Path('/fixture/no-write'),'CAKE')
        private.assert_called_once();shared.assert_not_called()
        self.assertEqual(captured['config']['task_id'],schema.REPAIR_TASK)
        self.assertEqual(captured['config']['attempt'],schema.REPAIR_ATTEMPT)
        self.assertEqual(captured['config']['generation_qualification_plan_sha256'],'e'*64)
        self.assertNotIn('qualification_plan_sha256',captured['config'])
        captured.clear()
        c['tracking_attempt']='cache-repair-r2'
        with patch.dict(os.environ,ENV,clear=True),patch.object(client,'init',fake_init),\
                patch.object(producer,'write'):
            producer.start_tracking(c,dict(source_commit='a'*40,config_sha256='b'*64),
                Path('/fixture/no-write'), 'BASE_MEMIT')
        self.assertEqual(captured['config']['attempt'],'cache-repair-r2')
        self.assertNotIn('PRIVATE',captured['config'])
        self.assertNotIn('repair',captured['config'])

    def test_client_progress_axis_does_not_pollute_edits_or_fit(self):
        t=tracker();t.config_values=config()
        self.assertTrue(t.log(current(100)))
        self.assertTrue(t.log(progress(0)))
        self.assertEqual(t.axis.edits,100);self.assertIsNone(t.axis.fit)
        self.assertTrue(t.log(current(200)))
        self.assertTrue(t.log(progress(1,completed=4,reused=1)))
        self.assertEqual(t.axis.edits,200);self.assertEqual(t.progress_axis.step,1)
        self.assertFalse(t.log(progress(0)))
        self.assertEqual(t.progress_axis.step,1)
        self.assertFalse(t.log(dict(progress(2),job_id='71003')))
        self.assertEqual(t.queue.qsize(),4)

    def test_rejected_queue_point_does_not_advance_progress_axis(self):
        t=tracker(capacity=1);t.config_values=config()
        self.assertTrue(t.log(progress(0)))
        self.assertFalse(t.log(progress(1,completed=1)))
        self.assertEqual(t.progress_axis.step,0)

    def test_fake_sdk_defines_separate_axis_and_bounded_finish_readback(self):
        sdk,emitted=exercise([progress(0),current(100),progress(1,4,1),current(200)])
        self.assertIn((('generation_progress/*',),dict(step_metric='generation_progress/step',step_sync=False)),sdk.axes)
        self.assertIn((('current/post/*',),dict(step_metric='edits',step_sync=False)),sdk.axes)
        final=emitted[-1]
        self.assertEqual(final['status'],'FINISHED_SDK_FLUSHED')
        self.assertEqual(final['generation_progress_readback']['status'],'REMOTE_BOUNDED_PROGRESS_VERIFIED')
        self.assertEqual(final['generation_progress_readback']['generation_step'],1)
        self.assertEqual(final['generation_progress_readback']['transport_step'],2)
        self.assertEqual(final['generation_progress_readback']['phase'],'W0_generation')
        self.assertEqual(final['method_readback']['rows'],1)
        self.assertFalse(final['scientific_completion_claim'])
        self.assertNotIn('W0_first2000/generation/planned_count',sdk.points[0])

    def test_worker_rejects_decreasing_axis_and_wrong_job_before_sdk_log(self):
        for rows in ([progress(1),progress(0)], [dict(progress(),job_id='71003')]):
            sdk=FakeSDK()
            with self.assertRaises(ValueError):exercise(rows,sdk)
            self.assertLessEqual(len(sdk.points),1)

    def test_readback_exact_phase_and_missing_row_are_unverified_without_error_text(self):
        sdk,_=exercise([progress(0)])
        expected=dict(step=0,values=progress(0))
        sdk.points[0]['phase']='PRIVATE_WRONG_PHASE'
        result=verify_last_progress(sdk,'https://api.wandb.ai',sdk.id,config(),sdk.name,expected)
        self.assertEqual(result['status'],'UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE')
        self.assertNotIn('PRIVATE',str(result))
        sdk.points.clear()
        self.assertEqual(verify_last_progress(sdk,'https://api.wandb.ai',sdk.id,config(),
            sdk.name,expected)['status'],'UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE')

    def test_raw_shared_summary_roundoff_and_edit_axes_remain_unchanged(self):
        raw=raw_summary();raw['reference_score']=math.nextafter(1.,math.inf)
        before=copy.deepcopy(raw)
        values=producer.generation_values('current/post',raw,100,0,100)
        self.assertEqual(schema.metrics(values,scientific=True,canonical=True),values)
        with self.assertRaisesRegex(ValueError,'GEN_COSINE_RANGE'):
            schema.metrics(values,scientific=True)
        self.assertEqual(raw,before)
        self.assertNotIn('generation_progress/step',values)

    def test_caller_rejected_progress_is_nonblocking_and_never_mutates_payload(self):
        t=tracker();t.config_values=config();values=dict(progress(),raw_tokens=[1,2])
        before=copy.deepcopy(values)
        with tempfile.TemporaryDirectory() as directory:
            t.spool=Path(directory)
            with patch('builtins.print'):
                self.assertFalse(producer.log_generation_progress(t,values))
        self.assertEqual(values,before)
        self.assertEqual(t.queue.qsize(),0)
        self.assertEqual(t.caller_dropped_points,1)


if __name__=='__main__':
    unittest.main()
