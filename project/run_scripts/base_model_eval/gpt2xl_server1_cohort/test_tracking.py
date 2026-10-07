"""Actual task-local assembly via fake SDK; CPU only, no online/model/Slurm."""
import copy
import json
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import Mock

from . import schema
from .tracking import Tracker,create_identity,chosen_run_id
from .worker import session
from .curves import DEFAULT_CONFIG,build_curves,attach_occurrence_ordinals
from project.run_scripts.experiment_tracking import schema as original


def fixture_config():
    return dict(DEFAULT_CONFIG,server='server1',task_id='base-model-gpt2xl-w0-cohort-curves',
        arm='W0_BASE_MODEL',attempt='fixture',source_sha='a'*40,config_sha='b'*64,
        model='gpt2xl',model_family='gpt2')


def fixtures():
    rows=[]
    for ordinal in range(2000):
        for kind,multiple in [('R',1),('P',2),('N',10)]:
            for panel in range(multiple):
                count=ordinal%3+1
                new=1. if kind!='N' else 3.
                row=dict(identity=f'{ordinal}/{kind}/{panel}',case_id=2000-ordinal,
                    kind=kind,prompt_index=panel,endpoint='W0',new_nll=new,true_nll=2.,
                    new_token_identity=f'new{ordinal}/{kind}/{panel}',true_token_identity=f'true{ordinal}/{kind}/{panel}',
                    new_token_count=count,new_token_correct=count-1,new_strict=False,
                    true_token_count=count,true_token_correct=count,true_strict=True,
                    margin_true_minus_new=2.-new)
                rows.append(row)
    rows=attach_occurrence_ordinals(rows)
    return build_curves(rows,copy.deepcopy(rows))


class FakeSDK:
    __version__=schema.SDK_VERSION
    def __init__(self,auth=True,log_error=False,finish_error=False,readback='ok'):
        self.auth=auth;self.log_error=log_error;self.finish_error=finish_error;self.readback=readback
        self.definitions=[];self.history=[];self.calls=[];self.scan_calls=[];self.next_step=0
        self.Settings=lambda **kw:kw
    def setup(self,**kw):self.settings=kw['settings']
    def login(self,**kw):
        assert kw['prompt'] is False and kw['verify'] is True
        return self.auth
    def init(self,**kw):
        self.calls.append(kw);self.id=kw['id'];self.name=kw['name'];self.config=kw['config'];self.offline=False
        self.url='https://wandb.ai/wkdguswns2256/layer%20allocation/runs/'+self.id
        return self
    def define_metric(self,*args,**kwargs):self.definitions.append((args,kwargs))
    def Api(self,**kw):return self
    def run(self,path):return self
    def log(self,values,step=None):
        if self.log_error:raise RuntimeError('SECRET_SENTINEL_NEVER_EMIT')
        actual=self.next_step if step is None else step;self.next_step=actual+1
        self.history.append(dict(values,_step=actual,_timestamp=1.,_runtime=1.))
    def finish(self,**kw):
        if self.finish_error:raise RuntimeError('SECRET_SENTINEL_NEVER_EMIT')
    def scan_history(self,**kw):
        self.scan_calls.append(kw)
        if self.readback=='error':raise RuntimeError('SECRET_SENTINEL_NEVER_EMIT')
        values=self.history[:-1] if self.readback=='missing' else self.history
        if self.readback=='wrong':
            values=copy.deepcopy(values);values[-1]['current/post/R/success_pct']=12.
        values=[row for row in values if kw.get('min_step',0)<=row['_step']<kw.get('max_step',float('inf'))]
        if kw.get('keys') is not None:
            values=[{key:row[key] for key in kw['keys']} for row in values if all(key in row for key in kw['keys'])]
        return iter(values)


def execute(payloads,sdk=None):
    sdk=sdk or FakeSDK();cfg=schema.bind_job_identity(fixture_config(),dict(SLURM_JOB_ID='42',
        SLURM_ARRAY_JOB_ID='40',SLURM_ARRAY_TASK_ID='0',SLURM_STEP_ID='-5'))
    request=dict(config=cfg,run_id='0123456789abcdef',spool='/tmp/no-fake-write',base_url='https://api.wandb.ai')
    commands=[dict(op='log',values=payload,step=None) for payload in payloads]
    commands.append(dict(op='finish',exit_code=0));output=[]
    session(sdk,request,commands,output.append)
    return sdk,output,cfg


class W0TrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.payloads=fixtures()

    def test_shared_validator_rejects_truthful_profile_own_branch_accepts(self):
        value=self.payloads[1];cfg=fixture_config()
        with self.assertRaises(ValueError):original.metrics(value,scientific=True)
        self.assertEqual(schema.metrics(value,cfg)['post_state_edits'],0)
        self.assertEqual(value['edits'],100)
        for change in ({'actual_applied_edits':100},{'writer':'memit'},{'reference_only':False},
                       {'w0_reference_schema':'anything'},{'role':'derived_comparison_snapshot'}):
            with self.assertRaises(ValueError):schema.config(dict(cfg,**change))

    def test_real_job_array_zero_signed_step_and_name_identity(self):
        sdk,out,cfg=execute(self.payloads)
        self.assertEqual(cfg['job_display_id'],'40_0');self.assertEqual(cfg['step_id'],'-5')
        self.assertIn('job40_0',out[0]['run_name'])
        self.assertEqual(sdk.calls[0]['config']['actual_model_edits'],0)
        with self.assertRaises(ValueError):schema.bind_job_identity(fixture_config(),{})
        with self.assertRaises(ValueError):schema.bind_job_identity(dict(fixture_config(),job_id='99'),{'SLURM_JOB_ID':'42'})

    def test_twenty_current_four_allseen_finish_readback_and_sdk_ack_separate(self):
        sdk,out,cfg=execute(self.payloads)
        final=out[-1]['cohort_readback']
        self.assertEqual(final['status'],'REMOTE_W0_CURVE_COVERAGE_VERIFIED')
        self.assertEqual(final['coverage']['current_coverage'],20)
        self.assertEqual(final['coverage']['all_seen_coverage'],4)
        self.assertEqual(final['coverage']['full_points'],[0])
        self.assertFalse(out[-1]['scientific_completion_claim'])
        self.assertTrue(all(row['delivery']=='SDK_ASYNC_NOT_REMOTE_ACK' for row in out if row['status']=='LOGGING_ACCEPTED'))
        self.assertIn((('current/post/*',),{'step_metric':'edits','step_sync':False}),sdk.definitions)
        self.assertEqual(sdk.scan_calls,[dict(min_step=0,max_step=21,page_size=26)])

    def test_missing_wrong_unavailable_readback_not_remote_pass(self):
        for mode in ('missing','wrong','error'):
            _,out,_=execute(self.payloads,FakeSDK(readback=mode))
            self.assertEqual(out[-1]['cohort_readback']['status'],'UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE')
            self.assertNotIn('SECRET_SENTINEL',json.dumps(out))

    def test_privacy_no_raw_tensor_config_or_unbounded_string(self):
        for payload in ({'raw_prompt':'private'},{'model_weight':[1.]},{'time/phase_seconds':float('nan')},
                        {'fit/loss':1.},{'evaluation_model_state':'private'}):
            with self.assertRaises(ValueError):schema.metrics(payload,fixture_config())
        for field in ('prompt','api_key','raw_env','argv'):
            with self.assertRaises(ValueError):schema.config(dict(fixture_config(),**{field:'private'}))
        sdk,_,_=execute(self.payloads)
        self.assertEqual(sdk.settings['console'],'off');self.assertFalse(sdk.settings['save_code'])
        self.assertTrue(sdk.settings['disable_git']);self.assertTrue(sdk.settings['x_disable_meta'])

    def test_log_rejection_explicit_and_queue_failure_does_not_advance(self):
        tracker=Tracker.__new__(Tracker);tracker.config_values=fixture_config();tracker.closed=False
        tracker.done=threading.Event();tracker.queue=queue.Queue(1);tracker.axis=schema.AxisState()
        tracker.log_lock=threading.Lock();tracker.dropped=0;tracker.last_rejection=None;tracker._receipt=Mock()
        tracker.queue.put('full')
        self.assertFalse(tracker.log(self.payloads[0]));self.assertIsNone(tracker.axis.last)
        self.assertTrue(tracker.last_rejection['point_not_enqueued'])
        tracker.queue.get();self.assertTrue(tracker.log(self.payloads[0]));self.assertEqual(tracker.axis.last,0)
        self.assertFalse(tracker.log({'raw_prompt':'private'}));self.assertEqual(tracker.dropped,2)

    def test_immutable_identity_and_progress_not_curve_coverage(self):
        _,out,cfg=execute(self.payloads)
        with tempfile.TemporaryDirectory() as folder:
            value=create_identity(folder,cfg,out[0],'0123456789abcdef')
            original_bytes=(Path(folder)/'identity.json').read_bytes()
            with self.assertRaises(FileExistsError):create_identity(folder,cfg,out[0],'0123456789abcdef')
            self.assertEqual((Path(folder)/'identity.json').read_bytes(),original_bytes)
            self.assertTrue(value['reference_only'])
        sdk,out,cfg=execute([dict(step=2000,edits=0,phase_id=2),*self.payloads])
        self.assertEqual(out[-1]['cohort_readback']['status'],'REMOTE_W0_CURVE_COVERAGE_VERIFIED')
        self.assertEqual(sdk.scan_calls,[dict(min_step=1,max_step=22,page_size=26)])

    def test_sealed_new_uuid_passes_without_replacing_it(self):
        self.assertEqual(chosen_run_id('0123456789abcdef'),'0123456789abcdef')
        self.assertNotEqual(chosen_run_id(),chosen_run_id())
        for value in ('job42','',42,'A'*16,'a'*15,'a'*17):
            with self.assertRaises(ValueError):chosen_run_id(value)

    def test_auth_and_transport_failures_no_secret_or_science_success(self):
        sdk,out,_=execute(self.payloads,FakeSDK(auth=False))
        self.assertFalse(sdk.calls);self.assertEqual(out[-1]['status'],'SETUP_READY_NEEDS_USER_LOGIN')
        for sdk in (FakeSDK(log_error=True),FakeSDK(finish_error=True)):
            _,out,_=execute(self.payloads,sdk)
            self.assertNotIn('SECRET_SENTINEL',json.dumps(out))
            self.assertNotEqual(out[-1].get('cohort_readback',{}).get('status'),'REMOTE_W0_CURVE_COVERAGE_VERIFIED')
        tracker=Tracker.__new__(Tracker);tracker.closed=False;tracker.queue=queue.Queue(1)
        tracker.done=threading.Event();tracker.result={};tracker.dropped=0;tracker.last_rejection=None
        tracker._stop=Mock(side_effect=OSError('SECRET_SENTINEL'));tracker._receipt=Mock()
        self.assertEqual(tracker.finish(timeout=.001)['local_status'],'LOGGING_DEGRADED_STOP')
        tracker.finish=Mock(side_effect=OSError('SECRET_SENTINEL'))
        self.assertFalse(tracker.__exit__(RuntimeError,RuntimeError('science'),None))


if __name__=='__main__':unittest.main()
