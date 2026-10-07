"""Production caller/transport fixtures only; no model/GPU qualification."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import run
from .common import NONCE
from project.run_scripts.experiment_tracking import schema,method


class Repair(unittest.TestCase):
    def payload(self,step=1):
        return dict(phase='W0_generation',**{'generation_progress/step':step,
            'generation_progress/completed_cases':1,'generation_progress/total_cases':2000,
            'generation_progress/completed_prompts':3,'generation_progress/total_prompts':6000,
            'generation_progress/generated_tokens':20,'generation_progress/elapsed_sec':1.5})

    def test_real_old_generation_metadata_path_not_RPN_identity(self):
        from . import prepare
        old=prepare.LOCAL/'attempt-register-r1'
        guard=prepare.LOCAL/'cache-repair-20261008-r1/old-cold-guard.json'
        if not (old/'config.json').is_file() or not guard.is_file():
            self.skipTest('Exact server1 old-attempt metadata is not present on this host')
        config=prepare.read(old/'config.json')
        bound=prepare.old_w0_reuse_binding(old,config,config['cold_W'],guard)
        path=Path(bound['observer_identity_member']['path'])
        self.assertEqual(path,old/'BASE_MEMIT/generation/observer-identity.json')
        self.assertNotEqual(path,Path(config['observer_identity']['path']))
        identity=prepare.read(prepare.verify(bound['observer_identity_member']))
        self.assertEqual(identity['identity_sha256'],prepare.digest(identity['identity']))
        self.assertEqual(identity['identity']['generation_source_sha'],config['generation']['generation_source_sha'])
        self.assertEqual(identity['identity']['route'],'UNPADDED_FULL_PREFIX_NO_CACHE')

    def test_public_phase_axis_and_scientific_axes_are_distinct(self):
        value=self.payload();schema.metrics(value,scientific=True)
        axis=method.AxisState();axis.accept(value)
        self.assertIsNone(axis.edits);self.assertIsNone(axis.fit)
        axis.accept({'edits':0});axis.accept({'fit/global_candidate':0,'batch':1,'candidate':0})
        with self.assertRaisesRegex(ValueError,'AXIS_DECREASE'):
            axis.accept(self.payload(0))
        for key in ('generated_text','case_id','prompt','token_ids'):
            with self.assertRaises(ValueError):schema.metrics(dict(value,**{key:'PRIVATE'}),scientific=True)
        with self.assertRaisesRegex(ValueError,'PHASE_AXIS|BUILTIN_FINITE'):
            schema.metrics(dict(value,phase='PRIVATE_PROMPT'),scientific=True)
        class EvilPhase:
            def __eq__(self,other):raise AssertionError('Phase equality must never invoke arbitrary code or GPU conversion')
        with self.assertRaisesRegex(ValueError,'BUILTIN_FINITE'):
            schema.metrics(dict(value,phase=EvilPhase()),scientific=True)

    def test_transport_failure_is_explicit_not_scientific_retry(self):
        tracker=SimpleNamespace(log=lambda _:False)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(run.log_generation_progress(tracker,self.payload(),Path(tmp)))
            result=run.json.loads((Path(tmp)/'generation-progress-transport/step-1.json').read_text())
            self.assertEqual(result['status'],'LOGGING_DEGRADED')
            self.assertTrue(result['scientific_result_not_restarted'])
        with self.assertRaisesRegex(RuntimeError,'PROGRESS_ONLY'):
            run.log_generation_progress(tracker,dict(self.payload(),**{'W0_first2000/fluency/ngram_entropy':2}))

    def fixture(self,tmp):
        root=Path(tmp);plan={'microbatch':8,'model_identity':{'model':'gpt2xl'}}
        run.write(root/'plan.json',plan)
        c=dict(generation=dict(primary_arm='BASE_MEMIT',qualification_plan_member=run.member(root/'plan.json'),
            qualification_plan_sha256=run.digest(plan),qualification_receipt=str(root/'qualification/qualification-actual.json'),
            shared_W0_root=str(root/'shared'),model_identity=plan['model_identity'],source_identity={'source':'a'*64}))
        return c,root

    def test_missing_or_failed_head_is_not_successful_W0_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,root=self.fixture(tmp)
            with self.assertRaisesRegex(RuntimeError,'READY_MISSING'):
                run.bind_runtime_generation(c,'CAKE',None,None,None,None,None,None,root,root/'CAKE')
            run.write(root/'shared/READY.json',{'completion_verified':False,'generation_repair_nonce':NONCE})
            with self.assertRaisesRegex(RuntimeError,'FAILED_OR_INCOMPLETE'):
                run.bind_runtime_generation(c,'CAKE',None,None,None,None,None,None,root,root/'CAKE')

    def test_actual_qualification_caller_binds_selected_route_without_edit_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,root=self.fixture(tmp);before=copy.deepcopy(c)
            actual=dict(qualification_pass=True,selected_route='UNPADDED_KV_SINGLETON',fixed_microbatch=1)
            run.write(root/'qualification/qualification-actual.json',actual)
            actual['member']=run.member(root/'qualification/qualification-actual.json')
            with patch.object(run,'run_qualification',return_value=actual) as qualify,\
                patch.object(run,'verify_actual_receipt',return_value=actual):
                # First primary must create-once, so remove the fixture's receipt
                # logically through the already-known existence mock; no local raw deletion.
                with patch.object(Path,'exists',return_value=False):
                    bound=run.bind_runtime_generation(c,'BASE_MEMIT',None,None,None,None,None,None,root,root/'MEMIT')
            self.assertEqual(qualify.call_count,1);self.assertEqual(c,before)
            self.assertEqual(bound['generation']['generation_route'],'UNPADDED_KV_SINGLETON')
            self.assertEqual(bound['generation']['generation_microbatch'],1)
            self.assertEqual(bound['generation']['qualification_receipt_member'],actual['member'])

    def test_same_production_worker_progress_axes_config_identity_readback(self):
        from project.run_scripts.experiment_tracking.test_tracking import SDK
        from project.run_scripts.experiment_tracking.worker import session
        class Fake(SDK):
            def __init__(self):super().__init__();self.axes=[];self.transport=[]
            def define_metric(self,name,**kwargs):self.axes.append((name,kwargs))
            def log(self,values,step=None):
                self.transport.append(dict(values,_step=len(self.transport) if step is None else step))
                return super().log(values,step)
            def scan_history(self,**kwargs):
                return iter(row for row in self.transport if kwargs['min_step']<=row['_step']<kwargs['max_step'])
        sdk=Fake();cfg=schema.bind_job_identity(dict(server='server1',task_id='repair',arm='MEMIT',
            attempt='r1',source_sha='a'*40,config_sha='b'*64,model='gpt2xl',model_family='gpt2',
            writer='memit',role='scientific',metric_schema=method.COMPARISON_SCHEMA,
            baseline='MEMIT',generation_metric_schema='counterfact-cake-generation-metrics-v1',
            generation_profile='cf-cake-prompt-inclusive-total100-eos-corrected-v1',generation_eval_seed=20261007,
            reference_assets_sha256='c'*64,generation_source_sha='d'*64,
            generation_qualification_plan_sha256='e'*64,generation_repair_instruction=NONCE),
            {'SLURM_JOB_ID':'80000','SLURM_STEP_ID':'-5'})
        request=dict(config=cfg,run_id='fakefixture',spool='/tmp/no-write-fake',smoke=False,base_url='https://api.wandb.ai')
        commands=[dict(op='log',values=self.payload(i),step=None) for i in (0,1)]
        commands.append(dict(op='finish',exit_code=0));out=[]
        session(sdk,request,commands,out.append)
        self.assertEqual(len(sdk.calls),1);self.assertIn('job80000',sdk.name)
        self.assertEqual(sdk.config['step_id'],'-5')
        self.assertIn(('generation_progress/*',dict(step_metric='generation_progress/step',step_sync=False)),sdk.axes)
        self.assertEqual(out[-1]['method_readback']['status'],'REMOTE_BOUNDED_ROWS_VERIFIED')
        self.assertEqual(out[-1]['method_readback']['checked'][0]['kind'],'generation_progress')


if __name__=='__main__':unittest.main()
