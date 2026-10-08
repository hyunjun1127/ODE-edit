"""CPU fake SDK/model fixtures only; no assets/GPU/Slurm/online run."""
import ast
import copy
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from . import repo_native_common as common
from . import repo_native_prepare as prepare
from . import repo_native_run as run
from . import repo_native_collect as collect
from project.run_scripts.experiment_generation_eval.native_observer import NativeGenerationObserver
from project.run_scripts.experiment_generation_eval.native_profile import PROFILE, ROUTE
from project.run_scripts.experiment_generation_eval.test_native_generator import Model, Tokenizer, prompt
from project.run_scripts.experiment_generation_eval.test_observer import FakeAssets, record
from project.run_scripts.jlz_realized_writer_sequential.review_completed import Reader


class RepoNativeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_authority_native_override_does_not_edit_global_policy(self):
        value,policy=common.authority()
        self.assertEqual(value['arms'],list(common.ARMS))
        self.assertEqual(common.ARMS,('BASE_MEMIT','BASE_ALPHAEDIT','ALPHAEDIT_BLUE'))
        self.assertEqual(policy['generation']['profile'],'cf-cake-prompt-inclusive-total100-eos-corrected-v1')
        self.assertEqual(value['generation_schedule'],'W20_ONLY_FIRST2000')

    def test_prepare_drops_old_qualification_reuse_preserves_native_bindings(self):
        prior=dict(source_configs={arm:dict(native_hparams=arm) for arm in common.ARMS},
            generation=dict(profile='old',schema='cf-generation-v1',qualification_plan_member={},
                qualification_plan_sha256='old',qualification_receipts={},generation_microbatch=8,
                shared_W0_root='old',old_w0_reuse={},primary_arm='old',model_identity={'model':'fake'},
                eval_seed=20261007,reference_assets_sha256='ref'))
        source=self.root/'source';source.mkdir()
        (source/'config.json').write_text('{}');(source/'execution.lock.json').write_text('{}')
        attempt=common.LOCAL/'attempt-native-repo-r1'
        c=prepare.prepared_config(prior,attempt,source,{'path':'cancel','sha256':'fixture'})
        self.assertEqual(c['source_configs'],prior['source_configs'])
        self.assertEqual(c['generation']['profile'],PROFILE)
        self.assertEqual(c['generation']['generation_route'],ROUTE)
        self.assertFalse(c['generation']['derived_route_qualification'])
        self.assertFalse(any(key.startswith('qualification_') for key in c['generation']))
        self.assertEqual(c['generation']['plan']['three_arm_case_observations_max'],6000)
        self.assertFalse(c['W0_generation_required'])
        for key,val in [('profile','old'),('shared_W0_root','old'),('qualification_plan_member',{})]:
            wrong=copy.deepcopy(c);wrong['generation'][key]=val
            with self.assertRaises(Exception):common.validate_config(wrong)

    def test_native20_commit_precedes_single_terminal_generation_no_fallback(self):
        source=inspect.getsource(run.native_loop)
        self.assertLess(source.index("write(folder/'commit.json',receipt)"),source.index('generation.observe('))
        self.assertEqual(source.count('generation.observe('),1)
        tree=ast.parse(source)
        loop=next(node for node in tree.body[0].body if isinstance(node,ast.For))
        self.assertFalse(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and node.func.attr=='observe' and isinstance(node.func.value,ast.Name)
            and node.func.value.id=='generation' for node in ast.walk(loop)))
        self.assertNotIn('run_qualification',inspect.getsource(run))
        self.assertNotIn('UNPADDED_FULL_PREFIX_NO_CACHE',inspect.getsource(run))

    def test_generation_failure_keeps_all_twenty_authoritative_native_commits(self):
        # This is a CPU control-flow fixture: no pretrained load/forward/nativefit.
        @dataclass
        class HP:
            native_fixture:bool=True
        class Engine:
            hp=HP()
            counter=0
            def history(self):return {}
            def context_snapshot(self):return ['fixed']
            def apply(self,current,number):
                self.counter+=1
                return model,{'delta':{'solves':0}}
        class Tx:
            def __init__(self,*args):self.done=False
            def __enter__(self):return self
            def finish(self):self.done=True
            def __exit__(self,*args):return False
        class BrokenGeneration:
            calls=0
            def observe(self,*args,**kwargs):
                self.calls+=1
                raise RuntimeError('fixture_generation_failure')
        model=object();engine=Engine();generation=BrokenGeneration();commits=[]
        view=type('View',(),{'weights':{'13':torch.ones(1)}})()
        records=[{'case_id':i} for i in range(2000)]
        c=dict(packs=[dict(ids=list(range(n*100,(n+1)*100))) for n in range(20)])
        physical=lambda *args:dict(W={'13':str(engine.counter)},H={})
        def observation(*args,**kwargs):return dict(summary={},current={})
        with patch.object(run,'state',side_effect=physical),patch.object(run,'observe',side_effect=observation),\
            patch.object(run,'rows',return_value=[]),patch.object(run,'NativeTransaction',Tx),\
            patch.object(run,'batches',side_effect=lambda rr:((n+1,rr[n*100:(n+1)*100],rr[:(n+1)*100]) for n in range(20))),\
            patch.object(run,'check_native'),patch.object(run,'finish_native_batch',side_effect=lambda *args:(physical(),{'prune_applied':False})),\
            patch.object(run,'log_batch',return_value=True),patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError,'fixture_generation_failure'):
                run.native_loop(c,{'source_commit':'CPUfixture'},self.root,'BASE_MEMIT',records,
                    model,view,engine,None,None,commits,generation)
        self.assertEqual(len(commits),20)
        self.assertEqual(generation.calls,1)
        self.assertEqual(len(list(self.root.glob('batch-*/commit.json'))),20)
        self.assertEqual(json.loads((self.root/'batch-20/commit.json').read_text())['after']['W'],{'13':'20'})
        self.assertFalse((self.root/'generation-W20.json').exists())

    def test_terminal_only_commit_flag_and_cold_history_source(self):
        commits=[dict(batch=n,generation_schedule=common.SCHEDULE,generation_measured=False,
            generation=None,terminal_generation_pending=n==20) for n in (1,20)]
        collect.require_only_terminal_generation(self.root,commits)
        commits[-1]['generation_measured']=True
        with self.assertRaises(Exception):collect.require_only_terminal_generation(self.root,commits)
        source=inspect.getsource(collect.review_arm)
        self.assertIn("if arm in ('CAKE','ALPHAEDIT_BLUE') else ()",source)

    def test_fake_native_endpoint_independent_reducer_state_and_stream(self):
        assets=FakeAssets();cfg=dict(model_identity={'model':'CPUfake'},generation_source_sha='fixture',
            profile=PROFILE,eval_seed=20261007,generation_route=ROUTE,reference_assets_sha256=assets.sha)
        physical=dict(W={'13':'same'},H={'13':'native'})
        records=[record(1,prompts=[prompt(98),prompt(99)]),record(2,501,prompts=[prompt(99)])]
        observer=NativeGenerationObserver(Model(stochastic=True),Tokenizer(),assets,cfg,self.root/'raw')
        observed=observer.observe(records,'W20','ALL_SEEN',dict(W=physical['W'],H={}))
        receipt=run.native_generation_receipt(observed)
        with patch.object(collect,'generation_assets',return_value=assets):
            value=collect.native_generation_endpoint(Reader(),receipt,dict(generation=cfg),records,physical)
        self.assertEqual(value['summary'],observed['summary'])
        for key,bad in [('physical_forward_calls',999),('prefill_query_tokens',999),
                        ('generated_tokens',999),('new_case_observations',1),('seconds',float('nan'))]:
            wrong=copy.deepcopy(receipt);wrong['work'][key]=bad
            with patch.object(collect,'generation_assets',return_value=assets):
                with self.assertRaisesRegex(Exception,'NATIVE_COLLECT_FRESH_WORK_AND_EXECUTION_COUNTERS'):
                    collect.native_generation_endpoint(Reader(),wrong,dict(generation=cfg),records,physical)
        # An internally re-hashed endpoint still cannot claim a different RNG
        # stream than the exact requested ordered records/source/runtime.
        wrong=copy.deepcopy(receipt);saved=json.loads(Path(receipt['rows']['path']).read_text())
        saved['identity']['sampling_stream_sha256']='0'*64
        saved['identity_sha256']=common.digest(saved['identity'])
        wrong['identity']=saved['identity'];wrong['identity_sha256']=saved['identity_sha256']
        with patch.object(Reader,'bound',return_value=saved):
            with self.assertRaisesRegex(Exception,'NATIVE_COLLECT_EXPECTED_GLOBAL_STREAM'):
                collect.native_generation_endpoint(Reader(),wrong,dict(generation=cfg),records,physical)
        wrong=copy.deepcopy(receipt);wrong['summary']['planned_count']=99
        with patch.object(collect,'generation_assets',return_value=assets):
            with self.assertRaises(Exception):collect.native_generation_endpoint(Reader(),wrong,
                dict(generation=cfg),records,physical)
        wrong=copy.deepcopy(receipt);wrong['identity']['sampling_stream_sha256']='wrong'
        with self.assertRaises(Exception):collect.native_generation_endpoint(Reader(),wrong,
            dict(generation=cfg),records,physical)

    def test_progress_phase_scalar_payload_not_remote_ack(self):
        class Tracker:
            def __init__(self):self.values=None
            def log(self,values):self.values=values;return True
        tracker=Tracker()
        value={'phase':'generation_evaluation','generation_progress/step':1,
            'generation_progress/completed_cases':2,'generation_progress/total_cases':2000}
        self.assertTrue(run.log_generation_progress(tracker,value))
        self.assertEqual(tracker.values['phase'],'W20_generation')
        value['prompt']='private'
        with self.assertRaises(Exception):run.log_generation_progress(tracker,value)


if __name__=='__main__':unittest.main()
