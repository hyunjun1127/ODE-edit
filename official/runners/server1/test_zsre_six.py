"""CPU fixtures only: six-arm control, native history, factual schedule/resume, strict transport."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch
from official.experiments import checkpoint
from official.experiments.prepare import digest,write_new
from official.runners.server1 import common,submit,zsre_run
from official.runners.server1.native import NativeEngine,NativeBindingError
from official.runners.server1 import test_run as fixtures
from official.runners.server1.test_run import FixtureModel,FixtureEngine,FixtureTracker,seed_CPU


class ZsreTests(unittest.TestCase):
    def test_six_arm_DAG_resource_edges_no_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);configs={}
            for method in submit.ZSRE_METHODS:
                cfg=dict(model='llama3',method=method,dataset='zsre',zsre_six=True,
                    scope_override='USER-DIRECT-SERVER1-ZSRE-SIX-20261009',
                    stream=dict(requests=2000,batch_size=100,batches=20))
                path=root/(method+'.json');write_new(path,cfg);configs[method,'zsre']=path
            plan=submit.build_zsre_six(configs,root/'runs',main_commit='a'*40,official_tree='b'*40,
                inputs=[submit.member(configs['FT','zsre'])],existing_frontier=['61677','61678'],cap=4)
            self.assertEqual(len(plan['jobs']),14)
            self.assertEqual(submit.graph_width(plan['jobs']),4)
            self.assertEqual(sum(j['mode']=='chain' for j in plan['jobs']),6)
            self.assertEqual(sum(j['mode']=='base_w0' for j in plan['jobs']),1)
            ids={j['key']:str(900+i) for i,j in enumerate(plan['jobs'])}
            later=next(j for j in plan['jobs'] if j['key']=='qual-zsre-memit_fe')
            _,edges=submit.sbatch_argv(plan,later,root/'a.sh',root,ids)
            self.assertIn(('afterany',ids['zsre-ft']),edges)
            self.assertIn(('afterok',ids['zsre-w0']),edges)
            self.assertIn('official.runners.server1.zsre_run',submit.runtime_argv(plan,later,root/'lock'))

    def test_BLUE_native_history_physical_layers_restore(self):
        engine=NativeEngine.__new__(NativeEngine);engine.method='ALPHAEDIT_BLUE'
        engine.hparams=SimpleNamespace(layers=[4,8]);engine.blue_H=torch.zeros(2,3,3)
        engine._refresh_history();self.assertEqual(set(engine.cache_c),{'4','8'})
        engine.restore_history({'4':torch.ones(3,3),'8':torch.full((3,3),8.)})
        self.assertTrue(torch.equal(engine.blue_H[1],torch.full((3,3),8.)))
        with self.assertRaises(NativeBindingError):engine.restore_history({'0':torch.zeros(3,3)})

    def test_BLUE_projector_uses_physical_slots_zero_four(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'P.pt';torch.save(torch.stack([torch.full((3,3),float(i)) for i in range(5)]),path)
            engine=NativeEngine.__new__(NativeEngine);engine.method='ALPHAEDIT_BLUE'
            engine.hparams=SimpleNamespace(layers=[4,8],nullspace_threshold=.02)
            engine.asset_manifest=dict(projector=dict(common.member(path),physical_layers=[4,5,6,7,8],threshold=.02),
                                       model=dict(identity=dict(intermediate=3)))
            engine._bind_projector()
            self.assertTrue(torch.equal(engine.blue_P[1],torch.full((3,3),4.)))
            self.assertTrue(torch.equal(engine.blue_H,torch.zeros(2,3,3)))

    def test_six_writers_real_strict_schema_current_and_cumulative(self):
        from official.runners.server1.test_tracking_binding import SchemaOnlyTransport,zsre_endpoint
        with tempfile.TemporaryDirectory() as tmp:
            config=dict(tracking=dict(module='official.tracking.client',entity='wkdguswns2256',
                project='layer allocation',attempt='CPU-fixture',env_file=str(Path(tmp)/'not-read')))
            for method in submit.ZSRE_METHODS:
                transport=SchemaOnlyTransport()
                with patch.object(common.importlib,'import_module',return_value=transport):
                    tracker=common.Tracking(config,Path(tmp)/method,dict(code_commit='a'*40,config_sha256='b'*64),
                        mode='chain',method=method,dataset='zsre')
                tracker.log(common.factual_payload(zsre_endpoint(2000),'W0_first2000',0))
                for b in range(1,21):
                    pre=common.factual_payload(zsre_endpoint(100),'current/pre',100*b)
                    pre.update(pre_state_edits=100*(b-1),post_state_edits=100*b);tracker.log(pre)
                    tracker.log(common.factual_payload(zsre_endpoint(100),'current/post',100*b))
                    if b in (5,10,15,20):tracker.log(common.factual_payload(zsre_endpoint(100*b),'all_seen/post',100*b))
                self.assertEqual(len(transport.calls[0]['tracker'].payloads),45)
                self.assertFalse(any(k.startswith('generation_') for k in transport.calls[0]['config']))

    def test_actual_state_schema_all_six(self):
        from official.baselines.registry import hparams
        for method in submit.ZSRE_METHODS:
            hp=hparams(method,'llama3')
            state=dict(method=method,successful_calls=3,contexts_sha256='c'*64,
                selected_weights={hp.rewrite_module_tmp.format(i)+'.weight':dict(sha256='a'*64,
                    shape=[4096,14336],dtype='torch.float32') for i in hp.layers},
                cache_c={str(i):dict(sha256='b'*64,shape=[14336,14336],dtype='torch.float32')
                    for i in hp.layers} if method in common.HISTORY_METHODS else {})
            state['identity_sha256']=digest(state);zsre_run.state_check(state,method,3)
            state['successful_calls']=2
            with self.assertRaises(ValueError):zsre_run.state_check(state,method,3)

    def test_twenty_current_and_four_cumulative_no_generation(self):
        fixture=fixtures.RunnerConnectorTests();fixture.setUp()
        try:
            args=fixture.arguments(fixture.root/'chain',dataset='zsre');output=args.output;output.mkdir()
            tracker=FixtureTracker();ref={'evaluation':fixture.endpoint(FixtureModel(),None,fixture.records,'zsre',fixture.external)}
            with ExitStack() as stack:
                stack.enter_context(patch('official.runners.server1.audit.audit_factual'))
                stack.enter_context(patch('official.runners.server1.native.NativeEngine',FixtureEngine))
                stack.enter_context(patch.object(zsre_run,'bindings',return_value=(fixture.assets,fixture.records,fixture.identity,fixture.external)))
                stack.enter_context(patch.object(zsre_run,'load_model',return_value=(FixtureModel(),None)))
                stack.enter_context(patch.object(zsre_run,'seed_edit',seed_CPU))
                stack.enter_context(patch.object(zsre_run,'factual',fixture.endpoint))
                stack.enter_context(patch.object(zsre_run,'reference',return_value=ref))
                stack.enter_context(patch.object(zsre_run,'verify_qualification'))
                zsre_run.chain(args,{},fixture.lock,output,tracker)
            rows=tracker.payloads
            self.assertEqual(sum('official/current/pre/requests' in r for r in rows),20)
            self.assertEqual(sum('official/current/post/requests' in r for r in rows),20)
            self.assertEqual([r['official/all_seen/post/requests'] for r in rows if 'official/all_seen/post/requests' in r],
                             [500,1000,1500,2000])
            for row in rows:
                if 'official/current/pre/requests' in row:
                    self.assertEqual(row['official/current/pre/requests'],100)
                    self.assertEqual(row['pre_state_edits'],row['edits']-100)
                self.assertFalse(any('generation' in k or 'fluency' in k for k in row))
            payload=checkpoint.load(output/'checkpoint',fixture.identity)
            self.assertEqual(payload['batch'],20)
            self.assertEqual(common.read(output/'COMPLETE.json')['generation_calls'],0)
        finally:fixture.doCleanups()

    def test_real_CPU_B3_vs_B2_resume_connector(self):
        fixture=fixtures.RunnerConnectorTests();fixture.setUp()
        try:
            config={};ref={'evaluation':{}}
            with ExitStack() as stack:
                stack.enter_context(patch('official.runners.server1.audit.audit_factual'))
                stack.enter_context(patch('official.runners.server1.native.NativeEngine',FixtureEngine))
                stack.enter_context(patch.object(zsre_run,'bindings',return_value=(fixture.assets,fixture.records,fixture.identity,fixture.external)))
                stack.enter_context(patch.object(zsre_run,'load_model',side_effect=lambda assets:(FixtureModel(),None)))
                stack.enter_context(patch.object(zsre_run,'seed_edit',seed_CPU))
                stack.enter_context(patch.object(zsre_run,'factual',fixture.endpoint))
                stack.enter_context(patch.object(zsre_run,'reference',return_value=ref))
                stack.enter_context(patch.object(zsre_run,'state_check'))
                for phase,name in [('continuous','continuous'),('stop','split'),('resume','split')]:
                    folder=fixture.root/name;folder.mkdir(exist_ok=True)
                    args=fixture.arguments(folder,stage=phase,dataset='zsre')
                    zsre_run.stage(args,config,fixture.lock,folder)
            continuous=common.read(fixture.root/'continuous'/'stage-complete.json')
            resumed=common.read(fixture.root/'split'/'stage-complete.json')
            zsre_run.compare_qualification(continuous,resumed)
            changed=deepcopy(resumed);changed['RNG_sha256']='different'
            with self.assertRaises(ValueError):zsre_run.compare_qualification(continuous,changed)
        finally:fixture.doCleanups()


if __name__=='__main__':unittest.main()
