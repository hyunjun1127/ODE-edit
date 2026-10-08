"""CPU fixtures only: no real model, Slurm, W&B or actual native PASS."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from official.experiments import checkpoint
from official.experiments.prepare import write_new
from official.runners.server2 import checkpoint_profile as profile, run, submit


class CheckpointOnlyTests(unittest.TestCase):
    def test_explicit_scope_and_legacy_default(self):
        self.assertFalse(profile.deferred({}))
        value = profile.profile()
        self.assertTrue(profile.deferred({'checkpoint_only_profile':value}))
        for key, replacement in [('dataset','zsre'), ('generation_W20',True),
                                 ('archive_delete_allowed',True), ('project_gpu_cap',4)]:
            broken = copy.deepcopy(value); broken[key] = replacement
            with self.assertRaisesRegex(ValueError,'EXACT_USER_PROFILE'):
                profile.deferred({'checkpoint_only_profile':broken})

    def test_truthful_tracking_schedule_not_enabled_generation(self):
        manifest = dict(tracking={'metric_schema':'official-baselines-scalar-v1'},
            checkpoint_only_profile=profile.profile(), code_commit='a'*40,
            official_tree_sha256='b'*40, model_revision='c'*40, tokenizer_sha256='d'*64,
            assets_identity_sha256='e'*64, streams={'cf':{'lock':{'stream_sha256':'f'*64}}})
        with patch.object(run.generation,'configuration',side_effect=AssertionError('generation config called')):
            cfg = run.tracking_config(manifest,run.configuration('MEMIT','cf'),'chain',Path('MEMIT'))
        self.assertEqual(cfg['dataset'],'cf')
        self.assertEqual(cfg['generation_schedule'],profile.SCHEDULE)
        self.assertNotIn('generation_profile',cfg)
        self.assertNotEqual(cfg['config_sha'], run.configuration('MEMIT','cf')['config_sha256'])
        # Exercise the actual shared schema and production worker using a fake
        # SDK, not just the caller's pre-validation config dictionary.
        from official.tracking.test_transport import execute, official
        sdk, receipts, bound = execute([official()], cfg=cfg)
        self.assertEqual(bound['generation_schedule'], profile.SCHEDULE)
        self.assertEqual(sdk.config['generation_schedule'], profile.SCHEDULE)
        self.assertEqual(receipts[-1]['method_readback']['status'], 'REMOTE_BOUNDED_ROWS_VERIFIED')
        self.assertFalse(receipts[-1]['scientific_completion_claim'])

    def test_cap3_six_pipeline_dag_and_real_collector_dependencies(self):
        roles = list(profile.METHODS); jobs = {}
        expected = [[],['100'],['100'],['100'],['101'],['102']]
        for i, role in enumerate(roles):
            dep = submit.stage_dependencies(role,'cf_checkpoint',roles,jobs,[],3)
            self.assertEqual(dep,expected[i])
            jobs[role] = str(100+i)
        self.assertEqual(submit.stage_dependencies('collector','cf_checkpoint',roles,jobs,[],3),list(jobs.values()))
        # FT single producer then MEMIT/Alpha/BLUE; FE and SPHERE replace
        # finished lanes. Every antichain has at most three GPU allocations.
        self.assertEqual(submit.roles('cf_checkpoint'),roles)

    def test_twenty_commits_save_final_without_any_generation_call(self):
        class Engine:
            method = 'MEMIT'
            batch = 0
            weight = torch.zeros(2,2)
            def weights(self): return {'weight':self.weight}
            def history(self): return {}
            def contexts(self, *args): return [['CPU fixture']]
            def state_identity(self): return {'batch':self.batch}
        engine = Engine()
        records = [dict(case_id=i, occurrence_index=i+1) for i in range(2000)]
        manifest = dict(checkpoint_only_profile=profile.profile(),code_commit='a'*40,
            base_manifest_sha256='a'*64,
            official_tree_sha256='b'*40,model_revision='c'*40,tokenizer_sha256='d'*64,
            assets_identity_sha256='e'*64,streams={'cf':{'lock':{'stream_sha256':'f'*64}}})
        def apply(engine, incoming, batch, tracker, cursor):
            self.assertEqual(len(incoming),100)
            engine.batch = batch; engine.weight.add_(1)
            return dict(batch=batch,method='MEMIT',requests=100,history_appends_expected=0)
        def factual(model,tok,manifest,incoming,dataset,out,endpoint,*args,**kwargs):
            path=out/(endpoint+'.json')
            value={'summary':{'requests':len(incoming)},'cases':incoming,'identity_sha256':'f'*64}
            write_new(path,value)
            return value,run.member(path)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); out=root/'MEMIT'; out.mkdir()
            w0=root/'W0.json'; write_new(w0, {'CPU_FIXTURE':True})
            ready_path=root/'READY.json'; write_new(ready_path, {'CPU_FIXTURE':True})
            manifest['W0_cf_ready_path']=str(ready_path)
            ready=dict(factual=run.member(w0),generation_READY=None)
            with patch.object(run,'LOCAL',root), patch.object(run,'read_w0',return_value=(ready,None)), \
                 patch.object(run,'native_apply',side_effect=apply),patch.object(run,'factual',side_effect=factual), \
                 patch.object(run,'current_subset',side_effect=lambda value,*_:value), \
                 patch.object(run,'evaluate_payload',return_value={}), \
                 patch.object(run.generation,'observe',side_effect=AssertionError('generation forbidden')):
                result=run.chain(None,None,engine,manifest,records,'cf',out,SimpleNamespace(log=lambda _:True))
            self.assertEqual(len(result['commits']),20)
            self.assertEqual(result['generation_status'],'DEFERRED_NOT_MEASURED')
            self.assertTrue(result['checkpoint_evaluation_consumer_pending'])
            payload=checkpoint.load(out/'checkpoints',result['checkpoint_identity'])
            self.assertEqual(payload['batch'],20)
            self.assertTrue(torch.equal(payload['weights']['weight'],torch.full((2,2),20.)))
            self.assertEqual(len(list((out/'checkpoints').glob('batch-*.pt'))),1)
            self.assertNotIn('generation_W20',payload['evaluation_cursor'])


if __name__=='__main__':
    unittest.main()
