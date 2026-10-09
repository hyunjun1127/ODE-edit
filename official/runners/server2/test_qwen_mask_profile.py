import ast
from copy import deepcopy
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from official.experiments.prepare import write_new,file_sha
from official.runners.server2 import qwen_mask_profile as profile,qwen_run,qwen_pipeline

class MaskTests(unittest.TestCase):
    def test_exact_eight_no_native_change(self):
        from official.runners.server2.qwen_plan import rows
        canonical={r['logical_main_row']:r['config'] for r in rows()}
        self.assertEqual(len(profile.rows()),8)
        for r in profile.rows():
            c=r['config'];self.assertEqual(qwen_run.validate_config(c),c)
            stripped={k:v for k,v in c.items() if k not in ('mask_repair_instruction','context_generator_sha256','generation_schedule','config_sha256')}
            self.assertEqual(stripped,{k:v for k,v in canonical[r['logical_main_row']].items() if k!='config_sha256'})
            bad=deepcopy(c);bad['hparams']['v_lr']=.123
            with self.assertRaises(ValueError):qwen_run.validate_config(bad)
            self.assertEqual(profile.deferred(c),c['dataset']=='cf')

    def test_actual_tracker_schema_deferred(self):
        from official.tracking.test_transport import execute,official
        with patch.dict('os.environ',{'ODEEDIT_WANDB_ENV_FILE':'fixture','ODEEDIT_ATTEMPT_ID':'fixture'}),patch('official.tracking.init') as init:
            qwen_run._tracker('/tmp/fixture',arm='MEMIT',writer='MEMIT',dataset='cf',source_sha='a'*40,config_sha='b'*64,assets={},deferred=True)
            cfg=init.call_args.kwargs['config']
        self.assertEqual([k for k in cfg if k.startswith('generation')],['generation_schedule'])
        self.assertNotIn('reference_assets_sha256',cfg)
        sdk,receipts,bound=execute([official()],cfg=cfg)
        self.assertEqual(bound['generation_schedule'],profile.DEFERRED)
        self.assertEqual(receipts[-1]['method_readback']['status'],'REMOTE_BOUNDED_ROWS_VERIFIED')

    def test_final_cp_with_generation_unmeasured(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'raw.json';write_new(p,[{}]*2000)
            c=dict(checkpoint_identity={'fixture':True},checkpoint_sha256='a'*64,
                factual=dict(cases_path=str(p),cases_sha256=file_sha(p)),generation=None)
            identity,evidence=qwen_pipeline.final_evidence(tmp,[c]*20,{'sha256':'a'*64},'cf',True)
            self.assertEqual(set(evidence),{'factual'})
            with self.assertRaises(TypeError):qwen_pipeline.final_evidence(tmp,[c]*20,{'sha256':'a'*64},'cf',False)

    def test_generator_source_and_native_routes(self):
        from official.baselines import registry
        from official.baselines.easyedit.util import generate
        self.assertEqual(file_sha(generate.__file__),profile.GENERATOR_SHA)
        for method in profile.METHODS:
            module,_,_=registry.implementation(method,'qwen25')
            # Qwen SPHERE registry selects EasyEdit SPHERE, not the separate
            # sphere package used by Llama MEMIT.
            expected='official.baselines.easyedit.util.generate'
            self.assertEqual(module.generate_fast.__module__,expected)
            native=SimpleNamespace(context_snapshot=lambda:None)
            profile.cold_guard(next(r['config'] for r in profile.rows() if r['config']['method']==method),native,False)
            with self.assertRaises(ValueError):profile.cold_guard(profile.rows()[0]['config'],native,True)

    def test_generation_call_is_under_deferred_guard(self):
        tree=ast.parse(inspect.getsource(qwen_run.execute))
        enclosing=[n for n in ast.walk(tree) if isinstance(n,ast.If) and 'DEFERRED_CHECKPOINT_EVALUATION' in ast.unparse(n.test)]
        self.assertEqual(len(enclosing),1)
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_generation' for n in ast.walk(enclosing[0])))

    def test_ft_uses_shared_restore_eval_only(self):
        from official.runners.server2 import qwen_mask_ft_eval as ft
        source=inspect.getsource(ft.execute)
        self.assertIn('load_and_restore',source)
        self.assertNotIn('native.apply',source)
        self.assertNotIn('checkpoint.save',source)
        self.assertEqual(ft.COUNTS,dict(rewrite=6691,paraphrase=6691,neighborhood=11476))

if __name__=='__main__': unittest.main()
