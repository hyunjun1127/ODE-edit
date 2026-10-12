import ast
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
from official.diagnostics.fe_qwen_current import author_basic_eot_benchmark as a
from official.diagnostics.fe_qwen_current import basic_eot_benchmark as old
from official.runners.server1.common import member
from official.tracking.schema import config as validate_config
from official.experiments.prepare import write_new


class Tests(unittest.TestCase):
    def setUp(self):
        root=Path(os.environ.get('FE_NATIVE_AUTHOR_ROOT','/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011/author'))
        if not root.exists():self.skipTest('pinned author YAML unavailable')
        self.b={'author_yaml':{'algs':member(root/'configs/algs/memit.yaml'),'llms':member(root/'configs/llms/qwen2.5-7b.yaml')}}
        self.profiles=a.author_profiles(self.b)
        self.historical=dict(model_dtype='bfloat16',bs=100,num_edits=2000,seed=0,cov_mode='default',lti=False,negetive_prompt_test=False,
            cache_dir='/unchanged/C0',data='old-heldout-label',**copy.deepcopy(self.profiles))
        self.historical['algs']['L2']=10

    def test_real_pinned_author_all_fields_actual_cfg_zero(self):
        before=copy.deepcopy(self.historical);cfg,receipt=a.execution_config(self.historical,self.b)
        self.assertEqual(self.historical,before);self.assertEqual(cfg.algs.L2,0)
        self.assertEqual(a.audit_settings(cfg,self.profiles)['llms'],self.profiles['llms'])
        self.assertEqual(cfg.llms.layers,[4,5,6,7,8]);self.assertEqual(cfg.llms.clamp_norm_factor,1)
        self.assertEqual(cfg.llms.v_lr,.5);self.assertEqual(cfg.llms.v_num_grad_steps,35)
        self.assertEqual(cfg.llms.v_weight_decay,.001);self.assertEqual(cfg.llms.kl_factor,.0625)
        self.assertEqual([float(x) for x in cfg.llms.mom2_update_weight],[15000]*5)
        self.assertEqual(receipt['historical_changes'],{'algs':{'L2':{'before':10,'after':0}},'llms':{}})
        self.assertFalse(receipt['selection_required']);self.assertEqual(cfg.cache_dir,before['cache_dir'])
        for group,key,value in [('algs','L2',10),('algs','add_old_keys',False),('llms','clamp_norm_factor',.5),('llms','v_lr',.1)]:
            bad=copy.deepcopy(cfg);bad[group][key]=value
            with self.assertRaises(ValueError):a.audit_settings(bad,self.profiles)

    def test_yaml_content_tamper_rejected_even_fresh_member(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'algs.json';write_new(p,dict(name='memit',add_old_keys=True,L2=10))
            b=copy.deepcopy(self.b);b['author_yaml']['algs']=member(p)
            with self.assertRaises(ValueError):a.author_profiles(b)

    def test_identity_old10_and_old_stage_rejected(self):
        identity=dict(stage='AUTHOR_BASIC_EOT_MAIN',input_policy=old.POLICY,L2=0,clamp_norm_factor=1.,no_mask=True,z_reference='batch_entry')
        a.settings(identity)
        for k,v in (('L2',10),('stage','BASIC_EOT_FINAL_MAIN'),('z_reference','W0'),('no_mask',False)):
            bad=dict(identity);bad[k]=v
            with self.assertRaises(ValueError):a.settings(bad)

    def test_same_native_evaluator_writer_checkpoint_callbacks(self):
        for name in ('evaluate','targets','write','snapshot'):
            self.assertIs(getattr(a.AuthorObserver,name),getattr(old.FinalObserver,name))
        # No old-source monkeypatch or AST rewrite: same c36 adapter object.
        self.assertIs(a.eot.InputAdapter,old.eot.InputAdapter)

    def test_no_selection_or_restore_in_actual_run_route(self):
        import inspect
        tree=ast.parse(inspect.getsource(a.run));calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ('audit_heldout','restore','load') for n in calls))
        self.assertIn('native.apply_memit_to_model(model,tok,native_records(stream),cfg)',inspect.getsource(a.run))
        self.assertLess(inspect.getsource(a.run).index('execution_config('),inspect.getsource(a.run).index('AutoModelForCausalLM'))
        # Missing all old selection members is valid configuration construction.
        with patch.object(old,'audit_heldout',side_effect=AssertionError('OLD_GATE')):
            cfg,_=a.execution_config(self.historical,self.b);self.assertEqual(cfg.algs.L2,0)

    def test_distinct_dataset_tracking_and_collector_identity(self):
        b=dict(execution=dict(server='rent',attempt='author-r1',source_commit='a'*40,job_id='121259',source_run_id='working-demo'),main_stream=dict(sha256='b'*64))
        for dataset in ('cf','zsre'):
            b['dataset']=dataset
            values=a.tracking(b,{}, {'ODEEDIT_RENT_JOB_ID':'121259'});validate_config(values)
            self.assertIn('L2-0',values['arm']);self.assertEqual(values['evaluation_profile'],a.PROFILE)
            self.assertEqual('generation_schedule' in values,dataset=='cf')
            with patch.dict(os.environ,{'ODEEDIT_RENT_JOB_ID':'122001'}):
                self.assertEqual(a.tracking(b,{}, {'ODEEDIT_RENT_JOB_ID':'121259'}),values)


if __name__=='__main__':unittest.main()
