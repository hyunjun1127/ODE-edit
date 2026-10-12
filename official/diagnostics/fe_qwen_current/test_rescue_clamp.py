import ast
import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from omegaconf import OmegaConf
from official.experiments.prepare import digest,write_new
from official.runners.server1.common import member
from official.diagnostics.fe_qwen_current import rescue_clamp as c,rescue as old,prepare,launch,observer
from official.diagnostics.fe_qwen_current import rescue_clamp_observer as obs
from official.diagnostics.fe_qwen_current import test_rescue
from official.tracking.schema import config as validate,bind_job_identity


class ClampTests(unittest.TestCase):
    def cfg(self):
        return dict(L2=100,clamp_norm_factor=.5,author_hparam_profile=c.SPEC['protocol'],
            selection_contract_sha256=digest(c.SPEC),server='rent',attempt='r1',config_sha256='a'*64,
            stream=dict(sha256='b'*64))
    def test_one_knob_new_contract_and_old_unchanged(self):
        previous=copy.deepcopy(old.SPEC)
        self.assertEqual(c.SPEC['L2_arms'],[100]);self.assertEqual(c.SPEC['clamp'],.5)
        self.assertNotEqual(digest(c.SPEC),digest(old.SPEC))
        with c.bindings():self.assertEqual(prepare.SPEC['promotable_L2'],[100])
        self.assertEqual(old.SPEC,previous)
        for key in ('layers','steps','lr','weight_decay','C0_weight','dtype','z_reference','early_stop','gate'):
            self.assertEqual(c.SPEC[key],previous[key])
    def test_old_clamp_or_relabelled_proof_rejected_before_result_read(self):
        for cfg in (dict(self.cfg(),clamp_norm_factor=1),dict(self.cfg(),selection_contract_sha256=digest(old.SPEC)),
                    dict(self.cfg(),author_hparam_profile=old.SPEC['protocol']),dict(self.cfg(),L2=10)):
            with self.assertRaisesRegex(ValueError,'CLAMP_RESCUE_PROFILE'):c.validate_profile(cfg)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_new(root/'c.json',dict(self.cfg(),clamp_norm_factor=1))
            write_new(root/'lock.json',dict(configs=[member(root/'c.json')]))
            with self.assertRaisesRegex(ValueError,'CLAMP_RESCUE_PROFILE'):c.collect(root)
    def test_actual_launch_tail_has_one_explicit_clamp_and_native_L2(self):
        tree=ast.parse(Path(launch.__file__).read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='launch')
        start=next(i for i,x in enumerate(fn.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in x.targets))
        code=compile(ast.fix_missing_locations(ast.Module(body=fn.body[start:],type_ignores=[])),'actual-launch-tail','exec')
        observed=[]
        namespace=dict(SPEC=c.SPEC,c=dict(self.cfg(),native_data=dict(path='/new/data/heldout-native.json')),
            lock=dict(python='/runtime/python',author_path='/new/author'),base=Path('/new'),out=Path('/new/run'),
            config='/new/config.json',Path=Path,env={},os=SimpleNamespace(chdir=lambda _:None,execve=lambda *a:observed.append(a)))
        exec(code,namespace);argv=observed[0][1]
        self.assertEqual(argv.count('llms.clamp_norm_factor=0.5'),1);self.assertIn('algs.L2=100',argv)
        self.assertIn('+diagnostic_z_reference=batch_entry',argv)
        self.assertFalse(any('dtype=' in a for a in argv))
        namespace['c']['clamp_norm_factor']=1
        with self.assertRaisesRegex(ValueError,'CLAMP_PROFILE'):exec(code,namespace)
    @unittest.skipUnless(os.environ.get('FE_NATIVE_AUTHOR_ROOT'),'native YAML required')
    def test_actual_native_yaml_override_changes_clamp_only(self):
        path=Path(os.environ['FE_NATIVE_AUTHOR_ROOT'])/'configs/llms/qwen2.5-7b.yaml'
        before=OmegaConf.load(path)
        after=OmegaConf.merge(OmegaConf.create(dict(llms=before)),OmegaConf.from_dotlist(['llms.clamp_norm_factor=0.5']))['llms']
        changed=[k for k in before if before[k]!=after[k]]
        self.assertEqual(changed,['clamp_norm_factor']);self.assertEqual(before.clamp_norm_factor,1)
        self.assertEqual(after.clamp_norm_factor,.5)
    def test_actual_native_state_guard_uses_new_clamp(self):
        tree=ast.parse(Path(observer.__file__).read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_state')
        assignment=next(x for x in fn.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected' for t in x.targets))
        namespace=dict(SPEC=c.SPEC);exec(compile(ast.Module(body=[assignment],type_ignores=[]),'actual-state-guard','exec'),namespace)
        self.assertEqual(namespace['expected']['clamp_norm_factor'],.5)
    def test_new_group_binds_both_knobs(self):
        with patch.dict(os.environ,{},clear=True):
            values=bind_job_identity(c.tracking_config(self.cfg(),'c'*40),{'ODEEDIT_RENT_JOB_ID':'120950'})
        validate(values)
        for key in ('task_id','arm','baseline','evaluation_profile'):self.assertIn('0p5',values[key])
        self.assertIn('100',values['arm'])
    def test_quality_classification_reuses_committed_stop_even_process_exit1(self):
        # Executes shared stop fixture under the actual new wrapper bindings.
        with patch.object(obs.shared,'SPEC',c.SPEC),patch.object(obs.shared,'tracking_config',c.tracking_config):
            test_rescue.RescueTests('test_stop_after_commit_preserves_checkpoint_and_not_failure').test_stop_after_commit_preserves_checkpoint_and_not_failure()


if __name__=='__main__':unittest.main()
