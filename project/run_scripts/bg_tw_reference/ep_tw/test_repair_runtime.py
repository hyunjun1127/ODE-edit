"""CPU fail-closed and early durability tests; not actual Llama parity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from .control import identity,save
from .repair_runtime import recorder,validate_repair_pass,config_from_lock,verify_members
from .repair_checks import RepairNumerics

class RuntimeGuards(unittest.TestCase):
    def setup_pass(self,root):
        lock={'source_head':'head','source_tree':'tree','source_archive':{},'model_revision':'revision',
              'teacher_manifest':{},'repair_numerics':json.loads(json.dumps(RepairNumerics().to_dict())),
              'repair_pass_path':str(root/'pass.json')}
        save(root/'science.json',lock);lock['_lock_path']=str(root/'science.json')
        check={obj:dict(direct={'status':'PASS'},fd={'status':'PASS','directions':{
            'self_gradient':{'status':'PASS'},'independent':{'status':'PASS'}}}) for obj in ('E','D')}
        check['status']='MODEL_TECHNICAL_CHECKS_PASS_NOT_G0'
        result={k:lock[k] for k in ('source_head','source_tree','source_archive','model_revision','teacher_manifest','repair_numerics')}
        result.update(status='SAVED_EPISODE_TECHNICAL_PASS_NOT_G0',checks=check,
            scientific_lock=identity(root/'science.json'),native_target_calls=0,native_solves=0,commits=0)
        return lock,result

    def test_missing_technical_receipt_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock,_=self.setup_pass(Path(tmp))
            with self.assertRaises(FileNotFoundError):validate_repair_pass(lock)

    def test_exact_pass_only_and_science_lock_drift_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);lock,result=self.setup_pass(p);save(p/'pass.json',result)
            self.assertEqual(validate_repair_pass(lock)[0]['commits'],0)
            with (p/'science.json').open('a') as f:f.write(' ')
            with self.assertRaises(AssertionError):validate_repair_pass(lock)

    def test_E_only_or_direction_unresolved_blocks(self):
        for field in ('D','independent'):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp);lock,result=self.setup_pass(p)
                if field=='D':result['checks']['D']['fd']['status']='UNRESOLVED'
                else:result['checks']['E']['fd']['directions']['independent']['status']='UNRESOLVED'
                save(p/'pass.json',result)
                with self.assertRaises(AssertionError):validate_repair_pass(lock)

    def test_early_tensor_saved_and_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);record=recorder(p);value=torch.arange(6,dtype=torch.float32).reshape(2,3)
            record('E',{'gradient':value,'E0':.2})
            member=torch.load(p/'E.pt',weights_only=True)
            self.assertTrue(torch.equal(member['payload/gradient'],value))
            with self.assertRaises(FileExistsError):record('E',{'gradient':value})

    def test_numerics_roundtrip_and_relaxation_block(self):
        lock={'repair_numerics':json.loads(json.dumps(RepairNumerics().to_dict()))}
        self.assertEqual(config_from_lock(lock).derivative_relative_tolerance,.15)
        lock['repair_numerics']['derivative_relative_tolerance']=.16
        with self.assertRaises(RuntimeError):config_from_lock(lock)

    def test_runtime_has_no_native_import_and_conditional_shell(self):
        root=Path(__file__).parent
        text=(root/'repair_runtime.py').read_text()
        self.assertNotIn('import NativeSingletonFitter',text)
        self.assertNotIn('import FrozenNativeMap',text)
        shell=(root/'repair.sbatch').read_text()
        self.assertIn('set -euo pipefail',shell)
        self.assertLess(shell.index('ep_tw.repair_runtime'),shell.index('ep_tw.runner'))
        self.assertNotIn('afterany',shell)
        runner=(root/'runner.py').read_text()
        self.assertLess(runner.index('validate_repair_pass(lock)'),runner.index('AutoModelForCausalLM.from_pretrained'))

if __name__=='__main__':unittest.main()
