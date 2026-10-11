"""CPU control-path tests, no model/GPU or scheduler mutations."""
import ast,unittest
from pathlib import Path
from official.runners.server2 import fe_qwen_working_submit as s
class Tests(unittest.TestCase):
 def test_exact_nonce(self):
  self.assertEqual(s.NONCE,'USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1')
 def test_native_launch_not_old_runner(self):
  source=Path(s.__file__).read_text()
  self.assertIn('official.diagnostics.fe_qwen_current.launch',source)
  self.assertNotIn('official.runners.fe_original',source)
 def test_explicit_four_arms(self):
  source=Path(s.__file__).read_text()
  self.assertIn("==[0,.1,1,10]",source)
  self.assertIn('FRONTIER_CHANGED_REQUIRES_EXACT_RESOURCE_PLAN',source)
 def test_no_global_cap_or_existing_job_mutation(self):
  tree=ast.parse(Path(s.__file__).read_text())
  strings=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
  for forbidden in ('scancel','requeue','hold','update','sacctmgr modify'):
   self.assertNotIn(forbidden,strings)
  self.assertIn('--hold',strings);self.assertIn('--no-requeue',strings)
 def test_physical_memory_preserved(self):
  source=Path(s.__file__).read_text()
  for value in ('--mem=60416M','--cpus-per-task=8','--gres=gpu:a6000:1','slurm_memory_policy.py'):
   self.assertIn(value,source)
 def test_capacity_arithmetic(self):
  self.assertEqual(7*7890010112+4*1024**3+32*1024**3,93884776448)
  self.assertEqual(5*7890010112+4*1024**3+32*1024**3,78104756224)
if __name__=='__main__':unittest.main()
