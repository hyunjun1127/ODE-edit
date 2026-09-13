"""CPU-only integration checks, never invoking a launcher/model/evaluator."""
import hashlib
import importlib
import inspect
from pathlib import Path
import subprocess
import unittest

class IntegrationChecks(unittest.TestCase):
    def test_existing_e01_source_preserved(self):
        for name in ('README.md','continuation.py','warm_plan.py','terminal_performance.py'):
            p='project/run_scripts/baseline_mechanism_first/'+name
            self.assertEqual(Path(p).read_bytes(),subprocess.check_output(['git','show','7d5bae2e:'+p]))

    def test_e01_observation_resume_is_original_source(self):
        for name in ('endpoint_observation_resume.py','prepare_fullseen_resume.py'):
            p='project/run_scripts/baseline_mechanism_first/'+name
            self.assertEqual(Path(p).read_bytes(),subprocess.check_output(['git','show','58f50a25:'+p]))

    def test_pdz_analysis_import_and_role_mapping(self):
        from project.run_scripts.ode_bf import p1r54_pdz_time_sweep_factual_analysis as a
        from project.run_scripts.ode_bf.p1r54_pdz_ablation_b100 import ROLES,RESULT_NAMES
        from project.run_scripts.ode_bf.p1r52_sequential_runtime import expected_p1r52_sequential_result_name
        from project.run_scripts.ode_bf.p1r52_sequential_runtime import P1R52_B100X10_SCALE
        self.assertEqual(a.SOURCE_HEAD,'c0dc8e3fcd1ea2c941441dd2dc5cfa91a28aa296')
        for role in ROLES:
            self.assertEqual(expected_p1r52_sequential_result_name('llama3-8b-inst',role,scale=P1R52_B100X10_SCALE),RESULT_NAMES[role])

    def test_optional_subcycle_callback_preserves_default(self):
        from project.run_scripts.ode_bf.p1_scalable_batched_experiment import _run_ode_arm
        from project.run_scripts.ode_bf.p1r52_joint_pc_fp32_runtime import _fp32_target_and_j0
        from project.run_scripts.ode_bf.p1r52_target_timescale_b100 import run_target_timescale_b100
        for f,key in ((_run_ode_arm,'p1r52_target_subcycle_runner'),(_fp32_target_and_j0,'target_subcycle_runner'),(run_target_timescale_b100,'target_subcycle_runner')):
            self.assertIsNone(inspect.signature(f).parameters[key].default)

    def test_sh1_a_source_matches_original(self):
        for p in Path('project/run_scripts/multilayer_joint_compensation').rglob('*.py'):
            if p.name=='aos_review.py' or 'aos_review' in p.name:
                continue
            old=subprocess.run(['git','show','ba91f274:'+str(p)],capture_output=True)
            if old.returncode==0:
                self.assertEqual(p.read_bytes(),old.stdout,str(p))

    def test_s1_analyzer_helpers_and_cli(self):
        from project.run_scripts.barrier_guided_ode import s1_analyzer as a
        self.assertEqual(a._vector_l2([1.,2.],[4.,6.]),5.)
        self.assertEqual(a._sum_nodes([{'x':{'y':2}}, {'x':{'y':3}}],('x','y')),5.)
        self.assertIn(b'x',a._csv_bytes([{'x':1}]))

if __name__=='__main__':unittest.main()
