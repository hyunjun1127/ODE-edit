"""CPU launcher/frontier protocol only, never calls sbatch or GPU."""
import unittest
from pathlib import Path
from .generation_submit import launcher,sbatch_argv,inspect_held
from .generation_plan import dependencies
from .generation_common import ARMS,SOURCE_ENV,TASK
class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.a=Path('/fixture/attempt')
        self.c=dict(runtime={'python':'/fixture/python'},resources=dict(cpu=6,collector_cpu=6,
            host_mib=59392,collector_host_mib=24576,wall='2-00:00:00',collector_wall='04:00:00'))
    def test_explicit_six_arm_launcher_no_cache_or_secret(self):
        for role in (*ARMS,'collector'):
            text=launcher(self.a,role,self.c,'a'*40)
            self.assertIn(SOURCE_ENV+'='+'a'*40,text)
            self.assertIn('OMP_NUM_THREADS=6',text)
            self.assertNotIn('API_KEY',text)
            self.assertIn('generation_collect' if role=='collector' else 'generation_run',text)
            self.assertNotIn('save_model',text)
    def test_every_submit_held_resources_no_requeue_export_none(self):
        for role in (*ARMS,'collector'):
            argv=sbatch_argv(self.a,role,self.c,['11'])
            self.assertIn('--hold',argv);self.assertIn('--export=NONE',argv)
            self.assertIn('--no-requeue',argv);self.assertIn('--cpus-per-task=6',argv)
            self.assertIn('--dependency=afterany:11',argv)
            self.assertEqual('--gres=gpu:1' in argv,role!='collector')
    def test_seven_jobs_all_exact_afterany_two_lane_concurrency(self):
        jobs={}
        for i,role in enumerate(ARMS):
            dep=dependencies(role,['9','10'],jobs,2)
            if i==0:self.assertEqual(dep,['9','10'])
            else:self.assertEqual(len(dep),1)
            jobs[role]=str(100+i)
        self.assertEqual(dependencies('collector',['9','10'],jobs,2),list(jobs.values()))
if __name__=='__main__':unittest.main()

