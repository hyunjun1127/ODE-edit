"""Metadata-only CPU regressions, no scheduler mutation or model import."""
import copy
import unittest
from pathlib import Path
from .resume_registration import config_diff,SCIENCE,NEW,RECALL
from .submit import order,ARMS,arguments,launcher,width,SOURCE_ENV

class Registration(unittest.TestCase):
    def test_control_only_changes(self):
        old=dict(attempt='old',run_instance={'attempt':'old'},science=dict(lr=.5,steps=20))
        new=copy.deepcopy(old);new.update(attempt='new',run_instance={'attempt':'new'},registration_recall={'nonce':RECALL})
        self.assertEqual(config_diff(old,new),['attempt','registration_recall','run_instance'])
        new['science']['lr']=.1
        with self.assertRaises(Exception):config_diff(old,new)
    def test_frozen_identity_new_control_path(self):
        script=launcher(NEW/'source',SCIENCE,'BASE_MEMIT',NEW)
        self.assertIn(SOURCE_ENV+'='+SCIENCE,script)
        self.assertIn(str(NEW),script)
        for denied in ('SLURM_JOB_ID=','WANDB_API_KEY','resume_registration'):self.assertNotIn(denied,script)
    def test_graph_combined_frontier(self):
        for cap in (1,2):
            jobs=[dict(job='10',gpus=1,resource_detail='Dependency=(null) ')]
            ids={};dag=order(cap)
            for i,arm in enumerate(ARMS):
                ids[arm]=str(100+i);dep=['10']+[ids[x] for x in dag[arm]]
                jobs.append(dict(job=ids[arm],gpus=1,resource_detail='Dependency=afterany:'+':'.join(dep)+' '))
            self.assertEqual(width(jobs),cap)
    def test_held_actual_source_and_privacy(self):
        r=dict(cpu=8,collector_cpu=8,host_mib=65536,collector_host_mib=24576,wall='2-00:00:00',collector_wall='04:00:00')
        argv=arguments('BASE_MEMIT',[],NEW,r)
        for expected in ('--hold','--export=NONE','--no-requeue','--gres=gpu:1','--chdir='+str(NEW/'source')):self.assertIn(expected,argv)
        self.assertNotIn('--gres=gpu:1',arguments('collector',['1','2'],NEW,r))
    def test_compile_control_only(self):
        path=Path(__file__).with_name('resume_registration.py');compile(path.read_bytes(),str(path),'exec')

if __name__=='__main__':unittest.main()
