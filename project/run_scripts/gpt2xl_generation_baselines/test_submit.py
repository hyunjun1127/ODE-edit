import shlex
import unittest
from pathlib import Path
from .submit import order,arguments,launcher,ARMS,SOURCE_ENV,dependencies,width

class Submission(unittest.TestCase):
    def test_real_six_lane_graph_and_W0_gate(self):
        for cap in (1,2):
            ids={};jobs=[];dag=order(cap)
            for i,role in enumerate(ARMS):
                ids[role]=str(i+1)
                dep=[ids[x] for x in dag[role]]
                jobs.append(dict(job=ids[role],gpus=1,resource_detail='Dependency='+('afterany:'+':'.join(dep) if dep else '(null)')+' '))
            self.assertEqual(width(jobs),cap)
        self.assertEqual(order(2)['BASE_ALPHAEDIT'],['BASE_MEMIT'])
        self.assertEqual(dependencies(jobs),['5','6'])
    def test_existing_frontier_preserved(self):
        jobs=[dict(job=str(i),gpus=1,resource_detail='Dependency='+d+' ') for i,d in
            [(1,'(null)'),(2,'(null)'),(3,'afterany:1'),(4,'afterany:2')]]
        self.assertEqual(dependencies(jobs),['3','4'])
        self.assertEqual(width(jobs),2)
    def test_actual_launchers_no_fake_job_or_secrets(self):
        r=dict(cpu=8,collector_cpu=8,host_mib=65536,collector_host_mib=24576,wall='2-00:00:00',collector_wall='04:00:00')
        args=arguments('BASE_ALPHAEDIT',['100'],Path('/attempt'),r)
        for value in ('--hold','--export=NONE','--no-requeue','--gres=gpu:1','--dependency=afterany:100'):self.assertIn(value,args)
        self.assertNotIn('--gres=gpu:1',arguments('collector',['100'],Path('/attempt'),r))
        script=launcher(Path('/source'),'a'*40,'CAKE',Path('/raid/attempt'))
        for value in ('TMPDIR=/raid/attempt/tmp/CAKE',SOURCE_ENV,'HF_HUB_OFFLINE=1'):self.assertIn(value,script)
        for value in ('WANDB_API_KEY','WANDB_DISABLED','SLURM_JOB_ID='):self.assertNotIn(value,script)
    def test_sidecar_forwards_only_approved_TMPDIR(self):
        from project.run_scripts.experiment_tracking import client
        import inspect
        source=inspect.getsource(client.Tracker.__init__)
        self.assertIn("'TMPDIR'",source)
        self.assertNotIn('env=os.environ.copy()',source)
    def test_compile_actual_production(self):
        for path in Path(__file__).parent.glob('*.py'):compile(path.read_bytes(),str(path),'exec')

if __name__=='__main__':unittest.main()
