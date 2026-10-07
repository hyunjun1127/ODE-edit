import shlex
import unittest
from pathlib import Path
from .submit import dependencies,arguments,launcher,ARMS,SOURCE_ENV
from project.run_scripts.jlz_price_gpt2xl.admission import width

class Submission(unittest.TestCase):
    def test_production_modules_compile(self):
        for path in Path(__file__).parent.glob('*.py'):compile(path.read_bytes(),str(path),'exec')
    def test_frontier_and_parallel_cap(self):
        jobs=[dict(job=str(j),gpus=1,resource_detail='Dependency='+d+' ') for j,d in
            [(1,'(null)'),(2,'(null)'),(3,'afterany:1'),(4,'afterany:2')]]
        self.assertEqual(dependencies(jobs),['3','4']);self.assertEqual(width(jobs),2)
        for j in (5,6):jobs.append(dict(job=str(j),gpus=1,resource_detail='Dependency=afterany:3:4 '))
        self.assertEqual(width(jobs),2)
    def test_stricter_serial_and_unknown_safe(self):
        jobs=[dict(job='1',gpus=1,resource_detail='Dependency=(null) '),
              dict(job='2',gpus=1,resource_detail='Dependency=afterany:1 ')]
        for j,d in [('3','2'),('4','2:3')]:jobs.append(dict(job=j,gpus=1,resource_detail='Dependency=afterany:'+d+' '))
        self.assertEqual(width(jobs),1)
        self.assertEqual(dependencies([dict(job='1',resource_detail='Dependency=afterok:9 ')]),['1'])
    def test_exact_resource_privacy_script(self):
        r=dict(cpu=8,collector_cpu=8,host_mib=65536,collector_host_mib=24576,wall='2-00:00:00',collector_wall='04:00:00')
        argv=arguments(ARMS[1],['123'],Path('/task'),r)
        self.assertIn('--dependency=afterany:123',argv);self.assertIn('--export=NONE',argv)
        self.assertIn('--no-requeue',argv);self.assertIn('--gres=gpu:1',argv)
        cpu=arguments('collector',['123','124'],Path('/task'),r);self.assertNotIn('--gres=gpu:1',cpu)
        text=launcher(Path('/source'),'a'*40,ARMS[1],Path('/task'))
        self.assertIn(SOURCE_ENV,text);self.assertIn('HF_HUB_OFFLINE=1',text)
        self.assertNotIn('WANDB_API_KEY',text);self.assertNotIn('WANDB_DISABLED',text)

if __name__=='__main__':unittest.main()
