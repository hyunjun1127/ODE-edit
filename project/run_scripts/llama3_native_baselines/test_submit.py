"""CPU scheduler metadata/launcher fixtures, never calls Slurm or fits a model."""
import unittest
from pathlib import Path

from .common import METHODS, TASK
from .submit import arguments, launcher, order
from project.run_scripts.jlz_price_gptj.admission import width


class SubmitTests(unittest.TestCase):
    def setUp(self):
        self.attempt=Path('/task/attempt')
        self.r=dict(cpu=8,collector_cpu=8,host_mib=59392,collector_host_mib=24576,
                    wall='2-00:00:00',collector_wall='04:00:00',qos='lab_gpu_s4')

    def test_exact_graph_width_and_common_W0(self):
        for parallel in (1,2,3):
            graph=order(parallel);jobs=[];ids={m:str(i+100) for i,m in enumerate(METHODS)}
            self.assertEqual(graph['MEMIT'],[])
            self.assertEqual(graph['collector'],list(METHODS))
            for method in METHODS:
                parents=graph[method]
                self.assertTrue(method=='MEMIT' or parents)
                jobs.append(dict(job=ids[method],gpus=1,resource_detail='Dependency='+
                    ('afterany:'+':'.join(ids[p] for p in parents) if parents else '(null)')+' '))
            self.assertEqual(width(jobs),parallel)

    def test_argv_exact_export_memory_gpu_dependency(self):
        argv=arguments('CAKE','afterany:123:124',self.attempt,self.r)
        for item in ('--hold','--export=NONE','--no-requeue','--gres=gpu:1',
                     '--mem=59392M','--dependency=afterany:123:124',
                     '--job-name='+TASK+'-CAKE'):
            self.assertIn(item,argv)
        cpu=arguments('collector','afterany:123',self.attempt,self.r)
        self.assertFalse(any('gres' in x for x in cpu))
        self.assertIn('--mem=24576M',cpu)

    def test_launcher_science_pin_and_private_nltk(self):
        value=launcher(Path('/sealed source'),'a'*40,'MEMIT',self.attempt,self.r,
                       dict(nltk_data='/private/nltk_data'))
        self.assertIn('NLTK_DATA=/private/nltk_data',value)
        self.assertIn('HF_HUB_OFFLINE=1',value)
        self.assertIn('--method MEMIT',value)
        self.assertNotIn('API_KEY',value)
        self.assertNotIn('WANDB_DISABLED',value)
        cpu=launcher(Path('/source'),'a'*40,'collector',self.attempt,self.r)
        self.assertIn("CUDA_VISIBLE_DEVICES=''",cpu)
        self.assertNotIn('--method',cpu)

    def test_bad_width_rejected(self):
        for bad in (0,4):
            with self.assertRaises(RuntimeError): order(bad)


if __name__=='__main__': unittest.main()
