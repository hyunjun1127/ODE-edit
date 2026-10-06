"""Narrow W0 mapping/count/axis/identity checks; no model/network/CUDA."""
import ast
import inspect
import unittest
from . import gptj_server2 as run
from . import gptj_server2_control as control
from project.run_scripts.experiment_tracking.schema import metrics, bind_job_identity, run_name
from project.run_scripts.jlz_realization.observe import reduce_rows


class W0Tests(unittest.TestCase):
    def summary(self):
        return {k:dict(denominator=n,numerator=n//2,rate=.5,strict_denominator=n,
                strict_numerator=n//4,desired_token_count=2*n,desired_token_correct=n,
                prompt_macro=.5,true_nll_mean=2.,new_nll_mean=1.) for k,n in run.COUNTS.items()}

    def test_mapping_axes(self):
        p=run.payload(self.summary());self.assertEqual(metrics(p,scientific=True),p)
        self.assertEqual(p['W0_first2000/success_harmonic_pct'],50.)
        self.assertEqual(p['W0_first2000/N/count'],20000)
        self.assertEqual(p['pre_state_edits'],0)
        self.assertFalse(any(k.startswith(('fit/','current/','all_seen/')) for k in p))

    def test_bad_counts(self):
        g=self.summary();g['R']['denominator']=100
        with self.assertRaises(ValueError):run.payload(g)

    def test_N_desired_true_tie_failure(self):
        r=dict(identity='x',kind='N',true_nll=1.,new_nll=1.,true_token_count=2,
            true_token_correct=2,true_strict=True,new_token_count=1,new_token_correct=0,new_strict=False)
        g=reduce_rows([r])['N'];self.assertEqual(g['numerator'],0);self.assertEqual(g['token_micro'],1.)

    def test_job_array_zero(self):
        c=dict(server='server2',task_id=run.TASK,arm='W0_BASE_MODEL',attempt='r1',source_sha='a'*40,
               config_sha='b'*64,model='gptj',model_family='gptj',writer='none',role='scientific',metric_schema='price-first2k-scalar-v1')
        actual=bind_job_identity(c,{'SLURM_JOB_ID':'123','SLURM_ARRAY_JOB_ID':'120','SLURM_ARRAY_TASK_ID':'0','SLURM_STEP_ID':'-1'})
        self.assertTrue(run_name(actual).endswith('job120_0'));self.assertEqual(actual['job_id'],'123')

    def test_runner_no_edit_no_save(self):
        tree=ast.parse(inspect.getsource(run))
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(attrs)&{'backward','save','save_pretrained','copy_','add_'})
        self.assertNotIn('C0', inspect.getsource(run.run).split('model =')[0])
        self.assertIn('use_cache=False',inspect.getsource(run.W0Adapter))

    def test_launcher(self):
        from pathlib import Path
        script=control.launcher(Path('/tmp/fake-w0'),'main',{'runtime':{'python':'/usr/bin/python3'}})
        self.assertIn('OMP_NUM_THREADS=6',script);self.assertIn('HF_HUB_OFFLINE=1',script)
        self.assertIn('gptj_server2 run',script);self.assertNotIn('C0',script)
        self.assertIn('CUDA_VISIBLE_DEVICES=',control.launcher(Path('/tmp/fake-w0'),'collector',{'runtime':{'python':'/usr/bin/python3'}}))


if __name__=='__main__':unittest.main()
