"""Small CPU-only W0 mapping/transport regression. No online fixture uploads."""
import ast
import inspect
import unittest
from pathlib import Path
from .gptj_server2_cohort_metrics import curves
from .gptj_server2_cohort_tracking import schema as s
from .gptj_server2_cohort_tracking.worker import session
from . import gptj_server2_cohort as runner
from . import gptj_server2_cohort_control as control
from project.run_scripts.experiment_tracking.test_tracking import SDK
from project.run_scripts.experiment_tracking.schema import metrics as general_metrics
from project.run_scripts.experiment_tracking.method import validate as general_validate

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[]
        for i in range(2000):
            for j,k in enumerate('RPP'+'N'*10):
                r=dict(ordinal=i,case_id=9000-i,identity=f'{i}:{j}',kind=k,new_nll=1. if i<100 else 3.,true_nll=2.)
                for label in ('new','true'):
                    n=1+i%3;c=n if i%2==0 else 0
                    r.update({label+'_token_count':n,label+'_token_correct':c,label+'_strict':c==n})
                cls.rows.append(r)
        cls.points=curves(cls.rows)
        cls.cfg=dict(server='server2',task_id=runner.TASK,arm='W0_BASE_MODEL',attempt='fixture',
            source_sha='a'*40,config_sha='b'*64,model='gptj',model_family='gptj',writer='none',role='scientific',
            metric_schema='price-first2k-scalar-v1',**s.EXTRA)
    def test_order_counts_micro(self):
        p=self.points;self.assertEqual(len(p),21)
        self.assertEqual(p[1]['current/post/R/success_count'],100)
        self.assertEqual(p[2]['current/post/R/success_count'],0)
        self.assertEqual(p[5]['all_seen/post/R/count'],500)
        self.assertNotIn('all_seen/post/R/count',p[4])
        rs=self.rows[:1300];rs=[r for r in rs if r['kind']=='R']
        ratio=100*sum(r['new_token_correct'] for r in rs)/sum(r['new_token_count'] for r in rs)
        self.assertEqual(p[1]['current/post/R/token_acc_pct'],ratio)
        self.assertEqual(p[1]['current/post/N/success_count'],0)
    def test_ordinal_not_case_threshold(self):
        broken=list(self.rows);broken[0]=dict(broken[0],ordinal=3)
        with self.assertRaises(RuntimeError):curves(broken)
    def test_reference_exception_narrow(self):
        self.assertEqual(s.metrics(self.points[1]),self.points[1])
        with self.assertRaisesRegex(ValueError,'POST_STATE_AXIS'):general_validate(self.points[1],True)
        for k in ('actual_model_edits','post_state_edits'):
            with self.assertRaises(ValueError):s.metrics(dict(self.points[1],**{k:100}))
        with self.assertRaises(ValueError):s.config(dict(self.cfg,reference_only=False))
        with self.assertRaises(ValueError):s.config(dict(self.cfg,task_id='other'))
        with self.assertRaises(ValueError):s.metrics(dict(self.points[1],prompt=1))
    def test_axes_order(self):
        a=s.AxisState()
        with self.assertRaises(ValueError):a.accept(self.points[1])
        for p in self.points:a.accept(p)
        with self.assertRaises(ValueError):a.accept(self.points[-1])
    def test_job_identity(self):
        c=s.bind_job_identity(self.cfg,dict(SLURM_JOB_ID='12',SLURM_ARRAY_JOB_ID='11',SLURM_ARRAY_TASK_ID='0',SLURM_STEP_ID='-1'))
        self.assertTrue(s.run_name(c).endswith('job11_0'))
        self.assertEqual(c['step_id'],'-1')
        with self.assertRaises(ValueError):s.bind_job_identity(dict(self.cfg,job_id='13'),{'SLURM_JOB_ID':'12'})
    def test_fake_production_full_readback(self):
        class Fake(SDK):
            def define_metric(self,*args,**kwargs):pass
            def log(self,v,step=None):self.points.append(dict(v,_step=len(self.points) if step is None else step))
        sdk=Fake();out=[]
        request=dict(config=s.bind_job_identity(self.cfg,{'SLURM_JOB_ID':'12'}),run_id='fixtureUUID',
                     base_url='https://api.wandb.ai',spool='/tmp/fake-only',smoke=False)
        commands=[dict(op='log',values=p,step=None) for p in self.points]+[dict(op='finish',exit_code=0)]
        session(sdk,request,commands,out.append)
        self.assertEqual(out[-1]['method_readback']['status'],'REMOTE_COHORT_CURVES_VERIFIED')
        self.assertEqual(out[-1]['method_readback']['current'],20)
        self.assertEqual(sdk.config['actual_model_edits'] if 'actual_model_edits' in sdk.config else 0,0)
        sdk=Fake();sdk.scan_history=lambda **kw:iter([]);out=[]
        session(sdk,request,commands,out.append)
        self.assertEqual(out[-1]['method_readback']['status'],'UNVERIFIED_INCOMPLETE_OR_UNAVAILABLE')
    def test_no_write_and_launcher(self):
        tree=ast.parse(inspect.getsource(runner))
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse(set(attrs)&{'backward','save','save_pretrained','copy_','add_'})
        launch=control.launcher(Path('/tmp/fixture'),'main',{'runtime':{'python':'/usr/bin/python3'}})
        self.assertIn('gptj_server2_cohort run',launch)
        self.assertIn('OMP_NUM_THREADS=6',launch)

if __name__=='__main__':
    import sys
    from project.run_scripts.jlz_realization.common import write,sha
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    root=Path(__file__).resolve().parents[3]
    files=sorted((root/'project/run_scripts/base_model_eval').glob('gptj_server2_cohort*.py'))
    files+=sorted((root/'project/run_scripts/base_model_eval/gptj_server2_cohort_tracking').glob('*.py'))
    write(control.LOCAL/'cpu-tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        GPU=0,online=False,owner_review=True,source_sha256={str(p.relative_to(root)):sha(p) for p in files}))
    sys.exit(0 if result.wasSuccessful() else 1)
