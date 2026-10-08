"""CPU schema/selection/real sidecar assembly checks; no online/GPU fixtures."""
import json
import tempfile
from pathlib import Path
import unittest
from official.ours.config import resolve,plain
from . import tracking
from .pipeline import select


def config(tier='tier1'):
    r=plain(resolve('qwen25','qwen25-Q0'))
    return dict(server='server4',task_id=tracking.TASK,arm='QWEN_Q0',attempt='cpu-fixture',
        source_sha='a'*40,config_sha='b'*64,model='qwen',model_family='QWEN',writer='memit',
        role='validation' if tier=='smoke' else 'scientific',metric_schema=tracking.SCHEMA,
        cohort_role='validation' if tier=='smoke' else 'heldout_tuning',tier=tier,
        slice_identity='c'*64,resolved_config=r,resolved_config_sha256=r['sha256'],preset='base',M1=True)


class Tests(unittest.TestCase):
    def test_real_worker_fakeSDK_and_identity(self):
        from project.run_scripts.experiment_tracking.test_method import MethodSDK,synthetic_rows
        from project.run_scripts.jlz_realization.observe import reduce_rows
        from project.run_scripts.jlz_interference_l1.comparison_bridge import metric_row
        from project.run_scripts.experiment_tracking import schema,worker,identity
        tracking.install()
        cfg=schema.bind_job_identity(config(),dict(SLURM_JOB_ID='42',SLURM_ARRAY_JOB_ID='40',SLURM_ARRAY_TASK_ID='0',SLURM_STEP_ID='-5'))
        w0=dict(edits=0,**metric_row('W0_first500',reduce_rows(synthetic_rows(500)),500))
        sdk=MethodSDK();output=[]
        worker.session(sdk,dict(config=cfg,run_id='uniqueFixtureID',spool='/tmp/fake-only',smoke=False,
            base_url='https://api.wandb.ai'),[dict(op='log',values=w0,step=None),dict(op='finish',exit_code=0)],output.append)
        self.assertEqual(output[-1]['method_readback']['status'],'REMOTE_BOUNDED_ROWS_VERIFIED')
        self.assertIn('job40_0',output[0]['run_name'])
        self.assertEqual(output[0]['config']['step_id'],'-5')
        self.assertTrue(any(a==('W0_first500/*',) and kw['step_metric']=='edits' for a,kw in sdk.definitions))
        with tempfile.TemporaryDirectory() as d:
            receipt=identity.create(d,cfg,output[0],'uniqueFixtureID')
            self.assertEqual(receipt['config']['cohort_role'],'heldout_tuning')
            with self.assertRaises(FileExistsError):identity.create(d,cfg,output[0],'uniqueFixtureID')

    def test_unknown_private_config_rejected(self):
        for key in ('api_key','prompt','secret','environment'):
            with self.assertRaises(ValueError):tracking.config(dict(config(),**{key:'do-not-upload'}))
        altered=config();altered['resolved_config']['price']['lr']=9
        with self.assertRaises(ValueError):tracking.config(altered)

    def test_validation_role_and_no_fake_job(self):
        self.assertEqual(tracking.config(config('smoke'))['role'],'validation')
        with self.assertRaises(ValueError):tracking.config(dict(config(),role='validation'))
        from project.run_scripts.experiment_tracking import schema
        tracking.install()
        cfg=schema.bind_job_identity(config('smoke'),{})
        self.assertNotIn('job_id',cfg)
        self.assertEqual(cfg['execution_backend'],'local')

    def test_w0_scope_and_privacy(self):
        from project.run_scripts.experiment_tracking.test_method import synthetic_rows
        from project.run_scripts.jlz_realization.observe import reduce_rows
        from project.run_scripts.jlz_interference_l1.comparison_bridge import metric_row
        good=dict(edits=0,**metric_row('W0_first500',reduce_rows(synthetic_rows(500)),500))
        self.assertEqual(tracking.metrics(good,scientific=True),good)
        for patch in ({'W0_first500/R/count':2000},{'edits':1},{'raw':[]},{'W0_first2000/R/count':2000}):
            with self.assertRaises(ValueError):tracking.metrics(dict(good,**patch),scientific=True)

    def test_selection_no_filling_failed_and_reference(self):
        rows={'Q0':dict(PS=90,NS_loss_pp=0),'Q1':dict(PS=91,NS_loss_pp=.5),
              'Q2':dict(PS=92,NS_loss_pp=1),'Q3':dict(PS=99,NS_loss_pp=1.6),
              'Q7-native':dict(PS=95,NS_loss_pp=4)}
        s=select(rows)
        self.assertEqual(s['selected'],['Q2','Q1','Q7-native'])
        self.assertEqual(s['selected_roles']['Q7-native'],'NATIVE_REFERENCE_ONLY')
        self.assertNotIn('Q3',s['selected'])
        rows['Q7-native']['NS_loss_pp']=5.01
        self.assertTrue(select(rows)['Q7_extreme_excluded'])
        self.assertEqual(select({'Q0':rows['Q0']})['selected'],[])


if __name__=='__main__':unittest.main()
