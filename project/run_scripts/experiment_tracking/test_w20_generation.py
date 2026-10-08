"""CPU/fake SDK checks; no actual online run, model, or Slurm submission."""
import unittest
from . import schema, method
from .test_tracking import SDK
from .worker import session

NONCE = 'USER-GH-SH1-GPT2XL-BLUE-PRUNE-RECT-W20-GENERATION-20261008-R1'

def config():
    return dict(server='server1',task_id='gpt2xl-blue-prune-rect-w20-generation',
        arm='PRUNE',attempt='w20-only-r1',source_sha='a'*40,config_sha='b'*64,
        model='gpt2xl',model_family='gpt2',writer='prune',role='scientific',
        metric_schema=method.COMPARISON_SCHEMA,baseline='PRUNE',
        generation_metric_schema='counterfact-cake-generation-metrics-v1',
        generation_profile='cf-cake-prompt-inclusive-total100-eos-corrected-v1',
        generation_schedule='W20_ONLY_FIRST2000',generation_eval_seed=20261007,
        reference_assets_sha256='c'*64,generation_source_sha='d'*64,
        generation_qualification_plan_sha256='e'*64,generation_repair_instruction=NONCE)

class W20Generation(unittest.TestCase):
    def test_config_exact_schedule_and_privacy(self):
        self.assertEqual(schema.config(config())['generation_schedule'],'W20_ONLY_FIRST2000')
        for key,value in (('generation_schedule','W5'),('generation_repair_instruction','UNAPPROVED'),
                          ('raw_prompt','PRIVATE')):
            c=config();c[key]=value
            with self.assertRaises(ValueError):schema.config(c)

    def test_public_phase_progress_separate_axis(self):
        p={'phase':'W20_generation','generation_progress/step':0,
           'generation_progress/completed_cases':0,'generation_progress/total_cases':2000}
        self.assertEqual(schema.metrics(p,scientific=True),p)
        p['edits']=2000
        with self.assertRaisesRegex(ValueError,'NOT_ENDPOINT_OR_FIT'):schema.metrics(p,scientific=True)

    def test_unknown_phase_rejected(self):
        with self.assertRaises(ValueError):
            schema.metrics({'phase':'RAW_PROMPT','generation_progress/step':0},scientific=True)

    def test_fake_sdk_identity_and_axes_no_remote_claim(self):
        c=schema.bind_job_identity(config(),{'SLURM_JOB_ID':'123','SLURM_ARRAY_JOB_ID':'123',
            'SLURM_ARRAY_TASK_ID':'0','SLURM_STEP_ID':'-1'})
        class Fake(SDK):
            def __init__(self):super().__init__();self.axes=[]
            def define_metric(self,key,**kwargs):self.axes.append((key,kwargs))
        sdk=Fake();out=[]
        request=dict(config=c,run_id='fixtureW20',spool='/tmp/fake-only-no-write',
            smoke=False,base_url='https://api.wandb.ai')
        values={'phase':'W20_generation','generation_progress/step':0}
        session(sdk,request,[dict(op='log',values=values,step=None)],out.append)
        self.assertIn('job123_0',sdk.name)
        self.assertEqual(sdk.config['step_id'],'-1')
        accepted=next(r for r in out if r['status']=='LOGGING_ACCEPTED')
        self.assertEqual(accepted['delivery'],'SDK_ASYNC_NOT_REMOTE_ACK')
        self.assertIn(('generation_progress/*',dict(step_metric='generation_progress/step',step_sync=False)),sdk.axes)
        self.assertEqual(sdk.points,[values])

if __name__=='__main__':unittest.main()
