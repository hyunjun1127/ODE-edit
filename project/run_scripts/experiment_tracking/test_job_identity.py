"""Fake IDs stay inside CPU fixtures; this suite never contacts W&B."""
import unittest
from .schema import bind_job_identity, run_name, SLURM_ENV
from .test_tracking import CFG, SDK
from .worker import session


class Tests(unittest.TestCase):
    def test_normal_and_caller_match(self):
        cfg=bind_job_identity(dict(CFG,job_id='59931'),{'SLURM_JOB_ID':'59931','PRIVATE':'ignored'})
        self.assertEqual(cfg['job_display_id'],'59931')
        self.assertTrue(run_name(cfg).endswith('-job59931'))
        self.assertNotIn('PRIVATE',cfg)

    def test_array_zero_step(self):
        cfg=bind_job_identity(CFG,dict(SLURM_JOB_ID='59932',SLURM_ARRAY_JOB_ID='59931',SLURM_ARRAY_TASK_ID='0',SLURM_STEP_ID='batch'))
        self.assertEqual(cfg['job_id'],'59932');self.assertEqual(cfg['array_task_id'],'0')
        self.assertTrue(run_name(cfg).endswith('-job59931_0'));self.assertEqual(cfg['step_id'],'batch')

    def test_local(self):
        cfg=bind_job_identity(CFG,{})
        self.assertNotIn('job_id',cfg);self.assertEqual(cfg['identity_source'],'NOT_APPLICABLE')
        with self.assertRaises(ValueError):bind_job_identity(dict(CFG,job_id='59931'),{})

    def test_optional_empty_and_signed_step(self):
        cfg=bind_job_identity(CFG,dict(SLURM_JOB_ID='59931',SLURM_STEP_ID=''))
        self.assertNotIn('step_id',cfg)
        for step in ('-1','-2','0','batch','extern'):
            cfg=bind_job_identity(CFG,dict(SLURM_JOB_ID='59931',SLURM_STEP_ID=step))
            self.assertEqual(cfg['step_id'],step)
            from .schema import config
            self.assertEqual(config(cfg)['step_id'],step)
        with self.assertRaisesRegex(ValueError,'INVALID_SLURM_STEP_ID'):
            bind_job_identity(CFG,dict(SLURM_JOB_ID='59931',SLURM_STEP_ID='not a step'))

    def test_missing_invalid_mismatch(self):
        for env in ({'SLURM_STEP_ID':'0'},{'SLURM_JOB_ID':''},{'SLURM_JOB_ID':'0'},
                    {'SLURM_JOB_ID':'59931','SLURM_ARRAY_TASK_ID':'0'}):
            with self.assertRaises(ValueError):bind_job_identity(CFG,env)
        with self.assertRaisesRegex(ValueError,'CALLER_ENV_MISMATCH'):
            bind_job_identity(dict(CFG,job_id='59930'),{'SLURM_JOB_ID':'59931'})

    def test_whitelist_only(self):
        class Env:
            def get(self,key):
                assert key in SLURM_ENV.values()
                return '59931' if key=='SLURM_JOB_ID' else None
            def __iter__(self):raise AssertionError('full env read forbidden')
        self.assertEqual(bind_job_identity(CFG,Env())['job_id'],'59931')

    def test_production_assembly_readback(self):
        cfg=bind_job_identity(CFG,{'SLURM_JOB_ID':'59931'})
        req=dict(config=cfg,run_id='distinctUUIDfixture',spool='/tmp/fake-only',smoke=False,base_url='https://api.wandb.ai')
        sdk=SDK();out=[]
        session(sdk,req,[dict(op='finish',exit_code=0)],out.append)
        self.assertEqual(out[0]['job_identity']['job_id'],'59931')
        self.assertEqual(sdk.calls[0]['name'],run_name(cfg));self.assertEqual(sdk.id,req['run_id'])
        for field in ('name','config'):
            sdk=SDK();original=sdk.run
            def bad(path,field=field):
                result=original(path)
                if field=='name':result.name='wrong'
                else:result.config=dict(result.config,job_id='59930')
                return result
            sdk.run=bad
            with self.assertRaisesRegex(ValueError,'REMOTE_JOB_'):session(sdk,req,[],lambda _:None)


if __name__=='__main__':unittest.main()
