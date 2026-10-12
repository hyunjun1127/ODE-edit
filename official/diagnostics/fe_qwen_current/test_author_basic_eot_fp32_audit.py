import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from official.diagnostics.fe_qwen_current import author_basic_eot_fp32_audit as a
from official.experiments.prepare import write_new
from official.runners.server1.common import member
from official.tracking.schema import job_identity,run_name


def fixture(server='server4',step='-5'):
    dataset='cf' if server=='server4' else 'zsre';job='63809' if server=='server4' else '121407'
    env={'SLURM_JOB_ID':job} if server=='server4' else {'ODEEDIT_RENT_JOB_ID':job}
    if server=='server4' and step is not None:env['SLURM_STEP_ID']=step
    b=dict(dataset=dataset,main_stream=dict(sha256=a.old.STREAMS[dataset][1]),execution=dict(server=server,job_id=job,
        source_commit='a'*40,source_run_id=a.old.REFERENCES[dataset],attempt='fp32-r1'))
    values=a.old.tracking(b,env);identity=dict(job_id=job,source='a'*40)
    runtime=dict(identity=identity,tracking=values);proof=dict(identity=identity,checkpoint=dict(identity=identity))
    jid=job_identity(values);name=run_name(values)
    startup=dict(status='READY_ONLINE',run_id='actualrun',url='https://example.invalid/run',config=values,job_identity=jid,run_name=name)
    immutable=dict(startup_remote_identity_verified=True,run_id='actualrun',url=startup['url'],config=values,job_identity=jid,run_name=name)
    receipt=dict(run_id='actualrun',job_identity=jid,status='FINISHED_ONLINE',dropped_points=0,startup_readback=startup,
        result=dict(job_identity=jid,run_id='actualrun',run_name=name))
    return b,runtime,proof,immutable,receipt


class Tests(unittest.TestCase):
    def test_negative_step_exact_reconstruction_not_ignored(self):
        args=fixture();before=copy.deepcopy(args);result=a.crosscheck_tracking(*args)
        self.assertEqual(result['reconstructed_environment'],{'SLURM_JOB_ID':'63809','SLURM_STEP_ID':'-5'})
        self.assertEqual(args,before);self.assertFalse(result['step_id_ignored'])
        # Reproduces the exact old collector bug while new auditor preserves step.
        self.assertNotEqual(a.old.tracking(args[0],{'SLURM_JOB_ID':'63809'}),args[1]['tracking'])

    def test_rent_absent_step_unchanged(self):
        args=fixture('rent');result=a.crosscheck_tracking(*args)
        self.assertEqual(result['reconstructed_environment'],{'ODEEDIT_RENT_JOB_ID':'121407'})
        self.assertNotIn('step_id',result['job_identity'])
        self.assertEqual(a.old.tracking(args[0],result['reconstructed_environment']),args[1]['tracking'])

    def test_each_independent_forgery_rejected(self):
        changes=[(1,('tracking','step_id'),'0'),(3,('job_identity','step_id'),'0'),
            (4,('job_identity','step_id'),'0'),(4,('startup_readback','job_identity','step_id'),'0'),
            (4,('startup_readback','config','step_id'),'0'),(4,('result','job_identity','step_id'),'0'),
            (3,('run_id',),'another'),(4,('run_id',),'another'),(0,('execution','job_id'),'63810'),
            (2,('checkpoint','identity','source'),'b'*40),(1,('tracking','config_sha'),'b'*64)]
        for pos,keys,value in changes:
            with self.subTest(pos=pos,keys=keys):
                # Independent copies intentionally prevent aliasing from making
                # a one-receipt tamper accidentally mutate all fixture evidence.
                args=[copy.deepcopy(x) for x in fixture()];node=args[pos]
                for k in keys[:-1]:node=node[k]
                node[keys[-1]]=value
                with self.assertRaises(ValueError):a.crosscheck_tracking(*args)

    def test_missing_step_evidence_rejected(self):
        args=[copy.deepcopy(x) for x in fixture()];del args[3]['job_identity']['step_id']
        with self.assertRaises(ValueError):a.crosscheck_tracking(*args)
        args=[copy.deepcopy(x) for x in fixture('rent')];args[4]['job_identity']['step_id']='-5'
        with self.assertRaises(ValueError):a.crosscheck_tracking(*args)

    def test_missing_or_partial_w20_before_any_model_load(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'partial.json';write_new(p,dict(status='QUALITY_EARLY_STOP',edits=1500,batches=15))
            b=root/'audit.json';write_new(b,dict(schema=a.SCHEMA,result=member(p)))
            with patch.object(a.old,'validate',side_effect=AssertionError('SCIENCE_MUST_NOT_START')):
                with self.assertRaises(ValueError):a.collect(b,root/'out.json')
            self.assertFalse((root/'out.json').exists())


if __name__=='__main__':unittest.main()
