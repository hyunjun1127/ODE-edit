"""CPU fake-SDK logging metadata checks; no network, GPU or science fixture."""
import os
import types
import unittest
from pathlib import Path
from unittest.mock import patch,mock_open
from . import comparison_publish as bridge
from . import tracking_readback


class TrackingIdentity(unittest.TestCase):
    def fixture(self, array=False):
        # Synthetic IDs exist only inside this fake SDK; never uploaded.
        identity=dict(job_id='123',job_display_id='120_0' if array else '123')
        if array:identity.update(array_job_id='120',array_task_id='0',step_id='batch')
        parent=types.SimpleNamespace(config=dict(source_sha='source',**identity))
        state={}
        class Run:
            def __init__(self,kwargs):
                self.config=kwargs['config'];self.name=kwargs['name'];self.summary={};self.rows=[]
            def define_metric(self,*a,**kw):pass
            def log(self,row):self.rows.append(row)
            def finish(self):pass
            def scan_history(self,**kw):return self.rows
        class API:
            def run(self,path):return parent if path.endswith('/parent') else state['run']
            def runs(self,*a,**kw):return []
        def init(**kwargs):state.update(kwargs=kwargs,run=Run(kwargs));return state['run']
        sdk=types.SimpleNamespace(Api=lambda **kw:API(),Settings=lambda **kw:kw,init=init)
        target=types.SimpleNamespace(root=Path('/fake'),out=Path('/fake/cell'),cell='MEMIT_CAP075',
            b=dict(job=123,source='source',writer='memit',model='GPTJ',config_sha256='config'),
            c=dict(task_id='jlz-price-gptj-2k'),mc=dict(observation_identity='tokens'),
            rows={1:{'progress/batch':1,'edits':100}},last=1,hashes={1:'commit'})
        def read(path):
            return {'jobs':{'MEMIT_CAP075':123}} if path.name=='submission.json' else {'run_id':'parent'}
        return target,parent,sdk,read,state

    def test_normal_and_array_zero_preserve_science_job(self):
        for array in (False,True):
            target,parent,sdk,read,state=self.fixture(array)
            with patch.dict('sys.modules',wandb=sdk),patch.object(bridge,'read',read), \
                 patch.dict(os.environ,SLURM_JOB_ID='999'):
                bridge.publish(target,Path('/fake/spool'))
            cfg=state['kwargs']['config'];self.assertEqual(cfg['job_id'],'123')
            self.assertIn('job'+parent.config['job_display_id'],state['kwargs']['name'])
            self.assertEqual(cfg['identity_source'],'VERIFIED_SCIENCE_SUBMISSION')
            self.assertNotIn('999',state['kwargs']['name'])
            if array:self.assertEqual(cfg['array_task_id'],'0')
            self.assertEqual(state['kwargs']['id'],'comparison-parent')

    def test_missing_display_and_conflicting_parent_block(self):
        for reason in ('missing','mismatch'):
            target,parent,sdk,read,state=self.fixture()
            if reason=='missing':parent.config.pop('job_display_id')
            else:parent.config['job_id']='124'
            with patch.dict('sys.modules',wandb=sdk),patch.object(bridge,'read',read):
                with self.assertRaises(ValueError):bridge.publish(target,Path('/fake/spool'))
            self.assertNotIn('kwargs',state)

    def test_bounded_readback_is_read_only_and_missing_rows_not_verified(self):
        identity=dict(run_id='fake',url='https://example.invalid/fake',source_sha='source',
            config=dict(model='gptj'),job_identity=dict(job_id='123',job_display_id='123'))
        for remote_rows,expected_status in [([{'edits':100,'current/post/R/count':100}],
                'REMOTE_PAYLOAD_VERIFIED'),([], 'LOGGING_INCOMPLETE_OR_MISMATCH')]:
            remote=types.SimpleNamespace(config=dict(model='gptj',job_id='123',job_display_id='123'),
                name='fake-job123',scan_history=lambda **kw:remote_rows)
            sdk=types.SimpleNamespace(Api=lambda **kw:types.SimpleNamespace(run=lambda path:remote))
            data='{"op":"log","values":{"edits":100,"current/post/R/count":100}}\n'
            with patch.dict('sys.modules',wandb=sdk),patch.object(tracking_readback,'read',return_value=identity), \
                 patch.object(Path,'open',mock_open(read_data=data)),patch.object(tracking_readback,'atomic'):
                receipt=tracking_readback.verify(Path('/fake'))
            self.assertEqual(receipt['status'],expected_status)
            self.assertFalse(receipt['scientific_completion_claim'])
            self.assertTrue(receipt['remote_run_not_modified'])


if __name__=='__main__':unittest.main()
