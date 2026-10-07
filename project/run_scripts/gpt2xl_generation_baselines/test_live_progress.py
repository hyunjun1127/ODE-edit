import json
import os
import unittest
from .live_progress import frame,progress_values,first_parent_frames_atomic
from project.run_scripts.experiment_tracking.method import AxisState
from project.run_scripts.experiment_tracking.test_tracking import SDK
from project.run_scripts.experiment_tracking.schema import bind_job_identity
from project.run_scripts.experiment_tracking.worker import session

class LiveProgress(unittest.TestCase):
    def test_progress_not_scientific_endpoint_or_axis(self):
        values=progress_values(175,{'start':0,'rss':4096})
        self.assertEqual(values['step'],175)
        self.assertNotIn('edits',values);self.assertNotIn('fit/global_candidate',values)
        state=AxisState();state.accept(values);self.assertIsNone(state.edits);self.assertIsNone(state.fit)
        for denied in ('prompt','tensor','WANDB_API_KEY','current/post/R/new_nll'):
            with self.assertRaises(ValueError):frame({denied:'forbidden'})
    def test_atomic_protocol_roundtrip(self):
        values={'phase_id':2,'step':175,'status_code':1,'time/elapsed_seconds':12.}
        encoded=frame(values);self.assertLessEqual(len(encoded),4096)
        r,w=os.pipe()
        try:
            self.assertEqual(os.write(w,encoded),len(encoded))
            self.assertEqual(json.loads(os.read(r,4096)),dict(op='log',values=values,step=None))
        finally:os.close(r);os.close(w)
    def test_same_production_worker_single_run_and_monotonic_followup(self):
        sdk=SDK();sdk.define_metric=lambda *a,**k:None
        cfg=bind_job_identity(dict(server='server1',task_id='fixture',arm='BASE_MEMIT',attempt='r1',
            source_sha='a'*40,config_sha='b'*64,model='gpt2xl',model_family='gpt2',writer='memit',
            role='scientific',metric_schema='price-first2k-scalar-v1'),{'SLURM_JOB_ID':'60928'})
        values=dict(phase_id=2,step=175,status_code=1)
        w0=dict(edits=0,batch=0)
        commands=[dict(op='log',values=values,step=None),dict(op='log',values=w0,step=None)]
        emitted=[]
        session(sdk,dict(config=cfg,run_id='existing123',spool='/no-write-fixture',smoke=False,
            base_url='https://api.wandb.ai'),commands,emitted.append)
        self.assertEqual(len(sdk.calls),1);self.assertEqual(sdk.points,[values,w0])
        self.assertEqual([x['points'] for x in emitted if x['status']=='LOGGING_ACCEPTED'],[1,2])
    def test_first_parent_frame_bound(self):
        first_parent_frames_atomic(dict(edits=0,batch=0))
        with self.assertRaises(RuntimeError):progress_values(2001,{'start':0,'rss':4096})

if __name__=='__main__':unittest.main()
