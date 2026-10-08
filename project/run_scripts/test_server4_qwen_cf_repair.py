"""CPU-only scheduler fixtures; never submits or changes a real job."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from project.run_scripts import server4_qwen_submit as s


class RepairControl(unittest.TestCase):
    def test_multi_frontier_controller_serialization(self):
        self.assertTrue(s.dependency_matches('afterany:61674(unfulfilled),afterany:61776(unfulfilled),afterany:61777(unfulfilled)', 'afterany:61674:61776:61777'))
        self.assertFalse(s.dependency_matches('afterany:61674(unfulfilled)', 'afterany:61674:61776:61777'))
        self.assertFalse(s.dependency_matches('afterok:61674(unfulfilled)', 'afterany:61674'))

    def test_started_unknown_and_allocated_are_rejected(self):
        old=Path('/fixture');item=dict(job_id='1',name='fixture',logical_main_row='qwen25-cf-ft',kind='gpu',dependency='afterok:9')
        base=dict(JobId='1',JobName='fixture',JobState='PENDING',Command='/fixture/scripts/qwen25-cf-ft-gpu.sh',
                  WorkDir='/fixture',ReqNodeList='server4',RunTime='00:00:00',StartTime='Unknown',Restarts='0',
                  Requeue='0',UserId='janghj(1025)',AllocTRES='(null)',NodeList='',Dependency='afterok:9(unfulfilled)')
        with patch.object(s,'job_fields',return_value=('fixture',base)):
            self.assertEqual(s.pending_original(old,item),'fixture')
        for key,value in [('JobState','RUNNING'),('StartTime','2026-10-09T00:00:00'),('AllocTRES','gres/gpu=1'),
                          ('UserId','other(1)'),('Command','/different'),('Restarts','1')]:
            with patch.object(s,'job_fields',return_value=('fixture',{**base,key:value})):
                with self.assertRaises(RuntimeError):s.pending_original(old,item)

    def test_hold_retained_head_then_downstream_cancel_only_cf(self):
        affected=[dict(job_id=str(i)) for i in (1,2,3,4)]
        retained=[dict(job_id='5'),dict(job_id='6')]
        inventory=dict(affected=affected,retained=retained)
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(s,'verify'),patch.object(s,'read',return_value={'main_cells':6}), \
                 patch.object(s,'cf_inventory',return_value=inventory), \
                 patch.object(s,'pending_original',return_value='Reason=JobHeldUser '), \
                 patch.object(s,'job_fields',return_value=('cancelled',{'JobState':'CANCELLED'})), \
                 patch.object(s,'cmd',side_effect=lambda a: calls.append(a) or ''):
                s.replace_pending_cf(root,Path('/old'))
        self.assertEqual(calls[:5],[['scontrol','hold',j] for j in ('5','4','3','2','1')])
        self.assertEqual(calls[5:],[['scancel',j] for j in ('4','3','2','1')])
        self.assertFalse(any(c==['scancel','5'] or c==['scancel','6'] for c in calls))

    def test_state_race_stops_before_cancel(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(s,'verify'),patch.object(s,'read',return_value={'main_cells':6}), \
                 patch.object(s,'cf_inventory',return_value=dict(affected=[{'job_id':'1'}],retained=[{'job_id':'2'}])), \
                 patch.object(s,'pending_original',side_effect=RuntimeError('STARTED')),patch.object(s,'cmd',side_effect=lambda a:calls.append(a)):
                with self.assertRaisesRegex(RuntimeError,'STARTED'):s.replace_pending_cf(Path(tmp),Path('/old'))
        self.assertEqual(calls,[])

if __name__=='__main__':unittest.main()
