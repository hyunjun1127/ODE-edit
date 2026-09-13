"""CPU-only queue parsing and exact resource request guards."""
from pathlib import Path
import os
import pwd
import unittest
from .sequential_submit import project_rows, inspect


class SubmissionTests(unittest.TestCase):
    def test_pending_and_completing_are_prior_admissions(self):
        text='11|janghj|odeedit_x|COMPLETING|1|gres/gpu:1|(null)\n12_[0-4%2]|janghj|odeedit_y|PENDING|1|gres/gpu:1|afterany:11\n13|other|unrelated|RUNNING|1|gres/gpu:1|(null)'
        rows=project_rows(text,['odeedit_*'])
        self.assertEqual([r['state'] for r in rows],['COMPLETING','PENDING'])

    def test_held_request_must_be_exact(self):
        owner=pwd.getpwuid(os.getuid()).pw_name
        root=Path('/task/attempt')
        source=root/'source/project/run_scripts/low_cost_write_donor_pilot/sequential.sbatch'
        text=f'JobId=100 UserId={owner}(0) JobName=odeedit_lowcost_seq10_s4 ArrayTaskId=0-5%2 NumCPUs=8 MinMemoryNode=59G TimeLimit=1-00:00:00 ReqNodeList=server4 Command={source} gres/gpu:rtx_pro_6000=1 JobState=PENDING Reason=JobHeldUser'
        inspect(text,'100',root)
        inspect(text.replace('gres/gpu:rtx_pro_6000=1','TresPerNode=gres/gpu:rtx_pro_6000:1'),'100',root)
        for bad in [text.replace('59G','60G'),text.replace('%2','%3'),text.replace('JobHeldUser','Resources'),text.replace('NumCPUs=8','NumCPUs=16')]:
            with self.assertRaises(ValueError):inspect(bad,'100',root)


if __name__=='__main__':unittest.main()
