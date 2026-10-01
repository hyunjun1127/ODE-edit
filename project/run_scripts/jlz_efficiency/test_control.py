import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from .measurement import write,timing
from .submit import capacity
from .collect import reduce_rows
from .run import compare_short

class Control(unittest.TestCase):
    def test_array_and_typed_aggregate_admission(self):
        def command(args):
            if args[0]=='squeue':
                self.assertIn('--array',args)
                return '10_0|janghj|odeedit_x|RUNNING\n10_1|janghj|odeedit_x|PENDING'
            return 'ReqTRES=cpu=8,mem=128G,gres/gpu=1,gres/gpu:a6000=1 '
        with patch('project.run_scripts.jlz_efficiency.submit.command',side_effect=command):
            with self.assertRaisesRegex(RuntimeError,'WAITING'):capacity()
            self.assertEqual(capacity(('10_1',))['active_plus_admitted'],1)
    def test_no_tensor_persistence_and_timing_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(TypeError,'NO_TENSOR'):write(Path(d)/'x.json',{'R':torch.zeros(2)})
        self.assertEqual(timing([1,2,3],[.8,1,2])['status'],'SPEED_UNRESOLVED')
        self.assertEqual(timing([2,3,4],[.8,1,1.1])['status'],'TIMING_SEPARATED')
    def test_short_branches_and_tie_failure(self):
        ref=dict(status='BUDGET_STOP',calls=12,accepted_steps=3,backtracks=7,branches=[True],point_trace=[{'x':'a'}])
        x=torch.ones(2,3)
        self.assertEqual(compare_short(ref,x,ref|{'boundary_ambiguities':['near']},x)['status'],'PASS')
        self.assertEqual(compare_short(ref,x,ref|{'accepted_steps':4},x)['status'],'UNQUALIFIED')
        rows=[dict(kind=k,new_nll=1.,true_nll=1.,new_strict=False,true_strict=False,new_token_correct=0,true_token_correct=0,new_token_count=2,true_token_count=3) for k in ('R','P','N')]
        self.assertEqual([r['successes'] for r in reduce_rows(rows).values()],[0,0,0])

if __name__=='__main__':unittest.main()
