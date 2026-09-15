"""Small routing mocks only: no torch/model/FD/numerical fixture execution."""
import ast
import copy
import unittest
from pathlib import Path
from .gate_skip import TASK,MODE,diagnostic,skip_enabled,parity_record

class SkipRouting(unittest.TestCase):
    def setUp(self):
        self.lock=dict(instruction_id=TASK,validation_mode=MODE,numerical_validation='NOT_ESTABLISHED',
                       validation_override={'instruction_id':TASK})
    def test_callback_never_executed(self):
        def forbidden():raise RuntimeError('DIAGNOSTIC_EXECUTED')
        self.assertEqual(diagnostic(self.lock,'FD_SELF_KL_MAP',forbidden)['calls'],0)
    def test_waiver_cannot_masquerade_as_repair(self):
        bad=copy.deepcopy(self.lock);bad['repair_pass_path']='anything'
        with self.assertRaises(AssertionError):skip_enabled(bad)
    def test_original_route_retained(self):
        self.assertEqual(diagnostic({},'original',lambda:17),17)
    def test_existing_float_difference_warns_without_forward(self):
        r=[dict(case_id=1,new_nll=1.001,new_strict=True)]
        obs=dict(rows=[dict(case_id=1,nll=1.)],strict_ids=[1])
        self.assertEqual(parity_record(self.lock,r,obs,[1])['status'],'WARNING')
        with self.assertRaises(AssertionError):parity_record(self.lock,r,obs,[2])
    def test_syntax_and_runner_skip_routes(self):
        source=(Path(__file__).parent/'runner.py').read_text();ast.parse(source)
        self.assertIn("if not skipped:\n            from .technical",source)
        self.assertIn("if not skipped and 'repair_pass_path' in lock",source)
        self.assertIn("if skipped:\n                    tech=diagnostic",source)
        self.assertNotIn('repair_checks',source)
        self.assertIn('INITIAL_EXECUTION_OBSERVED_WITH_VALIDATION_SKIPPED',source)

if __name__=='__main__':unittest.main()
