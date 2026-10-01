import tempfile
import unittest
from pathlib import Path
from .initial_review import review
from .common import write

class InitialBoundaryTests(unittest.TestCase):
    def test_submission_and_pilot_are_not_main_initial_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            write(p/'submission.json',{'jobs':{'main-A':'1','main-B':'2'}})
            write(p/'config.json',{})
            write(p/'execution.lock.json',{})
            write(p/'pilot-A/INITIAL_VALID.json',{'main_representative':False})
            self.assertEqual(review(p)['state'],'INITIAL_NOT_OBSERVED')
            self.assertFalse((p/'handoff.json').exists())

if __name__=='__main__':unittest.main()
