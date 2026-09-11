import unittest
from unittest.mock import patch
from project.run_scripts.multilayer_joint_compensation.track_b.launch import admission

class TestAdmission(unittest.TestCase):
    def check(self,rows,exclude=None):
        with patch('project.run_scripts.multilayer_joint_compensation.track_b.launch.command',side_effect=[
            dict(exit=0,stdout='ALLOW',stderr=''),dict(exit=0,stdout=rows,stderr='')]):
            return admission(exclude_job=exclude)
    def test_pending_included(self):
        ok,r=self.check('1|odeedit_old|gres/gpu:1|RUNNING\n2|odeedit_new|gres/gpu:1|PENDING\n')
        self.assertFalse(ok);self.assertEqual(r['existing_including_pending'],2)
    def test_new_held_excluded_at_release(self):
        ok,r=self.check('1|odeedit_old|gres/gpu:1|RUNNING\n2|odeedit_multilayer_b_s2|gres/gpu:1|PENDING\n','2')
        self.assertTrue(ok);self.assertEqual(r['existing_including_pending'],1)
    def test_unresolved_array_failclosed(self):
        ok,r=self.check('3_[0-2]|odeedit_other|gres/gpu:1|PENDING\n');self.assertFalse(ok)
    def test_unresolved_gpu_failclosed(self):
        ok,r=self.check('3|odeedit_other|N/A|RUNNING\n');self.assertFalse(ok)

if __name__=='__main__':unittest.main()
