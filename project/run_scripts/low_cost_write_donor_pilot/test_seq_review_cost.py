import unittest
from .seq_review_cost import segments, SCHED
class CostTests(unittest.TestCase):
 def test_native_segments(self):
  self.assertEqual(segments(['Computing right vector','loss 1','loss 2','Computing right vector','loss 1']),[2,1])
 def test_incomplete_rejected(self):
  with self.assertRaises(ValueError):segments(['Computing right vector'])
 def test_limits(self):
  with self.assertRaises(ValueError):segments(['Computing right vector']+['loss x']*26)
 def test_observed_allocation(self):
  self.assertEqual(sum(x[0] for x in SCHED),33475)
if __name__=='__main__':unittest.main()
