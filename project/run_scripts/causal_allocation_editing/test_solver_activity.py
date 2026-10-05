"""Scalar activity receipts do not modify plans or advance rejected state."""
import unittest
import torch
from .solver import block_activity


class ActivityTests(unittest.TestCase):
    def test_activity_is_against_last_accepted_and_nonmutating(self):
        accepted = {4:torch.tensor([[0., .75, .1],[0.,0.,0.]])}
        trial = {4:torch.tensor([[.2,0.,.1],[0.,0.,0.]])}
        before = accepted[4].clone()
        row = block_activity(trial,accepted)['4']
        self.assertEqual(row['active_blocks'],2)
        self.assertEqual(row['reentry_from_previous_accepted'],1)
        self.assertEqual(row['zero_blocks'],1)
        self.assertTrue(torch.equal(accepted[4],before))
        # Rejection advances no cache: another trial uses the same accepted u.
        self.assertEqual(block_activity(trial,accepted),block_activity(trial,accepted))
        self.assertEqual(block_activity(accepted)['4']['cap_contacts'],1)


if __name__ == '__main__':
    unittest.main()
