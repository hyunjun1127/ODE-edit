import tempfile
import unittest
from pathlib import Path
import torch
from .seq_review_state import tensor_sha, tensor_inventory, norm, file_member, digest, validate_commit

class StateReviewTests(unittest.TestCase):
    def test_hash_conventions_separate(self):
        x = torch.arange(8, dtype=torch.float32)
        self.assertNotEqual(tensor_sha(x), tensor_sha(x, False))
        self.assertNotEqual(tensor_sha(x), tensor_sha(x.reshape(2,4)))
        self.assertEqual(tensor_sha(x,False), tensor_sha(x.reshape(2,4),False))

    def test_finite_gate(self):
        with self.assertRaises(AssertionError):
            tensor_inventory({'x': torch.tensor([float('nan')])})

    def test_norm_is_actual_difference(self):
        x = torch.tensor([3.,4.])
        self.assertEqual(norm(x),5.)
        self.assertEqual(norm(x,x),0.)

    def test_recursive_inventory(self):
        out=tensor_inventory({'a':[torch.ones(2, dtype=torch.float32)]})
        self.assertEqual(out[0]['key'],'a[0]')
        self.assertEqual(out[0]['dtype'],'torch.float32')

    def test_regular_file_and_link_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_bytes(b'abc')
            self.assertEqual(file_member(p)['bytes'],3)
            q=Path(d)/'link';q.symlink_to(p)
            with self.assertRaises(AssertionError):file_member(q)

    def test_canonical_order(self):
        self.assertEqual(digest({'a':1,'b':2}),digest({'b':2,'a':1}))
        self.assertNotEqual(digest([1,2]),digest([2,1]))

    def test_history_mutation_rejected(self):
        import copy
        prior = dict(M4='a',M8='b',P4='c',P8='d',contexts='e')
        fit = dict(layer=4,compute_z=100,solve=1,history_append=0,
                   target_mode='same-host-fresh-current-state',
                   source={'optimizer_or_solve_equation_changes':0})
        c = dict(entry=prior,evaluation_nonmutation=True,materialization={'alpha':1.},
                 first_fit=fit,second_fit=None,partial_state=copy.deepcopy(prior),
                 history=[dict(layer=4,history_append=1)])
        entry = dict(state=prior,previous_commit_exact=True)
        validate_commit(c,entry,prior,1.,None)
        c['partial_state']['M4']='changed'
        with self.assertRaises(AssertionError):validate_commit(c,entry,prior,1.,None)

if __name__=='__main__': unittest.main()
