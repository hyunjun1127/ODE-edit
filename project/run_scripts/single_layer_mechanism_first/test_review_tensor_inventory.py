import unittest
import torch
from .review_tensor_inventory import inspect_payload


class Tests(unittest.TestCase):
    def test_small_scientific_factors(self):
        rows=inspect_payload({'keys':torch.ones(7,3),'nested':[torch.zeros(2,4)],'label':'scientific'})
        self.assertEqual(len(rows),2)
        self.assertTrue(all(r['finite'] for r in rows))

    def test_full_writer_and_memory_rejected_without_allocating(self):
        for shape in ((4096,14336),(1,14336,14336)):
            with self.assertRaisesRegex(ValueError,'FULL_WRITER_OR_MEMORY'):
                inspect_payload(torch.zeros(1).expand(shape))

    def test_nonfinite_factor_rejected(self):
        with self.assertRaisesRegex(ValueError,'NONFINITE'):
            inspect_payload({'factor':torch.tensor([1.,float('nan')])})


if __name__=='__main__':unittest.main()
