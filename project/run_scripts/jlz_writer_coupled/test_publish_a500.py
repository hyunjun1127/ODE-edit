"""Small CPU-only reducer checks; not model or runtime validation."""
import unittest
from .publish_a500 import reduce


def fixture():
    return [dict(identity=k, kind=k, new_nll=1., true_nll=2.,
                 margin_true_minus_new=1., new_token_count=2,
                 new_token_correct=1, new_strict=False, true_token_count=1,
                 true_token_correct=1, true_strict=True) for k in ('R','P','N')]


class ReductionTest(unittest.TestCase):
    def test_target_direction_and_tf(self):
        out = reduce(fixture())
        self.assertEqual(out['R']['rate'], 1)
        self.assertEqual(out['N']['rate'], 0)
        self.assertEqual(out['R']['token_micro'], .5)
        self.assertEqual(out['N']['token_micro'], 1)

    def test_ties_fail(self):
        rows = fixture()
        for r in rows: r['true_nll'] = 1.; r['margin_true_minus_new'] = 0.
        self.assertTrue(all(x['numerator'] == 0 for x in reduce(rows).values()))

    def test_nonfinite_duplicate_and_strict_corruption_block(self):
        for mutation in ('nan', 'duplicate', 'strict'):
            rows = fixture()
            if mutation == 'nan': rows[0]['new_nll'] = float('nan')
            if mutation == 'duplicate': rows.append(dict(rows[0]))
            if mutation == 'strict': rows[0]['new_strict'] = True
            with self.assertRaises(AssertionError): reduce(rows)


if __name__ == '__main__':
    unittest.main()
