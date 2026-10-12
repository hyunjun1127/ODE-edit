import unittest
from .fe_fp32_host_prepare import budget, GIB


class HostBudgetTests(unittest.TestCase):
    def test_actual_fp32_shapes(self):
        b = budget(3584, 18944, 5, 3*GIB, 2*GIB)
        self.assertEqual(b['W_bytes'], 1357905920)
        self.assertEqual(b['H_bytes'], 7177502720)
        self.assertEqual(b['tensor_bytes'], 8535408640)
        self.assertEqual(b['operating_reserve_bytes'], 8*GIB)
        self.assertEqual(b['required_free_bytes'], 2*(8535408640+32*1024**2)+13*GIB)

    def test_margin_is_additive_not_payload_discount(self):
        a = budget(3584, 18944, 5, GIB, GIB)
        b = budget(3584, 18944, 5, 2*GIB, 3*GIB)
        self.assertEqual(b['required_free_bytes']-a['required_free_bytes'], 3*GIB)


if __name__ == '__main__':
    unittest.main()
