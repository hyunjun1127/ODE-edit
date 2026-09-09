import unittest
import torch
from .directions import group_indices,normalized_rows,soft_filter,physical_unit,frobenius_finite_change,operator_gradient

class DirectionTests(unittest.TestCase):
    def test_groups(self):
        groups=group_indices(list(range(100)))
        self.assertEqual(sorted(sum(groups,[])),list(range(100)))
        self.assertTrue(all(len(g)==10 for g in groups))
    def test_filter_formula_and_sign(self):
        torch.manual_seed(20260910);j=torch.randn(10,12,dtype=torch.float64);g=torch.randn(3,4,dtype=torch.float64)
        result,r=soft_filter(g,j)
        expected=g.flatten()-j.T@torch.linalg.solve(j@j.T+.1*torch.eye(10,dtype=torch.float64),j@g.flatten())
        torch.testing.assert_close(result.flatten(),expected)
        u,_=torch.linalg.qr(torch.randn(7,4,dtype=torch.float64))
        unit,_=physical_unit(-result,u);torch.testing.assert_close((unit@u.T).norm(),torch.tensor(1.,dtype=torch.float64))
        torch.testing.assert_close((-unit)@u.T,-(unit@u.T),rtol=0,atol=0)
    def test_zero_row_kept_and_finite_change(self):
        g=torch.zeros(10,2,3);g[0,0,0]=1
        j,r=normalized_rows(g);self.assertEqual(r['zero_rows'],9);self.assertEqual(j.shape,(10,6))
        torch.manual_seed(2);w=torch.randn(3,4,dtype=torch.float64);ref=torch.randn_like(w);d=torch.randn_like(w)
        change=frobenius_finite_change(w,ref,d)
        self.assertAlmostEqual(change['actual'],change['first_order']+change['second_order'],places=12)
    def test_operator_diagonal(self):
        d=torch.diag(torch.tensor([4.,2.,1.],dtype=torch.float64));g,r=operator_gradient(d)
        expected=torch.zeros_like(d);expected[0,0]=4
        torch.testing.assert_close(g,expected,atol=1e-10,rtol=1e-10)
    def test_exact_WN_offset_penalty(self):
        torch.manual_seed(7)
        delta=torch.randn(3,6,dtype=torch.float64);u,_=torch.linalg.qr(torch.randn(6,4,dtype=torch.float64))
        a=torch.randn(3,4,dtype=torch.float64);m=torch.randn(6,6,dtype=torch.float64);m=m@m.T
        from .algebra import action,reduced_metric
        expanded=((a@reduced_metric(u,m))*a).sum()+2*(a*((delta@m+delta)@u)).sum()+action(delta,m)
        torch.testing.assert_close(expanded,action(delta+a@u.T,m))

if __name__=='__main__':unittest.main()
