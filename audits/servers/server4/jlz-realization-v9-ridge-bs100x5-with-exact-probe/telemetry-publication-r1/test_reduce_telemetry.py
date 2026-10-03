"""게시용 reducer의 단위·분모·null 처리 회귀검사. GPU 사용 없음."""
import unittest
from reduce_telemetry import stats, decomposition_rows


class ReductionTests(unittest.TestCase):
    def test_null_and_signed_values(self):
        r = stats([None, -1, 1, 2])
        self.assertEqual((r['n'],r['valid'],r['null']),(4,3,1))
        self.assertEqual(r['mean'],2/3)
        self.assertEqual(r['q50'],1)
        self.assertAlmostEqual(r['q95'],1.9)

    def test_all_null(self):
        r=stats([None,None])
        self.assertIsNone(r['mean'])
        self.assertIsNone(r['q95'])

    def test_nonfinite_is_not_silently_dropped(self):
        for v in (float('nan'),float('inf')):
            with self.assertRaises(ValueError): stats([v])

    def test_decomposition_signed_cross_terms(self):
        d=dict(request=[0,0],context_index=[0,1],canonical=[True,False],
               norms=[[1,1,0],[1,1,0]],cross_dots=[[-.5,0,0],[-.5,0,0]],
               virtual_pre_gap_norm=[1,1],virtual_actual_gap=[1,1],sum_error_max=0)
        for route in ('ideal_direction','FP32_direction'):
            d[route]=dict(gamma=[.5,.5],norm_ratio=[.5,.5],fit_relative=[.5,.5])
        rows=decomposition_rows(dict(decomposition=[{'4':d}]),'A',1,1,2)
        self.assertEqual([r['n_context_rows'] for r in rows],[2,1,1])
        self.assertEqual(rows[0]['sum_parts_ideal_gap_RMSnorm'],1)
        self.assertEqual(rows[0]['inherited_mean_dot_mean'],-.5)
        d['context_index']=[0,0]
        with self.assertRaises(ValueError):
            decomposition_rows(dict(decomposition=[{'4':d}]),'A',1,1,2)


if __name__=='__main__':
    unittest.main()
