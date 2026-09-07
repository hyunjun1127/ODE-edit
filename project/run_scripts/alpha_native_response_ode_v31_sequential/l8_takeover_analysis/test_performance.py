import unittest
from .performance import bits,stats,published_margins,MISSING


class PerformanceTests(unittest.TestCase):
    def test_preference_direction_and_tie(self):
        public={'rows':[]}
        for prefix in ('rewrite','rephrase','locality'):
            for case,new,true in [(0,1.,2.),(1,2.,1.),(2,1.,1.)]:
                for target,value in [('new',new),('true',true)]:
                    public['rows'].append(dict(case_id=case,prompt_index=0,kind=prefix+'_target_'+target,nll=value,input_identity_sha256=target))
        self.assertEqual([v['success'] for v in bits(public,'rewrite').values()],[True,False,False])
        self.assertEqual([v['success'] for v in bits(public,'locality').values()],[False,True,False])

    def test_marginal_quantiles_never_subtracted(self):
        row={}
        for kind in ('rewrite','rephrase','locality'):
            row[kind+'_target_true_nll_mean']=4.;row[kind+'_target_new_nll_mean']=1.;row[kind+'_target_new_row_count']=10
        r=published_margins(row)
        self.assertEqual(r['rewrite_margin_mean'],3.)
        self.assertEqual(r['locality_margin_mean'],-3.)
        self.assertEqual(r['rephrase_margin_p90'],MISSING)

    def test_nonfinite_not_imputed(self):
        with self.assertRaises(ValueError):stats([1.,float('nan')])
        with self.assertRaises(ValueError):stats([])
        self.assertEqual(stats([1.,2.,3.])['median'],2.)


if __name__=='__main__':unittest.main()
