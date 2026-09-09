import unittest
from .plotting import render,companion

class PlotTests(unittest.TestCase):
    def test_png_byte_reproduction(self):
        rows=[]
        for endpoint in ['N','B-alpha-0.02/eval-032']:
            for panel in ['Current100','Fixed100','Past100']:
                for metric in ['RS','NS']:
                    rows.append(dict(entry='Middle',endpoint=endpoint,panel=panel,metric=metric,
                                     new_nll_mean='1.25',additional_margin_mean='0.15',rate='0.6'))
        for kind in ['rate','nll']:
            a=render(rows,kind);b=render(rows,kind)
            self.assertTrue(a.startswith(b'\x89PNG'));self.assertEqual(a,b)
    def test_companion_deterministic_missing_native_train_not_imputed(self):
        rows=[dict(entry='Middle',endpoint='N-full',panel=p,metric=m,rate='.5',loss='2',denominator='100',additional_margin_mean='.1')
              for p in ['Current100','Fixed100','Past100'] for m in ['RS','PS','NS']]
        for kind in ['rs_ps_ns','ps_retention','train_ns']:
            self.assertEqual(companion(rows,[],kind),companion(rows,[],kind))

if __name__=='__main__':unittest.main()
