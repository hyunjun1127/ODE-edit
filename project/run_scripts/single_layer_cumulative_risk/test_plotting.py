import unittest
from .plotting import render

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

if __name__=='__main__':unittest.main()
