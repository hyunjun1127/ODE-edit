"""Synthetic visualization regression only; not model or scientific evidence."""
from pathlib import Path
import tempfile
import unittest
from .common import sha
from .reduce import write_csv
from .plot_completed import plots


class PlotTests(unittest.TestCase):
    def test_repeat_bytes_and_fixed_measured_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'synthetic';source.mkdir();rows=[]
            for name in ['W0','BASE_ALPHAEDIT_W001','BASE_MEMIT_W001']:
                for panel,kind,n in [('N_diag1000','N',1000),('H_diag_B1_R100_P200','R',100),
                    ('H_diag_B1_R100_P200','P',200),('BaseEval256','BASE',256)]:
                    rows.append(dict(endpoint=name,panel=panel,kind=kind,count=n,success=n//2))
            write_csv(source/'endpoint-summary.csv',rows)
            rows=[]
            for key in ['BASE_ALPHAEDIT_s001_t050','BASE_MEMIT_s001_t010']:
                for i,v in enumerate(['dose_0p5','dose_1','dose_minus1','rotation_2026092401','rotation_2026092402','rotation_2026092403']):
                    rows.append(dict(contrast=key+'/'+v+' minus actual11',panel='N_diag1000',kind='N',desired_nll_delta=i*.1-.2))
            write_csv(source/'path-patch-paired.csv',rows)
            plots(source,root/'a');plots(source,root/'b')
            for name in ['endpoint-preferences.png','patch-neighborhood-nll.png']:
                self.assertEqual(sha(root/'a'/name),sha(root/'b'/name))
                self.assertGreater((root/'a'/name).stat().st_size,1000)


if __name__=='__main__':unittest.main()
