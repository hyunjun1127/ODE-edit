import tempfile
import unittest
from pathlib import Path
from .analysis import write_csv
from .discussion import build
from .report import build_a,table

class ReportTests(unittest.TestCase):
    def test_measured_schema_rendering_and_table_width(self):
        rows=[]
        for entry in ['Early','Middle','Late']:
            for endpoint in ['W0-full','ENTRY-full','N-full','B-alpha-0.02/eval-032','C-alpha-0.02/eval-032']:
                for panel in ['Current100','Fixed100','Past100']:
                    for metric in ['RS','PS','NS']:
                        rows.append(dict(entry=entry,endpoint=endpoint,panel=panel,metric=metric,resolution='full',
                             numerator=1,denominator=2,rate=.5,strict_numerator=1,strict_denominator=2,
                             loss=0,recovery=0,entry_success_denominator=1,entry_failure_denominator=1,
                             new_nll_delta_mean=0.,additional_margin_mean=0.,
                             **{f'{kind}_nll_{stat}':1. for kind in ['new','true'] for stat in ['mean','median','p90','max']}))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'auxiliary').mkdir()
            write_csv(root/'paired-summary.csv',rows);write_csv(root/'trajectory.csv',[])
            write_csv(root/'auxiliary/compute-summary.csv',[])
            completion=dict(stage='A',status='COMPLETE',writers=13,fullbatch_steps=320,native_scales=12,
                 selections={s:dict(alpha=.02,scores={.02:1.}) for s in ['B','C']})
            self.assertIn('neighbor true TF exact',build_a(root,completion))
            self.assertIn('scientific_promotion=false',build(root,completion))
        with self.assertRaisesRegex(ValueError,'TABLE_COLUMN'):
            table(['a'],[[1,2]])

if __name__=='__main__':unittest.main()
