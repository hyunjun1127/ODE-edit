"""Structural selected-arm regression only: no numerical toy/model/Slurm."""
import copy
import unittest
from .common import CELLS
from .submit import selected_cells


class SingleRetryScopeTests(unittest.TestCase):
    def config(self):
        return dict(selected_cells=['ALPHA_CAP075'],execution_override=dict(
            instruction_id='USER-SH4-GPTJ-ALPHA-CAP075-CAST-REPAIR-20261008',
            failed_job='60619',selected_cells=['ALPHA_CAP075'],same_hparams_and_method=True))

    def test_legacy_six_default(self):
        self.assertEqual(selected_cells({}),CELLS)

    def test_exact_explicit_single_retry(self):
        self.assertEqual(selected_cells(self.config()),('ALPHA_CAP075',))

    def test_not_an_arbitrary_subset_or_new_arm(self):
        for selected in ([],['MEMIT_CAP075'],['ALPHA_CAP100'],['ALPHA_CAP075','ALPHA_CAP100']):
            config=self.config();config['selected_cells']=selected
            with self.assertRaises(RuntimeError):selected_cells(config)

    def test_exact_failed_job_and_authority_required(self):
        for field,value in (('failed_job','60620'),('instruction_id','OTHER'),
                            ('same_hparams_and_method',False),('selected_cells',list(CELLS))):
            config=copy.deepcopy(self.config());config['execution_override'][field]=value
            with self.assertRaises(RuntimeError):selected_cells(config)


if __name__=='__main__':unittest.main()
