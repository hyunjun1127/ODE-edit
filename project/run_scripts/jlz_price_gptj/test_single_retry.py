"""Structural selected-arm regression only: no numerical toy/model/Slurm."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
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

    def test_release_preserves_immutable_held_submission(self):
        from . import single_retry as module
        from .common import write
        with tempfile.TemporaryDirectory() as folder:
            attempt=Path(folder)
            ids={'ALPHA_CAP075':'61003','collector':'61004'}
            held=dict(status='HELD_NOT_RELEASED',jobs=ids,mapping={
                role:dict(dependency=None,argv=[]) for role in ids})
            write(attempt/'submission.json',held)
            write(attempt/'config.json',dict(resources={}))
            proof=attempt/'proof.json'
            write(proof,dict(retry_job='61003',maximum_possible_GPU_concurrency=2,
                running_mutations=0,baseline_after_ours=True))
            original=(attempt/'submission.json').read_bytes()
            with patch.object(module,'NEW',attempt),patch.object(module,'boundary'),\
                    patch.object(module,'effective_cap',return_value=(2,60416)),\
                    patch.object(module,'verify_frozen'),patch.object(module,'inspect'),\
                    patch.object(module,'command',return_value='') as command:
                result=module.release(proof)
            self.assertEqual(result['status'],'RELEASED')
            self.assertEqual((attempt/'submission.json').read_bytes(),original)
            self.assertEqual(json.loads((attempt/'release.json').read_text())['status'],'RELEASED')
            self.assertEqual([call.args[0] for call in command.call_args_list[:2]],
                [['scontrol','release','61004'],['scontrol','release','61003']])


if __name__=='__main__':unittest.main()
