"""CPU-only metadata routing tests; no model, new run, or scientific evaluation."""
import unittest
from .cap_tracking import model_scoped_arm

class ModelRouting(unittest.TestCase):
    def test_existing_cells_unchanged(self):
        for model in ('LLAMA','QWEN'):
            for writer in ('','AE_'):
                for arm in ('CAP075','CAP100','FREE100'):
                    cell=model+'_'+writer+arm
                    self.assertEqual(model_scoped_arm({'model_profile':model},cell),cell)

    def test_gptj_six_future_cells(self):
        for writer in ('MEMIT','ALPHA'):
            for arm in ('CAP075','CAP100','FREE100'):
                cell=writer+'_'+arm
                self.assertEqual(model_scoped_arm({'model_profile':'GPTJ'},cell),'GPTJ_'+cell)

    def test_unknown_not_guessed(self):
        self.assertEqual(model_scoped_arm({},'UNBOUND'),'UNBOUND')

    def test_cross_model_mismatch_rejected(self):
        with self.assertRaises(Exception):
            model_scoped_arm({'model_profile':'QWEN'},'LLAMA_CAP075')

if __name__=='__main__':unittest.main()
