"""Small CPU-only config/route/storage checks; not Llama execution validation."""
import unittest
from .preflight import check_import_only_patch, storage_plan, check_config, SCORES, LAYERS


class PreparationTests(unittest.TestCase):
    def test_exact_storage_floor(self):
        x = storage_plan(4096, 14336, 39098310656)
        self.assertEqual(x['checkpoint_tensor_floor_bytes'], 63417876480)
        self.assertEqual(x['tensor_floor_shortfall_bytes'], 24319565824)
        self.assertEqual(x['status'], 'DISK_HOLD')

    def test_reserve_not_just_tensor_floor(self):
        self.assertEqual(storage_plan(4096, 14336, 63417876480)['status'], 'DISK_HOLD')

    def test_unused_import_only(self):
        a = 'from notebooks.util import hparams\ndef f(hparams):\n return hparams.layers\n'
        check_import_only_patch(a, a.replace('from notebooks.util import hparams\n', ''))

    def test_math_patch_rejected(self):
        a = 'from notebooks.util import hparams\ndef f(hparams):\n return hparams.L2\n'
        with self.assertRaises(ValueError):
            check_import_only_patch(a, 'def f(hparams):\n return 1\n')

    def test_live_import_rejected(self):
        a = 'from notebooks.util import hparams\nx=hparams.layers\n'
        with self.assertRaises(ValueError):
            check_import_only_patch(a, a.replace('from notebooks.util import hparams\n', ''))

    def test_original_config(self):
        hp = dict(layers=LAYERS, L2=10, v_weight_decay=.4, v_num_grad_steps=25,
                  v_lr=.1, clamp_norm_factor=.5, nullspace_threshold=.02,
                  temperature=.1, causal_scores=SCORES)
        check_config(hp)
        for field, value in [('L2',1), ('causal_scores', {str(i+4):v for i,v in enumerate(SCORES.values())})]:
            with self.assertRaises(ValueError):
                check_config(dict(hp, **{field:value}))


if __name__ == '__main__':
    unittest.main()
