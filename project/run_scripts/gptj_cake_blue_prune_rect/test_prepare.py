"""Narrow CPU metadata tests; no preparation staging, tensor I/O or GPU calls."""
import ast
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import common, prepare


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = common.authority()
        cls.old = common.read(prepare.PARENT / 'config-submission.json')
        cls.checks = common.read(prepare.PARENT / 'asset-checks.json')

    def test_common_authority_four_arms(self):
        self.assertEqual(self.contract['scope']['arms'], list(common.ARMS))
        self.assertEqual(common.ARM_LAYERS['ALPHAEDIT_BLUE'], (3, 8))
        self.assertEqual(common.ARM_LAYERS['CAKE'], common.LAYERS)

    def test_prior_small_receipts_exact(self):
        self.assertEqual(common.small_member(prepare.PARENT / 'config-submission.json')['sha256'],
                         prepare.PARENT_CONFIG_SHA)
        self.assertEqual(common.small_member(prepare.PARENT / 'asset-checks.json')['sha256'],
                         prepare.PARENT_ASSET_CHECKS_SHA)

    def test_actual_reuse_seals_without_payload_hash(self):
        with patch.object(common, 'member', side_effect=AssertionError('no large member hash')):
            stats = prepare.validate_reused_assets(self.old, self.checks, self.contract)
        self.assertEqual(set(stats), set(map(str, common.LAYERS)))
        self.assertEqual(stats['8']['bytes'], 1073743054)

    def test_reuse_rejects_wrong_projector_slots(self):
        checks = copy.deepcopy(self.checks)
        checks['projector_slots'][1]['physical_layer'] = 8
        with self.assertRaisesRegex(RuntimeError, 'REUSED_PROJECTOR_SCHEMA_FINITE_RECEIPT'):
            prepare.validate_reused_assets(self.old, checks, self.contract)

    def test_reuse_rejects_count100000(self):
        checks = copy.deepcopy(self.checks)
        checks['stats'][0]['count'] = 100000.0
        with self.assertRaisesRegex(RuntimeError, 'REUSED_RAW_MOM2_SCHEMA_COUNT'):
            prepare.validate_reused_assets(self.old, checks, self.contract)

    def test_reuse_rejects_gpt2_runtime_model(self):
        old = copy.deepcopy(self.old)
        old['model_revision'] = 'GPT2XL_NOT_ALLOWED'
        with self.assertRaisesRegex(RuntimeError, 'REUSED_MODEL_INPUT_IDENTITY'):
            prepare.validate_reused_assets(old, self.checks, self.contract)

    def test_stat_seal_detects_mutation(self):
        with tempfile.TemporaryDirectory(prefix='gptj-fourarm-stat-') as folder:
            file = Path(folder) / 'small-source.py'
            file.write_text('original\n')
            row = common.small_member(file)
            self.assertEqual(common.stat_seal(row), row)
            file.write_text('changed-source\n')
            with self.assertRaisesRegex(RuntimeError, 'ASSET_STAT_CHANGED'):
                common.stat_seal(row)

    def test_stat_seal_rejects_symlink(self):
        with tempfile.TemporaryDirectory(prefix='gptj-fourarm-symlink-') as folder:
            file = Path(folder) / 'small-source.py'
            file.write_text('source\n')
            row = common.small_member(file)
            link = Path(folder) / 'source-link.py'
            link.symlink_to(file)
            with self.assertRaisesRegex(RuntimeError, 'ASSET_SAFE_FILE'):
                common.stat_seal(dict(row, path=str(link)))

    def test_actual_input_twenty_packs_without_native_execution(self):
        records, packs, schedule = prepare.load_input_packs(self.old, self.contract)
        self.assertEqual(len(records), 2000)
        self.assertEqual([pack['batch'] for pack in packs], list(range(1, 21)))
        self.assertTrue(all(pack['counts'] == dict(requests=100, R=100, P=200, N=1000)
                            for pack in packs))
        self.assertEqual(schedule['sha256'], self.contract['input']['schedule_sha256'])
        self.assertEqual(len(list(common.batches(records))), 20)
        with self.assertRaisesRegex(RuntimeError, 'EXACT_FIRST2K'):
            list(common.batches(records[:100]))

    def test_W0_reference_is_scalar_and_read_only(self):
        binding, sources = prepare.bind_W0_reference(self.old)
        self.assertEqual(binding['status'], 'COMPLETE_REFERENCE_IDENTITY_NOT_NEW_SCIENCE')
        self.assertEqual(len(binding['chunks']), 40)
        self.assertTrue(binding['scalar_bridge_only'])
        self.assertFalse(binding['history_or_editor_resume'])
        self.assertEqual(binding['new_model_evaluations'], 0)
        self.assertEqual([row['relative'] for row in sources], self.old['evaluator_sources'])

    def test_preparation_has_no_numerical_or_scheduler_calls(self):
        tree = ast.parse(Path(prepare.__file__).read_text())
        imported = {alias.name.split('.')[0] for node in ast.walk(tree)
                    if isinstance(node, ast.Import) for alias in node.names}
        self.assertTrue(imported.isdisjoint({'torch', 'numpy', 'subprocess', 'requests', 'wandb'}))
        call_names = {ast.unparse(node.func) for node in ast.walk(tree)
                      if isinstance(node, ast.Call)}
        self.assertFalse(any('torch.load' in name or 'isfinite' in name
                             or 'sbatch' in name or 'from_pretrained' in name
                             for name in call_names))


if __name__ == '__main__':
    unittest.main()
