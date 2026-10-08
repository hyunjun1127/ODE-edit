"""Narrow CPU controls for S2 binding; not native model qualification."""
import copy
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from official.experiments.prepare import ROOT, digest, read
from official.runners.server2 import assets


class AssetControls(unittest.TestCase):
    def test_official_gptj_scope_and_layer_maps(self):
        contract, _ = assets.load_plan()
        self.assertEqual(contract['models']['gptj']['revision'], assets.REVISION)
        self.assertEqual(contract['models']['gptj']['layers'], list(assets.LAYERS))
        self.assertEqual([assets.LAYERS[i] for i in (0, 5)], [3, 8])
        self.assertEqual(contract['stream']['requests'], 2000)

    def test_default_paths_use_assets_not_algorithm_imports(self):
        paths = assets.defaults()
        self.assertEqual(paths['easyedit_root'], '/mnt/raid5/janghj/EasyEdit')
        self.assertEqual(Path(paths['model_snapshot']).name, assets.REVISION)
        self.assertEqual(Path(paths['zsre_source']).name, 'zsre_mend_eval.json')
        self.assertIn('local/datasets/counterfact-fixed-10k-v1', paths['cf_source'])
        self.assertNotIn('sys.path', Path(assets.__file__).read_text())

    def test_layer_moment_header_and_count_without_dense_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'small.npz'
            np.savez(path, **{'mom2.constructor': 'SecondMoment',
                'mom2.count': assets.COUNT, 'mom2.mom2': np.zeros((2, 2), np.float32),
                'sample_size': 100000})
            with mock.patch.object(assets, 'SHAPE', [2, 2]):
                value = assets.stats_metadata(path)
            self.assertEqual(value['count'], 54924275)
            self.assertEqual(value['native_covariance'], 'mom2_sum / count')
            self.assertEqual(value['dtype'], 'float32')

    def test_invalid_count_is_not_sample_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'small.npz'
            np.savez(path, **{'mom2.constructor': 'SecondMoment', 'mom2.count': 100000,
                'mom2.mom2': np.zeros((2, 2), np.float32), 'sample_size': 100000})
            with mock.patch.object(assets, 'SHAPE', [2, 2]):
                with self.assertRaisesRegex(assets.AssetBindingError, 'C0_COUNT_CHANGED'):
                    assets.stats_metadata(path)

    def test_reused_full_hash_requires_exact_stat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'asset'
            path.write_bytes(b'old')
            row = assets.member(path)
            self.assertEqual(assets.verify_member(row)['verification'],
                'PRIOR_FULL_SHA256_PLUS_CURRENT_UNCHANGED_STAT')
            path.write_bytes(b'new-and-longer')
            with self.assertRaisesRegex(assets.AssetBindingError, 'SEALED_ASSET_STAT_CHANGED'):
                assets.verify_member(row)

    def test_missing_file_typed_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'missing.pt'
            with self.assertRaises(assets.AssetBindingError) as caught:
                assets.member(path)
            self.assertEqual(caught.exception.code, 'ASSET_MISSING')
            self.assertEqual(caught.exception.path, str(path))

    def test_stream_exact_order_boundaries(self):
        for dataset in ('cf', 'zsre'):
            locked = read(ROOT / 'hparams' / f'{dataset}-stream.lock.json')
            assets.validate_stream_lock(copy.deepcopy(locked), locked)
            altered = copy.deepcopy(locked)
            altered['batches'][0]['end_exclusive'] = 99
            with self.assertRaisesRegex(assets.AssetBindingError, 'STREAM_LOCK_MISMATCH'):
                assets.validate_stream_lock(altered, locked)
        self.assertEqual(read(ROOT / 'hparams/cf-stream.lock.json')['ordered_case_ids_sha256'],
                         '0b912d11659eb087ee71a391b7bea1e02ecc9d999a48254eb8559434965640f4')

    def test_tokenizer_identity_canonical_sha(self):
        expected = read(ROOT / 'hparams/tokenizers.lock.json')['audits']['gptj-cf']
        self.assertEqual(digest(expected['tokenizer_files_sha256']), expected['tokenizer_sha256'])
        assets.validate_tokenizer_audit(copy.deepcopy(expected), expected)
        altered = copy.deepcopy(expected)
        altered['lookup_zero_count'] = 0
        with self.assertRaisesRegex(assets.AssetBindingError, 'TOKENIZER_AUDIT_MISMATCH'):
            assets.validate_tokenizer_audit(altered, expected)

    def test_output_scope(self):
        self.assertEqual(assets._output(assets.DEFAULT_OUT), assets.DEFAULT_OUT)
        with self.assertRaisesRegex(assets.AssetBindingError, 'OUTPUT_OUTSIDE'):
            assets._output('/tmp/unrelated')

    def test_manifest_mutation_rejected_before_assets_open(self):
        value = dict(schema='official-server2-assets-v1', GPU=0)
        value['assets_sha256'] = digest(value)
        value['GPU'] = 1
        with self.assertRaisesRegex(assets.AssetBindingError, 'ASSET_MANIFEST_IDENTITY'):
            assets.verify(value)

    def test_imported_frozen_source_bytes_not_preparation_wt_inode(self):
        with tempfile.TemporaryDirectory() as tmp:
            original, archive = Path(tmp)/'old-official', Path(tmp)/'archive-official'
            original.mkdir(); archive.mkdir()
            old = original/'source.py'; old.write_text('EXACT_PIN=1\n')
            row = assets.member(old)
            copied = archive/'source.py'; copied.write_text('EXACT_PIN=1\n')
            old.unlink()  # frozen runner must not depend on live preparation WT
            with mock.patch.object(assets,'ROOT',archive):
                assets.verify_official_source(row,original)
                copied.write_text('EXACT_PIN=2\n')
                with self.assertRaisesRegex(assets.AssetBindingError,'IMPORTED_OFFICIAL_SOURCE_SHA_CHANGED'):
                    assets.verify_official_source(row,original)
                with self.assertRaisesRegex(assets.AssetBindingError,'OFFICIAL_SOURCE_MEMBER_SCOPE'):
                    assets.verify_official_source(dict(row,path=str(Path(tmp)/'other.py')),original)


if __name__ == '__main__':
    unittest.main()
