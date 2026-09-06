"""Focused portability tests; no model or scientific kernel execution."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import server4_takeover as binding


class TakeoverTests(unittest.TestCase):
    def test_only_l8_mapping(self):
        self.assertEqual(binding.CHAINS[4:6], (('llama3-8b-inst', 'L8_ONLY_NATIVE'),
                                            ('qwen2.5-7b-inst', 'L8_ONLY_NATIVE')))
        for index in range(4):
            with self.assertRaisesRegex(RuntimeError, 'ONLY_TWO_L8_CHAINS'):
                binding.validate_handoff(Path('/unused'), 'main', index)

    def test_create_once_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'receipt'
            binding.write_bytes(target, b'first')
            with self.assertRaises(FileExistsError):
                binding.write_bytes(target, b'second')
            self.assertEqual(target.read_bytes(), b'first')
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)

    def test_path_only_binding(self):
        with patch.object(binding.assets, 'OFFICIAL_EASYEDIT_ROOT'), \
             patch.object(binding.assets, 'EASYEDIT_ARTIFACT_ROOT'), \
             patch.object(binding.assets, 'HF_HUB_CACHE_ROOT'):
            binding.bind_paths()
            self.assertEqual(binding.assets.OFFICIAL_EASYEDIT_ROOT, binding.OFFICIAL)
            self.assertEqual(binding.assets.EASYEDIT_ARTIFACT_ROOT, binding.ARTIFACTS)
            self.assertEqual(binding.assets.HF_HUB_CACHE_ROOT, binding.HUB)


if __name__ == '__main__':
    unittest.main()
