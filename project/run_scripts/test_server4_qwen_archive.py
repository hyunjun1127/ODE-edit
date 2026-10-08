import unittest
from unittest.mock import patch
from project.run_scripts.server4_qwen_archive import archive


class ArchiveDraftTests(unittest.TestCase):
    def test_archive_draft_cannot_transfer_or_delete(self):
        with patch('subprocess.run', side_effect=AssertionError('network forbidden')):
            with self.assertRaisesRegex(RuntimeError, 'ARCHIVE_CALLER_INTEGRATION_NOT_VALIDATED'):
                archive('/nonexistent', 'qwen25-cf-ft')


if __name__ == '__main__':
    unittest.main()
