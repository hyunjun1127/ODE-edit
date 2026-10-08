import unittest
from unittest.mock import patch
from official.runners.server4.qwen_submission_plan import plan, require_execution_enabled
from official.runners.server4.qwen_pipeline import run


class PreparationTests(unittest.TestCase):
    def test_twelve_serial_cells_no_real_ids(self):
        p = plan()
        self.assertFalse(p['submission_enabled'])
        self.assertEqual(p['job_ids'], [])
        self.assertEqual(p['project_GPU_cap'], 2)
        nodes = p['nodes']
        self.assertEqual(len(nodes), 24)
        self.assertEqual(sum(n['GPUs'] for n in nodes), 12)
        for i, n in enumerate(nodes):
            self.assertIsNone(n['job_id'])
            if i:
                self.assertEqual(n['dependency_symbol'], nodes[i-1]['symbol'])

    def test_hold_precedes_files_and_processes(self):
        with patch('subprocess.run', side_effect=AssertionError('process forbidden')):
            with self.assertRaisesRegex(RuntimeError, 'USER_PREPARATION_ONLY'):
                run('/nonexistent', 'qwen25-cf-ft')
            with self.assertRaisesRegex(RuntimeError, 'USER_PREPARATION_ONLY'):
                require_execution_enabled()

    def test_archive_draft_cannot_transfer_or_delete(self):
        from project.run_scripts.server4_qwen_archive import archive
        with patch('subprocess.run', side_effect=AssertionError('network forbidden')):
            with self.assertRaisesRegex(RuntimeError, 'USER_PREPARATION_ONLY'):
                archive('/nonexistent', 'qwen25-cf-ft')


if __name__ == '__main__':
    unittest.main()
