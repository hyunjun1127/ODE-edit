import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from official.runners.server4.qwen_submission_plan import plan, require_execution_enabled
from official.runners.server4.qwen_pipeline import run


class PreparationTests(unittest.TestCase):
    def test_qualification_entry_is_explicitly_disabled_not_pass(self):
        from official.runners.server4.qwen_run import qualify, execute
        from types import SimpleNamespace
        with self.assertRaisesRegex(RuntimeError, 'NOT_RUN_USER_DISABLED'):
            qualify(None)
        for args in (SimpleNamespace(qualify_b3_metrics=True), SimpleNamespace(stop_after_batch=1)):
            with self.assertRaisesRegex(RuntimeError, 'NOT_RUN_USER_DISABLED'):
                execute(args)

    def test_main_only_no_gpu_validation_or_resume_gate(self):
        from official.runners.server4.qwen_plan import rows
        for dataset in ('cf', 'zsre'):
            row = next(r for r in rows() if r['config']['dataset'] == dataset and r['config']['method'] == 'FT')
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); (root/'configs').mkdir()
                logical = row['logical_main_row']
                (root/'configs'/f'{logical}.json').write_text(json.dumps(row['config']))
                calls = []
                def child(root, command, output, **kwargs):
                    calls.append((command, kwargs.get('extra', ())))
                    if command == 'w0':
                        output.mkdir(parents=True)
                        (output/'w0-receipt.json').write_text('{}')
                    else:
                        (output/'checkpoint').mkdir(parents=True)
                        (output/'checkpoint/latest.json').write_text(json.dumps({'batch':20, 'final_W20':True}))
                        (output/'commits').mkdir()
                        for b in range(1,21):
                            (output/'commits'/f'b{b:02d}.json').write_text(json.dumps({'completed_batch':b}))
                with patch.dict('os.environ', SLURM_JOB_ID='CPU_FIXTURE_NOT_SUBMITTED'), patch('official.runners.server4.qwen_pipeline.child', child):
                    run(root, logical)
                self.assertEqual(calls, [('w0', ()), ('execute', ())])
                self.assertFalse((root/'qualification').exists())
                receipt=json.loads((root/'validation'/f'{logical}.json').read_text())
                self.assertEqual(receipt['qualification'],'NOT_RUN_USER_DISABLED')
                self.assertEqual(receipt['GPU_resume_equivalence'],'NOT_RUN_USER_DISABLED')

    def test_twelve_serial_cells_no_real_ids(self):
        p = plan()
        self.assertTrue(p['submission_enabled'])
        self.assertEqual(p['qualification'], 'NOT_RUN_USER_DISABLED')
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
        with patch('official.runners.server4.qwen_submission_plan.USER_SUBMISSION_HOLD', True), patch('subprocess.run', side_effect=AssertionError('process forbidden')):
            with self.assertRaisesRegex(RuntimeError, 'USER_PREPARATION_ONLY'):
                run('/nonexistent', 'qwen25-cf-ft')
            with self.assertRaisesRegex(RuntimeError, 'USER_PREPARATION_ONLY'):
                require_execution_enabled()


if __name__ == '__main__':
    unittest.main()
