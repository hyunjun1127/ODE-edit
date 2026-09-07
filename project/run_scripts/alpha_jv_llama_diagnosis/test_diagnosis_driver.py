"""Minimal preparatory-driver fixtures; no external data/model/runtime calls."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from .contracts import PUBLICATION_HEAD
from .diagnosis_driver import (FrozenDiagnosisBundle, DiagnosisInputBoundary,
                              analyze_frozen_bundle, read_publication_availability)


class DiagnosisDriverTests(unittest.TestCase):
    def arguments(self):
        generator = torch.Generator().manual_seed(20260907)
        target = torch.randn((3, 3), dtype=torch.float32, generator=generator)
        entry = torch.zeros_like(target)
        evidence = dict(track='D', fixture='CPU-D-B10', model='CPU_SYNTHETIC',
            provenance_mode='CPU_SYNTHETIC_FIXTURE', source_sha='0'*40,
            qN_ref=2., cache_c_new=False)
        for name in ('entry_W_sha256', 'entry_M_sha256', 'fixed_z_sha256',
                     'request_order_sha256', 'semantic_inventory_sha256',
                     'context_sha256', 'metric_identity_sha256'):
            evidence[name] = 'a'*64
        return dict(target32=target, entry_captures32=(entry, entry.clone(), entry.clone()),
            raw_responses32=torch.randn((2, 3, 3), dtype=torch.float32, generator=generator),
            q_layers=torch.tensor([.4, .6], dtype=torch.float64),
            metric=torch.eye(2, dtype=torch.float64), layer_ids=(4, 8),
            request_ids=('a', 'b', 'c'), evidence=evidence)

    def test_frozen_workflow_tables_and_missing_endpoint_status(self):
        bundle = FrozenDiagnosisBundle.seal(**self.arguments())
        result = analyze_frozen_bundle(bundle)
        self.assertEqual(result['status'], 'CPU_FROZEN_RAW_ATTRIBUTION_COMPLETE')
        self.assertEqual(len(result['tables']['request_response_contributions']), 6)
        self.assertEqual(len(result['tables']['all_row_influence']), 6)
        self.assertEqual(len(result['tables']['normalization_same_state']), 2)
        self.assertEqual(result['endpoint_status'], 'NOT_RECORDED_ACTUAL_ENDPOINT_CAPTURE')
        self.assertFalse(result['S_blocked_by_D'])
        self.assertEqual(result['actual_write_count'], 0)
        self.assertEqual(result['receipt_identity_sha256'], analyze_frozen_bundle(bundle)['receipt_identity_sha256'])

    def test_provided_endpoint_not_entry_replacement(self):
        args = self.arguments()
        args['endpoint_terminals'] = {'D-N0': args['target32']*.3}
        result = analyze_frozen_bundle(FrozenDiagnosisBundle.seal(**args), include_all_row=False)
        self.assertEqual(set(result['endpoint_cross_objective']), {'D-N0'})
        self.assertEqual(result['tables']['all_row_influence'], [])
        self.assertNotEqual(result['entry_cross_objective']['endpoint_terminal_sha256'],
                            result['endpoint_cross_objective']['D-N0']['endpoint_terminal_sha256'])

    def test_missing_bundle_does_not_block_s(self):
        result = analyze_frozen_bundle(None, availability={'status': 'HISTORICAL_EXACT_STATE_UNAVAILABLE_ON_SERVER1'})
        self.assertFalse(result['S_blocked_by_D'])
        self.assertEqual(result['tables'], {})
        self.assertEqual(result['replay_action_count'], 0)

    def test_input_alias_and_tamper_guard(self):
        args = self.arguments()
        bundle = FrozenDiagnosisBundle.seal(**args)
        args['target32'].zero_()
        bundle.assert_frozen()
        bundle.raw_responses32[0, 0, 0] += 1
        with self.assertRaisesRegex(DiagnosisInputBoundary, 'MUTATED'):
            analyze_frozen_bundle(bundle)

    def test_evidence_dtype_and_request_rejection(self):
        args = self.arguments(); args['evidence']['qN_ref'] = float('nan')
        with self.assertRaises(DiagnosisInputBoundary):
            FrozenDiagnosisBundle.seal(**args)
        args = self.arguments(); args['target32'] = args['target32'].double()
        with self.assertRaises(DiagnosisInputBoundary):
            FrozenDiagnosisBundle.seal(**args)
        args = self.arguments(); args['request_ids'] = ('a', 'a', 'b')
        with self.assertRaises(DiagnosisInputBoundary):
            FrozenDiagnosisBundle.seal(**args)

    def test_git_publication_reader_pins_commit_and_never_fetches(self):
        checkpoint = b'alias,arm,batch,path,sha256,bytes,journal_replay_parity\n'
        contexts = b'chain,raw_local_path,file_sha256,bytes\n'
        with patch('subprocess.check_output', side_effect=[PUBLICATION_HEAD+'\n', checkpoint, contexts]) as call:
            result = read_publication_availability('/read-only-repo')
        self.assertEqual(result['publication_head'], PUBLICATION_HEAD)
        self.assertEqual(len(result['publication_members']), 2)
        self.assertFalse(result['availability']['S_blocked_by_D'])
        for args in call.call_args_list:
            command = args.args[0]
            self.assertEqual(command[0], 'git')
            self.assertFalse(any(value in command for value in ('fetch', 'checkout', 'pull', 'push')))

    def test_driver_no_model_or_writer_imports(self):
        tree = ast.parse((Path(__file__).parent/'diagnosis_driver.py').read_text())
        imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        for module in imported:
            self.assertFalse(any(name in (module or '') for name in ('runtime', 'fixture', 'overlay', 'trajectory')))
        attrs = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertFalse(any(value in attrs for value in ('load', 'from_pretrained', 'cuda', 'run_joint', 'run_official')))


if __name__ == '__main__':
    unittest.main()
