"""CPU fake-observer crash tests; no model load/evaluator/GPU execution."""
import json
from pathlib import Path
import tempfile
import unittest

import torch

from project.run_scripts.single_layer_zflow.durable import CheckpointStore, prepare_state
from project.run_scripts.single_layer_zflow.runner import (
    atomic_json, check_resume_latest, ensure_observation, next_private_attempt,
    reconcile_observations,
)
from project.run_scripts.single_layer_zflow.runtime import file_sha, save


class RunnerRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = CheckpointStore(self.root / 'checkpoints')
        self.records = [{'case_id': i} for i in range(1000)]
        self.source = {'source_head': 'fixture-source', 'source_tree': 'fixture-tree'}
        self.calls = []
        self.model = {'state': None}
        self.w = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.m = torch.zeros(3, 3)

    def checkpoints(self, count=1):
        parent = None
        for index in range(1, count + 1):
            state = prepare_state(self.w, self.m, self.w.clone(), torch.zeros(2, 100),
                                  torch.zeros(100, 3), torch.zeros(100, 100), torch.zeros(3, 100),
                                  accepted=0, request_count=100)
            parent = self.store.publish(f'B{index:03d}', parent, index + 1, state,
                config={'main': True}, source=self.source, context=[['{}']],
                rng={'cpu': torch.Generator().manual_seed(10).get_state()},
                ledger={'flow': {'status': 'RESOURCE_STOP', 'accepted': 0, 'rejected': 2,
                                 'oracle_calls': 3, 'flow_seconds': 1.},
                        'preparation': {'total_preparation_seconds': 2.}},
                cache_resume_fingerprint='fixture-cache')
        self.model['state'] = parent['payload_sha256']
        return parent

    def observer(self, model, tok, records, history, path, *, state_binding):
        self.assertEqual(model['state'], state_binding['checkpoint_payload_sha256'])
        self.calls.append((state_binding['kind'], len(records)))
        metrics = {}
        for kind, multiple in [('RS', 1), ('PS', 2), ('NS', 10)]:
            rows = []
            for record in records:
                for prompt in range(multiple):
                    new, true = (2., 1.) if kind == 'NS' else (1., 2.)
                    rows.append(dict(identity=f'{kind}-{record["case_id"]}-{prompt}',
                        case_id=record['case_id'], prompt_index=prompt, new_nll=new, true_nll=true,
                        success=True, new_strict=True, true_strict=False))
            metrics[kind] = dict(rows=rows, numerator=len(rows), denominator=len(rows), rate=1.)
        save(path, dict(metrics=metrics, state_binding=state_binding,
            request_order=state_binding['request_order'], before_after_exact=True,
            evaluator_controller_influence=0, new_forward=True, evaluation_seconds=.5,
            weight_state={'weight': 'fixture'}, cache_sha256='fixture-cache'))

    def reconcile(self, receipt, **kwargs):
        return reconcile_observations(self.root, int(receipt['batch_id'][1:]), receipt,
            self.source, self.records, self.model, None, self.m,
            observer=kwargs.pop('observer', self.observer), **kwargs)

    def test_crash_after_checkpoint_before_observation_recovers_without_edit(self):
        receipt = self.checkpoints()
        checkpoint_sha = file_sha(self.root / 'checkpoints/B001/state.pt')
        first = self.reconcile(receipt)
        self.assertEqual(self.calls, [('current', 100)])
        self.assertEqual(first['metrics']['NS']['denominator'], 1000)
        self.assertTrue(first['recovery'])
        self.assertIsNone(first['commit_seconds'])
        self.assertEqual(first['missing_interrupted_timing'], 'NOT_RECORDED')
        second = self.reconcile(receipt)
        self.assertEqual(self.calls, [('current', 100)])
        self.assertEqual(first, second)
        self.assertEqual(checkpoint_sha, file_sha(self.root / 'checkpoints/B001/state.pt'))

    def test_complete_unregistered_raw_is_salvaged_without_duplicate_forward(self):
        receipt = self.checkpoints()
        def crash(point):
            if point == 'after_observation_result':
                raise RuntimeError('injected interruption')
        with self.assertRaises(RuntimeError):
            self.reconcile(receipt, crash_hook=crash)
        self.assertFalse((self.root / 'B001/COMPLETE.json').exists())
        self.reconcile(receipt)
        self.assertEqual(self.calls, [('current', 100)])
        self.assertTrue((self.root / 'B001/observation-registry/current/000001.json').is_file())

    def test_registered_result_survives_crash_before_complete_marker(self):
        receipt = self.checkpoints()
        def crash(point):
            if point == 'after_observation_registry':
                raise RuntimeError('injected interruption')
        with self.assertRaises(RuntimeError):
            self.reconcile(receipt, crash_hook=crash)
        self.reconcile(receipt)
        self.assertEqual(len(self.calls), 1)

    def test_partial_raw_is_preserved_and_new_private_observation_runs(self):
        receipt = self.checkpoints()
        def broken(model, tok, records, history, path, *, state_binding):
            with Path(path).open('x') as handle:
                handle.write('{"partial":')
            raise RuntimeError('injected write interruption')
        with self.assertRaises(RuntimeError):
            self.reconcile(receipt, observer=broken)
        partial = self.root / 'B001/observation-attempts/current-r0001/result.json'
        before = partial.read_bytes()
        summary = self.reconcile(receipt)
        self.assertEqual(partial.read_bytes(), before)
        self.assertIn('current-r0002', summary['artifacts']['current']['relative_path'])
        self.assertEqual(len(self.calls), 1)

    def test_sealed_corruption_is_blocked_not_silently_re_evaluated(self):
        receipt = self.checkpoints()
        summary = self.reconcile(receipt)
        raw_path = self.root / 'B001' / summary['artifacts']['current']['relative_path']
        with raw_path.open('a') as handle:
            handle.write('corrupt')
        with self.assertRaisesRegex(RuntimeError, 'SEALED_OBSERVATION_CORRUPT'):
            self.reconcile(receipt)
        self.assertEqual(len(self.calls), 1)

    def test_wrong_checkpoint_binding_never_reuses_sealed_metrics(self):
        receipt = self.checkpoints()
        self.reconcile(receipt)
        changed = dict(receipt, payload_sha256='other-state')
        with self.assertRaisesRegex(RuntimeError, 'SEALED_OBSERVATION_CORRUPT'):
            self.reconcile(changed)
        self.assertEqual(len(self.calls), 1)

    def test_w5_requires_current_and_seen_first500_before_complete(self):
        receipt = self.checkpoints(5)
        summary = self.reconcile(receipt)
        self.assertEqual(self.calls, [('current', 100), ('seen-full', 500)])
        self.assertEqual(summary['artifacts']['seen-full']['metrics']['PS']['denominator'], 1000)
        self.assertEqual(set(summary['artifacts']), {'current', 'seen-full'})

    def test_w10_derived_first500_requires_no_extra_forward_and_preserves_denominators(self):
        receipt = self.checkpoints(10)
        summary = self.reconcile(receipt)
        self.assertEqual(self.calls, [('current', 100), ('seen-full', 1000)])
        self.assertEqual(summary['artifacts']['seen-full']['metrics']['NS']['denominator'], 10000)
        self.assertEqual(summary['artifacts']['first500']['metrics']['NS']['denominator'], 5000)
        self.assertFalse(summary['artifacts']['first500']['new_forward'])
        self.reconcile(receipt)
        self.assertEqual(len(self.calls), 2)

    def test_older_resume_is_rejected_and_private_attempts_do_not_advance_state(self):
        check_resume_latest(self.store, 'W0')
        self.checkpoints(2)
        with self.assertRaisesRegex(RuntimeError, 'OLDER_RESUME'):
            check_resume_latest(self.store, 'B001')
        with self.assertRaisesRegex(RuntimeError, 'OLDER_RESUME'):
            check_resume_latest(self.store, 'W0')
        check_resume_latest(self.store, 'B002')
        (self.store.root / '.B003.partial-interrupted').mkdir()
        check_resume_latest(self.store, 'B002')

    def test_uncommitted_edit_attempt_is_not_overwritten(self):
        root = self.root / 'B002/edit-attempts'
        first = next_private_attempt(root, 'attempt')
        save(first / 'flow.json', {'failed': True})
        second = next_private_attempt(root, 'attempt')
        self.assertNotEqual(first, second)
        self.assertEqual(json.loads((first / 'flow.json').read_text()), {'failed': True})

    def test_atomic_json_create_once_preserves_existing_receipt(self):
        target = self.root / 'events/one.json'
        atomic_json(target, {'state': 1})
        with self.assertRaises(FileExistsError):
            atomic_json(target, {'state': 2})
        self.assertEqual(json.loads(target.read_text()), {'state': 1})


if __name__ == '__main__':
    unittest.main()
