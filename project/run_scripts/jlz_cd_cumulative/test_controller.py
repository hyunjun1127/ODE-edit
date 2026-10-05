"""Two-batch RAM state links, rollback, whole horizon and scalar DAG fixtures."""
import unittest
from unittest.mock import patch
import torch
from project.run_scripts.jlz_realization.writer import Transaction
from .common import batches, selected_for_post, state
from .submit import admission, arguments

class FakeAdapter:
    def __init__(self):
        self.weights = {4: torch.zeros(2, 3)}
    def guard(self):
        return 'UNCHANGED_NONSELECTED'

class ControllerTests(unittest.TestCase):
    def test_two_successful_batches_and_failed_third_preserve_prefix(self):
        a = FakeAdapter(); H = {4: torch.zeros(3, 3)}; joined = []
        for i in (1, 2):
            before = state(a, H)
            with Transaction(a, H) as tx:
                a.weights[4].add_(1); H[4].add_(torch.eye(3)); tx.finish()
            joined.append((before, state(a, H)))
        self.assertEqual(joined[0][1], joined[1][0])
        final = state(a, H)
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            with Transaction(a, H) as tx:
                a.weights[4].add_(123); H[4].add_(456)
                raise RuntimeError('fixture')
        self.assertEqual(state(a, H), final); self.assertTrue(tx.rollback_verified)
    def test_horizon_and_seen_prefix(self):
        records = list(range(2000)); parts = list(batches(records, 100))
        self.assertEqual(len(parts), 20); self.assertEqual(parts[-1][0], 20)
        self.assertEqual(sum(len(cur) for _, cur, _ in parts), 2000)
        for b, cur, seen in parts:
            self.assertEqual(len(seen), b * 100)
            self.assertEqual(selected_for_post(cur, seen, b), seen if b in (5, 10, 15, 20) else cur)
    def test_single_slot_producer_first_no_deadlock(self):
        inventory = {'jobs': []}
        barrier, serial = admission(inventory, 3, 'CfgTRES=cpu=128,gres/gpu=8\nAllocTRES=cpu=34,gres/gpu=7')
        self.assertEqual(barrier, []); self.assertTrue(serial)
        barrier, serial = admission({'jobs': [{'job':'123', 'gpus':3}]}, 3,
                                    'CfgTRES=gres/gpu=8\nAllocTRES=gres/gpu=7')
        self.assertEqual(barrier, ['123']); self.assertTrue(serial)
    def test_explicit_resources(self):
        from pathlib import Path
        r = dict(host_mib=59392, collector_host_mib=24576, wall='2-00:00:00', collector_wall='04:00:00')
        argv = arguments('CD_C', 'afterany:123', Path('/exact/task/attempt-r1'), r)
        self.assertIn('--mem=59392M', argv); self.assertIn('--export=NONE', argv)
        self.assertIn('--no-requeue', argv); self.assertIn('--dependency=afterany:123', argv)
        self.assertIn('--gres=gpu:1', argv)

if __name__ == '__main__':
    unittest.main()
