import json
from pathlib import Path
import tempfile
import types
import unittest

import torch

from project.run_scripts.multilayer_joint_compensation.history import (
    HistoryBoundary, capture_native_keys, finalize_history_once, reconstruct_history)
from project.run_scripts.multilayer_joint_compensation.contracts import Ledger, sha


def records(n):
    return [dict(case_id=i, requested_rewrite=dict(subject=str(i % 17), prompt='{} is',
        target_new={'str': 'x'}, relation_id='r')) for i in range(n)]


def keys(rr):
    return torch.tensor([[r['case_id'] % 7 + .125 for r in rr],
                         [r['case_id'] % 13 + .25 for r in rr]], dtype=torch.float32)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model = torch.nn.Linear(2, 2).eval()
        self.tok = types.SimpleNamespace(padding_side='right')
        self.hp = types.SimpleNamespace(rewrite_module_tmp='layer.{}', fact_token='subject_last')
        self.contexts = [['{}'], ['prefix {}', 'second {}']]
        self.rr = records(300)

    def tearDown(self):
        self.tmp.cleanup()

    def run_history(self, *, out=None, capture=keys, offset=300, rr=None, entry=None, layer=8, progress=None):
        return reconstruct_history(self.model, self.tok, None, self.hp, self.contexts,
            self.rr if rr is None else rr, offset, Ledger(), self.root / 'history' if out is None else out,
            entry_identity={'We': 'sealed'} if entry is None else entry,
            source_identity={'native': 'source-sha'}, input_width=2,
            capture=capture, layer=layer, progress=progress)

    def test_raw_chronological_fp32_native_grouping_and_readonly(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        gram, receipt = self.run_history()
        expected = torch.zeros(2, 2)
        for i in range(0, 300, 100):
            k = keys(self.rr[i:i + 100]); expected += k @ k.T
        self.assertTrue(torch.equal(gram, expected))
        self.assertEqual(receipt['completed_requests'], 300)  # not 17 active facts
        for k, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[k]))

    def test_complete_resume_does_not_recapture_or_change_bytes(self):
        events = []
        first, r1 = self.run_history(progress=events.append)
        self.assertEqual([e['completed_requests'] for e in events], [100, 200, 300])
        self.assertTrue(all(not e['reused'] for e in events))
        files = {p.name: sha(p) for p in (self.root / 'history').iterdir()}
        events = []
        second, r2 = self.run_history(capture=lambda rr: self.fail('forward duplicated'), progress=events.append)
        self.assertTrue(all(e['reused'] for e in events))
        self.assertTrue(torch.equal(first, second)); self.assertEqual(r1, r2)
        self.assertEqual(files, {p.name: sha(p) for p in (self.root / 'history').iterdir()})

    def test_partial_resume_preserves_finished_prefix(self):
        seen = []
        def failing(rr):
            seen.append(rr[0]['case_id'])
            if len(seen) == 2: raise RuntimeError('technical interruption')
            return keys(rr)
        with self.assertRaisesRegex(RuntimeError, 'technical interruption'):
            self.run_history(capture=failing)
        prefix = sha(self.root / 'history/batch-001-keys.pt')
        resumed = []
        def remaining(rr):
            resumed.append(rr[0]['case_id']); return keys(rr)
        _, receipt = self.run_history(capture=remaining)
        self.assertEqual(resumed, [100, 200]); self.assertEqual(receipt['completed_batches'], 3)
        self.assertEqual(prefix, sha(self.root / 'history/batch-001-keys.pt'))

    def test_changed_input_or_entry_fails_before_capture(self):
        self.run_history()
        rr = records(300); rr[0]['requested_rewrite']['subject'] = 'changed'
        with self.assertRaisesRegex(HistoryBoundary, 'INPUT_MISMATCH'):
            self.run_history(rr=rr)
        with self.assertRaisesRegex(HistoryBoundary, 'INPUT_MISMATCH'):
            self.run_history(entry={'We': 'other'})

    def test_corrupted_member_rejected(self):
        self.run_history()
        path = self.root / 'history/batch-001-keys.pt'
        with path.open('ab') as f: f.write(b'corrupt')
        with self.assertRaisesRegex(HistoryBoundary, 'KEY_MEMBER_HASH'):
            self.run_history()

    def test_orphan_file_preserved_and_no_forward(self):
        def fail(rr): raise RuntimeError('before capture')
        with self.assertRaisesRegex(RuntimeError, 'before capture'):
            self.run_history(capture=fail)
        orphan = self.root / 'history/batch-001-keys.pt'
        orphan.write_bytes(b'immutable orphan')
        with self.assertRaisesRegex(HistoryBoundary, 'UNRECEIPTED'):
            self.run_history(capture=lambda rr: self.fail('forward on orphan'))
        self.assertEqual(orphan.read_bytes(), b'immutable orphan')

    def test_empty_and_original_m4_rejected(self):
        g, r = self.run_history(offset=0, capture=lambda rr: self.fail('empty forward'))
        self.assertTrue(torch.equal(g, torch.zeros(2, 2))); self.assertEqual(r['completed_batches'], 0)
        with self.assertRaisesRegex(HistoryBoundary, 'ORIGINAL_M4'):
            self.run_history(layer=4)

    def test_native_scoped_reader_and_arbitrary_batch(self):
        called = []
        def reader(model, tok, contexts, idxs, layer, module, track='in'):
            called.append(len(contexts))
            return torch.tensor([[float(x), float(x) + 1] for x in contexts])
        module = types.SimpleNamespace(get_reprs_at_idxs=reader)
        def compute(model, tok, rr, hp, layer, ctx):
            return module.get_reprs_at_idxs(model, tok, [str(r['case_id']) for r in rr],
                                           [0] * len(rr), layer, 'layer.{}')
        native = types.SimpleNamespace(compute_ks=compute)
        k = capture_native_keys(self.model, self.tok, native, self.hp, self.contexts,
            self.rr[:7], 8, Ledger(), input_width=2, physical_batch=2, repr_module=module)
        self.assertEqual(tuple(k.shape), (2, 7)); self.assertEqual(called, [2, 2, 2, 1])
        self.assertIs(module.get_reprs_at_idxs, reader)
        def fail(*a): raise RuntimeError('forward fail')
        native.compute_ks = fail
        with self.assertRaisesRegex(RuntimeError, 'forward fail'):
            capture_native_keys(self.model, self.tok, native, self.hp, self.contexts,
                self.rr[:1], 8, Ledger(), input_width=2, repr_module=module)
        self.assertIs(module.get_reprs_at_idxs, reader)

    def test_mutated_live_state_rejected(self):
        def capture(rr):
            with torch.no_grad(): self.model.weight.add_(1)
            return keys(rr)
        with self.assertRaisesRegex(HistoryBoundary, 'ENTRY_STATE_CHANGED'):
            self.run_history(capture=capture)

    def test_terminal_once_and_original_history_unchanged(self):
        histories = {4: torch.eye(2), 8: torch.ones(2, 2)}
        original = {l: m.clone() for l, m in histories.items()}
        args = (self.model, self.tok, None, self.hp, self.contexts, self.rr[:7], histories,
                Ledger(), self.root / 'terminal')
        kw = dict(endpoint_identity={'W': 'end'}, source_identity={'native': 'sha'},
                  capture=lambda layer, rr: keys(rr))
        result, receipt = finalize_history_once(*args, **kw)
        self.assertEqual(receipt['terminal_batch_finalizations'], 1)
        self.assertEqual(receipt['terminal_layer_appends'], 2)
        for l in histories:
            self.assertTrue(torch.equal(histories[l], original[l]))
            self.assertTrue(torch.equal(result[l], original[l] + keys(self.rr[:7]) @ keys(self.rr[:7]).T))
        with self.assertRaisesRegex(HistoryBoundary, 'ALREADY_ATTEMPTED'):
            finalize_history_once(*args, **kw)

    def test_symlink_namespace_rejected(self):
        outside = self.root / 'outside'; outside.mkdir()
        link = self.root / 'link'; link.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(HistoryBoundary, 'UNSAFE_DIRECTORY'):
            self.run_history(out=link / 'nested')
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == '__main__': unittest.main()
