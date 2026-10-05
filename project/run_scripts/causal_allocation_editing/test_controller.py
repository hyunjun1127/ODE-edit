"""Small CPU fixtures for only the new controller state boundaries."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from . import batches,digest
from .run import BatchTransaction,commit_payload,Events
from .entry import make_native_rows

class Bench:
    contexts=[['{}'],['prefix {}']]

class FakeAdapter:
    def __init__(self):
        self.weights={4:torch.nn.Parameter(torch.zeros(2,3),requires_grad=False)}
        self.last_virtual={'old':'cache'};self.capture_virtual=False
    def guard(self):return 'immutable-other-parameters'
    def hook_signature(self):return ()

class ControllerTests(unittest.TestCase):
    def test_full_kl_native_tokens_not_lookup_crop(self):
        pack=dict(row_kind=['rewrite','kl'],row_request=[0,0],lookup=[1,1],
            tokens=dict(input_ids=torch.tensor([[10,11,12,13],[20,21,22,23]]),
                        attention_mask=torch.ones(2,4,dtype=torch.long)),
            targets=torch.tensor([[1,2,3,4],[-100,-100,-100,-100]]))
        rows=make_native_rows(pack)
        self.assertEqual(rows[1]['tokens']['input_ids'].tolist(),[20,21,22,23])
        self.assertEqual(rows[1]['lookup'],1)
        self.assertEqual(rows[1]['target'].shape,(4,))
    def test_native_full_rows_remove_only_right_padding(self):
        pack=dict(row_kind=['kl'],row_request=[0],lookup=[1],
            tokens=dict(input_ids=torch.tensor([[20,21,22,0]]),attention_mask=torch.tensor([[1,1,1,0]])),
            targets=torch.full((1,4),-100))
        self.assertEqual(make_native_rows(pack)[0]['tokens']['input_ids'].tolist(),[20,21,22])
    def test_twenty_and_tail_no_extra_entry(self):
        records=list(range(2000));groups=list(batches(records))
        self.assertEqual(len(groups),20);self.assertEqual(groups[-1][0],20)
        self.assertEqual(groups[-1][1],list(range(1900,2000)));self.assertEqual(groups[-1][2],records)
        self.assertEqual([len(g[1]) for g in batches(list(range(203)),100)],[100,100,3])
    def test_empty_is_noop(self):self.assertEqual(list(batches([])),[])
    def test_invalid_batch(self):
        with self.assertRaises(RuntimeError):list(batches([1],0))
    def test_failed_batch_restores_only_current_entry_and_cursor(self):
        a=FakeAdapter();H={4:torch.eye(3)};bench=Bench();bench.contexts=copy.deepcopy(Bench.contexts);cursor=[1,2]
        with self.assertRaisesRegex(ValueError,'technical'):
            tx=BatchTransaction(a,H,bench,cursor)
            with tx:
                a.weights[4].add_(2);H[4].add_(3);cursor.append(3);bench.contexts[0][0]='changed'
                a.last_virtual={'bad':'cache'};raise ValueError('technical')
        self.assertTrue(tx.rollback_verified);self.assertEqual(cursor,[1,2]);self.assertEqual(bench.contexts,Bench.contexts)
        self.assertTrue(torch.equal(a.weights[4],torch.zeros(2,3)));self.assertTrue(torch.equal(H[4],torch.eye(3)))
        self.assertEqual(a.last_virtual,{'old':'cache'})
    def test_success_persists_next_entry_and_history(self):
        a=FakeAdapter();H={4:torch.zeros(3,3)};bench=Bench();cursor=[]
        with BatchTransaction(a,H,bench,cursor) as tx:
            a.weights[4].add_(2);H[4].add_(3);cursor.append(17);tx.finish()
        self.assertTrue(torch.equal(a.weights[4],torch.full((2,3),2.)));self.assertEqual(cursor,[17])
        with BatchTransaction(a,H,bench,cursor) as second:
            self.assertTrue(torch.equal(second.W[4],torch.full((2,3),2.)))
            second.finish()
    def test_context_mutation_blocks_finish(self):
        a=FakeAdapter();H={4:torch.zeros(3,3)};bench=Bench();bench.contexts=copy.deepcopy(Bench.contexts)
        with self.assertRaisesRegex(RuntimeError,'TRANSACTION_CONTEXT'):
            with BatchTransaction(a,H,bench,[]) as tx:
                bench.contexts.append(['future']);tx.finish()
    def test_scalar_events_independent_namespace(self):
        with tempfile.TemporaryDirectory() as d:
            e=Events(Path(d)/'events.jsonl',2);e(dict(candidate_id=0,accepted=True));e.emit('cost',dict(Q=1.))
            import json
            rows=[json.loads(line) for line in e.path.read_text().splitlines()]
            self.assertTrue(all(r['batch']==2 and r['experiment']=='causal-allocation-editing' for r in rows))
    def test_terminal_no_cp_code_path(self):
        import inspect
        from . import run
        source=inspect.getsource(run)
        self.assertNotIn('torch.save(',source)
        self.assertIn("w.copy_(value)",source)
        self.assertIn("H[l].add_(K@K.T)",source)
        self.assertIn("number<=profile['batches']",source)
        from .profile import execution
        self.assertEqual(execution({})['batches'],20)

if __name__=='__main__':unittest.main()
