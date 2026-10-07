"""CPU production-loop/rollback fixtures, never native fitting or GPU parity."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from . import run


class View:
    def __init__(self):
        self.model = torch.nn.Module()
        self.weights = {}
        for layer in (13,14,15,16,17):
            value = torch.nn.Parameter(torch.zeros((2,2),dtype=torch.float32))
            self.model.register_parameter('weight'+str(layer),value)
            self.weights[layer] = value
        self.model.register_parameter('bias',torch.nn.Parameter(torch.zeros(1)))
    def hook_signature(self): return ()


class Engine:
    def __init__(self,view,arm):
        self.view,self.arm,self.context = view,arm,[['{}'],['a {}']]
        self.H = {l:torch.zeros(2,2) for l in run.ARM_LAYERS[arm]}
        self.counts = {k:0 for k in run.EXPECTED[arm]}
        self.calls = []
    def history(self): return self.H
    def context_snapshot(self): return copy.deepcopy(self.context)
    def restore_context(self,value): self.context = copy.deepcopy(value)
    def restore_history(self,clones):
        for l,value in clones.items(): self.H[l].copy_(value)
    def apply(self,requests,batch):
        self.calls.append((batch,requests))
        with torch.no_grad():
            for l in run.ARM_LAYERS[self.arm]:
                self.view.weights[l].add_(1)
                self.H[l].add_(1)
        for k,value in run.EXPECTED[self.arm].items(): self.counts[k] += value
        return self.view.model,dict(arm=self.arm,batch=batch,delta=run.EXPECTED[self.arm],
            same_model_returned=True,native_has_history=True,caller_history_appends=0,
            cache_template=None,native_z_disk_cache=False,checkpoint_saved=False)


class ProductionTests(unittest.TestCase):
    def test_alpha_selected_guard_includes_unselected_middle_and_bias(self):
        view = View()
        original = run.nonselected(view,'ALPHAEDIT_BLUE')
        with torch.no_grad(): view.weights[13].add_(1);view.weights[17].add_(1)
        self.assertEqual(original,run.nonselected(view,'ALPHAEDIT_BLUE'))
        with torch.no_grad(): view.weights[14].add_(1)
        self.assertNotEqual(original,run.nonselected(view,'ALPHAEDIT_BLUE'))

    def test_observer_error_restores_edited_W_H_context_in_RAM(self):
        view = View();engine = Engine(view,'ALPHAEDIT_BLUE')
        bench = SimpleNamespace(contexts=[['{}']])
        before = run.state(view,engine.history())
        tx = run.NativeTransaction(view,engine,bench,'ALPHAEDIT_BLUE')
        with self.assertRaisesRegex(ValueError,'observer'):
            with tx:
                engine.apply([],1);engine.context=[['changed']]
                raise ValueError('observer')
        self.assertTrue(tx.rollback_verified)
        self.assertEqual(before,run.state(view,engine.history()))
        self.assertEqual(engine.context,[['{}'],['a {}']])

    def test_receipt_IO_failure_rolls_back_before_commit_ledger(self):
        view = View();engine = Engine(view,'CAKE');bench = SimpleNamespace(contexts=[['{}']])
        before = run.state(view,engine.history())
        tx = run.NativeTransaction(view,engine,bench,'CAKE')
        with self.assertRaises(OSError):
            with tx:
                engine.apply([],1);tx.finish()
                try: raise OSError('disk')
                except BaseException: tx.done=False;raise
        self.assertTrue(tx.rollback_verified)
        self.assertEqual(before,run.state(view,engine.history()))

    def test_success_preserves_native_history_without_caller_append(self):
        view = View();engine = Engine(view,'CAKE');bench = SimpleNamespace(contexts=[['{}']])
        with run.NativeTransaction(view,engine,bench,'CAKE') as tx:
            engine.apply([],1);tx.finish()
        self.assertTrue(all(float(h.sum()) == 4 for h in engine.H.values()))
        self.assertFalse(tx.rollback_verified)

    def test_wrong_counts_or_history_rejected(self):
        view=View();engine=Engine(view,'ALPHAEDIT_BLUE')
        with self.assertRaisesRegex(RuntimeError,'COUNTS'):
            run.check_native('ALPHAEDIT_BLUE',dict(delta=run.EXPECTED['CAKE']),engine,1)
        engine.H = {}
        with self.assertRaisesRegex(RuntimeError,'HISTORY'):
            run.check_native('ALPHAEDIT_BLUE',dict(delta=run.EXPECTED['ALPHAEDIT_BLUE']),engine,1)

    def test_twenty_cumulative_calls_and_current100_at_milestones(self):
        records=[dict(case_id=i,requested_rewrite=dict(prompt='{} works',subject='x',
                    target_new={'str':'new'})) for i in range(2000)]
        view=View();engine=Engine(view,'ALPHAEDIT_BLUE')
        bench=SimpleNamespace(contexts=[['{}']],prepare=lambda rs:
            dict(record_ids=[r['case_id'] for r in rs],identity=str(rs[0]['case_id'])))
        packs=[dict(ids=list(range(i*100,(i+1)*100)),identity=None) for i in range(20)]
        c=dict(packs=packs,native=dict(effective_hparams=dict(ALPHAEDIT_BLUE={})))
        lock=dict(source_commit='a'*40)
        observations=[]
        def observe(view,bench,seen,selected,H,endpoint,out,current_ids):
            observations.append((endpoint,len(selected),len(current_ids)))
            return dict(summary={'requests':len(selected)},current={'requests':len(current_ids)})
        chunks=lambda rs:((i+1,rs[i*100:(i+1)*100],rs[:(i+1)*100]) for i in range(20))
        with tempfile.TemporaryDirectory() as tmp, patch.object(run,'batches',chunks),\
             patch.object(run,'rows',return_value=[]),patch.object(run,'observe',observe),\
             patch.object(run,'native_requests',side_effect=lambda x:x),\
             patch.object(run,'log_batch',return_value=True),patch.object(run.gc,'collect'),\
             patch('builtins.print'):
            commits=[]
            final=run.native_loop(c,lock,Path(tmp),'ALPHAEDIT_BLUE',records,
                view.model,view,engine,bench,SimpleNamespace(),commits)
        self.assertEqual([b for b,_ in engine.calls],list(range(1,21)))
        self.assertEqual(len(commits),20)
        self.assertEqual(final,commits[-1]['after'])
        self.assertEqual([n for ep,n,_ in observations if ep.startswith('W')],
            [i*100 if i in (5,10,15,20) else 100 for i in range(1,21)])
        self.assertTrue(all(current == 100 for _,_,current in observations))
        self.assertEqual(engine.counts['history_appends'],40)
        self.assertEqual(engine.counts['native_z'],4000)
        self.assertEqual(engine.calls[0][1][0]['requested_rewrite']['target_new'],{'str':'new'})
        self.assertTrue(all(commits[i]['before'] == commits[i-1]['after'] for i in range(1,20)))


if __name__ == '__main__': unittest.main()
