"""CPU transaction/terminal-order regression; no native model fitting."""
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
        self.model=torch.nn.Module();self.weights={}
        for l in (13,14,15,16,17):
            w=torch.nn.Parameter(torch.zeros(2,2))
            self.model.register_parameter('w'+str(l),w);self.weights[l]=w
        self.model.register_parameter('bias',torch.nn.Parameter(torch.zeros(1)))
    def hook_signature(self): return ()


class Engine:
    def __init__(self,view,arm):
        self.view,self.arm=view,arm;self.next_batch=1
        self.context=[['{}'],['a {}']];self.cold={};self.pruned=False;self.calls=[]
        self.counts={k:0 for k in run.EXPECTED[arm]}
    def history(self): return {}
    def context_snapshot(self): return copy.deepcopy(self.context)
    def snapshot_native_state(self):
        return dict(context=copy.deepcopy(self.context),cold=dict(self.cold),
                    next_batch=self.next_batch,pruned=self.pruned)
    def restore_native_state(self,value):
        self.context=copy.deepcopy(value['context']);self.cold=dict(value['cold'])
        self.next_batch,self.pruned=value['next_batch'],value['pruned']
    def native_state_signature(self):
        return dict(context=self.context_snapshot(),keys=sorted(self.cold),
                    next_batch=self.next_batch,pruned=self.pruned)
    def apply(self,requests,batch):
        self.calls.append(batch)
        if self.arm=='PRUNE' and batch==1:
            self.cold={l:w.detach().cpu().clone() for l,w in self.view.weights.items()}
        with torch.no_grad():
            for w in self.view.weights.values(): w.add_(1)
        self.next_batch+=1
        for k,v in run.EXPECTED[self.arm].items():self.counts[k]+=v
        return self.view.model,dict(arm=self.arm,batch=batch,delta=run.EXPECTED[self.arm],
            same_model_returned=True,native_has_history=False,caller_history_appends=0,
            native_z_disk_cache=False,cache_template=None,checkpoint_saved=False)
    def finish_batch(self,batch):
        if self.arm=='PRUNE' and batch==20:
            self.pruned=True
            with torch.no_grad():
                for w in self.view.weights.values():w.mul_(.5)
            return dict(prune_applied=True,repair='PRUNE_TERMINAL_BASE_FIX',repair_authorized=True,
                upstream_bitwise_equivalence=False,terminal_transforms=1,final_base='SAVED_COLD_W0',
                spectrum_formula_changed=False,checkpoint_saved=False)
        return dict(prune_applied=False,terminal_transforms=0,no_model_mutation=True)


class RunTests(unittest.TestCase):
    def test_native_no_history_counts_and_bias_guard(self):
        v=View();e=Engine(v,'RECT');before=run.nonselected(v,'RECT')
        _,receipt=e.apply([],1);run.check_native('RECT',receipt,e,1)
        self.assertEqual(before,run.nonselected(v,'RECT'));self.assertEqual(e.history(),{})
        with torch.no_grad():v.model.bias.add_(1)
        self.assertNotEqual(before,run.nonselected(v,'RECT'))

    def test_B1_observer_failure_restores_RAM_cold_cache_and_model(self):
        v=View();e=Engine(v,'PRUNE');b=SimpleNamespace(contexts=[['{}']])
        state=run.state(v,{})
        tx=run.NativeTransaction(v,e,b,'PRUNE')
        with self.assertRaisesRegex(ValueError,'observer'):
            with tx:e.apply([],1);raise ValueError('observer')
        self.assertTrue(tx.rollback_verified);self.assertEqual(run.state(v,{}),state)
        self.assertEqual(e.cold,{});self.assertEqual(e.next_batch,1)
        self.assertEqual(e.counts['native_z'],100) # failed work is never erased

    def test_partial_terminal_failure_restores_flag_and_dense_entry(self):
        v=View();e=Engine(v,'PRUNE');e.next_batch=20;e.cold={13:torch.zeros(2,2)}
        b=SimpleNamespace(contexts=[['{}']]);state=run.state(v,{})
        tx=run.NativeTransaction(v,e,b,'PRUNE')
        with self.assertRaises(OSError):
            with tx:
                e.apply([],20);run.finish_native_batch(e,'PRUNE',20,v)
                tx.finish()
                try:raise OSError('receipt')
                except BaseException:tx.done=False;raise
        self.assertEqual(run.state(v,{}),state);self.assertFalse(e.pruned)
        self.assertEqual(e.next_batch,20);self.assertTrue(tx.rollback_verified)

    def test_terminal_before_only_W20_post_and_twenty_native_calls(self):
        records=[dict(case_id=i,requested_rewrite=dict(prompt='{} works',subject='x',target_new={'str':'new'}))
                 for i in range(2000)]
        v=View();e=Engine(v,'PRUNE');b=SimpleNamespace(contexts=[['{}']])
        c=dict(packs=[dict(ids=list(range(i*100,(i+1)*100)),identity=None) for i in range(20)],
               native=dict(effective_hparams=dict(PRUNE={})))
        seen=[]
        def observe(view,bench,all_records,selected,H,endpoint,out,current_ids):
            seen.append((endpoint,len(selected),len(current_ids),e.pruned))
            return dict(summary={'requests':len(selected)},current={'requests':len(current_ids)})
        chunks=lambda rs:((i+1,rs[i*100:(i+1)*100],rs[:(i+1)*100]) for i in range(20))
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'batches',chunks),\
             patch.object(run,'rows',return_value=[]),patch.object(run,'observe',observe),\
             patch.object(run,'native_requests',side_effect=lambda x:x),\
             patch.object(run,'log_batch',return_value=True),patch.object(run.gc,'collect'),patch('builtins.print'):
            commits=[]
            run.native_loop(c,dict(source_commit='a'*40),Path(tmp),'PRUNE',records,
                            v.model,v,e,b,SimpleNamespace(),commits)
        self.assertEqual(e.calls,list(range(1,21)))
        self.assertEqual([i for i,x in enumerate(commits,1) if x['prune_applied']],[20])
        self.assertEqual(seen[-1],('W20',2000,100,True))
        self.assertEqual(seen[-2],('B20_PRE',100,100,False))
        self.assertTrue(all(not pruned for endpoint,_,_,pruned in seen[:-1]))
        self.assertEqual(e.counts['native_z'],2000)
        self.assertNotEqual(commits[-1]['native_after'],commits[-1]['after'])

    def test_nonterminal_unapproved_transform_blocks(self):
        v=View();e=Engine(v,'RECT')
        e.finish_batch=lambda n:dict(prune_applied=True)
        with self.assertRaisesRegex(RuntimeError,'TERMINAL_ONLY'):
            run.finish_native_batch(e,'RECT',1,v)


if __name__=='__main__':unittest.main()
