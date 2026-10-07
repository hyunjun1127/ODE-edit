"""CPU production-loop/transaction integration; no native fitting/GPU."""
import copy
from dataclasses import dataclass
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from . import run,metrics


@dataclass
class HP:
    layers: tuple=(13,14,15,16,17)


class View:
    def __init__(self):
        self.model=torch.nn.Module();self.weights={}
        for layer in (13,14,15,16,17):
            value=torch.nn.Parameter(torch.zeros(2,2));self.weights[layer]=value
            self.model.register_parameter('w'+str(layer),value)
        self.model.register_parameter('bias',torch.nn.Parameter(torch.zeros(1)))
    def hook_signature(self):return ()
    def guard(self):return tuple((n,id(p),p.data_ptr(),p._version) for n,p in self.model.named_parameters())


class Engine:
    def __init__(self,view,arm):
        self.view,self.arm=view,arm;self.hp=HP();self.next_batch=1;self.context=[['{}'],['a {}']]
        self.H={l:torch.zeros(2,2) for l in run.HISTORY_LAYERS[arm]} if arm in ('CAKE','ALPHAEDIT_BLUE') else {}
        self.cold={};self.pruned=False;self.calls=[];self.counts={k:0 for k in run.EXPECTED[arm]}
    def history(self):return self.H
    def context_snapshot(self):return copy.deepcopy(self.context)
    def snapshot_native_state(self):return dict(context=self.context_snapshot(),next_batch=self.next_batch,cold=dict(self.cold),pruned=self.pruned)
    def restore_native_state(self,value):
        self.context=copy.deepcopy(value['context']);self.next_batch=value['next_batch'];self.cold=dict(value['cold']);self.pruned=value['pruned']
    def native_state_signature(self):return dict(context=self.context_snapshot(),next_batch=self.next_batch,cold=sorted(self.cold),pruned=self.pruned)
    def restore_history(self,value):
        if value:
            with torch.no_grad():
                for l,h in self.H.items():h.copy_(value[l])
        else:self.H={}
    def apply(self,records,batch):
        self.calls.append(batch);self.next_batch+=1
        if self.arm=='PRUNE' and batch==1:self.cold={l:w.detach().clone() for l,w in self.view.weights.items()}
        if self.arm=='BASE_ALPHAEDIT' and not self.H:self.H={l:torch.zeros(2,2) for l in run.HISTORY_LAYERS[self.arm]}
        with torch.no_grad():
            for l in run.ARM_LAYERS[self.arm]:self.view.weights[l].add_(1)
            for h in self.H.values():h.add_(1)
        for k,v in run.EXPECTED[self.arm].items():self.counts[k]+=v
        return self.view.model,dict(arm=self.arm,batch=batch,requests=100,delta=run.EXPECTED[self.arm],
            same_model_returned=True,native_has_history=bool(run.HISTORY_LAYERS[self.arm]),caller_history_appends=0,
            cache_template=None,native_z_disk_cache=False,checkpoint_saved=False)
    def finish_batch(self,batch):
        if self.arm=='PRUNE' and batch==20:
            self.pruned=True
            with torch.no_grad():
                for w in self.view.weights.values():w.mul_(.5)
            return dict(prune_applied=True,repair='PRUNE_TERMINAL_BASE_FIX',repair_authorized=True,
                upstream_bitwise_equivalence=False,terminal_transforms=1,final_base='SAVED_COLD_W0',
                spectrum_formula_changed=False,checkpoint_saved=False)
        return dict(prune_applied=False,terminal_transforms=0,no_model_mutation=True)


def summary(n):
    return dict(planned_count=n,fluency_count=n,consistency_count=n,fluency_sum=float(n),consistency_sum=.25*n,
        ngram_entropy=1.,reference_score=.25,generation_prompt_count=n,generated_token_count=n,
        missing_reason_counts={r:0 for r in metrics.GENERATION_REASONS})


class Generation:
    def __init__(self,out):self.out=Path(out);self.index=0;self.calls=[];self.runtime_sha='a'*64
    def value(self,records,endpoint,cohort,state,subset=False):
        self.index+=1;self.calls.append((endpoint,len(records),subset))
        rows=[dict(occurrence=r['occurrence_index'],case_id=r['case_id']) for r in records]
        identity=dict(runtime=self.runtime_sha,endpoint=endpoint,cohort=cohort,state_sha256=run.digest(state),
            ordered_occurrences=[r['occurrence_index'] for r in records])
        path=self.out/(str(self.index)+'.json');run.write(path,dict(identity=identity,rows=rows))
        return dict(rows_path=str(path),rows=rows,identity=identity,identity_sha256=run.digest(identity),summary=summary(len(records)),
            RNG_restored=True,observer_no_mutation=True,work=dict(new_case_observations=0 if subset else len(records),
                cached_case_observations=len(records) if subset else 0,generation_forwards=0 if subset else len(records),
                full_prefix_token_work=0,seconds=0.))
    def observe(self,records,endpoint,cohort=None,state_identity=None):return self.value(records,endpoint,cohort,state_identity)
    def subset(self,value,records,endpoint,cohort=None):
        require_ids={r['occurrence'] for r in value['rows']}
        assert all(r['occurrence_index'] in require_ids for r in records)
        return self.value(records,endpoint,cohort,{},True)


class Tests(unittest.TestCase):
    def records(self):return [dict(case_id=i,generation_prompts=['text'],requested_rewrite=dict(prompt='{} works',subject='s',target_new={'str':'new','id':'n'})) for i in range(2000)]
    def test_six_native_history_counts_and_two_layer_guard(self):
        for arm in run.ARMS:
            v=View();e=Engine(v,arm);before=run.nonselected(v,arm);_,receipt=e.apply(self.records()[:100],1)
            run.check_native(arm,receipt,e,1);self.assertEqual(before,run.nonselected(v,arm))
            if arm=='ALPHAEDIT_BLUE':self.assertTrue(all(torch.count_nonzero(v.weights[l])==0 for l in (14,15,16)))
    def test_all_arm_observer_failure_restores_native_history_cold_cache_rng(self):
        for arm in run.ARMS:
            v=View();e=Engine(v,arm);b=SimpleNamespace(contexts=[['{}']]);before=run.state(v,e.history())
            tx=run.NativeTransaction(v,e,b,arm)
            with self.assertRaisesRegex(ValueError,'observer'):
                with tx:e.apply([],1);raise ValueError('observer')
            self.assertTrue(tx.rollback_verified);self.assertEqual(run.state(v,e.history()),before)
            self.assertEqual(e.next_batch,1);self.assertEqual(e.cold,{})
            self.assertGreater(e.counts['native_z'],0) # failed cost retained
    def test_terminal_IO_failure_restores_PRUNE_metadata_and_model(self):
        v=View();e=Engine(v,'PRUNE');e.next_batch=20;e.cold={13:torch.zeros(2,2)}
        b=SimpleNamespace(contexts=[['{}']]);before=run.state(v,e.history());tx=run.NativeTransaction(v,e,b,'PRUNE')
        with self.assertRaises(OSError):
            with tx:
                e.apply([],20);run.finish_native_batch(e,'PRUNE',20,v);tx.finish()
                try:raise OSError('IO')
                except BaseException:tx.done=False;raise
        self.assertEqual(before,run.state(v,e.history()));self.assertEqual(e.next_batch,20);self.assertFalse(e.pruned)
    def test_actual_production_six_20apply_generation_schedule_and_PRUNE_final(self):
        records=self.records();chunks=lambda rs:((i+1,rs[i*100:(i+1)*100],rs[:(i+1)*100]) for i in range(20))
        for arm in run.ARMS:
            v=View();e=Engine(v,arm);b=SimpleNamespace(contexts=[['{}']]);observed=[]
            def obs(view,bench,all_records,selected,H,endpoint,out,current_ids):
                observed.append((endpoint,len(selected),len(current_ids),e.pruned))
                return dict(summary=dict(requests=len(selected)),current=dict(requests=len(current_ids)))
            c=dict(packs=[dict(ids=list(range(i*100,(i+1)*100))) for i in range(20)])
            with tempfile.TemporaryDirectory() as tmp,patch.object(run,'batches',chunks),patch.object(run,'rows',return_value=[]),\
                patch.object(run,'observe',obs),patch.object(run,'log_batch',return_value=True),\
                patch.object(run,'log_generation',return_value=True),patch.object(run.gc,'collect'),patch('builtins.print'):
                g=Generation(Path(tmp)/'generation');wg=g.value(run.generation_records(records),'W0','FIRST2000',{})
                commits=[];run.native_loop(c,dict(source_commit='a'*40),Path(tmp),arm,records,v.model,v,e,b,SimpleNamespace(),commits,g,wg)
                self.assertEqual(e.calls,list(range(1,21)));self.assertEqual(len(commits),20)
                self.assertTrue(all(ncurrent==100 for _,_,ncurrent,_ in observed))
                self.assertEqual([n for n,value in enumerate(commits,1) if value['prune_applied']], [20] if arm=='PRUNE' else [])
                self.assertEqual(observed[-1],('W20',2000,100,arm=='PRUNE'))
                for value in commits:
                    self.assertEqual(value['generation']['current']['summary']['planned_count'],100)
                    self.assertEqual(value['generation']['pre']['summary']['planned_count'],100)
                    self.assertEqual(value['generation']['current']['work']['generation_forwards'],0)
                self.assertEqual([x[1] for x in g.calls if x[0] in ('W5','W10','W15','W20')],[500,1000,1500,2000])
                self.assertEqual([x[2] for x in g.calls if x[0]=='B1_PRE'],[True])
    def test_missing_W0_ready_blocks_secondary_without_observer(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=Generation(Path(tmp)/'obs');c=dict(generation=dict(shared_W0_root=str(Path(tmp)/'shared'),primary_arm='BASE_MEMIT'))
            with self.assertRaisesRegex(RuntimeError,'READY_MISSING'):
                run.install_generation_W0(c,'CAKE',g,run.generation_records(self.records()),dict(W={}),Path(tmp))
            self.assertEqual(g.calls,[])
    def test_generation_mapping_missing_not_zero_and_six_public_reasons(self):
        item=summary(2);item.update(fluency_count=0,fluency_sum=0.);item.pop('ngram_entropy')
        payload=metrics.generation_payload('current/post',item)
        self.assertNotIn('current/post/fluency/ngram_entropy',payload)
        self.assertNotIn('current/post/generation/missing_asset_not_available_count',payload)
        self.assertEqual(payload['current/post/consistency/reference_score'],.25)
        item['reference_score']=float('nan')
        with self.assertRaisesRegex(RuntimeError,'MEAN_SUM_COUNT'):metrics.generation_payload('current/post',item)
    def test_generation_log_axis_state_and_explicit_rejection(self):
        seen=[]
        tracker=SimpleNamespace(log=lambda payload:seen.append(payload) or False)
        self.assertFalse(run.log_generation(tracker,[('current/pre',summary(100)),('current/post',summary(100))],3,300))
        self.assertEqual(seen[0]['edits'],300);self.assertEqual(seen[0]['pre_state_edits'],200)
        self.assertEqual(seen[0]['post_state_edits'],300)
    def test_native_apply_dispatch_does_not_convert_nonstock_dict_target(self):
        sentinel=SimpleNamespace()
        fake=SimpleNamespace(apply=lambda records,batch:(sentinel,records))
        adapter=run.EngineAdapter.__new__(run.EngineAdapter);adapter.inner=fake
        records=self.records()[:100];returned,seen=adapter.apply(records,1)
        self.assertIs(returned,sentinel);self.assertIs(seen,records)
        self.assertIsInstance(seen[0]['requested_rewrite']['target_new'],dict)
    def test_reference_manifest_pointer_and_actual_sha_are_hard_startup_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manifest.json';run.write(path,dict(status='READY'))
            c=dict(generation=dict(assets_manifest=str(path),assets_manifest_member=run.member(path),reference_assets_sha256='a'*64))
            with patch.object(run,'load_assets',return_value=SimpleNamespace(sha='b'*64)) as loader:
                with self.assertRaisesRegex(RuntimeError,'ASSET_IDENTITY'):run.generation_assets(c)
                loader.assert_called_once()
            c['generation']['assets_manifest']=str(Path(tmp)/'replacement.json')
            with patch.object(run,'load_assets') as loader:
                with self.assertRaisesRegex(RuntimeError,'MANIFEST_POINTER'):run.generation_assets(c)
                loader.assert_not_called()
    def test_returned_pre_generation_phase_survives_later_observer_failure(self):
        records=self.records();v=View();e=Engine(v,'BASE_ALPHAEDIT');b=SimpleNamespace(contexts=[['{}']])
        before=run.state(v,e.history());chunks=lambda rs:((1,rs[:100],rs[:100]),)
        with tempfile.TemporaryDirectory() as tmp,patch.object(run,'batches',chunks),patch.object(run,'rows',return_value=[]),\
            patch.object(run,'observe',return_value=dict(summary={},current={})),patch.object(run.gc,'collect'):
            g=Generation(Path(tmp)/'gen');wg=g.value(run.generation_records(records),'W0','FIRST2000',{})
            g.observe=lambda *args,**kw:(_ for _ in ()).throw(ValueError('post-generation'))
            c=dict(packs=[dict(ids=list(range(100)))])
            with self.assertRaisesRegex(ValueError,'post-generation'):
                run.native_loop(c,dict(source_commit='a'*40),Path(tmp),'BASE_ALPHAEDIT',records,
                    v.model,v,e,b,SimpleNamespace(),[],g,wg)
            self.assertTrue((Path(tmp)/'generation-work/B1_PRE.json').exists())
            self.assertFalse((Path(tmp)/'batch-01/commit.json').exists())
            self.assertEqual(run.state(v,e.history()),before);self.assertEqual(e.next_batch,1)


if __name__=='__main__':unittest.main()
