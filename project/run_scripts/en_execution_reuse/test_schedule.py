"""CPU wiring checks: real optimizer/guard/invariant/session, toy suffix/loss."""
import copy
from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .config import ARMS
from .schedule import run_schedule
from .generated_oracle import FiniteTrialModelOverflow
from .parity import compare_exact
from .test_current_observation import ToyOracle
from project.run_scripts.single_layer_edit_preserving_correction.geometry import RightSpace
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng,restore_rng


class ToyReference:
    def __init__(self):self.work={};self.last_sweep=None;self.calls=[]
    def reset_counters(self):
        old=self.work;self.work={'gradient_sweeps':0,'kl_sweeps':0};return old
    def kl(self,w,*,gradient,session=None,handle=None,allow_trial_overflow=False):
        self.calls.append(dict(gradient=gradient,resident=session is not None))
        if session is not None:
            with session.readonly(handle):
                if not torch.equal(session.evaluate_handle(handle),w):raise ValueError('toy endpoint')
        g=w.double().clone();g[:,:2]=0
        loss=float(g.square().sum()/2)
        rows=[dict(index=i,loss=loss,scored_positions=256) for i in range(512)]
        self.work['gradient_sweeps']+=int(gradient);self.work['kl_sweeps']+=1
        self.last_sweep=dict(gradient=gradient,coverage=dict(documents=512,backward_documents=512 if gradient else 0,
            positions=512*256,vocabulary_size=7,complete=True))
        return loss,g if gradient else None,rows


class OverflowReference(ToyReference):
    def kl(self,w,**kwargs):
        if kwargs['allow_trial_overflow'] and len(self.calls)==1:
            self.calls.append(dict(gradient=False,resident=kwargs['session'] is not None))
            raise FiniteTrialModelOverflow(dict(gradient=False,partial_rows=[],complete=False,
                coverage=dict(documents=0,positions=0,backward_documents=0,vocabulary_size=7,complete=False)))
        return super().kl(w,**kwargs)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.rng=capture_rng();self.addCleanup(restore_rng,self.rng)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.WN=torch.arange(12,dtype=torch.float32).reshape(3,4)/16
        self.current=ToyOracle();self.current.weight_work={};self.current.collect_weight_timing=lambda:{}
        self.rows=[dict(cache=0,positions=[2,5,17],labels=[1,2,1],sequence_id='1:canonical:new',branch='new',kind='canonical'),
            dict(cache=0,positions=[2],labels=[3],sequence_id='1:canonical:old',branch='old',kind='canonical'),
            dict(cache=1,positions=[0,3],labels=[0,1],sequence_id='2:native:new',branch='new',kind='native')]
        self.K=torch.cat([c.valid_keys() for c in self.current.caches],dim=1)
        self.allowed=RightSpace(np.eye(4),np.empty((4,0)),'RESOLVED')
        self.space=RightSpace(np.eye(4),np.eye(4)[:,:2],'RESOLVED')
        store=SimpleNamespace(receipt={'manifest_sha256':'teacher'},indices=lambda role:list(range(512)),
            capsule=lambda i:dict(source_row_id=f'row-{i}',actual_length=256))
        self.rt=SimpleNamespace(lock=dict(reference_inputs={'sha256':'inputs'},execution={'commit':'source'},sample_order=[1,2]),
            identity={'runtime':'CPU_FIXTURE'},generated_store=store,rng=self.rng,model=object(),
            copy_weight=lambda w:None,guard=lambda:None)

    def run_one(self,arm,reference=None,space=None):
        with patch('project.run_scripts.en_execution_reuse.schedule.model_guard',return_value=('CPU_FIXTURE',)), \
             patch.object(torch.cuda,'synchronize',return_value=None):
            return run_schedule(self.rt,arm,self.WN,reference or ToyReference(),self.current,self.rows,self.K,
                self.allowed,space or self.space,{'K':'fixture'},Path(self.tmp.name)/arm)

    def test_independent_gradients_full_unchanged_controller_exact_receipts(self):
        references=[ToyReference(),ToyReference()];values=[]
        for arm,ref in zip(ARMS,references):
            result,exact,work=self.run_one(arm,ref);exact['history_appends']=1;values.append(exact)
            self.assertEqual(result.counters['gradient_sweeps'],1)
            self.assertEqual(result.counters['accepted_rounds'],1)
            self.assertEqual(result.counters['attempted_trial_slots'],1)
            self.assertEqual(len(ref.calls),2)
            self.assertEqual(work['Past_actual'],'EMPTY_NA')
        self.assertEqual(compare_exact(*values)['status'],'EXACT_RECEIPT_MATCH')
        self.assertFalse(references[0].calls[0]['resident']);self.assertTrue(references[1].calls[0]['resident'])
        self.assertEqual(self.current.work['cached_suffix_forwards'],14) # legacy10 + reuse4

    def test_empty_space_is_normal_no_gradient_not_relabelled_full_coverage(self):
        empty=RightSpace(np.eye(4),np.eye(4),'REPAIR_SPACE_EMPTY')
        exact=[]
        for arm in ARMS:
            ref=ToyReference();r,e,_=self.run_one(arm,ref,empty);e['history_appends']=1;exact.append(e)
            self.assertFalse(ref.calls);self.assertEqual(r.stop_reason,'REPAIR_SPACE_EMPTY')
        value=compare_exact(*exact)
        self.assertEqual(value['status'],'EXACT_RECEIPT_MATCH')
        self.assertEqual(value['gradient_coverage'],'NOT_RUN_NORMAL_EMPTY_OR_UNRESOLVED_SPACE')

    def test_B2_arm_not_injected(self):
        with self.assertRaisesRegex(ValueError,'ALLOWLIST'):self.run_one('SEQUENTIAL_B2')

    def test_finite_trial_overflow_rechecks_all_teachers_and_rejects_not_complete_sweep(self):
        visits=[]
        @contextmanager
        def document(index):
            visits.append(index);yield SimpleNamespace(logp=np.zeros((1,7),dtype=np.float32))
        self.rt.generated_store.document=document
        r,exact,_=self.run_one(ARMS[1],OverflowReference())
        self.assertEqual(visits,list(range(512)))
        self.assertEqual(r.counters['nonfinite_objective_trials'],1)
        self.assertEqual(r.trials[0]['reason'],'FINITE_WEIGHT_NONFINITE_OBJECTIVE')
        self.assertIsNone(exact['full_sweep_rows'][1]['loss'])
        self.assertFalse(exact['full_sweep_rows'][1]['coverage']['complete'])
        self.assertEqual(r.counters['accepted_rounds'],1)

    def test_teacher_corruption_never_becomes_normal_backtrack(self):
        @contextmanager
        def document(index):yield SimpleNamespace(logp=np.array([[np.nan]],dtype=np.float32))
        self.rt.generated_store.document=document
        with self.assertRaisesRegex(Exception,'OBJECTIVE_CALLBACK_FAILED'):
            self.run_one(ARMS[1],OverflowReference())


if __name__=='__main__':unittest.main()
