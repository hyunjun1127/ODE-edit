"""CPU production fixtures. These are not pretrained/GPU qualification."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import torch
from project.run_scripts.jlz_realized_subject.test_core import fixture,record
from project.run_scripts.jlz_realized_subject.optimize import fit
from project.run_scripts.jlz_realized_subject.observe import scores
from project.run_scripts.jlz_realized_subject.writer import Transaction
from .instrument import RAMObserver,replace
from .lookup import masks,panel
from .masked_observer import score,select_linears
from .qualify import qualify,mask_parity
from .common import state

torch.set_num_threads(1)

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)
        self.a,self.b,self.H,self.e,self.stats=fixture(self.path)
    def tearDown(self):self.tmp.cleanup()
    def test_frozen_fit_instrumented_25_24_exact_and_RAM(self):
        source=Path('/mnt/raid5/janghj/.codex/worktrees/odeedit-jlz-v5-a-w5-review-20261002-v1/local/jlz-v10-a-review/20261003-r1/snapshot/attempt-v2/source/project/run_scripts/jlz_realized_subject/optimize.py')
        spec=importlib.util.spec_from_file_location('project.run_scripts.jlz_realized_subject.frozen_optimize',source)
        original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        ref,rs=original.fit(self.a,self.e,'A',self.path/'reference',1)
        observer=RAMObserver(self.path/'instrument');actual,report=fit(self.a,self.e,'A',self.path/'observed',1,observer=observer)
        self.assertEqual(rs,report);self.assertEqual(set(observer.snapshots),{9,13,17,21,25})
        for l in self.a.sites:self.assertTrue(torch.equal(ref['weights'][l],actual['weights'][l]))
        for c in range(1,26):
            x=json.loads((self.path/f'reference/candidate-{c:02d}.json').read_text())
            y=json.loads((self.path/f'observed/candidate-{c:02d}.json').read_text())
            for key in ('losses','total_mean','gradient','energy','layer','Adam_updates_after'):
                self.assertEqual(x[key],y[key])
        self.assertFalse(list(self.path.rglob('*.pt')))
        self.assertEqual(json.loads((self.path/'instrument/step-c25.json').read_text())['layers']['0']['q_step_norm'],0.)
    def test_extra_adjoint_and_snapshot_fixed_qualification(self):
        r=qualify(self.a,self.e,self.H,self.path/'qual');self.assertTrue(r['passed'])
    def test_mask_partition_multitoken_leftpad(self):
        x=masks([([1,3,4],[8,9,10]),([1,7],[11])],[2,1],5,'cpu')
        self.assertTrue(torch.equal(x['ALL'],x['SUBJECT_ONLY']|x['NONSUBJECT_ONLY']))
        self.assertFalse(bool((x['SUBJECT_ONLY']&x['NONSUBJECT_ONLY']).any()))
        self.assertTrue(x['NONSUBJECT_ONLY'][0,4]);self.assertFalse(x['ALL'][1,0])
        with self.assertRaisesRegex(RuntimeError,'LOOKUP'):masks([([1,3],[4])],[2],2,'cpu')
    def test_actual_Flinear_ALL_NONE_and_restore(self):
        W0={l:w.detach().clone() for l,w in self.a.weights.items()}
        W1={l:w+.0001 for l,w in W0.items()};pairs=[('Joe was',' B'),('Anna was',' C')]
        lookups=[3,4];before=state(self.a,self.H);sig=self.a.hook_signature()
        none=score(self.a,self.b,pairs,lookups,W0,W1,'NONE',1);ref=scores(self.a,self.b,pairs,1)
        self.assertEqual(none,ref)
        all_=score(self.a,self.b,pairs,lookups,W0,W1,'ALL',1)
        with Transaction(self.a,self.H):
            replace(self.a,W1);effective=scores(self.a,self.b,pairs,1)
        self.assertEqual(all_,effective);self.assertEqual(state(self.a,self.H),before);self.assertEqual(sig,self.a.hook_signature())
    def test_mask_hook_exception_restored(self):
        W={l:w.detach() for l,w in self.a.weights.items()};sig=self.a.hook_signature()
        with self.assertRaisesRegex(ValueError,'fault'):
            with select_linears(self.a,W,W,torch.ones(1,4,dtype=torch.bool)):raise ValueError('fault')
        self.assertEqual(sig,self.a.hook_signature())
    def test_exact_template_not_edited_subject_and_ambiguity(self):
        r=record(0);r['neighborhood_prompts']=['Bob was','Not matched']
        result=panel([r],[r],self.b)
        self.assertEqual(result['identifiable'],1);self.assertEqual(result['rows'][0]['subject'],'Bob')
        alternative=record(1);alternative['requested_rewrite']['prompt']='B{} was'
        result=panel([r],[r,alternative],self.b)
        self.assertEqual(result['identifiable'],0)
    def test_wrong_snapshot_rejected_and_transaction_restores(self):
        before=state(self.a,self.H)
        with self.assertRaisesRegex(RuntimeError,'SCHEMA'):
            with Transaction(self.a,self.H):
                wrong={l:torch.zeros_like(w) for l,w in self.a.weights.items()}
                wrong[max(wrong)]=torch.ones(1);replace(self.a,wrong)
        self.assertEqual(state(self.a,self.H),before)

if __name__=='__main__':unittest.main()
