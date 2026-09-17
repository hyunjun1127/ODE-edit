"""CPU preparation plumbing/negative tests; actual Llama is NOT_RUN here."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from .common import LAYERS, digest, save
from .native import tensor_sha
from .technical import (Journal, MANDATORY, NUMERICS, TechnicalHold,
    _gates_and_repeats, coverage_evidence, finite_json, native_totals,
    repeated_scores, run, weight_comparison, check_locked_numerics)
from .controller import Controller, run_arm
from .test_controller import ToyBackend


def score(value=.01, **changes):
    fields=dict(training_e=value,base_kl=value/100,past_h=value/2,
        current_strict=frozenset({1,2}),current_pair=frozenset({1}),
        past_strict=frozenset({101}),past_pair=frozenset({101}),canonical_e=value)
    fields.update(changes)
    return dict(controller=fields,E=fields['training_e'],H=fields['past_h'],B=fields['base_kl'])


def numerical_lock():
    return dict(E_H_repeat=5e-5,B_repeat=5e-7,native_weight_max_abs=5e-6,
        native_weight_relative_Frobenius=5e-5,strict_pair_repeat='EXACT_ID_SET',
        E_H_allowance=1e-4,B_tie=1e-6,current_plateau=False)


class FakeRuntime:
    """Only gate/repeat RAM mechanics; no fake scientific endpoint receipt."""
    def __init__(self):
        self.records=[dict(case_id=i) for i in range(1000)]
        self.cpuW={l:torch.ones(2,2)*l for l in LAYERS}
        self.M={l:torch.zeros(1,2,2) for l in LAYERS}
        self.calls=0
        self.metrics=SimpleNamespace(score=self.score)
    def state(self):
        return dict(W={str(l):tensor_sha(w) for l,w in self.cpuW.items()},
            M={str(l):tensor_sha(m) for l,m in self.M.items()},contexts='ctx',rng='fixed')
    def snapshot(self):
        return dict(W=dict(self.cpuW),M=dict(self.M),rng={},state=self.state(),
            whash={l:tensor_sha(w) for l,w in self.cpuW.items()},mhash={l:tensor_sha(m) for l,m in self.M.items()})
    def restore(self,s):self.cpuW=dict(s['W']);self.M=dict(s['M'])
    def adopt_native_weights(self,weights,rng):self.cpuW.update(weights)
    def observe(self,fn):
        before=self.state();result=fn()
        assert before==self.state()
        return result
    def score(self,current,past):
        self.calls+=1
        assert len(current)==100 and len(past)==64
        return score(float(self.cpuW[4].mean())/100)


class TechnicalTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_runtime_numerical_lock_matches_preregistered_constants(self):
        lock=dict(numerical=numerical_lock());check_locked_numerics(lock)
        lock['numerical']['B_repeat']*=2
        with self.assertRaisesRegex(ValueError,'EXECUTION_LOCK_MISMATCH'):check_locked_numerics(lock)

    def test_exact_ids_not_same_count_and_fixed_scalar_threshold(self):
        a=score();b=score(training_e=.01004,past_h=.00504,base_kl=.0001004)
        self.assertTrue(repeated_scores(a,b)['passed'])
        self.assertFalse(repeated_scores(a,score(current_strict=frozenset({2,3})))['passed'])
        self.assertFalse(repeated_scores(a,score(training_e=.010051))['passed'])
        self.assertFalse(repeated_scores(a,score(base_kl=.00010051))['passed'])
        self.assertFalse(repeated_scores(a,score(past_pair=None))['passed'])
        self.assertFalse(repeated_scores(a,score(training_e=float('nan')))['passed'])

    def test_weight_comparison_checks_all_layers_schema_finite_and_both_tolerances(self):
        x={l:torch.ones(3,3) for l in LAYERS};y=deepcopy(x)
        self.assertTrue(weight_comparison(x,y)['passed'])
        y[6][0,0]+=.001
        self.assertFalse(weight_comparison(x,y)['passed'])
        y=deepcopy(x);y[7][0,0]=float('nan')
        self.assertFalse(weight_comparison(x,y)['passed'])
        small={l:torch.full((3,3),1e-8) for l in LAYERS}
        doubled={l:v*2 for l,v in small.items()}
        self.assertFalse(weight_comparison(small,doubled)['passed'])

    def test_failed_stage_receipt_written_before_raise_and_READY_not_possible(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);j=Journal(root)
            with self.assertRaises(TechnicalHold):
                with j.step('NATIVE_N4'):
                    j.check('NATIVE_N4',dict(status='MISMATCH',value=float('nan')),False)
            result=json.loads((root/'NATIVE_N4/result.json').read_text())
            self.assertEqual(result['status'],'HOLD')
            self.assertFalse(result['evidence']['value']['valid'])
            self.assertTrue((root/'NATIVE_N4/failure.json').exists())
            self.assertFalse(json.loads((root/'NATIVE_N4/elapsed.json').read_text())['completed_check'])
            with self.assertRaises(TechnicalHold):j.ready()
            self.assertFalse((root/'READY.json').exists())

    def test_partial_native_failure_is_persisted_without_weight_checkpoint(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);j=Journal(root);exc=RuntimeError('native-first-error')
            exc.native_partial={'failed_requests':[dict(case_id=1,iteration=2,losses=[1.,.5])],
                                'completed_requests':[dict(target=torch.ones(3))]}
            with self.assertRaisesRegex(RuntimeError,'native-first-error'):
                with j.step('NATIVE_N4'):raise exc
            payload=torch.load(root/'NATIVE_N4/native-failure-evidence.pt',weights_only=True)
            self.assertEqual(payload['failed_requests'][0]['iteration'],2)
            self.assertNotIn('W',payload);self.assertNotIn('M',payload)

    def test_mandatory_coverage_blocks_READY_even_if_other_stages_pass(self):
        with TemporaryDirectory() as tmp:
            j=Journal(tmp);j.passed={name:True for name in MANDATORY if name!='C45678_BOUNDED_SEARCH'}
            with self.assertRaisesRegex(TechnicalHold,'C45678_BOUNDED_SEARCH'):j.ready()
            j.passed['C45678_BOUNDED_SEARCH']=True;j.ready()

    def test_gates_fixed_forward_reverse_repeat_and_RAM_restore(self):
        rt=FakeRuntime();entry=rt.snapshot();rt.adopt_native_weights({4:entry['W'][4]+1},{})
        n4=rt.snapshot();rt.restore(entry)
        with TemporaryDirectory() as tmp:
            root=Path(tmp);j=Journal(root)
            _gates_and_repeats(rt,rt.records[:100],entry,n4,root,j)
            self.assertTrue(j.passed['GATES_REPLAY_REPEAT']);self.assertEqual(rt.calls,4)
            self.assertEqual(rt.state(),entry['state'])
            self.assertFalse(list(root.rglob('*.pt')))
            result=json.loads((root/'GATES_REPLAY_REPEAT/result.json').read_text())
            self.assertFalse(result['evidence']['Past_fixture_only']['scientific_controller_input'])

    def test_actual_CPU_solver_coverage_is_only_count_proxy_no_budget_change(self):
        import scipy
        if scipy.__version__!='1.15.3':self.skipTest('Pinned SciPy 1.15.3 is not installed')
        backend=ToyBackend(LAYERS)
        c=Controller(backend,LAYERS,namespace='CPU_ONLY_TECHNICAL_FIXTURE',history_layers=LAYERS)
        chosen,stop=run_arm(c,'C45678',finalize=False)
        evidence=coverage_evidence(c,stop)
        self.assertEqual(evidence['passed'],c.coverage()['adequate_by_count_proxy'])
        self.assertFalse(evidence['budget_expanded']);self.assertFalse(evidence['solver_simplex_or_optimality_certified'])
        self.assertLessEqual(evidence['counts']['endpoints'],28)
        self.assertLessEqual(evidence['counts']['suffix_fits'],40)
        self.assertEqual(backend.history,{l:0 for l in LAYERS})

    def test_nontraced_reference_cost_not_zero_Adam_claim(self):
        with TemporaryDirectory() as tmp:
            rows=[save(Path(tmp)/'a.json',dict(compute_z=100,solve=1,compute_ks=1,seconds=1)),
                  save(Path(tmp)/'b.json',dict(compute_z=100,solve=1,compute_ks=1,seconds=2,
                      adam_updates=20,loss_evaluations=120,clamp_hits=3))]
            totals=native_totals(rows)
            self.assertEqual(totals['native_target_calls'],200)
            self.assertEqual(totals['actual_adam_recorded'],20)
            self.assertEqual(totals['target_calls_without_adam_loss_trace'],100)
            self.assertEqual(totals['native_fit_inclusive_seconds'],3)

    def test_load_failure_never_creates_READY_or_calls_new_model(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);lock_path=root/'lock.json'
            save(lock_path,dict(instruction_id='TEST',source_head='TEST',numerical=numerical_lock()))
            with patch('project.run_scripts.sequential_local_z_allocation.technical.ROOT',root), \
                 patch('project.run_scripts.sequential_local_z_allocation.technical.verify'), \
                 patch('project.run_scripts.sequential_local_z_allocation.runtime.Runtime',side_effect=RuntimeError('CPU_fake_load_error')):
                with self.assertRaisesRegex(RuntimeError,'CPU_fake_load_error'):run(lock_path)
            failure=json.loads((root/'technical/attempt-v1/failure.json').read_text())
            self.assertEqual(failure['stage'],'LOAD');self.assertFalse(failure['science_ready'])
            self.assertFalse((root/'technical/attempt-v1/READY.json').exists())

    def test_nonfinite_evidence_is_labelled_not_silently_changed(self):
        value=finite_json(dict(a=float('inf'),b=[float('nan')],c=1.))
        self.assertFalse(value['a']['valid']);self.assertEqual(value['c'],1.)
        json.dumps(value,allow_nan=False)


if __name__=='__main__':unittest.main()
