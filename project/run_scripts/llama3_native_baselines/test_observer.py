"""CPU-only observer/transaction mechanics, not a scientific toy or LM PASS."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from project.run_scripts.jlz_realization.common import digest, member, write
from project.run_scripts.jlz_realization.writer import rng_equal, rng_snapshot
from . import metrics
from .metrics import ObservationView, observe, rows, state
from .native import NativeEngine
from .transaction import NativeTransaction


class MechanicalView:
    """Small tensors test state machinery only; never emulate LM inference."""
    def __init__(self):
        self.model = torch.nn.Module()
        self.model.register_parameter('selected4', torch.nn.Parameter(torch.zeros(2, 3)))
        self.model.register_parameter('selected8', torch.nn.Parameter(torch.zeros(2, 3)))
        self.model.register_parameter('other', torch.nn.Parameter(torch.ones(2)))
        self.sites = (4, 8)
        self.weights = {4: self.model.selected4, 8: self.model.selected8}

    guard = ObservationView.guard
    hook_signature = ObservationView.hook_signature


class MechanicalEngine:
    def __init__(self, initialized=False):
        self.H = {4: torch.zeros(3, 3), 8: torch.zeros(3, 3)} if initialized else {}
        self.context = [['prefix {}']]
        self.module = SimpleNamespace(COV_CACHE={}, cache_c_new=False, P_loaded=True,
                                      P_loaded_from='mechanical-fixed-input')
        self.P = None
        self.saved_cold_weights = {}
        self.next_batch, self.prune_applied, self.current = 1, False, None
        self.cumulative = {'public_applies': 0, 'fit_forwards': 0}

    snapshot_ledger = NativeEngine.snapshot_ledger
    restore_ledger = NativeEngine.restore_ledger
    ledger_identity = NativeEngine.ledger_identity

    def history(self):
        return self.H

    def contexts(self):
        return self.context

    def snapshot_context(self):
        return copy.deepcopy(self.context)

    def restore_context(self, value):
        self.context = copy.deepcopy(value)

    def reset_history_to_uninitialized(self):
        self.H = {}


class MechanicalBench:
    def __init__(self):
        self.contexts = []

    def panels(self, record):
        return {'R': ['r' + str(record['case_id'])],
                'P': ['p' + str(record['case_id'])],
                'N': ['n' + str(record['case_id'])]}


def records(count):
    return [dict(case_id=index, requested_rewrite=dict(subject='s' + str(index),
        relation_id='relation', target_new={'str': 'new'}, target_true={'str': 'true'}))
        for index in range(count)]


def fake_scores(view, bench, pairs, microbatch):
    # Deliberately consume RNG to prove observer restoration on success/failure.
    random.random(); np.random.rand(); torch.rand(1)
    return [dict(nll=1. if target == 'new' else 2., token_count=2,
                 token_correct=1 if target == 'new' else 2,
                 strict=target != 'new', token_identity=digest([prompt, target]))
            for prompt, target in pairs]


class ObserverMechanics(unittest.TestCase):
    def test_llama_layout_and_blue_physical_sites_meta_only(self):
        modules = [torch.nn.Identity() for _ in range(9)]
        for layer in (4, 8):
            modules[layer] = torch.nn.Module()
            modules[layer].mlp = torch.nn.Module()
            modules[layer].mlp.down_proj = torch.nn.Linear(14336, 4096, bias=False, device='meta')
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList(modules)
        model.config = SimpleNamespace(model_type='llama')
        view = ObservationView(model, (4, 8))
        self.assertEqual(tuple(view.weights), (4, 8))
        model.config.model_type = 'gptj'
        with self.assertRaisesRegex(RuntimeError, 'OBSERVER_LLAMA_MODEL'):
            ObservationView(model, (4, 8))

    def test_empty_history_and_wrong_history_layout(self):
        view = MechanicalView()
        self.assertEqual(state(view, {})['H'], {})
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_HISTORY_PHYSICAL_SITES'):
            state(view, {4: torch.zeros(1)})

    def test_current_and_seen_reducers_rng_nonmutation(self):
        view, bench = MechanicalView(), MechanicalBench()
        before = state(view, {})
        rng = rng_snapshot()
        with tempfile.TemporaryDirectory() as temp, patch.object(metrics, 'scores', fake_scores):
            result = observe(view, bench, records(3), records(3), {}, 'W5', temp,
                             current_ids=[2])
            self.assertEqual(result['summary']['R']['denominator'], 3)
            self.assertEqual(result['current']['R']['denominator'], 1)
            self.assertEqual(result['summary']['R']['rate'], 1.)
            self.assertEqual(result['summary']['N']['rate'], 0.)
            self.assertEqual(result['summary']['N']['token_micro'], 1.)
            self.assertEqual(len(rows(temp, before)), 9)
            self.assertTrue(rng_equal(rng))
            self.assertEqual(state(view, {}), before)

    def test_scoring_failure_preserves_rng_no_false_summary(self):
        view, bench = MechanicalView(), MechanicalBench()
        rng = rng_snapshot()
        def fail(*args):
            fake_scores(*args)
            raise ValueError('original score failure')
        with tempfile.TemporaryDirectory() as temp, patch.object(metrics, 'scores', fail):
            with self.assertRaisesRegex(ValueError, 'original score failure'):
                observe(view, bench, records(1), records(1), {}, 'B1_PRE', temp)
            self.assertFalse((Path(temp) / 'summary.json').exists())
            self.assertTrue(rng_equal(rng))

    def test_observer_hook_mutation_is_blocked(self):
        view, bench = MechanicalView(), MechanicalBench()
        def mutate(*args):
            view.model.register_forward_hook(lambda *unused: None)
            return fake_scores(*args)
        with tempfile.TemporaryDirectory() as temp, patch.object(metrics, 'scores', mutate):
            with self.assertRaisesRegex(RuntimeError, 'NATIVE_OBSERVER_MUTATION'):
                observe(view, bench, records(1), records(1), {}, 'W1', temp)
            self.assertFalse((Path(temp) / 'summary.json').exists())

    def test_current_must_be_actual_selected_occurrences(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, 'OBSERVER_CURRENT_OCCURRENCES'):
                observe(MechanicalView(), MechanicalBench(), records(2), records(1),
                        {}, 'W1', temp, current_ids=[1])

    def test_unqualified_w0_manifest_fails_without_scoring(self):
        from .common import ROOT
        names = ('observe.py', 'inputs.py', 'common.py')
        evaluator = []
        for name in names:
            relative = 'project/run_scripts/jlz_realization/' + name
            row = member(ROOT / relative)
            # Both sides are current local fixture bytes, not a historical
            # source-parity claim; qualification must still reject the manifest.
            evaluator.append(dict(relative=relative, current=row, historical=row))
        c = {'W0_reuse': {'status': 'UNVERIFIED'}, 'W0_evaluator': evaluator}
        with tempfile.TemporaryDirectory() as temp, patch.object(metrics, 'scores') as call:
            with self.assertRaisesRegex(RuntimeError, 'W0_REUSE_NOT_QUALIFIED'):
                metrics.install_W0(c, MechanicalView(), {}, temp, MechanicalBench(), records(1))
            call.assert_not_called()

    def test_missing_w0_evaluator_binding_blocks_without_scoring(self):
        c = {'W0_reuse': {'status': 'QUALIFIED_EXACT_REUSE'}}
        with tempfile.TemporaryDirectory() as temp, patch.object(metrics, 'scores') as call:
            with self.assertRaisesRegex(RuntimeError, 'W0_CURRENT_EVALUATOR_NOT_BOUND'):
                metrics.install_W0(c, MechanicalView(), {}, temp, MechanicalBench(), records(1))
            call.assert_not_called()

    def test_reuse_native_vs_source_history_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source_state = {'W': {'4': 'cold'}, 'H': {'4': 'zero'}}
            actual_state = {'W': {'4': 'cold'}, 'H': {}}
            # Reuse readback is a state-schema fixture, not measured W0 evidence.
            row = dict(identity='r', kind='N', case_id=0, prompt_index=0,
                new_nll=2., true_nll=1., new_token_count=1, new_token_correct=0,
                new_strict=False, true_token_count=1, true_token_correct=1, true_strict=True)
            write(folder / 'source.json', dict(state=source_state, optimizer_feedback=False, rows=[row]))
            manifest = {'cold_state': source_state, 'chunks': [member(folder / 'source.json')]}
            write(folder / 'reuse.json', dict(manifest=manifest, manifest_sha256=digest(manifest),
                actual_native_state=actual_state, scalar_bridge_only=True, history_or_editor_resume=False))
            self.assertEqual(rows(folder, actual_state)[0]['identity'], 'r')
            with self.assertRaisesRegex(RuntimeError, 'W0_NATIVE_STATE'):
                rows(folder, source_state)
            changed = json.loads((folder / 'source.json').read_text())
            changed['rows'][0]['new_nll'] = 3.
            (folder / 'source.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, 'CHANGED:'):
                rows(folder, actual_state)


class TransactionMechanics(unittest.TestCase):
    def test_success_finish_persists_selected_weights_and_history(self):
        view, engine, bench = MechanicalView(), MechanicalEngine(True), MechanicalBench()
        with NativeTransaction(view, engine, bench) as tx:
            with torch.no_grad():
                view.weights[4].add_(1.)
                engine.H[4].add_(2.)
            engine.next_batch = 2
            tx.finish()
        self.assertTrue(bool((view.weights[4] == 1.).all()))
        self.assertTrue(bool((engine.H[4] == 2.).all()))
        self.assertFalse(tx.rollback_verified)
        self.assertFalse(tx.W or tx.H)
        self.assertEqual(engine.next_batch, 2)

    def test_io_failure_after_finish_restores_entry_and_cursor(self):
        view, engine, bench = MechanicalView(), MechanicalEngine(True), MechanicalBench()
        cursor = [0]
        before, rng = state(view, engine.history()), rng_snapshot()
        with self.assertRaisesRegex(OSError, 'commit IO'):
            with NativeTransaction(view, engine, bench, cursor) as tx:
                with torch.no_grad():
                    view.weights[4].add_(1.); engine.H[4].add_(2.)
                cursor.append(1)
                random.random(); np.random.rand(); torch.rand(1)
                tx.finish()
                raise OSError('commit IO')
        self.assertEqual(state(view, engine.history()), before)
        self.assertEqual(cursor, [0])
        self.assertTrue(rng_equal(rng) and tx.rollback_verified)

    def test_empty_to_initialized_history_failure_resets_native_cache(self):
        view, engine, bench = MechanicalView(), MechanicalEngine(), MechanicalBench()
        before = state(view, {})
        with self.assertRaises(ValueError):
            with NativeTransaction(view, engine, bench) as tx:
                engine.H = {4: torch.ones(3, 3), 8: torch.ones(3, 3)}
                with torch.no_grad():
                    view.weights[8].add_(2.)
                raise ValueError('native apply')
        self.assertEqual(state(view, engine.history()), before)
        self.assertTrue(tx.rollback_verified)

    def test_context_and_hook_failure_restores_entry(self):
        view, engine, bench = MechanicalView(), MechanicalEngine(), MechanicalBench()
        hooks = view.hook_signature()
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_COMMIT_GUARD'):
            with NativeTransaction(view, engine, bench) as tx:
                bench.contexts.append('bad')
                engine.context.append(['bad'])
                view.model.register_forward_hook(lambda *unused: None, always_call=True)
                tx.finish()
        self.assertEqual(bench.contexts, [])
        self.assertEqual(engine.contexts(), [['prefix {}']])
        self.assertEqual(view.hook_signature(), hooks)
        self.assertTrue(tx.rollback_verified)

    def test_nonselected_mutation_is_not_verified_rollback(self):
        view, engine, bench = MechanicalView(), MechanicalEngine(), MechanicalBench()
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_ROLLBACK_MISMATCH'):
            with NativeTransaction(view, engine, bench) as tx:
                with torch.no_grad():
                    view.model.other.add_(1.)
                tx.finish()
        self.assertFalse(tx.rollback_verified)
        self.assertFalse(tx.W or tx.H)

    def test_apply_then_observer_or_io_failure_restores_complete_logical_ledger(self):
        view, engine, bench = MechanicalView(), MechanicalEngine(True), MechanicalBench()
        engine.next_batch = 20
        engine.current = {'batch': 19, 'fit_trace': [{'loss': 0.7}],
                          'seconds': {'fit': 2.}}
        cov = torch.ones(2, 2)
        engine.module.COV_CACHE = {('model', 'layer4'): {'tensor': cov,
                                  'entry_metadata': {'order': [4]}}}
        engine.module.cache_c_new = True
        engine.module.cache_c = engine.H[4]
        original_history = engine.H
        before = state(view, engine.history())
        ledger_before = engine.ledger_identity()
        cursor = list(range(19))
        with self.assertRaisesRegex(OSError, 'postapply observer IO'):
            with NativeTransaction(view, engine, bench, cursor) as tx:
                with torch.no_grad():
                    view.weights[4].add_(1.)
                    engine.H[4].add_(3.)
                engine.next_batch, engine.prune_applied = 21, True
                engine.current['fit_trace'][0]['loss'] = 4.
                engine.current['fit_trace'].append({'loss': 5.})
                engine.module.COV_CACHE[('model', 'layer4')]['entry_metadata']['order'].append(8)
                engine.module.COV_CACHE[('model', 'layer8')] = torch.zeros(2, 2)
                engine.module.cache_c = {'foreign': torch.zeros(1)}
                engine.module.cache_c_new = False
                engine.module.P_loaded_from = 'wrong-input'
                engine.cumulative['public_applies'] += 1
                engine.cumulative['fit_forwards'] += 100
                cursor.append(19)
                tx.finish()
                raise OSError('postapply observer IO')
        self.assertEqual(state(view, engine.history()), before)
        self.assertEqual(engine.ledger_identity(), ledger_before)
        self.assertIs(engine.H, original_history)
        self.assertIs(engine.module.cache_c, original_history[4])
        self.assertIs(engine.module.COV_CACHE[('model', 'layer4')]['tensor'], cov)
        self.assertEqual(engine.next_batch, 20)
        self.assertFalse(engine.prune_applied)
        self.assertEqual(cursor, list(range(19)))
        self.assertEqual(engine.cumulative, {'public_applies': 1, 'fit_forwards': 100})
        self.assertTrue(tx.rollback_verified and tx.rollback_ledger_verified)

    def test_missing_logical_ledger_api_blocks_before_mutation(self):
        engine = SimpleNamespace(history=lambda: {}, contexts=lambda: [])
        view = MechanicalView()
        before = state(view, {})
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_LOGICAL_LEDGER_API_REQUIRED'):
            with NativeTransaction(view, engine, MechanicalBench()):
                self.fail('Missing API must fail before entry body')
        self.assertEqual(state(view, {}), before)

    def test_protected_cache_mutation_is_not_a_verified_rollback(self):
        view, engine, bench = MechanicalView(), MechanicalEngine(True), MechanicalBench()
        cov = torch.ones(2, 2)
        engine.module.COV_CACHE = {'fixed-c0': cov}
        before = state(view, engine.history())
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_LOGICAL_LEDGER_ROLLBACK_MISMATCH'):
            with NativeTransaction(view, engine, bench) as tx:
                with torch.no_grad():
                    view.weights[4].add_(1.)
                    engine.H[4].add_(2.)
                    cov.add_(1.)
                engine.next_batch = 2
                raise ValueError('immutable input mutated')
        self.assertEqual(state(view, engine.history()), before)
        self.assertEqual(engine.next_batch, 1)
        self.assertFalse(tx.rollback_verified or tx.rollback_ledger_verified)


if __name__ == '__main__':
    unittest.main()
