"""CPU mocked-dispatch checks: no language model, target fit, GPU or SH1 math."""
import ast
import contextlib
import copy
import inspect
import io
import random
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from . import generation_run as runner


def records():
    return [dict(case_id=index, requested_rewrite=dict(prompt='{} works in',
        subject='fixture-subject', target_new={'str': ' new'}, target_true={'str': ' old'}),
        generation_prompts=['fixture-generation']) for index in range(2000)]


def config():
    scientific = dict(model='fixture-local-model', model_revision='fixture-revision',
        seed=20261002, runtime=dict(torch='fixture', transformers='fixture'),
        stream='/fixture/counterfact.json', native={'binding': 'family-specific'},
        cold_W={str(layer): 'fixture-cold' for layer in runner.ARM_LAYERS['BASE_MEMIT']})
    result = dict(scientific, task_id=runner.TASK, instruction_id=runner.NONCE,
        authority={'path': '/fixture/authority'}, resources={'reserve_bytes': 0, 'cpu': 6},
        tracking={'env_file': '/fixture/env'}, noCP=True, z_disk_cache=False,
        exact_resume='NOT_AVAILABLE')
    result['arm_configs'] = {arm: dict(copy.deepcopy(scientific), task_id='old-family-task')
                            for arm in runner.ARMS}
    return result


class MockEngine:
    """Schema/counter sentinel only, never invokes native fitting functions."""
    def __init__(self, arm, model):
        self.arm, self.model = arm, model
        self.module = types.SimpleNamespace(CONTEXT_TEMPLATES_CACHE=None)
        self.current, self.progress, self.applies, self.terminal_calls = None, None, [], 0
        self._counts = {key: 0 for key in runner.expected_counts(arm)}
        self._history = {layer: torch.tensor(0.) for layer in runner.ARM_LAYERS[arm]}
        if arm not in ('CAKE', 'ALPHAEDIT_BLUE'):
            self._history = {}
        else:
            self.module.CONTEXT_TEMPLATES_CACHE = [['{}'], ['fixture. {}'] * 5]

    @property
    def counts(self):
        return dict(self._counts)

    def history(self):
        return self._history

    def contexts(self):
        return copy.deepcopy(self.module.CONTEXT_TEMPLATES_CACHE)

    def apply(self, requests, number):
        if number != len(self.applies) + 1 or len(requests) != 100:
            raise AssertionError('Unexpected native dispatch')
        expected_type = str if self.arm in runner.STOCK_ARMS else dict
        if not all(isinstance(row['target_new'], expected_type) for row in requests):
            raise AssertionError('Wrong family request schema')
        self.applies.append(number)
        self.model.number = number
        if self.arm in runner.HISTORY_ARMS and not self._history:
            self._history = {layer: torch.tensor(0.) for layer in runner.ARM_LAYERS[self.arm]}
        for value in self._history.values():
            value.add_(1)
        self.module.CONTEXT_TEMPLATES_CACHE = [['{}'], ['fixture. {}'] * 5]
        counts = runner.expected_counts(self.arm)
        self._counts = {key: self._counts[key] + value for key, value in counts.items()}
        return self.model, dict(delta=counts, prune_applied=False, checkpoint_saved=False)

    def terminal_prune(self):
        if self.arm != 'PRUNE' or self.model.number != 20 or self.terminal_calls:
            raise AssertionError('Wrong terminal dispatch')
        self.terminal_calls += 1
        self.model.pruned = True
        return dict(explicit_repair=runner.BASE_FIX, prune_applied=True, native_svd_calls=12)


class MockGeneration:
    def __init__(self, model, all_records, *, mutate=None):
        self.model, self.records, self.mutate = model, all_records, mutate
        self.loads, self.endpoints, self.subsets = 0, [], []

    def load_W0(self):
        self.loads += 1
        return dict(summary={'planned_count': 2000},
                    cases=[{'case_id': record['case_id']} for record in self.records],
                    identity={'mock': 'one-model-cold-cache'})

    def endpoint(self, selected, *, out, endpoint, model_state, cohort_label):
        self.endpoints.append(dict(endpoint=endpoint, count=len(selected),
            number=self.model.number, pruned=self.model.pruned,
            state=copy.deepcopy(model_state), cohort=cohort_label))
        if self.mutate == endpoint:
            self.model.number += 1
        if self.mutate == 'rng:' + endpoint:
            random.random()
        return dict(summary={'planned_count': len(selected)},
                    cases=[{'case_id': record['case_id']} for record in selected],
                    identity={'mock_endpoint': endpoint})

    def subset(self, case_rows, selected):
        self.subsets.append((len(case_rows), len(selected)))
        if not set(record['case_id'] for record in selected) <= set(row['case_id'] for row in case_rows):
            raise AssertionError('Subset without measured overlap')
        return {'planned_count': len(selected)}

    def subset_receipt(self, observed, selected, *, endpoint, cohort_label, out):
        return dict(summary=self.subset(observed['cases'],selected),
                    cases=[{'case_id':r['case_id']} for r in selected],
                    identity={'mock_endpoint':endpoint})


class MockTransaction:
    def __init__(self, view, engine, bench):
        self.view, self.engine, self.done = view, engine, False
        self.rollback_verified = False

    def __enter__(self):
        self.before = (self.view.model.number, self.view.model.pruned,
                       {layer: value.clone() for layer, value in self.engine.history().items()})
        return self

    def finish(self):
        self.done = True

    def __exit__(self, kind, value, tb):
        if not self.done:
            self.view.model.number, self.view.model.pruned, self.engine._history = self.before
            self.rollback_verified = True


def fixture(arm, *, mutation=None, fail_commit=False):
    model = types.SimpleNamespace(number=0, pruned=False)
    engine = MockEngine(arm, model)
    view = types.SimpleNamespace(model=model, sites=runner.ARM_LAYERS[arm], weights={},
                                 guard=lambda: ('mock-storage-guard',), hook_signature=lambda: ())
    bench = types.SimpleNamespace(contexts=[])
    all_records = records()
    generation = MockGeneration(model, all_records, mutate=mutation)
    written, logged, transactions = {}, [], []

    def physical(view, history):
        return dict(W={str(layer): f'mock-W{model.number}-pruned{model.pruned}' for layer in view.sites},
                    H={str(layer): float(value) for layer, value in history.items()})

    def observe(view, bench, all_records, selected, history, endpoint, out, current_ids=None):
        summary = {'mock_RPN_requests': len(selected)}
        return dict(summary=summary, current={'mock_RPN_requests': len(current_ids)},
                    state=physical(view, history))

    def persist(path, value):
        if fail_commit and Path(path).name == 'commit.json':
            raise OSError('MOCK_COMMIT_WRITE_FAILURE')
        written[str(path)] = copy.deepcopy(value)

    def transaction(*args):
        tx = MockTransaction(*args)
        transactions.append(tx)
        return tx

    def log_generation(tracker, prefix, summary, edits, pre_edits, post_edits):
        if prefix != 'W0_first2000':
            if model.number * 100 != post_edits:
                raise AssertionError('Prospective post state logged before commit')
            if f'/mock/{arm}/batch-{model.number:02d}/commit.json' not in written:
                raise AssertionError('Generation logged before successful commit')
        logged.append((prefix, dict(summary), edits, pre_edits, post_edits))

    ops = types.SimpleNamespace(state=physical, observe=observe,
        install_W0=lambda *args: {'summary': {'mock_RPN_requests': 2000}},
        rows=lambda *args: [], transaction=transaction, write=persist,
        log_w0=lambda *args: None, log_batch=lambda *args: None,
        log_generation=log_generation, safe_log=lambda tracker, builder, phase: builder(),
        reserve=lambda *args: None, cleanup=lambda: None,
        batches=lambda rows: ((index + 1, rows[index * 100:(index + 1) * 100],
                              rows[:(index + 1) * 100]) for index in range(20)))
    return model, engine, view, bench, all_records, generation, ops, written, logged, transactions


def execute_fixture(arm, **kwargs):
    parts = fixture(arm, **kwargs)
    model, engine, view, bench, all_records, generation, ops, *_ = parts
    with contextlib.redirect_stdout(io.StringIO()):
        terminal = runner.execute_chain(config(), {'source_commit': 'fixture-source'},
            Path('/mock') / arm, arm, model, None, view, engine, bench,
            all_records, generation, object(), ops=ops)
    return terminal, parts


class SixArmDispatchTests(unittest.TestCase):
    def test_six_factories_keep_native_configuration_and_request_schema(self):
        original = config()
        before = copy.deepcopy(original)
        for arm in runner.ARMS:
            projected = runner.arm_configuration(original, arm)
            self.assertEqual(projected['native'], original['arm_configs'][arm]['native'])
            self.assertEqual(projected['task_id'], runner.TASK)
            self.assertEqual(projected['model_revision'], original['model_revision'])
            _, normalize = runner.family_api(arm)
            normalized = normalize(records()[:100])
            self.assertIsInstance(normalized[0]['target_new'], str if arm in runner.STOCK_ARMS else dict)
        self.assertEqual(original, before)
        wrong = copy.deepcopy(original)
        wrong['arm_configs']['BASE_MEMIT']['seed'] += 1
        with self.assertRaisesRegex(RuntimeError, 'GENERATION_FAMILY_DRIFT:seed'):
            runner.arm_configuration(wrong, 'BASE_MEMIT')

    def test_all_six_20_dispatches_one_w0_and_no_duplicate_milestone_generation(self):
        for arm in runner.ARMS:
            with self.subTest(arm=arm):
                terminal, parts = execute_fixture(arm)
                model, engine, view, bench, all_records, generation, ops, written, logged, txs = parts
                self.assertEqual(terminal['status'], 'COMPLETED')
                self.assertEqual(engine.applies, list(range(1, 21)))
                self.assertEqual(engine.counts, {key: value * 20 for key, value in runner.expected_counts(arm).items()})
                self.assertEqual(generation.loads, 1)
                self.assertEqual(len(generation.endpoints), 40)
                self.assertEqual([row['count'] for row in generation.endpoints if row['cohort'] == 'ALL_SEEN'],
                                 [500, 1000, 1500, 2000])
                self.assertEqual(generation.subsets, [(500, 100), (1000, 100), (1500, 100), (2000, 100)])
                self.assertEqual(sum(row['count'] for row in generation.endpoints), 8600)
                self.assertEqual(len([row for row in logged if row[0] == 'all_seen/post']), 4)
                self.assertTrue(all(tx.done for tx in txs))
                commit = written[f'/mock/{arm}/batch-20/commit.json']
                self.assertEqual(commit['post_current']['mock_RPN_requests'], 100)
                self.assertEqual(commit['gen_current']['summary']['planned_count'], 100)
                self.assertEqual(commit['gen_prefix']['summary']['planned_count'], 2000)
                self.assertNotEqual(commit['gen_current']['identity'], commit['gen_prefix']['identity'])
                self.assertEqual(commit['gen_current']['model_state'],commit['gen_prefix']['model_state'])
                self.assertEqual(commit['gen_current']['raw_directory'],commit['gen_prefix']['raw_directory'])
                self.assertTrue(commit['gen_current']['derived_subset'])
                self.assertTrue(commit['gen_current']['derived_subset'])
                self.assertFalse(commit['checkpoint_saved'])

    def test_pre_is_actual_entry_and_prune_terminal_precedes_both_post_observers(self):
        terminal, parts = execute_fixture('PRUNE')
        model, engine, view, bench, all_records, generation, ops, written, logged, txs = parts
        for number in range(1, 21):
            pre, post = generation.endpoints[2 * (number - 1):2 * number]
            self.assertEqual((pre['number'], post['number']), (number - 1, number))
            self.assertFalse(pre['pruned'])
            self.assertEqual(post['pruned'], number == 20)
            row = [row for row in logged if row[0] == 'current/pre' and row[2] == number * 100][0]
            self.assertEqual(row[3:], ((number - 1) * 100, number * 100))
        self.assertEqual(engine.terminal_calls, 1)
        commit = written['/mock/PRUNE/batch-20/commit.json']
        self.assertEqual(commit['native']['terminal_prune']['explicit_repair'], runner.BASE_FIX)
        self.assertNotIn('repair_label', commit['native']['terminal_prune'])
        self.assertTrue(commit['native']['prune_applied'])
        self.assertIn('prunedTrue', next(iter(commit['after']['W'].values())))

    def test_generation_mutation_and_commit_failure_rollback_without_refit(self):
        for mode in ('post_mutation', 'commit_failure'):
            parts = fixture('BASE_ALPHAEDIT', mutation='W1' if mode == 'post_mutation' else None,
                            fail_commit=mode == 'commit_failure')
            model, engine, view, bench, all_records, generation, ops, written, logged, txs = parts
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises((RuntimeError, OSError)):
                runner.execute_chain(config(), {'source_commit': 'fixture-source'},
                    '/mock/BASE_ALPHAEDIT', 'BASE_ALPHAEDIT', model, None, view, engine, bench,
                    all_records, generation, object(), ops=ops)
            self.assertEqual(engine.applies, [1])
            self.assertEqual(model.number, 0)
            self.assertEqual(engine.history(), {})
            self.assertTrue(txs[-1].rollback_verified)
            self.assertEqual(engine.counts['native_z'], 100)
            self.assertFalse(any(row[0] == 'current/pre' for row in logged))

    def test_rng_mutation_before_fit_is_typed_stop_no_native_call(self):
        parts = fixture('BASE_MEMIT', mutation='rng:B1_PRE')
        model, engine, view, bench, all_records, generation, ops, *_ = parts
        before = random.getstate()
        try:
            with self.assertRaisesRegex(RuntimeError, 'GENERATION_NATIVE_RNG_MUTATION'):
                runner.execute_chain(config(), {'source_commit': 'fixture-source'},
                    '/mock/BASE_MEMIT', 'BASE_MEMIT', model, None, view, engine, bench,
                    all_records, generation, object(), ops=ops)
            self.assertEqual(engine.applies, [])
        finally:
            random.setstate(before)

    def test_stock_fit_axis_uses_measured_evaluations_and_updates_not_z_calls(self):
        engine = types.SimpleNamespace(current={'fit_trace': []})
        seen = []
        progress = runner.make_progress(engine, object(), 'BASE_MEMIT',
            lambda tracker, builder, phase: seen.append(builder()))
        for index, evaluations, updates in [(1, 25, 24), (2, 3, 2)]:
            engine.current['fit_trace'].append(dict(request_index=index, evaluations=evaluations,
                Adam_updates=updates, loss=.2, nll_loss=.1, kl_loss=.01))
            progress(dict(batch=1, request_index=index, native_z=index, native_z_completed=index))
        self.assertEqual([row['fit/global_candidate'] for row in seen], [25, 28])
        self.assertEqual([row['optimizer/calls'] for row in seen], [24, 26])
        self.assertEqual(seen[-1]['fit/nll'], .1)

    def test_no_checkpoint_writers_or_generation_math_and_no_gpu_initialized(self):
        tree = ast.parse(inspect.getsource(runner))
        forbidden = {'save', 'savez', 'savez_compressed', 'save_pretrained', 'dump'}
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & forbidden)
        self.assertNotIn('TfidfVectorizer', inspect.getsource(runner))
        self.assertNotIn('word_tokenize', inspect.getsource(runner))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
