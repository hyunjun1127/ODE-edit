"""CPU-only scalar fixtures. No tokenizer, model load, GPU or scheduler."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from . import metrics
from .collect import endpoint, _counter_delta, _metric_rows, collect, allocation_once
from .common import TASK, NONCE, ARMS, digest, member, sha, write
from project.run_scripts.jlz_price_gpt2xl.tracking import batch_values
from project.run_scripts.jlz_realization.observe import reduce_rows
from project.run_scripts.jlz_realized_writer_sequential.review_completed import Reader


def make_row(case, kind, index, endpoint_name='W0'):
    # Deliberately make N's desired true accuracy differ from new accuracy.
    return dict(identity=digest([case, kind, index]), case_id=case, kind=kind,
        prompt_index=index, endpoint=endpoint_name, active_at_endpoint=True,
        new_token_identity=digest(['new', case, kind, index]),
        true_token_identity=digest(['true', case, kind, index]),
        new_nll=2. if kind == 'N' else 1., true_nll=1. if kind == 'N' else 2.,
        new_token_count=2, new_token_correct=0 if kind == 'N' else 2,
        new_strict=kind != 'N', true_token_count=2,
        true_token_correct=2 if kind == 'N' else 0, true_strict=kind == 'N',
        margin_true_minus_new=-1. if kind == 'N' else 1.,
        margin_new_minus_true=1. if kind == 'N' else -1.)


def case_rows(case, name='W0'):
    return [make_row(case, kind, index, name) for kind, count in (('R', 1), ('P', 2), ('N', 10))
            for index in range(count)]


def summary(rows, state, name, requests):
    return dict(endpoint=name, state=state, requests=requests, row_count=len(rows),
        row_order=digest([row['identity'] for row in rows]), summary=reduce_rows(rows),
        seconds=3., no_mutation=True, optimizer_feedback=False)


class MetricsTests(unittest.TestCase):
    def accounting_fixture(self, root):
        lock = dict(source_commit='a' * 40, owner='fixture-owner')
        write(root / 'execution.lock.json', lock)
        write(root / 'submission.json', dict(instruction_id=NONCE, task_id=TASK,
            source_commit=lock['source_commit'], lock=member(root / 'execution.lock.json'),
            jobs=dict(BASE_MEMIT='12345', BASE_ALPHAEDIT='12346', collector='12347')))
        return lock

    def test_accounting_one_query_exact_gpu_parents_and_no_double_count(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lock = self.accounting_fixture(root)
            output = '\n'.join([
                f'12345|{TASK}-BASE_MEMIT|fixture-owner|COMPLETED|0:0|7|cpu=8,gres/gpu=1,gres/gpu:rtx_a6000=1|',
                '12345.batch|batch|fixture-owner|COMPLETED|0:0|6|cpu=8|',
                f'12346|{TASK}-BASE_ALPHAEDIT|fixture-owner|FAILED|1:0|9|cpu=8,gres/gpu=1|'])
            calls = []
            def runner(argv, **kwargs):
                calls.append((argv, kwargs))
                return SimpleNamespace(returncode=0, stdout=output)
            value = allocation_once(Reader(), root, lock, runner, 'fixture-owner')
            self.assertEqual(value['status'], 'RECORDED')
            self.assertEqual(value['queries'], 1)
            self.assertEqual(calls[0][0][4], '12345,12346')
            self.assertEqual(calls[0][1]['timeout'], 20)
            self.assertEqual([record['allocated_GPU_seconds'] for record in value['records']], [7, 9])
            self.assertTrue(all(record['child_steps_excluded'] for record in value['records']))
            self.assertNotIn('12347', ' '.join(calls[0][0]))

    def test_accounting_owner_mismatch_is_not_science_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lock = self.accounting_fixture(root)
            output = f'12345|{TASK}-BASE_MEMIT|other-owner|COMPLETED|0:0|7|gres/gpu=1|'
            value = allocation_once(Reader(), root, lock,
                lambda *a, **k: SimpleNamespace(returncode=0, stdout=output), 'fixture-owner')
            self.assertEqual(value['status'], 'NOT_RECORDED')
            self.assertEqual(value['queries'], 1)
            self.assertNotIn('other-owner', json.dumps(value))

    def test_accounting_invalid_or_duplicate_id_has_no_scheduler_call(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lock = self.accounting_fixture(root)
            submission = json.loads((root / 'submission.json').read_text())
            submission['jobs']['BASE_ALPHAEDIT'] = '12345'
            with patch.object(Reader, 'json', return_value=submission):
                with patch('subprocess.run') as runner:
                    value = allocation_once(Reader(), root, lock, runner, 'fixture-owner')
            self.assertEqual(value['status'], 'NOT_RECORDED')
            self.assertEqual(value['queries'], 0)
            runner.assert_not_called()

    def test_accounting_timeout_has_no_retry_or_scheduler_raw(self):
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            lock = self.accounting_fixture(root)
            def runner(*a, **k):
                raise subprocess.TimeoutExpired('sacct', 20, output='PRIVATE_STDOUT')
            value = allocation_once(Reader(), root, lock, runner, 'fixture-owner')
            self.assertEqual((value['status'], value['queries']), ('NOT_RECORDED', 1))
            self.assertTrue(value['no_retry'])
            self.assertNotIn('PRIVATE_STDOUT', json.dumps(value))

    @staticmethod
    def transaction_fixture(initial_history=False):
        import torch
        from . import run
        selected = {layer: torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
                    for layer in metrics.SITES}
        other = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        model = type('Model', (), {'named_parameters': lambda self:
            iter([(f'w{layer}', value) for layer, value in selected.items()] + [('other', other)])})()
        view = type('View', (), {'weights': selected, 'model': model,
                                 'hook_signature': lambda self: ()})()
        class Engine:
            def __init__(self):
                self.H = {layer: torch.ones(2, 2) for layer in metrics.SITES} if initial_history else {}
                self.resets = 0
            def history(self):
                return self.H
            def reset_history_to_uninitialized(self):
                self.H = {}; self.resets += 1
        return run, torch, view, Engine(), type('Bench', (), {'contexts': [['{}']]})()

    def test_transaction_observer_exception_restores_initial_alpha_without_H(self):
        run, torch, view, engine, bench = self.transaction_fixture()
        before = metrics.state(view, {})
        with patch.object(run, 'rng_snapshot', return_value='fixture'), \
             patch.object(run, 'rng_restore') as restore, \
             patch.object(run, 'rng_equal', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'CPU_OBSERVER_EXCEPTION'):
                with run.NativeTransaction(view, engine, bench) as tx:
                    with torch.no_grad():
                        for value in view.weights.values(): value.add_(1)
                    engine.H = {layer: torch.ones(2, 2) for layer in metrics.SITES}
                    raise RuntimeError('CPU_OBSERVER_EXCEPTION')
        self.assertTrue(tx.rollback_verified)
        self.assertEqual(metrics.state(view, engine.history()), before)
        self.assertEqual(engine.resets, 1)
        restore.assert_called_once_with('fixture')

    def test_transaction_commit_IO_exception_restores_existing_alpha_H(self):
        run, torch, view, engine, bench = self.transaction_fixture(True)
        before = metrics.state(view, engine.history())
        with patch.object(run, 'rng_snapshot', return_value='fixture'), \
             patch.object(run, 'rng_restore'), patch.object(run, 'rng_equal', return_value=True):
            with self.assertRaisesRegex(OSError, 'CPU_COMMIT_WRITE_FAILURE'):
                with run.NativeTransaction(view, engine, bench) as tx:
                    with torch.no_grad():
                        for value in view.weights.values(): value.add_(1)
                        for value in engine.H.values(): value.add_(1)
                    tx.finish()
                    try:
                        raise OSError('CPU_COMMIT_WRITE_FAILURE')
                    except OSError:
                        tx.done = False
                        raise
        self.assertTrue(tx.rollback_verified)
        self.assertEqual(metrics.state(view, engine.history()), before)
        self.assertEqual(engine.resets, 0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_MEMIT_state_has_no_synthetic_H(self):
        view = type('View', (), {'weights': {layer: object() for layer in metrics.SITES}})()
        with patch.object(metrics, 'tensor_sha', return_value='weight') as hashed:
            actual = metrics.state(view, {})
        self.assertEqual(actual['H'], {})
        self.assertEqual(hashed.call_count, 5)

    def test_wrong_alpha_history_layers_block(self):
        view = type('View', (), {'weights': {13: object()}})()
        with self.assertRaises(RuntimeError):
            metrics.state(view, {13: object()})

    def test_independent_N_desired_true_and_ties_fail(self):
        raw = case_rows(1)
        reduced = metrics.independent_reduce(raw)
        self.assertEqual(reduced['N']['desired_token_correct'], 20)
        self.assertEqual(reduced['N']['numerator'], 10)
        tied = copy.deepcopy(raw)
        for row in tied:
            row['new_nll'] = row['true_nll'] = 1.
            row['margin_true_minus_new'] = row['margin_new_minus_true'] = 0.
        self.assertTrue(all(group['numerator'] == 0 for group in metrics.independent_reduce(tied).values()))

    def test_missing_not_zero(self):
        self.assertNotIn('P', metrics.independent_reduce([make_row(1, 'R', 0)]))
        with tempfile.TemporaryDirectory() as folder:
            self.assertIsNone(endpoint(Reader(), folder, [], [], 'W0', dict(W={}, H={})))

    def test_duplicate_identity_blocks(self):
        with self.assertRaises(ValueError):
            metrics.independent_reduce([make_row(1, 'R', 0), make_row(1, 'R', 0)])

    def test_native_counts_no_ours_KKT_or_Adam_inference(self):
        self.assertIsNone(_counter_delta({}))
        data = dict(native_z=100, write_keys=5, history_keys=0, solves=5, history_appends=0)
        self.assertEqual(_counter_delta(dict(native_counts=data)), data)
        with self.assertRaises(RuntimeError):
            _counter_delta(dict(native_counts=dict(data, solves=-1)))

    def test_mapping_pct_nats_harmonic(self):
        rows = list(_metric_rows('BASE_MEMIT', 'W1_CURRENT', 100,
                                 metrics.independent_reduce(case_rows(1))))
        self.assertTrue(all(row['success_pct'] == row['success_harmonic_pct'] == 100. for row in rows))
        self.assertEqual(next(row for row in rows if row['kind'] == 'N')['margin_true_minus_new'], -1.)

    def test_milestone_current100_and_all_seen500_distinct(self):
        current = reduce_rows([row for case in range(100) for row in case_rows(case)])
        seen = reduce_rows([row for case in range(500) for row in case_rows(case)])
        values = batch_values(current, current, seen, 5)
        self.assertEqual([values[f'current/post/{kind}/count'] for kind in ('R', 'P', 'N')],
                         [100, 200, 1000])
        self.assertEqual([values[f'all_seen/post/{kind}/count'] for kind in ('R', 'P', 'N')],
                         [500, 1000, 5000])
        self.assertEqual((values['edits'], values['pre_state_edits'], values['post_state_edits']),
                         (500, 400, 500))
        ordinary = batch_values(current, current, None, 4)
        self.assertFalse(any(key.startswith('all_seen/') for key in ordinary))
        with self.assertRaises(RuntimeError):
            batch_values(current, current, seen, 4)

    def test_collector_missing_science_is_not_completed(self):
        with tempfile.TemporaryDirectory() as folder:
            attempt = Path(folder)
            write(attempt / 'identity.json', dict(rows=[]))
            write(attempt / 'stream.json', [dict(case_id=case, requested_rewrite=dict(
                subject=f's{case}', relation_id='r', target_new=dict(str='new', id=case)))
                for case in range(2000)])
            config = dict(task_id=TASK, observer_identity=member(attempt / 'identity.json'),
                stream=str(attempt / 'stream.json'), assets=[member(attempt / 'stream.json')],
                cold_W={str(layer): 'W' for layer in metrics.SITES},
                packs=[dict(ids=list(range(i * 100, (i + 1) * 100))) for i in range(20)])
            write(attempt / 'config.json', config)
            write(attempt / 'execution.lock.json', dict(source_commit='a' * 40,
                                                       config_sha256=sha(attempt / 'config.json')))
            result = collect(attempt)
            self.assertEqual(result['collector_status'], 'COMPLETED')
            self.assertFalse(result['scientific_complete'])
            terminal = json.loads((attempt / 'collector/terminal.json').read_text())
            self.assertTrue((attempt / 'collector/report-ko.md').exists())
            self.assertTrue((attempt / 'collector/manifest.json').exists())
            self.assertFalse(terminal['scientific_complete'])
            self.assertEqual(terminal['new_model_forwards'], 0)
            with self.assertRaises(RuntimeError):
                collect(attempt)

    def test_full_W0_reference_projection_keeps_original_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root / 'source', root / 'arm'
            source.mkdir(); out.mkdir()
            W = {str(layer): 'W' + str(layer) for layer in metrics.SITES}
            old_state = dict(W=W, H={str(layer): 'ZERO' for layer in metrics.SITES})
            actual_state = dict(W=W, H={})
            raw = [row for case in range(2000) for row in case_rows(case)]
            chunks = []
            for i in range(40):
                path = source / f'chunk-{50 * i:04d}.json'
                write(path, dict(state=old_state, optimizer_feedback=False,
                                 rows=raw[650 * i:650 * (i + 1)]))
                chunks.append(member(path))
            old_summary = summary(raw, old_state, 'W0', 2000)
            write(source / 'summary.json', old_summary)
            runtime = {key: 'bound' for key in metrics.RUNTIME_FIELDS}
            write(source / 'runtime.json', runtime)
            write(out / 'runtime.json', runtime)
            write(root / 'identities.json', dict(rows=[{key: row[key] for key in
                ('identity', 'case_id', 'kind', 'prompt_index', 'new_token_identity', 'true_token_identity')}
                for row in raw]))
            manifest = dict(status='QUALIFIED_EXACT_REUSE', cold_state=old_state,
                chunks=chunks, summary=member(source / 'summary.json'),
                runtime=member(source / 'runtime.json'), observation_identity='exact', source_folder=str(source))
            c = dict(W0_reuse=manifest, observation_identity='exact', cold_W=W,
                observer_identity=member(root / 'identities.json'),
                packs=[dict(ids=list(range(i * 100, (i + 1) * 100))) for i in range(20)])
            first_bytes = Path(chunks[0]['path']).read_bytes()
            with patch.object(metrics, 'state', return_value=actual_state):
                installed = metrics.install_W0(c, object(), {}, out, None)
            self.assertEqual(installed['state'], actual_state)
            self.assertEqual(installed['source_observed_state'], old_state)
            self.assertEqual(installed['new_forwards'], 0)
            self.assertEqual(metrics.rows(out / 'W0', actual_state), raw)
            self.assertEqual(Path(chunks[0]['path']).read_bytes(), first_bytes)
            audited = endpoint(Reader(), out / 'W0', json.loads((root / 'identities.json').read_text())['rows'],
                               list(range(2000)), 'W0', actual_state)
            self.assertEqual(audited['summary']['N']['denominator'], 20000)
            broken_state = dict(W=dict(W, **{'13': 'CHANGED'}), H={})
            with self.assertRaises(RuntimeError):
                metrics.rows(out / 'W0', broken_state)
            runtime['torch'] = 'different'
            # New output has no overwritten original receipt or fixture member.
            another = root / 'another'; another.mkdir()
            write(another / 'runtime.json', runtime)
            with patch.object(metrics, 'state', return_value=actual_state):
                with self.assertRaises(RuntimeError):
                    metrics.install_W0(c, object(), {}, another, None)


if __name__ == '__main__':
    unittest.main()
