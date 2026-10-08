"""CPU-only six-native dispatch/commit controls, not actual GPU observations."""
import ast
import contextlib
import copy
import inspect
import io
import random
import unittest
from unittest.mock import patch

import torch

from . import generation_native_run as runner
from . import generation_run as original
from .generation_native_common import TASK, NONCE, SCHEDULE
from .test_generation_run import config, fixture, records


def native_config():
    value = config()
    value.update(task_id=TASK, instruction_id=NONCE)
    value['generation'] = dict(evaluation_schedule=SCHEDULE)
    return value


def native_fixture(arm, **kwargs):
    parts = fixture(arm, **kwargs)
    generation = parts[5]
    endpoint = generation.endpoint
    def observe(*args, **kw):
        result = endpoint(*args, **kw)
        result.update(native_execution_member={'CPU_fixture':True}, generation_native_runtime_member={},
            native_commit_members=[{'CPU_fixture':True}], committed_batch20_member={'CPU_fixture':True},
            generation_profile='CPU_NATIVE_PROFILE_FIXTURE', generation_route='CPU_NATIVE_ROUTE_FIXTURE',
            sampling_scope='ENDPOINT_GLOBAL_BATCH_STREAM', qualification_performed=False, no_fallback=True)
        return result
    generation.endpoint = observe
    return parts


def execute(arm, **kwargs):
    parts = native_fixture(arm, **kwargs)
    model, engine, view, bench, rows, generation, ops, *_ = parts
    rpn, commits, stages = [], [], []
    original_observe = ops.observe
    def observe(*args, **kw):
        rpn.append((args[5], len(args[3]), model.number, model.pruned))
        return original_observe(*args, **kw)
    ops.observe = observe
    with contextlib.redirect_stdout(io.StringIO()):
        terminal = runner.execute_chain(native_config(), {'source_commit':'CPU_SOURCE_FIXTURE'},
            '/mock/'+arm, arm, model, None, view, engine, bench, rows, generation, object(),
            ops=ops, on_commit=lambda r:commits.append(copy.deepcopy(r)), on_stage=stages.append)
    return terminal, parts, rpn, commits, stages


class NativeGenerationRunnerTests(unittest.TestCase):
    def test_projection_preserves_science_replaces_only_identity_and_old_helper_untouched(self):
        value = native_config()
        before = copy.deepcopy(value)
        for arm in runner.ARMS:
            projected = runner.arm_configuration(value, arm)
            self.assertEqual(projected['task_id'], TASK)
            self.assertEqual(projected['instruction_id'], NONCE)
            self.assertEqual(projected['native'], value['arm_configs'][arm]['native'])
            for key in ('model', 'model_revision', 'runtime', 'seed', 'stream', 'cold_W'):
                self.assertEqual(projected[key], value[key])
            _, normalize = original.family_api(arm)
            actual = normalize(records()[:100])
            self.assertIsInstance(actual[0]['target_new'], str if arm in original.STOCK_ARMS else dict)
        self.assertEqual(value, before)
        wrong = copy.deepcopy(value)
        wrong['arm_configs']['CAKE']['seed'] += 1
        with self.assertRaisesRegex(RuntimeError, 'FAMILY_DRIFT:seed'):
            runner.arm_configuration(wrong, 'CAKE')
        with self.assertRaisesRegex(RuntimeError, 'PROFILE_IDENTITY'):
            original.arm_configuration(value, 'CAKE')

    def test_six_arms20_native_commits_unchanged_rpn_single_generation(self):
        for arm in runner.ARMS:
            with self.subTest(arm=arm):
                terminal, parts, rpn, commits, stages = execute(arm)
                model, engine, view, bench, rows, generation, ops, written, logged, txs = parts
                self.assertEqual(engine.applies, list(range(1, 21)))
                self.assertEqual(engine.counts, {key:value*20 for key,value in runner.expected_counts(arm).items()})
                self.assertEqual([row[1] for row in rpn[::2]], [100]*20)
                self.assertEqual([row[1] for row in rpn[1::2]],
                    [n*100 if n in original.MILESTONES else 100 for n in range(1, 21)])
                self.assertEqual(generation.loads, 0)
                self.assertEqual(generation.subsets, [])
                self.assertEqual(len(generation.endpoints), 1)
                self.assertEqual(generation.endpoints[0]['endpoint'], 'W20')
                self.assertEqual(generation.endpoints[0]['count'], 2000)
                self.assertEqual(generation.endpoints[0]['number'], 20)
                self.assertEqual([point[0] for point in logged], ['all_seen/post'])
                self.assertEqual(logged[0][2:], (2000, 1900, 2000))
                self.assertEqual(len(commits), 20)
                self.assertTrue(all(tx.done and not tx.rollback_verified for tx in txs))
                self.assertTrue(all(r['task'] == TASK and r['generation_schedule'] == SCHEDULE for r in commits))
                final = written['/mock/'+arm+'/generation-final.json']
                self.assertEqual(final['committed_batch20_member'], {'CPU_fixture':True})
                self.assertIn('native_execution_member', final)
                self.assertFalse(final['qualification_performed'])
                self.assertEqual(terminal['generation_W0_endpoints'], 0)
                self.assertEqual(terminal['generation_intermediate_endpoints'], 0)
                self.assertTrue(terminal['final_generation_completed'])
                self.assertEqual(stages[-2:], ['FINAL_W20_GENERATION', 'FINAL_W20_GENERATION_COMPLETE'])

    def test_terminal_prune_precedes_final_rpn_and_generation(self):
        terminal, parts, rpn, commits, stages = execute('PRUNE')
        self.assertEqual(parts[1].terminal_calls, 1)
        self.assertEqual(rpn[-1], ('W20', 2000, 20, True))
        self.assertTrue(parts[5].endpoints[0]['pruned'])
        self.assertLess(stages.index('TERMINAL_PRUNE_BASE_FIX'), stages.index('B20_POST_RPN'))
        self.assertLess(stages.index('B20_POST_RPN'), stages.index('FINAL_W20_GENERATION'))

    def test_generation_failure_preserves_all_twenty_commits_and_rng_state_violations_detected(self):
        for mode in ('exception', 'rng', 'state'):
            parts = native_fixture('BASE_MEMIT', mutation={'rng':'rng:W20', 'state':'W20'}.get(mode))
            model, engine, view, bench, rows, generation, ops, written, logged, txs = parts
            saved = random.getstate()
            if mode == 'exception':
                generation.endpoint = lambda *args, **kw: (_ for _ in ()).throw(OSError('CPU_GENERATION_FAIL'))
            commits = []
            try:
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises((RuntimeError, OSError)):
                    runner.execute_chain(native_config(), {'source_commit':'CPU_SOURCE_FIXTURE'},
                        '/mock/BASE_MEMIT', 'BASE_MEMIT', model, None, view, engine, bench, rows,
                        generation, object(), ops=ops, on_commit=commits.append)
                self.assertEqual(len(commits), 20)
                self.assertTrue(all(tx.done and not tx.rollback_verified for tx in txs))
                self.assertIn('/mock/BASE_MEMIT/batch-20/commit.json', written)
                self.assertNotIn('/mock/BASE_MEMIT/generation-final.json', written)
                self.assertEqual(logged, [])
            finally:
                random.setstate(saved)

    def test_commit_write_failure_rolls_back_and_never_generates(self):
        parts = native_fixture('BASE_ALPHAEDIT', fail_commit=True)
        model, engine, view, bench, rows, generation, ops, written, logged, txs = parts
        with self.assertRaises(OSError):
            runner.execute_chain(native_config(), {'source_commit':'CPU_SOURCE_FIXTURE'},
                '/mock/BASE_ALPHAEDIT', 'BASE_ALPHAEDIT', model, None, view, engine, bench, rows,
                generation, object(), ops=ops)
        self.assertEqual(engine.applies, [1])
        self.assertEqual(model.number, 0)
        self.assertTrue(txs[-1].rollback_verified)
        self.assertFalse(generation.endpoints)

    def test_entry_has_no_legacy_qualification_w0_or_scientific_checkpoint_calls(self):
        source = inspect.getsource(runner)
        calls = {node.func.attr for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {'save', 'savez', 'savez_compressed', 'save_pretrained', 'dump'})
        self.assertEqual(source.count('generation.endpoint('), 1)
        self.assertNotIn('generation.load_W0(', source)
        self.assertNotIn('generation.subset', source)
        self.assertNotIn('qualify_and_link', source)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
