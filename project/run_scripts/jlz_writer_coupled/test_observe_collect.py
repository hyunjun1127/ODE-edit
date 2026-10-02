"""Bounded CPU observer and saved-evidence collection tests; no pretrained model."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from .collect import (EXPECTED, NOT_RECORDED, _completion_problems, collect,
                      transition)
from .common import digest, sha, write
from .observe import active_flags, observe, reduce_rows, scores


def record(case, target='new', subject='subject'):
    return dict(case_id=case, requested_rewrite=dict(prompt='{} private prompt', subject=subject,
                relation_id='R1', target_new=dict(id=target, str=target),
                target_true=dict(id='true', str='true')))


class Bench:
    tokenizer = SimpleNamespace(pad_token_id=0)

    def panels(self, value):
        return {'R': [value['requested_rewrite']['prompt'].format(value['requested_rewrite']['subject'])],
                'P': ['private paraphrase 0', 'private paraphrase 1'],
                'N': [f'private neighborhood {i}' for i in range(10)]}

    def evaluation_ids(self, prompt, target):
        return ([1, 2] if prompt == 'short' else [1, 1, 2],
                [2, 2] if target == 'new' else [3])


class Backbone(torch.nn.Module):
    def forward(self, input_ids, attention_mask, use_cache):
        assert use_cache is False
        return SimpleNamespace(last_hidden_state=torch.nn.functional.one_hot(input_ids, 5).float() * 3.)


class Adapter:
    def __init__(self):
        self.device = torch.device('cpu')
        self.model = SimpleNamespace(model=Backbone(), lm_head=torch.nn.Identity(),
                                     config=SimpleNamespace(max_position_embeddings=64))
        self.weights = {0: torch.tensor([[1., 2.]], dtype=torch.float32)}
        self.revision, self.hooks = 0, 0

    def guard(self):
        return self.revision

    def hook_signature(self):
        return self.hooks


class Memory:
    def __init__(self):
        self.revision = 0

    def summary(self):
        return dict(state_hash=str(self.revision))


def fake_scores(adapter, bench, pairs, microbatch):
    return [dict(nll=1. if target == 'new' else 2., token_count=2,
                 token_correct=2 if target == 'new' else 1, strict=target == 'new',
                 token_identity=digest([prompt, target])) for prompt, target in pairs]


def rows(case_ids=(0,), endpoint='W01', success=True):
    result = []
    for case in case_ids:
        for kind, count in (('R', 1), ('P', 2), ('N', 10)):
            for index in range(count):
                new, true = (1., 2.) if (success != (kind == 'N')) else (2., 1.)
                result.append(dict(case_id=case, kind=kind, prompt_index=index,
                    identity=digest([case, kind, index]), endpoint=endpoint, active_at_endpoint=True,
                    new_nll=new, true_nll=true, new_token_count=2, new_token_correct=2,
                    new_strict=True, true_token_count=3, true_token_correct=2, true_strict=False,
                    new_token_identity=digest([case, kind, index, 'new']),
                    true_token_identity=digest([case, kind, index, 'true'])))
    return result


def commit(batch, ids):
    return dict(batch=batch, actual_B=len(ids), current_ids=list(ids), candidate_count=3,
                backward_count=2, accepted_updates=1, rejected_trials=1,
                history_appends=5, seconds=1.5)


def endpoint(root, arm, batch, data, ids, summary=True):
    base = root / f'arm-{arm}' / 'main'
    write(base / f'batch-{batch:02d}' / 'commit.json', commit(batch, ids))
    path = base / f'observe-W{batch:02d}'
    write(path / 'chunk-0000.json', dict(rows=data, optimizer_feedback=False))
    if summary:
        write(path / 'summary.json', dict(summary=reduce_rows(data),
              current=reduce_rows([r for r in data if r['case_id'] in set(ids)]),
              row_count=len(data), row_order=digest([r['identity'] for r in data]),
              no_mutation=True, optimizer_feedback=False))


class ObserverTests(unittest.TestCase):
    def test_scores_full_vocab_target_tokens_and_microbatch_parity(self):
        adapter, bench = Adapter(), Bench()
        pairs = [('short', 'new'), ('long', 'true'), ('long', 'new')]
        one, all_at_once = scores(adapter, bench, pairs, 1), scores(adapter, bench, pairs, 3)
        self.assertEqual(one, all_at_once)
        self.assertEqual([x['token_count'] for x in one], [2, 1, 2])
        self.assertEqual([x['token_correct'] for x in one], [2, 0, 2])
        self.assertEqual([x['strict'] for x in one], [True, False, True])
        self.assertEqual(one[0]['token_identity'], digest([[1, 2], [2, 2]]))
        with self.assertRaisesRegex(RuntimeError, 'MICROBATCH'):
            scores(adapter, bench, pairs, 0)
        adapter.model.config.max_position_embeddings = 1
        with self.assertRaisesRegex(RuntimeError, 'OVERFLOW_NO_TRUNCATION'):
            scores(adapter, bench, pairs, 2)

    def test_active_flags_unicode_alias_and_target_supersession(self):
        values = [record(0, 'a', ' e\u0301  subject '), record(1, 'a', 'é subject'),
                  record(2, 'b', 'é subject'), record(3, 'a', 'é subject')]
        self.assertEqual(active_flags(values), {0: False, 1: False, 2: False, 3: True})

    def test_reducer_ties_fail_and_preserves_denominators_tf_means(self):
        data = rows()
        data[0]['new_nll'] = data[0]['true_nll']
        reduced = reduce_rows(data)
        self.assertEqual(reduced['R']['numerator'], 0)
        self.assertEqual(reduced['P']['denominator'], 2)
        self.assertEqual(reduced['N']['numerator'], 10)
        self.assertEqual(reduced['N']['token_micro'], 2 / 3)
        self.assertEqual(reduced['N']['strict_numerator'], 0)
        self.assertEqual(reduce_rows([]), {})
        for bad in (data + [data[0]], [dict(data[0], new_nll=float('nan'))],
                    [dict(data[0], new_token_count=0)], [dict(data[0], new_strict=False)]):
            with self.assertRaises(RuntimeError):
                reduce_rows(bad)

    def test_observe_preserves_state_and_writes_only_scalar_identity_rows(self):
        adapter, memory, bench = Adapter(), Memory(), Bench()
        history = {0: torch.zeros(2, 2)}
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
                'project.run_scripts.jlz_writer_coupled.observe.scores', fake_scores), mock.patch('builtins.print'):
            out = Path(tmp)
            result = observe(adapter, bench, [record(0)], [record(0)], history, 'W01', out,
                             current_ids=[], memory=memory)
            self.assertTrue(result['no_mutation'])
            self.assertTrue(result['memory_no_mutation'])
            self.assertEqual(result['current'], {})
            self.assertEqual(result['row_count'], 13)
            text = (out / 'chunk-0000.json').read_text()
            self.assertNotIn('private', text)
            self.assertNotIn('target_new', text)
            self.assertNotIn('input_ids', text)
            self.assertEqual(json.loads((out / 'summary.json').read_text())['summary'], result['summary'])

    def test_observe_detects_weights_history_guard_hooks_and_memory_mutation(self):
        for component in ('weights', 'history', 'guard', 'hooks', 'memory'):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as tmp:
                adapter, memory, history = Adapter(), Memory(), {0: torch.zeros(2, 2)}

                def mutate(a, b, pairs, size):
                    if component == 'weights': a.weights[0].add_(1)
                    elif component == 'history': history[0].add_(1)
                    elif component == 'guard': a.revision += 1
                    elif component == 'hooks': a.hooks += 1
                    else: memory.revision += 1
                    return fake_scores(a, b, pairs, size)

                with mock.patch('project.run_scripts.jlz_writer_coupled.observe.scores', mutate), \
                        mock.patch('builtins.print'), self.assertRaisesRegex(RuntimeError, 'OBSERVER.*MUTATION'):
                    observe(adapter, Bench(), [record(0)], [record(0)], history, 'W01', Path(tmp), memory=memory)
                self.assertFalse((Path(tmp) / 'summary.json').exists())

    def test_observer_failure_preserves_completed_chunks_no_summary(self):
        calls = 0

        def fail_second(adapter, bench, pairs, size):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('synthetic later failure')
            return fake_scores(adapter, bench, pairs, size)

        with tempfile.TemporaryDirectory() as tmp, mock.patch('builtins.print'), mock.patch(
                'project.run_scripts.jlz_writer_coupled.observe.scores', fail_second):
            records = [record(i) for i in range(51)]
            with self.assertRaisesRegex(RuntimeError, 'later failure'):
                observe(Adapter(), Bench(), records, records, {0: torch.zeros(2, 2)}, 'W01', Path(tmp))
            self.assertEqual(len(json.loads((Path(tmp) / 'chunk-0000.json').read_text())['rows']), 650)
            self.assertFalse((Path(tmp) / 'summary.json').exists())


class CollectorTests(unittest.TestCase):
    def test_transition_lost_gained_ties_missing_and_nll_delta(self):
        before = rows((0, 1), success=True)
        before.extend(rows((2,), success=False))
        after = rows((0,), success=False) + rows((2,), success=True)
        result = transition(before, after)
        self.assertEqual(result['N']['lost'], 10)
        self.assertEqual(result['N']['gained'], 10)
        self.assertEqual(result['N']['missing_rows'], 10)
        self.assertEqual(result['R']['birth_conditional_retention'], 0.)
        self.assertFalse(result['R']['coverage_complete'])
        self.assertEqual(result['R']['new_nll_delta'], 0.)
        self.assertEqual(transition(rows(success=False), rows())['R']['birth_conditional_retention'], NOT_RECORDED)

    def test_completion_requires_every_commit_panel_milestone_and_receipt(self):
        commits = {b: commit(b, list(range((b - 1) * 100, b * 100))) for b in range(1, 21)}
        current = {b: {k: dict(denominator=v // 20) for k, v in EXPECTED.items()} for b in commits}
        endpoints = {b: dict(verified=True, summary={k: dict(denominator=v * b // 20)
                        for k, v in EXPECTED.items()}) for b in commits}
        terminal = dict(status='COMPLETED')
        self.assertEqual(_completion_problems(commits, endpoints, current, terminal, []), [])
        for changed in ('missing_commit', 'wrong_n', 'summary_unverified', 'terminal', 'bad_budget', 'errors'):
            c, e, u, t, errors = copy.deepcopy(commits), copy.deepcopy(endpoints), copy.deepcopy(current), dict(terminal), []
            if changed == 'missing_commit': del c[4]
            elif changed == 'wrong_n': e[20]['summary']['N']['denominator'] -= 1
            elif changed == 'summary_unverified': e[12]['verified'] = False
            elif changed == 'terminal': t['status'] = 'TECHNICAL_FAILED'
            elif changed == 'bad_budget': c[2]['candidate_count'] = 26
            else: errors.append(dict(error='broken receipt'))
            self.assertTrue(_completion_problems(c, e, u, t, errors), changed)

    def test_empty_or_failed_attempt_reports_partial_without_wait_or_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / 'arm-A' / 'terminal.json', dict(status='TECHNICAL_FAILED'))
            with mock.patch('subprocess.run', side_effect=AssertionError('no Slurm/model subprocess')):
                result = collect(root, root / 'report-1')
            self.assertEqual(result['status'], 'PARTIAL_OR_TECHNICAL_FAILED')
            self.assertEqual(result['arms']['A']['commits'], 0)
            self.assertEqual(result['arms']['B']['cost']['candidate_count'], NOT_RECORDED)
            self.assertTrue((root / 'report-1' / 'artifact-index.json').is_file())
            self.assertTrue((root / 'report-1' / 'terminal.json').is_file())
            with self.assertRaisesRegex(RuntimeError, 'NEW_OR_EMPTY'):
                collect(root, root / 'report-1')

    def test_current_and_allseen_raw_reduction_birth_w0_active_and_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            birth = rows((0,), 'W01', True)
            final = rows((0,), 'W05', False) + rows((1,), 'W05', True)
            for row in final:
                row['active_at_endpoint'] = row['case_id'] == 1
            endpoint(root, 'A', 1, birth, [0])
            endpoint(root, 'A', 5, final, [1])
            raw_w0 = root / 'historic-raw.json'
            write(raw_w0, dict(rows=rows((0, 1), 'W0', True)))
            write(root / 'W0-reuse.json', dict(observations=dict(path=str(raw_w0), sha256=sha(raw_w0))))
            write(root / 'arm-A' / 'terminal.json', dict(status='TECHNICAL_FAILED', cost=dict(entry_seconds=5.)))
            write(root / 'arm-A' / 'initial.json', dict(main_B1_committed=True, observer_no_mutation=True,
                                                       main_B2_entry=dict(state=dict(W='hash', H='hash'))))
            result = collect(root, root / 'report')
            self.assertEqual(result['w0']['status'], 'REUSED_HISTORICAL_NO_NEW_FORWARD')
            self.assertEqual(result['arms']['A']['cost']['candidate_count'], 6)
            self.assertEqual(result['arms']['A']['component_cost'], {'entry_seconds': 5.})
            paired = json.loads((root / 'report' / 'paired.json').read_text())['A']
            self.assertEqual(paired['atwrite_to_endpoint']['R']['lost'], 1)
            self.assertEqual(paired['W0_to_endpoint']['N']['lost'], 10)
            self.assertEqual(paired['active']['R']['denominator'], 1)
            self.assertEqual(paired['superseded']['R']['denominator'], 1)
            text = (root / 'report' / 'metrics.csv').read_text()
            self.assertIn('allseen', text)
            self.assertIn('current', text)
            for path in (root / 'report').glob('*.json'):
                self.assertNotIn('private prompt', path.read_text())
                self.assertNotIn('"rows"', path.read_text())

    def test_missing_summary_and_corrupt_later_chunk_still_publish_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            endpoint(root, 'A', 1, rows(), [0], summary=False)
            broken = root / 'arm-A' / 'main' / 'observe-W01' / 'chunk-0050.json'
            broken.write_text('{')
            result = collect(root, root / 'report')
            self.assertEqual(result['arms']['A']['endpoints']['W01']['summary']['R']['denominator'], 1)
            self.assertFalse(result['arms']['A']['endpoints']['W01']['verified'])
            self.assertTrue(result['arms']['A']['errors'])
            self.assertEqual(result['status'], 'PARTIAL_OR_TECHNICAL_FAILED')

    def test_independent_summary_mismatch_and_changed_w0_never_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            endpoint(root, 'A', 1, rows(), [0])
            path = root / 'arm-A' / 'main' / 'observe-W01' / 'summary.json'
            payload = json.loads(path.read_text())
            payload['summary']['R']['numerator'] = 99
            path.write_text(json.dumps(payload))
            raw_w0 = root / 'historic-raw.json'
            write(raw_w0, dict(rows=rows()))
            write(root / 'w0-reuse.json', dict(observations=dict(path=str(raw_w0), sha256='not-the-hash')))
            result = collect(root, root / 'report')
            self.assertTrue(result['arms']['A']['errors'])
            self.assertEqual(result['errors'][0]['error'], 'W0_RAW_IDENTITY_MISMATCH')
            self.assertEqual(result['w0']['status'], NOT_RECORDED)

    def test_collector_cli_import_and_empty_attempt_do_not_import_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = ('import sys; from project.run_scripts.jlz_writer_coupled.collect import main; '
                      'assert "torch" not in sys.modules; main(); assert "torch" not in sys.modules')
            completed = subprocess.run([sys.executable, '-c', script, '--attempt', tmp,
                                        '--report', str(Path(tmp) / 'report')],
                                       capture_output=True, text=True, check=True)
            self.assertIn('PARTIAL_OR_TECHNICAL_FAILED', completed.stdout)


if __name__ == '__main__':
    unittest.main()
