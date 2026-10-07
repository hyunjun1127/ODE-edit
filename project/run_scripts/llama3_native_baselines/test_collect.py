"""CPU scalar/raw identity fixtures, never editing/model/scheduler experiments."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project.run_scripts.jlz_realized_writer_sequential.review_completed import Reader, reduce_rows
from .common import TASK, digest, member, write
from .collect import (HISTORY_SITES, _identity, endpoint, exact_scalars, fit_events,
                      generation_reduce, generation_scalar, native_counts, review_arm, scalar_rpn)


def raw_rows(case=0, endpoint_name='B1_PRE'):
    result = []
    for kind, count in (('R', 1), ('P', 2), ('N', 10)):
        for index in range(count):
            new, true = (2., 1.) if kind == 'N' else (1., 2.)
            result.append(dict(case_id=case, kind=kind, prompt_index=index,
                identity=digest([case, kind, index]), endpoint=endpoint_name,
                new_token_identity='new-token', true_token_identity='true-token',
                new_nll=new, true_nll=true, margin_true_minus_new=true-new,
                new_token_count=2, new_token_correct=0, new_strict=False,
                true_token_count=2, true_token_correct=2, true_strict=True))
    return result


def gen(case, entropy=None, cosine=None, reasons=None):
    return dict(case_id=case, ngram_entropy=entropy, reference_score=cosine,
                generation_prompt_count=1, generated_token_count=4,
                reason_counts=reasons or {})


class CollectorFixtures(unittest.TestCase):
    def test_generation_valid_zero_and_missing_are_not_confused(self):
        measured = generation_reduce([gen(0, 0., 0.)], [0])
        missing = generation_reduce([gen(0, reasons={'missing_reference': 1})], [0])
        values = generation_scalar('current/post', measured)
        self.assertEqual(values['current/post/fluency/ngram_entropy'], 0.)
        self.assertEqual(values['current/post/consistency/reference_score'], 0.)
        absent = generation_scalar('current/post', missing)
        self.assertNotIn('current/post/fluency/ngram_entropy', absent)
        self.assertNotIn('current/post/consistency/reference_score', absent)
        self.assertEqual(absent['current/post/generation/planned_count'], 1)

    def test_generation_occurrence_macro_sums_and_nonexclusive_reasons(self):
        reduced = generation_reduce([gen(0, 2., .25), gen(1, 4., None,
            {'missing_reference': 1, 'length_cap_no_continuation': 1})], [0, 1])
        self.assertEqual(reduced['fluency_sum'], 6.)
        self.assertEqual(reduced['consistency_sum'], .25)
        values = generation_scalar('all_seen/post', reduced)
        self.assertEqual(values['all_seen/post/fluency/ngram_entropy'], 3.)
        self.assertEqual(values['all_seen/post/consistency/reference_score'], .25)
        self.assertEqual(reduced['reason_counts']['length_cap_no_continuation'], 1)
        with self.assertRaisesRegex(RuntimeError, 'CASE_ORDER'):
            generation_reduce([gen(1), gen(0)], [0, 1])
        with self.assertRaisesRegex(RuntimeError, 'COSINE'):
            generation_reduce([gen(0, 1., 1.01)], [0])

    def test_shared_cosine_roundoff_policy_preserves_raw_and_denominator(self):
        value = 1 + math.ulp(1.)
        reduced = generation_reduce([gen(0, 1., value), gen(1, 1., value)], [0, 1])
        self.assertEqual(reduced['planned_count'], 2)
        self.assertEqual(reduced['consistency_count'], 2)
        self.assertEqual(reduced['consistency_sum'], 2 * value)
        mapped = generation_scalar('current/post', reduced)
        self.assertEqual(mapped['current/post/consistency/reference_score'], value)
        self.assertGreater(mapped['current/post/consistency/reference_score'], 1.)
        boundary = 1 + 4 * math.ulp(1.)
        self.assertEqual(generation_reduce([gen(0, 1., boundary)], [0])['consistency_sum'], boundary)
        with self.assertRaisesRegex(RuntimeError, 'COSINE'):
            generation_reduce([gen(0, 1., 1 + 5 * math.ulp(1.))], [0])

    def test_independent_RPN_percent_harmonic_and_N_true_target(self):
        reduced = reduce_rows(raw_rows())
        values = scalar_rpn('current/post', reduced, 1)
        self.assertEqual(values['current/post/N/token_acc_pct'], 100.)
        self.assertEqual(values['current/post/R/token_acc_pct'], 0.)
        self.assertEqual(values['current/post/success_harmonic_pct'], 100.)
        ties = raw_rows()
        ties[0].update(new_nll=2., margin_true_minus_new=0.)
        self.assertEqual(scalar_rpn('x', reduce_rows(ties), 1)['x/success_harmonic_pct'], 0.)
        with self.assertRaisesRegex(RuntimeError, 'DENOMINATOR'):
            scalar_rpn('current/post', reduced, 100)

    def test_exact_producer_keys_and_values_not_missing_zero(self):
        exact_scalars({'valid': 0., 'count': 1}, {'valid': 0., 'count': 1})
        with self.assertRaisesRegex(RuntimeError, 'SCALAR_KEYS'):
            exact_scalars({'valid': 0.}, {})
        with self.assertRaisesRegex(RuntimeError, 'SCALAR_VALUE'):
            exact_scalars({'valid': 0.}, {'valid': 1.})

    def test_native_history_and_fit_counts_are_method_specific(self):
        self.assertEqual(HISTORY_SITES, dict(MEMIT=0, PRUNE=0, RECT=0,
                         ALPHAEDIT=5, ALPHAEDIT_BLUE=2, CAKE=5))
        for method, history in HISTORY_SITES.items():
            sites = 2 if method == 'ALPHAEDIT_BLUE' else 5
            expected = dict(native_z=100*sites if method == 'ALPHAEDIT_BLUE' else 100,
                write_keys=sites, history_keys=history, solves=sites, history_appends=history)
            native = dict(method=method, batch=1, requests=100, same_model_returned=True,
                caller_history_appends=0, checkpoint_saved=False, cache_template=None,
                counts=expected, cumulative=expected)
            self.assertEqual(native_counts(native, method, 1, {key: 0 for key in expected}), expected)
            native['counts'] = dict(expected, history_appends=history+1)
            with self.assertRaisesRegex(RuntimeError, 'BATCH_COUNTS'):
                native_counts(native, method, 1, {key: 0 for key in expected})

    def test_unavailable_arm_remains_not_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {'native': {'MEMIT': {'layers': [4, 5, 6, 7, 8]}}}
            result = review_arm(Reader(), temporary, config, {}, 'MEMIT', 'fixture-not-uploaded', [], [])
            self.assertEqual(result['scientific_status'], 'NOT_AVAILABLE')
            self.assertEqual(result['commits'], 0)
            self.assertEqual(result['missing'], ['ARM_OUTPUT_NOT_AVAILABLE'])

    def test_source_config_actual_job_identity_is_strict(self):
        lock = dict(source_commit='a'*40, config_sha256='b'*64)
        receipt = dict(source='a'*40, config='b'*64, job='123', task_id=TASK,
                       method='CAKE', model='llama3')
        _identity(receipt, 'CAKE', lock, {}, '123', 'FIXTURE_IDENTITY')
        for field, value in (('job', '124'), ('model', 'gptj'), ('config', 'c'*64)):
            changed = dict(receipt, **{field: value})
            with self.assertRaises(RuntimeError):
                _identity(changed, 'CAKE', lock, {}, '123', 'FIXTURE_IDENTITY')

    def test_raw_endpoint_order_state_tokens_and_reducer(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rows, state = raw_rows(), {'W': {'4': 'cold'}, 'H': {}}
            reduced = reduce_rows(rows)
            write(folder/'chunk-0000.json', dict(state=state, optimizer_feedback=False, rows=rows))
            write(folder/'summary.json', dict(endpoint='B1_PRE', state=state, requests=1,
                row_count=13, row_order=digest([row['identity'] for row in rows]),
                summary=reduced, seconds=0., no_mutation=True, optimizer_feedback=False))
            self.assertEqual(endpoint(Reader(), folder, rows, [0], 'B1_PRE', state)['summary'], reduced)
            bad = copy.deepcopy(rows); bad[0]['new_token_identity'] = 'wrong'
            (folder/'chunk-0000.json').write_text(json.dumps(dict(state=state,
                optimizer_feedback=False, rows=bad)))
            with self.assertRaisesRegex(ValueError, 'TOKEN_IDENTITY'):
                endpoint(Reader(), folder, rows, [0], 'B1_PRE', state)

    def test_authoritative_fit_scalar_axis_and_terminal_counter(self):
        terminal = dict(batch=1, request_index=1, evaluations=2, Adam_updates=1,
                        loss=.01, nll_loss=.01, kl_loss=0., weight_decay=0.,
                        stop='TOTAL_LOSS_BELOW_005', fit_global_candidate=2, fit_updates=1)
        events = [dict(batch=1, fit_global_candidate=1, fit_updates=0),
                  dict(batch=1, fit_global_candidate=2, fit_updates=1), terminal]
        trace = {key: value for key, value in terminal.items() if key not in
                 ('batch', 'fit_global_candidate', 'fit_updates')}
        native = dict(counts=dict(native_z=1, fit_forwards=2, fit_updates=1, fit_trace=[trace]),
                      cumulative=dict(fit_forwards=2, fit_updates=1))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'events.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in events))
            receipt = dict(member=member(path), line_count=3, authoritative_stream=True)
            result = fit_events(Reader(), receipt, native, 1, 0, 0)
            self.assertEqual(result['native_Adam_updates'], 1)
            changed = copy.deepcopy(native); changed['counts']['fit_trace'][0]['Adam_updates'] = 2
            with self.assertRaisesRegex(RuntimeError, 'TERMINAL_TRACE'):
                fit_events(Reader(), receipt, changed, 1, 0, 0)

    def test_later_identity_fault_preserves_only_verified_commit_prefix(self):
        # Stored-metadata parser fixture, not a native fit or model experiment.
        records = [dict(case_id=index) for index in range(2000)]
        cold = dict(W={str(layer): 'cold-' + str(layer) for layer in range(4, 9)}, H={})
        after = dict(W={key: 'edited-' + key for key in cold['W']}, H={})
        config = dict(cold_W=cold['W'], native={'MEMIT': dict(layers=list(range(4, 9)),
            source_commit='native-source', native_has_history=False,
            scientific_fields={'v_lr': .1}, context_reuse={'contexts': {'sha256': 'context'}})})
        lock = dict(source_commit='a'*40, config_sha256='b'*64)
        expected = dict(native_z=100, write_keys=5, history_keys=0, solves=5, history_appends=0)
        trace = [dict(request_index=index, evaluations=1, Adam_updates=0,
            loss=.01, nll_loss=.01, kl_loss=0., weight_decay=0., stop='TOTAL_LOSS_BELOW_005')
            for index in range(1, 101)]
        native = dict(method='MEMIT', batch=1, requests=100, same_model_returned=True,
            caller_history_appends=0, checkpoint_saved=False, cache_template=None,
            layers=list(range(4, 9)), native_source_commit='native-source', native_has_history=False,
            hparams={'v_lr': .1}, context={'contexts_sha256': 'context'}, seconds=0.,
            counts=dict(expected, fit_forwards=100, fit_updates=0, fit_trace=trace, seconds={}),
            cumulative=dict(expected, fit_forwards=100, fit_updates=0))
        def observed(reader, folder, identities, ids, name, state, seen=None):
            raw = [row for case in ids for row in raw_rows(case, name)]
            return dict(rows=raw, summary=reduce_rows(raw), seconds=0., reference_only=name == 'W0')
        def gen_read(reader, ref, c, ids, state, name):
            aggregate = generation_reduce([gen(case, 0., 0., {key: 0 for key in
                ('missing_generation_prompts', 'missing_reference', 'zero_generated_vector',
                 'zero_reference_vector', 'nonfinite_score', 'length_cap_no_continuation')}) for case in ids], ids)
            return dict(summary=aggregate, work={}, pure_metric_rescore='FIXTURE_NOT_ACTUAL_EVIDENCE')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); out = root/'MEMIT'; folder = out/'batch-01'
            write(out/'W0/summary.json', dict(state=cold))
            write(out/'generation-W0.json', {})
            identity = dict(source='a'*40, config='b'*64, job='123', task_id=TASK,
                            method='MEMIT', model='llama3')
            entry = dict(identity, batch=1, requests=100, ids=list(range(100)), seen_ids=list(range(100)),
                         state=cold, RNG='entry-rng', context='entry-context', ledger='entry-ledger')
            write(folder/'entry.json', entry); write(folder/'native.json', native)
            lines = []
            for index, terminal in enumerate(trace, 1):
                lines.append(dict(batch=1, fit_global_candidate=index, fit_updates=0))
                lines.append(dict(terminal, batch=1, fit_global_candidate=index, fit_updates=0))
            event_path = folder/'events.jsonl'
            event_path.write_text(''.join(json.dumps(row)+'\n' for row in lines))
            write(folder/'generation.json', dict(task_id=TASK, method='MEMIT', model='llama3', batch=1,
                endpoints=dict(pre={}, current={}, w0_current={}), native_history_identity_separate={},
                raw_local_only=True))
            pre = observed(None, None, None, list(range(100)), 'B1_PRE', cold)['summary']
            current = observed(None, None, None, list(range(100)), 'B1_POST', after)['summary']
            values = dict(edits=100, batch=1, pre_state_edits=0, post_state_edits=100)
            values.update(scalar_rpn('current/pre', pre, 100)); values.update(scalar_rpn('current/post', current, 100))
            for prefix in ('current/pre', 'current/post', 'w0/current'):
                values.update(generation_scalar(prefix, gen_read(
                    None, None, None, list(range(100)), None, None)['summary']))
            mapping = scalar_rpn('_', pre, 100)
            values.update({'w0/current/N' + key[len('_/N'):]: value for key, value in mapping.items()
                           if key.startswith('_/N/')})
            write(folder/'metrics.json', values)
            commit = dict(identity, batch=1, requests=100, state_before=cold, state_after=after,
                transaction_finished=True, RNG_before='entry-rng', RNG_after='next-rng',
                entry_context='entry-context', exit_context='entry-context', ledger_before='entry-ledger',
                ledger_after='next-ledger', native=member(folder/'native.json'), native_counts=expected,
                history_appends=0, pre=pre, post=current, post_current=current, seconds=0.,
                generation=member(folder/'generation.json'), metrics=member(folder/'metrics.json'),
                events=dict(member=member(event_path), line_count=200, authoritative_stream=True))
            write(folder/'commit.json', commit)
            write(out/'batch-02/commit.json', dict(commit, batch=2, job='different-job'))
            write(out/'batch-02/entry.json', dict(entry, batch=2))
            progress = {}
            with patch('project.run_scripts.llama3_native_baselines.collect.endpoint', observed):
                with self.assertRaisesRegex(RuntimeError, 'COMMIT_IDENTITY_JOB'):
                    review_arm(Reader(), root, config, lock, 'MEMIT', '123', [], records,
                               generation_reader=gen_read, progress=progress)
            self.assertEqual(progress['commits'], 1)
            self.assertEqual(progress['requests'], 100)
            self.assertEqual(progress['state_links'], 0)
            self.assertEqual(progress['measured_native_counts']['history_appends'], 0)


if __name__ == '__main__':
    unittest.main()
