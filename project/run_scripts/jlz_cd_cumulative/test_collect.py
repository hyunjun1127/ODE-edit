"""Small stdlib-only regression fixtures; never load a model or scheduler."""
import copy
import unittest
from .common import expected_rows, batches, selected_for_post
from .collect import validate_fit, validate_actions, validate_commit, parse_accounting
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    reduce_rows, validate_rows, paired, harmonic)


def row(case=7, kind='R', new=1., true=2., correct=1, count=2):
    return dict(identity=f'{case}:{kind}', case_id=case, kind=kind, prompt_index=0, endpoint='W1',
        new_token_identity='new', true_token_identity='true', new_nll=new, true_nll=true,
        new_token_count=count, new_token_correct=correct, new_strict=correct == count,
        true_token_count=count, true_token_correct=correct, true_strict=correct == count,
        margin_true_minus_new=true - new, active_at_endpoint=True)


def fit_fixture():
    ids, layers = [7, 9], [4, 8]; trace = []; events = []
    for i in range(2):
        J = .1 if i == 0 else .01; reason = None if i == 0 else 'OBJECTIVE_THRESHOLD'
        t = dict(candidate=i, logical_evaluation=i + 1, updates_completed=i, alpha=0,
            costs={str(l): dict(Q=0., c=0., cumulative=0., Pi=0., weighted_cost=0.) for l in layers},
            native_J=[J, J], native_sum=2 * J, objective_mean=J, common_terminal_reason=reason,
            norm_added_once=i == 0, allocation_added_once=i == 0)
        trace.append(t)
        def emit(kind, payload, request=None, layer=None):
            events.append(dict(arm='CD_Q', batch=1, event=kind, request=request, candidate=i, layer=layer, payload=payload))
        emit('coupled_candidate', t)
        for r in ids:
            emit('candidate_request', dict(evaluation_ordinal=i + 1, updates_completed=i, will_backward=i == 0,
                common_candidate=True, permanent_request_freeze=False, nll=J, kl=0., F=J, J=J,
                price=.01, budget=0., norms=[0., 0.]), r)
            for l in layers:
                emit('candidate_layer', dict(gradients={} if i == 0 else None), r, l)
                if i == 0:
                    emit('optimizer_layer', dict(update=1, stored_budget=0., violation=0.,
                        moments_preserved_after_projection=True, eligible_for_reentry=True), r, l)
            if i == 1:
                emit('terminal_request', dict(accepted_candidate_index=1, logical_evaluations=2,
                    optimizer_updates=1, backward_calls=1, gradient_kkt_status='NO_BACKWARD_TERMINAL'), r)
    fit = dict(schema='JLZ_CD_CUMULATIVE_FIT_V1', terminal_candidate=1, requests=2,
        logical_evaluations=2, request_evaluations=4, optimizer_updates=1, request_updates=2,
        terminal_extra_forward=0, terminal_extra_backward=0, norm_gradient_additions=1, allocation_gradient_additions=1,
        synchronous_candidate=True, independent_request_freeze=False, checkpoint_saved=False,
        alpha=0, lambda_identity='lambda', candidates={str(r): dict(candidate=1) for r in ids}, candidate_trace=trace)
    return fit, events, ids, layers


class CollectorTests(unittest.TestCase):
    def test_ties_failure_and_N_true_desired(self):
        out = reduce_rows([row(kind='R', new=1, true=1), row(kind='N', new=2, true=1)])
        self.assertEqual(out['R']['numerator'], 0); self.assertEqual(out['N']['numerator'], 1)
        self.assertEqual(out['N']['desired_token_count'], 2)

    def test_prompt_macro_not_token_micro(self):
        out = reduce_rows([row(case=1, correct=1, count=1), row(case=2, correct=0, count=3)])['R']
        self.assertEqual(out['token_micro'], .25); self.assertEqual(out['prompt_macro'], .5)

    def test_nonfinite_and_strict_inconsistency_block(self):
        for bad in (row(new=float('nan')), row(correct=3, count=2)):
            with self.assertRaises(ValueError): validate_rows([bad])

    def test_exact_order_not_only_set(self):
        a, b = row(case=1), row(case=2)
        with self.assertRaises(ValueError): validate_rows([b, a], [a, b])
        refs = [a, b]; self.assertEqual([r['case_id'] for r in expected_rows(refs, [2, 1])], [2, 1])

    def test_paired_tokens_and_conservation(self):
        before, after = row(new=1, true=2), row(new=3, true=2)
        self.assertEqual(paired([before], [after])['R']['preference']['lost'], 1)
        after['new_token_identity'] = 'different'
        with self.assertRaises(ValueError): paired([before], [after])

    def test_common_fit_events(self):
        f, e, ids, ls = fit_fixture()
        out = validate_fit(f, e, ids, ls, 'source', 'CD_Q', 1, 'lambda')
        self.assertEqual(out['request_updates'], 2)

    def test_terminal_optimizer_and_missing_layer_block(self):
        f, e, ids, ls = fit_fixture()
        bad = copy.deepcopy(e); bad.append(dict(arm='CD_Q', batch=1, event='optimizer_layer', request=7, candidate=1, layer=4, payload={}))
        with self.assertRaises(RuntimeError): validate_fit(f, bad, ids, ls, 'source', 'CD_Q', 1, 'lambda')
        bad = [x for x in e if not (x['event'] == 'candidate_layer' and x['request'] == 7 and x['candidate'] == 0 and x['layer'] == 4)]
        with self.assertRaises(RuntimeError): validate_fit(f, bad, ids, ls, 'source', 'CD_Q', 1, 'lambda')

    def test_allocation_is_not_common_stop(self):
        f, e, ids, ls = fit_fixture(); f = copy.deepcopy(f); f['candidate_trace'][1]['native_J'] = [.1, .1]
        f['candidate_trace'][1]['native_sum'] = .2; f['candidate_trace'][1]['objective_mean'] = .1
        with self.assertRaises(RuntimeError): validate_fit(f, e, ids, ls, 'source', 'CD_Q', 1, 'lambda')

    def test_history_action_duplicate_rows_block(self):
        pack = dict(ids=[7], native_rows=2)
        item = dict(layer=4, global_row=0, owner=0, case_id=7, kind='rewrite', canonical=True,
            reference='ACTUAL_FIT_PROJECTED_Y', actual={})
        with self.assertRaises(RuntimeError): validate_actions(dict(rows=[item, copy.deepcopy(item)]), pack, [4], 'CD_Q')

    def test_action_lookup_mapping_is_bound_not_only_count(self):
        mapping = [dict(global_row=0, owner=0, kind='rewrite', lookup=3, canonical=True),
                   dict(global_row=1, owner=0, kind='kl', lookup=2, canonical=False)]
        pack = dict(ids=[7], native_rows=2, row_map=mapping)
        rows = [dict(layer=4, case_id=7, reference='ACTUAL_FIT_PROJECTED_Y', actual={}, **r) for r in mapping]
        validate_actions(dict(rows=rows), pack, [4], 'CD_Q')
        rows[1]['lookup'] = 3
        with self.assertRaises(RuntimeError): validate_actions(dict(rows=rows), pack, [4], 'CD_Q')

    def test_no_B21_and_milestone_selection(self):
        records = list(range(2000)); bs = list(batches(records, 100))
        self.assertEqual(len(bs), 20); self.assertEqual(bs[-1][0], 20)
        self.assertEqual(len(selected_for_post(bs[4][1], bs[4][2], 5)), 500)
        self.assertEqual(len(selected_for_post(bs[5][1], bs[5][2], 6)), 100)

    def test_accounting_parent_once_and_owner(self):
        jobs = dict(CD_Q=11, CD_C=12, collector=13)
        data = '11|owner|Q|COMPLETED|10|cpu=8,gres/gpu=1\n11.batch|owner|step|COMPLETED|10|gres/gpu=1\n12|owner|C|FAILED|4|gres/gpu=1\n13|owner|cpu|RUNNING|1|cpu=8\n'
        out = parse_accounting(data, jobs, 'owner')
        self.assertEqual(out['allocated_GPU_seconds'], 14); self.assertEqual(len(out['parents']), 3)
        with self.assertRaises(RuntimeError): parse_accounting(data, jobs, 'another')

    def test_duplicate_history_append_cannot_pass_layer_set(self):
        previous = dict(W={'4': 'w0'}, H={'4': 'h0'})
        after = dict(W={'4': 'w1'}, H={'4': 'h1'})
        entry = dict(source='s', config='c', arm='CD_Q', batch=1, ids=[7], native_pack='pack', state=previous, RNG='rng')
        commit = dict(entry, before=previous, after=after, RNG_before='rng', RNG_after='rng',
            lambda_identity='lambda', fit_count=1, observer_no_mutation=True, checkpoint_saved=False, history_appends=1)
        item = dict(layer=4, append_count=1, columns=1, rewrite_only=True, KL_in_history=False,
            CPU_FP32=True, before='h0', after='h1')
        writer = dict(history_appends=1, history=[item], layers={'4': dict(weight_after='w1',
            solver=dict(numerical_projection_verified=True), ideal_effective_parity=dict(pass_=True))})
        expected = dict(ids=[7], identity='pack')
        self.assertEqual(validate_commit(commit, writer, entry, expected, previous, 's', 'c', 'CD_Q', 1, [4], 'lambda'), after)
        writer['history'].append(copy.deepcopy(item))
        with self.assertRaises(RuntimeError): validate_commit(commit, writer, entry, expected, previous, 's', 'c', 'CD_Q', 1, [4], 'lambda')


if __name__ == '__main__': unittest.main()
