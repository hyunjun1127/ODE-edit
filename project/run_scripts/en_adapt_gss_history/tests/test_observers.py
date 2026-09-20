"""CPU fixtures for metrics parity, identities, overwrite and report boundaries."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from project.run_scripts.en_adaptive_nullspace import metrics as parent
from project.run_scripts.en_adapt_gss_history import observers as obs
from project.run_scripts.en_adapt_gss_history import report


def row(case, family='NS', flags=(True,), success=True, index=0):
    result = dict(case_id=case, family=family, prompt_index=index, identity=f'{case}-{family}-{index}',
                  token_identity=str(len(flags)), success=success, new_nll=2., true_nll=1.,
                  desired_nll=1., desired_margin=1. if success else -1.)
    for side in ('new', 'true', 'desired'):
        result[side + '_token_correct'] = list(flags)
        result[side + '_token_count'] = len(flags)
        result[side + '_strict'] = all(flags)
    return result


def result(ids, *, bad=()):
    return obs.summarize([row(case, family, success=case not in bad, index=i)
                          for family, count in [('RS',1),('PS',2),('NS',2)]
                          for case in ids for i in range(count)], ids)


def inventory(n=200):
    records, versions = [], []
    for i in range(n):
        records.append(dict(case_id=i, requested_rewrite=dict(subject=f'S{i}', relation_id='P1',
            target_new={'id':str(i),'str':f'T{i}'},target_true={'str':'old'},prompt='{} is'),
            paraphrase_prompts=['p1','p2'], neighborhood_prompts=['n1','n2']))
        versions.append(dict(version_id=f'v{i}',fact=[f'S{i}','P1'],created_batch=i//100+1,
            representative_case_id=i,latest_case_id=i,occurrence_case_ids=[i],active=True))
    return records, dict(versions=versions, active_version_ids=[v['version_id'] for v in versions],bank_ids=[],pending_ids=[])


class ObserverTests(unittest.TestCase):
    def test_linear_summary_matches_parent_all_definitions(self):
        rows = [row(1,'RS'),row(1,'PS',flags=(True,False)),row(1,'PS',index=1),
                row(2,'NS',flags=(True,)),row(3,'NS',flags=(False,False,True),success=False)]
        ids = [1,2,3]
        self.assertEqual(obs.summarize(rows, ids), parent.summarize(rows, ids))
        a = obs.summarize(rows, ids)['aggregates']['NS']
        self.assertEqual(a['tf_token_micro']['rate'], .5)
        self.assertAlmostEqual(a['tf_prompt_macro']['value'], 2/3)
        self.assertEqual(a['tf_strict']['rate'], .5)

    def test_exact_neural_family_stream_and_layout_preserved(self):
        calls = []
        def pairs(records):
            return {family + '_' + side: [(r['case_id'], family, side) for r in records]
                    for family in ('rewrite','rephrase','locality') for side in ('target_new','target_true')
                    if family != 'locality' or side == 'target_true'}
        def eval_pairs(model, tok, values, device, microbatch_size):
            calls.append((copy.deepcopy(values), device, microbatch_size))
            return [dict(case_id=c, prompt_index=0, prompt=f'{c}-{family}',target=side,
                         target_token_ids=[3],nll=1. if side=='target_new' else 2.,
                         token_correct=[True],all_tokens_correct=True) for c,family,side in values]
        modules = {'evaluator':SimpleNamespace(counterfact_pairs=pairs,evaluate_pairs=eval_pairs),
                   'locality':SimpleNamespace(counterfact_locality_target_new_pairs=lambda records:
                       [(r['case_id'],'locality','target_new') for r in records]),'receipt':{'fixture':True}}
        binder = SimpleNamespace(_binding=lambda:modules,_device=lambda model:'cpu')
        records = [{'case_id':i} for i in range(37)]
        with mock.patch.object(parent,'_binding',return_value=binder):
            old = parent.evaluate(None,SimpleNamespace(padding_side='right'),records)
            old_calls = copy.deepcopy(calls)
            calls.clear()
            new = obs.evaluate(None,SimpleNamespace(padding_side='right'),records)
        self.assertEqual(calls,old_calls)
        self.assertEqual(len(calls),6)
        for key in ('rows','metrics','aggregates','joint','request_ids','request_order','source_binding','evaluator_layout'):
            self.assertEqual(new[key],old[key])

    def test_plan_dedup_fixed_order_and_arm_bank_independence(self):
        records, ledger = inventory()
        a = obs.plan_records(records,ledger,2)
        other = copy.deepcopy(ledger)
        other['bank_ids'] = ['v1','v4']
        b = obs.plan_records(records,other,2)
        self.assertEqual(a['request_ids'],list(range(200)))
        self.assertEqual(a['cohorts'],b['cohorts'])
        self.assertEqual(a['panel']['selected_version_ids'],b['panel']['selected_version_ids'])
        self.assertEqual(len(a['cohorts']['history_panel']),100)
        self.assertEqual(len(a['records']),200)
        self.assertEqual(obs.plan_records(records[:100],{'versions':ledger['versions'][:100]},1)['request_ids'],list(range(100)))

    def test_overwrite_and_repeat_keep_occurrence_denominators(self):
        records, ledger = inventory()
        records[100]['requested_rewrite']['subject'] = 'S0'
        records[101]['requested_rewrite']['subject'] = 'S1'
        records[101]['requested_rewrite']['target_new'] = records[1]['requested_rewrite']['target_new']
        versions = ledger['versions']
        versions[0]['active'] = False
        versions[100]['fact'] = ['S0','P1']
        versions[1].update(latest_case_id=101, occurrence_case_ids=[1,101])
        del versions[101]
        ledger['active_version_ids'] = [v['version_id'] for v in versions if v['active']]
        plan = obs.plan_records(records,ledger,2)
        self.assertEqual(plan['superseded_occurrence_case_ids'],[0])
        self.assertEqual(len(plan['cohorts']['full_latest_valid']),198)
        self.assertNotIn(1,plan['cohorts']['full_latest_valid'])
        self.assertIn(101,plan['cohorts']['full_latest_valid'])
        self.assertEqual(plan['cohorts']['cumulative_neighborhood'],list(range(200)))
        self.assertEqual(plan['case_metadata'][101]['created_batch'],1)
        self.assertEqual(plan['case_metadata'][101]['age_batches'],1)
        self.assertEqual(plan['case_metadata'][101]['version_original_case_id'],1)
        self.assertNotIn('v1',plan['panel']['selected_version_ids'])

    def test_paired_equal_totals_preserve_lost_gained(self):
        before, after = result([1,2],bad=[2]),result([1,2],bad=[1])
        pair = obs._pair(before,after,[1,2])['families']['RS']
        self.assertEqual(pair['lost_ids'],[[1,0]])
        self.assertEqual(pair['gained_ids'],[[2,0]])
        with self.assertRaises(ValueError):
            obs._pair(before,after,[3])

    def test_state_bank_and_entry_missing_are_explicit(self):
        records, ledger = inventory(100)
        state = obs.ObserverState('EN_ADAPT_H_RES')
        plan = state.plan(records,ledger,1)
        self.assertEqual(plan['missing_W0_case_ids'],list(range(100)))
        receipt = state.record(1,plan,baseline=result(range(100)),native=result(range(100),bad=[1]),
            selected=result(range(100)),bank_before_current=[],bank_after_commit=['v1'])
        self.assertEqual(receipt['comparisons']['current']['native_damage_entry_to_WN']['status'],'NOT_OBSERVED')
        self.assertEqual(receipt['observed_groups']['objective_bank_out']['requests'],100)
        self.assertEqual(receipt['observed_groups']['postcommit_bank_in']['requests'],1)
        self.assertFalse(receipt['bootstrap']['performed'])
        self.assertEqual(receipt['precision_status'],'NOT_ESTABLISHED')
        with self.assertRaises(ValueError):
            state.record(1,plan,native=result(range(100)),selected=result(range(100)))

    def test_same_case_different_baseline_is_not_replaced(self):
        state = obs.ObserverState('EN_ADAPT_H_GSS_REC')
        state.remember_baseline(result([1]))
        with self.assertRaises(ValueError):
            state.remember_baseline(result([1],bad=[1]))
        with self.assertRaises(ValueError):
            obs.merge_results([result([1]),result([1])])

    def test_report_reads_gzip_and_never_claims_terminal(self):
        records, ledger = inventory(100)
        state = obs.ObserverState('EN_ADAPT_H_RES')
        plan = state.plan(records,ledger,1)
        receipt = state.record(1,plan,baseline=result(range(100)),native=result(range(100)),selected=result(range(100)))
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)/'run'; batch = output/'B001'; batch.mkdir(parents=True)
            (batch/'observer.json.gz').write_bytes(gzip.compress(json.dumps(receipt).encode()))
            reduced = report.build(output,state.arm,bootstrap=False)
            self.assertEqual(reduced['observed_batches'],[1])
            self.assertEqual(reduced['missing_batches'],list(range(2,101)))
            self.assertEqual(reduced['actual_scheduler_terminal'],'NOT_OBSERVED')
            self.assertEqual(reduced['status'],'INCOMPLETE_OBSERVER_EVIDENCE')
            destination = Path(td)/'report'
            report.write(reduced,destination)
            self.assertTrue((destination/'report-ko.md').is_file())
            with self.assertRaises(FileExistsError):
                report.write(reduced,destination)

    def test_method_report_keeps_selected_scalars_and_latest_counters(self):
        arm = 'EN_ADAPT_H_GSS_REC'
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)/'output'; root.mkdir()
            (root.parent/'execution.lock.json').write_text(json.dumps({'source_commit':'fixture'}))
            for batch in (1, 2):
                directory = root/f'B{batch:03d}'; directory.mkdir()
                members = {
                    'batch-cost': dict(batch=batch, arm=arm, seconds=10., native=2., geometry=1., gradient=3.,
                        controller=1., commit_teacher=1., observer=2., cumulative={'native_batches':batch},
                        history_counts={'KL_VJPs':batch*100}, reference_counts={'reference_gradient':batch}),
                    'SELECTION_SEALED': dict(batch=batch, arm=arm, status='SEARCH_LIMIT_NO_ACCEPTED_CANDIDATE',
                        evaluations=2, selected_trial=None, objective={'J':.1,'L_R':.02,'L_H':.08}),
                    'spectrum': dict(spectrum={'exact_energy':.4,'eigenvalues':[9]*100,'mode_energies':[7]*100},
                        geometry={'rank':3,'singular_values':[4]*100}, selected={'released_modes':2,'eta':.5,
                        'caps':{'linear_zero_loss':.5},'active_caps':['linear_zero_loss']}, frontiers={'large':[8]*100}),
                    'history-selection': dict(pool_size=612, selected_ids=['a','b'], weights=[.4,.6],
                        selection_exercised=True,NLL_VJPs=612,KL_VJPs=612, pruning={'trace':[{'huge':[1]*100}],
                        'removed':[3]},diagnostics={'sketch':{'status':'SKETCH_SELECTION_UNRESOLVED',
                        'pair_rank_spearman':.6,'exact_factor_norms':[4]*64}}),
                    'controller': dict(status='SEARCH_LIMIT_NO_ACCEPTED_CANDIDATE',alias='native',ledger=[]),
                    'teacher-bindings': dict(receipt={'new_versions':1},bindings={'v1':{
                        'at_write_TF_strict':False,'at_write_TF_valid_tokens':2,'at_write_TF_token_correct':[True,False]}})
                }
                for name, value in members.items():
                    (directory/(name+'.json.gz')).write_bytes(gzip.compress(json.dumps(value).encode()))
            reduced = report.build(root,arm,bootstrap=False)
            totals = reduced['method_totals']
            self.assertEqual(totals['latest_cumulative_counts']['native_batches'],2)
            self.assertEqual(totals['phase_seconds']['seconds'],20.)
            self.assertEqual(totals['NLL_fact_VJPs'],1224)
            self.assertEqual(totals['fallback_batches'],[1,2])
            self.assertEqual(reduced['execution_lock']['source_commit'],'fixture')
            self.assertIsNone(reduced['terminal_marker'])
            method = reduced['methods'][0]
            self.assertNotIn('eigenvalues',method['geometry']['spectrum_scalars'])
            self.assertNotIn('trace',method['history']['pruning'])
            self.assertNotIn('exact_factor_norms',method['history']['diagnostics']['sketch'])
            self.assertEqual(method['teacher']['atwrite_TF_strict_failures'],1)
            self.assertEqual(method['teacher']['atwrite_TF_token_micro'],.5)
            self.assertEqual(method['teacher_technical_failures'],'NOT_RECORDED_IN_SUCCESS_RECEIPT')


if __name__ == '__main__':
    unittest.main()
