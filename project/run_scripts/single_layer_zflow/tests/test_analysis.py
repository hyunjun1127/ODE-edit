"""Small CPU fixtures only; no actual result/model/evaluator/Slurm access."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from project.run_scripts.single_layer_zflow import analysis as a
from project.run_scripts.single_layer_zflow.durable import CheckpointStore, prepare_state, tensor_sha256


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)


def records(count=1000):
    return [dict(case_id=i, requested_rewrite=dict(prompt='{} lives in', subject=f's{i}',
                target_new={'str': ' New'}, target_true={'str': ' Old'}),
                paraphrase_prompts=[f'p{i}-{j}' for j in range(2)],
                neighborhood_prompts=[f'n{i}-{j}' for j in range(10)]) for i in range(count)]


def observation(selected, binding=None, *, fail_first=False):
    metrics = {}
    for tag in a.TAGS:
        rows = []
        for index, (case, prompt, identity) in enumerate(a.expected_items(selected, tag)):
            new, true = (2., 1.) if tag == 'NS' else (1., 2.)
            if fail_first and index == 0:
                new = true  # ties are failures, including NS.
            success = true < new if tag == 'NS' else new < true
            rows.append(dict(case_id=case, prompt_index=prompt, identity=identity,
                new_nll=new, true_nll=true, margin=true-new, success=success,
                new_strict=True, true_strict=False, new_token_correct=2,
                new_token_count=2, true_token_correct=0, true_token_count=1))
        n = sum(r['success'] for r in rows)
        metrics[tag] = dict(rows=rows, numerator=n, denominator=len(rows), rate=n/len(rows),
                            bit_order_sha256=a.digest([(r['identity'], r['success']) for r in rows]))
    out = dict(requests=len(selected), metrics=metrics, new_forward=True, evaluation_seconds=.2,
               before_after_exact=True, evaluator_controller_influence=0,
               weight_state={a.WEIGHT: 'native-tensor-digest'}, cache_sha256='native-cache-digest')
    if binding is not None:
        out.update(state_binding=binding, request_order=binding['request_order'])
    return out


def flow_fixture():
    events = [dict(oracle=i+1, edit_nll=2.+i, essence_kl=0., L=2.+i, seconds=.02,
                   forward_loss_seconds=.01, backward_seconds=.01,
                   x_norm=float(i), gradient_norm=1.) for i in range(2)]
    return dict(status='NUMERICAL_STOP', accepted=0, rejected=1, oracle_calls=2,
                flow_seconds=.1, oracle_events=events,
                trace=[dict(nfe=2., accepted=0., step=1., F_before=2., F_trial=3.,
                            cost=0., barrier_cap=None, barrier_dual=0.,
                            stationarity_before=1., roundoff_limited=0.)],
                terminal=dict(L=2., C=0., F=2., residual=1., normalized_complementarity=0.,
                              normalized_feasibility=0., stationarity_threshold=.00001),
                work=dict(oracle_calls=2, suffix_forward_microbatches=2,
                          suffix_backward_microbatches=2, suffix_forward_loss_seconds=.02,
                          suffix_backward_seconds=.02))


class ReductionTests(unittest.TestCase):
    def setUp(self):
        self.records = records(2)

    def test_strict_token_and_native_ns_margin_are_distinct(self):
        raw = observation(self.records, fail_first=True)
        summary, _ = a.reduce_observation(raw, self.records)
        ns = next(r for r in summary if r['metric'] == 'NS')
        self.assertEqual((ns['numerator'], ns['denominator'], ns['ties']), (19, 20, 1))
        self.assertEqual(ns['new_tf_strict_num'], 20)
        self.assertEqual(ns['new_token_den'], 40)
        self.assertGreater(ns['success_margin_mean'], 0.)
        self.assertLess(raw['metrics']['NS']['rows'][1]['margin'], 0.)

    def test_quantile_linear_interpolation(self):
        q = a.quantiles([0., 2., 4., 6.])
        self.assertEqual(q['median'], 3.)
        self.assertEqual(q['p25'], 1.5)
        self.assertAlmostEqual(q['p90'], 5.4)

    def test_superseded_is_input_only_later_batch_not_same_batch_order(self):
        selected = records(101)
        for row in selected:
            row['requested_rewrite']['relation_id'] = 'relation'
        selected[100]['requested_rewrite']['subject'] = 's0'
        selected[100]['requested_rewrite']['target_new'] = {'str': 'Different'}
        selected[2]['requested_rewrite']['subject'] = 's1'
        selected[2]['requested_rewrite']['target_new'] = {'str': 'Conflict'}
        del selected[3]['requested_rewrite']['relation_id']
        populations, summary = a.request_populations(selected)
        self.assertEqual(populations['SUPERSEDED_CANDIDATE_LATER_BATCH'], {0})
        self.assertEqual(populations['WITHIN_BATCH_TARGET_CONFLICT'], {1, 2})
        self.assertEqual(populations['UNRESOLVED_MISSING_RELATION'], {3})
        self.assertIn(100, populations['ACTIVE_NO_LATER_DIFFERENT_TARGET'])
        self.assertEqual(sum(summary['counts'].values()), 101)
        self.assertFalse(summary['outcome_filtering'])

    def test_empty_population_is_na_not_zero_performance(self):
        raw = observation(self.records)
        _, items = a.reduce_observation(raw, self.records)
        rows = a.population_pairs(items, items, {'empty': set()}, 'fixture')
        self.assertTrue(all(r['status'] == 'EMPTY_INPUT_POPULATION' and r['denominator'] == 0 for r in rows))
        self.assertTrue(all('delta_pp' not in r for r in rows))

    def test_ties_fail_all_primary_metrics(self):
        summary, _ = a.reduce_observation(observation(self.records, fail_first=True), self.records)
        self.assertTrue(all(r['numerator'] == r['denominator'] - 1 for r in summary))

    def test_exact_identity_order_not_row_index_only(self):
        raw = observation(self.records)
        raw['metrics']['RS']['rows'].reverse()
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'identity or order'):
            a.reduce_observation(raw, self.records)

    def test_nonfinite_negative_nll_and_wrong_bool_rejected(self):
        for field, value in [('new_nll', float('nan')), ('true_nll', -1.), ('success', 1)]:
            raw = observation(self.records)
            raw['metrics']['RS']['rows'][0][field] = value
            with self.assertRaises(a.AnalysisIntegrityError):
                a.reduce_observation(raw, self.records)

    def test_strict_cannot_disagree_with_token_count(self):
        raw = observation(self.records)
        raw['metrics']['RS']['rows'][0]['new_token_correct'] = 1
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'strict/token'):
            a.reduce_observation(raw, self.records)

    def test_wrong_success_margin_and_numden_are_fail_closed(self):
        for mutation in ('success', 'margin', 'denominator', 'numerator', 'rate', 'bit_order_sha256'):
            raw = observation(self.records)
            metric = raw['metrics']['NS']
            if mutation == 'success': metric['rows'][0]['success'] = False
            elif mutation == 'margin': metric['rows'][0]['margin'] = 1.
            elif mutation == 'bit_order_sha256': metric[mutation] = 'bad'
            else: metric[mutation] = 999
            with self.assertRaises(a.AnalysisIntegrityError):
                a.reduce_observation(raw, self.records)

    def test_pairing_counts_losses_recoveries_and_nll_deltas(self):
        _, before = a.reduce_observation(observation(self.records, fail_first=True), self.records)
        _, after = a.reduce_observation(observation(self.records), self.records)
        pairs = a.paired(before, after, comparison='fixture')
        self.assertTrue(all(r['lost'] == 0 and r['gained'] == 1 for r in pairs))
        after['PS'].reverse()
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'paired identity'):
            a.paired(before, after, comparison='wrong')

    def test_flow_carry_and_accept_reject_accounting(self):
        config = {'objective': {'price': 1., 'beta_essence': .0625}, 'integrator': {'max_oracle_calls': 25}}
        flow = flow_fixture()
        rows, compute = a.flow_rows(flow, config, 'B001')
        self.assertEqual([r['phase'] for r in rows], ['initial', 'rejected'])
        self.assertEqual([r['calls'] for r in compute], [1, 0, 1])
        flow['accepted'] = 1
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'N_oracle'):
            a.flow_rows(flow, config, 'B001')

    def test_accepted_flow_terminal_must_be_last_accepted_not_rejected(self):
        config = {'objective': {'price': 1., 'beta_essence': .0625}, 'integrator': {'max_oracle_calls': 25}}
        flow = flow_fixture(); flow['accepted'] = 1; flow['rejected'] = 0
        flow['trace'][0]['accepted'] = 1.
        flow['oracle_events'][1].update(edit_nll=1., L=1.)
        flow['trace'][0]['F_trial'] = 1.
        flow['terminal'].update(L=1., F=1.)
        rows, _ = a.flow_rows(flow, config, 'B001')
        self.assertEqual(rows[-1]['phase'], 'accepted')
        flow['terminal']['F'] = 2.
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'terminal F'):
            a.flow_rows(flow, config, 'B001')


class CompleteChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'; self.run.mkdir()
        self.inputs = self.root / 'inputs'; self.inputs.mkdir()
        self.records = records()
        contract_source = Path(__file__).resolve().parents[4] / a.CONTRACT_REL
        self.config = json.loads(contract_source.read_text())
        contract = self.inputs / a.CONTRACT_REL
        contract.parent.mkdir(parents=True); contract.write_bytes(contract_source.read_bytes())
        dataset = self.inputs / 'counterfact.json'; put(dataset, self.records)
        contexts = self.inputs / 'contexts.json'; put(contexts, [['{}'], ['x. {}']])
        dummy_source = self.inputs / 'frozen.py'; dummy_source.write_text('# frozen fixture\n')
        source_lock = self.inputs / 'source.lock.json'
        put(source_lock, {'head': 'source-head', 'tree': 'source-tree', 'members': [a.member(dummy_source)]})
        ids = [r['case_id'] for r in self.records]
        self.lock = self.inputs / 'input.lock.json'
        put(self.lock, dict(source_head='source-head', source_tree='source-tree', source_root=str(self.inputs),
            source_lock=a.member(source_lock), dataset_root=str(self.inputs), contexts_path=str(contexts),
            contexts=dict(sha256=a.sha256(contexts), semantic_sha256=a.digest(json.loads(contexts.read_text()))),
            sample=dict(members=[a.member(dataset)], case_ids=ids, case_order_sha256=a.digest(ids),
                        prefix_records_sha256=a.digest(self.records),
                        batch_boundaries=[dict(batch_id=i+1, start=i*100, stop=(i+1)*100,
                            case_order_sha256=a.digest(ids[i*100:(i+1)*100])) for i in range(10)])))
        self.source = dict(input_lock_sha256=a.sha256(self.lock), source_head='source-head', source_tree='source-tree')
        w = torch.arange(6, dtype=torch.float32).reshape(2, 3); m = torch.zeros(3, 3)
        put(self.run / 'runtime.json', dict(source=self.source, fresh_pretrained_W0_cold_M0=True,
            technical_state_carried=False, base_selected_weight_sha256=tensor_sha256(w), model_dtype='float32',
            attention='eager', tf32_matmul=False, tf32_cudnn=False, writer_add_bos=False, writer_padding='right',
            evaluator_packing='native manual left / microbatch16', transformers='4.44.2'))
        store = CheckpointStore(self.run / 'checkpoints'); parent = None; completes = []
        for index in range(1, 11):
            batch = f'B{index:03d}'; root = self.run / batch
            flow = flow_fixture(); preparation = {'total_preparation_seconds': .2}
            keys = ('max_logit_abs', 'logit_relative_l2', 'edit_abs_error', 'kl_abs_error')
            parity = dict(passed=True, physical_parameter_copy=True, entry_restored_exact=True,
                          nonselected_pointer_version_unchanged=True, inner_history_append=0,
                          checks={k: True for k in keys}, tolerance={k: .001 for k in keys},
                          **{k: 0. for k in keys})
            cost = dict(passed=True, actual_cost=0., predicted_cost=0., relative_error=0.,
                        tolerance=.001, actual_delta_norm=0.)
            state = prepare_state(w, m, w.clone(), torch.zeros(2, 100), torch.zeros(100, 3),
                                  torch.zeros(100, 100), torch.zeros(3, 100), accepted=0, request_count=100,
                                  parity_evidence=parity, cost_evidence=cost)
            parent = store.publish(batch, parent, index+1, state, config=self.config, source=self.source,
                context=json.loads(contexts.read_text()), rng={'cpu': torch.Generator().manual_seed(2).get_state()},
                ledger=dict(flow=flow, preparation=preparation), cache_resume_fingerprint='fixture-cache')
            inventories = {'current': self.records[(index-1)*100:index*100]}
            if index in (5, 10): inventories['seen-full'] = self.records[:index*100]
            if index == 10: inventories['first500'] = self.records[:500]
            artifacts = {}
            for kind, selected in inventories.items():
                binding = a._binding(parent, self.source, kind, selected)
                raw = observation(selected, binding)
                if kind == 'first500':
                    raw['new_forward'] = False; raw['evaluation_seconds'] = 0.
                    raw['source_full_sha256'] = artifacts['seen-full']['sha256']
                path = root / 'observation-attempts' / f'{kind}-r0001' / 'result.json'; put(path, raw)
                summary = {tag: {key: raw['metrics'][tag][key] for key in ('numerator', 'denominator', 'rate')} for tag in a.TAGS}
                artifacts[kind] = dict(kind=kind, binding=binding, relative_path=str(path.relative_to(root)),
                    bytes=path.stat().st_size, sha256=a.sha256(path), evaluation_seconds=raw['evaluation_seconds'],
                    new_forward=raw['new_forward'], metrics=summary)
                put(root / 'observation-registry' / kind / '000001.json', artifacts[kind])
            complete = dict(batch=batch, batch_index=index, request_count=100, source=self.source,
                checkpoint=parent, accepted=0, rejected=1, oracle_calls=2, history_append=0,
                solver_status='NUMERICAL_STOP', metrics=artifacts['current']['metrics'], artifacts=artifacts,
                flow_seconds=.1, preparation_seconds=.2, commit_seconds=.01, evaluation_seconds=.2,
                total_seconds=.51, peak_gpu_allocated=0, peak_gpu_reserved=0, recovery=False,
                missing_interrupted_timing=None)
            put(root / 'COMPLETE.json', complete); completes.append(complete)
        put(self.run / 'TERMINAL.json', dict(status='SEQ1000_COMPLETED', source=self.source, batches=10,
            unique_requests=1000, attempted_requests=1000, completed=completes, final_checkpoint=parent,
            accepted=0, rejected=10, oracle_calls=20, history_appends=0, no_update_batches=10,
            latest_process_seconds=6., scientific_promotion=False))

    def analyze(self, **kwargs):
        return a.analyze(self.run, self.lock, input_sha256=a.sha256(self.lock), **kwargs)

    def test_complete_chain_full_checkpoint_and_independent_reduction(self):
        result = self.analyze()
        self.assertEqual(len(result['batch_rows']), 10)
        self.assertEqual(len(result['node_rows']), 20)
        self.assertEqual(len(result['metric_rows']), 39)
        self.assertEqual(result['terminal_counts']['no_update_batches'], 10)
        self.assertEqual(result['paired_rows'][-1]['status'], 'NOT_MEASURED_NO_N4_RAW')
        self.assertTrue(all(r['lost'] == r['gained'] == 0 for r in result['paired_rows'] if 'lost' in r))

    def test_optional_n4_requires_exact_file_and_item_identity(self):
        n4 = self.root / 'n4.json'; put(n4, observation(self.records, fail_first=True))
        result = self.analyze(n4_path=n4, n4_sha256=a.sha256(n4))
        pairs = [r for r in result['paired_rows'] if r['comparison'] == 'N4_W10_TO_SL_ZFLOW_W10']
        self.assertTrue(all(r['gained'] == 1 for r in pairs))
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'N4 raw SHA'):
            self.analyze(n4_path=n4, n4_sha256='wrong')

    def test_raw_corruption_fails_hash_before_metrics(self):
        path = self.run / 'B003/observation-attempts/current-r0001/result.json'
        with path.open('a') as stream: stream.write(' ')
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'SHA/size'):
            self.analyze()

    def test_incomplete_chain_not_promoted_to_final(self):
        (self.run / 'B010/COMPLETE.json').unlink()
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'missing/nonregular'):
            self.analyze()

    def test_wrong_initial_weight_and_terminal_count_fail(self):
        runtime = a.read_json(self.run / 'runtime.json')
        runtime['base_selected_weight_sha256'] = 'wrong'; put(self.run / 'runtime.json', runtime)
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'W0 selected weight'):
            self.analyze()

    def test_parent_link_is_independently_checked(self):
        store = CheckpointStore(self.run / 'checkpoints')
        def wrong_parent(batch):
            loaded = store.load(batch)
            if batch == 'B002': loaded['metadata']['parent']['payload_sha256'] = 'bad'
            return loaded
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'parent receipt'):
            self.analyze(checkpoint_loader=wrong_parent)

    def test_checkpoint_tensor_corruption_not_only_metadata_validation(self):
        path = self.run / 'checkpoints/B003/state.pt'
        with path.open('ab') as stream: stream.write(b'extra')
        with self.assertRaisesRegex(RuntimeError, 'SHA/size'):
            self.analyze()

    def test_source_and_input_lock_pin_checked(self):
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'input lock SHA'):
            a.analyze(self.run, self.lock, input_sha256='wrong')
        (self.inputs / 'frozen.py').write_text('# changed\n')
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'SHA/size'):
            self.analyze()

    def test_public_output_has_no_item_rows_and_is_create_once(self):
        result = self.analyze(); output = self.root / 'public'; private = self.root / 'local/private'
        a.write_outputs(result, output, private_output=private)
        self.assertTrue((private / 'per-item.json').is_file())
        for path in output.glob('*.csv'):
            text = path.read_text()
            for forbidden in ('case_id', 'target_new', 'target_true', 'native-tensor-digest', 'New', 'Old'):
                self.assertNotIn(forbidden, text)
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'output exists'):
            a.write_outputs(result, output)
        with self.assertRaisesRegex(a.AnalysisIntegrityError, 'overlap'):
            a.write_outputs(result, self.root / 'local/shared', private_output=self.root / 'local/shared/items')


class FileSafetyTests(unittest.TestCase):
    def test_json_duplicate_nonfinite_symlink_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); path = root / 'one.json'
            path.write_text('{"same": 1, "same": 2}')
            with self.assertRaisesRegex(a.AnalysisIntegrityError, 'duplicate JSON'):
                a.read_json(path)
            path.write_text('{"x": NaN}')
            with self.assertRaisesRegex(a.AnalysisIntegrityError, 'nonfinite JSON'):
                a.read_json(path)
            with self.assertRaisesRegex(a.AnalysisIntegrityError, 'unsafe relative'):
                a.child(root, '../other')
            link = root / 'link'; link.symlink_to(path)
            with self.assertRaisesRegex(a.AnalysisIntegrityError, 'symlink'):
                a.child(root, 'link')


if __name__ == '__main__':
    unittest.main()
