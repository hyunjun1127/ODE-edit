"""CPU mocked shared-API bridge fixtures; no model, generator, asset or network load."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from project.run_scripts.experiment_generation_eval.common import (
    EVAL_SEED, PROFILE, SCHEMA, digest, immutable_write, require,
)
from project.run_scripts.experiment_generation_eval.metrics import reduce_cases
from . import generation_bridge as bridge


def records():
    return [dict(case_id=5000 - index, generation_prompts=['fixture prompt'],
        requested_rewrite=dict(prompt='{} fixture', subject='fixture', relation_id='r',
            target_new={'id': 't', 'str': ' new'}, target_true={'str': ' old'}))
        for index in range(2000)]


class MockSharedObserver:
    """Only models the published receipt protocol; never invokes a model."""
    saved = {}
    calls = []
    mutation = None
    failure = False

    def __init__(self, model, tokenizer, assets, config, out, state_callback=None):
        self.out, self.callback = Path(out), state_callback
        self.runtime_sha = digest(dict(config=config, reference=assets.sha))

    def _receipt(self, rows, endpoint, cohort, state_sha, *, fresh):
        identity = dict(runtime=self.runtime_sha, state_sha256=state_sha,
            endpoint=endpoint, cohort=cohort, ordered_occurrences=[row['occurrence'] for row in rows],
            observation_identities=[row['identity_sha256'] for row in rows])
        receipt = dict(identity=identity, identity_sha256=digest(identity), rows=copy.deepcopy(rows),
            summary=reduce_cases(rows), RNG_restored=True, observer_no_mutation=True, raw_local_only=True)
        path = self.out / 'endpoints' / (receipt['identity_sha256'] + '.json')
        immutable_write(path, receipt)
        receipt['rows_path'] = str(path.resolve())
        receipt['work'] = dict(new_case_observations=len(rows) if fresh else 0,
            cached_case_observations=0 if fresh else len(rows), generation_forwards=len(rows) if fresh else 0,
            full_prefix_token_work=10 * len(rows) if fresh else 0, seconds=1.25 if fresh else 0.)
        self.saved[str(path.resolve())] = copy.deepcopy(receipt)
        return receipt

    def observe(self, selected, endpoint, cohort=None, state_identity=None):
        before = self.callback()
        self.calls.append((endpoint, copy.deepcopy(selected), copy.deepcopy(state_identity)))
        if self.failure:
            raise RuntimeError('FIXTURE_SHARED_OBSERVE_FAILED')
        if type(self).mutation is not None:
            type(self).mutation()
        require(self.callback() == before, 'GENERATION_NATIVE_STATE_MUTATION')
        rows = []
        for record in selected:
            identity = digest([state_identity, record['occurrence_index'], record['case_id']])
            rows.append(dict(occurrence=record['occurrence_index'], case_id=record['case_id'],
                identity_sha256=identity, payload_sha256=digest(identity),
                observation_path='/fixture/raw/' + identity + '.json', metrics=dict(
                    ngram_entropy=0., reference_score=None, fluency_valid=True,
                    consistency_valid=False, reasons=['missing_reference'],
                    generation_prompt_count=1, generated_token_count=0,
                    length_cap_no_continuation_count=0)))
        return self._receipt(rows, endpoint, cohort, digest(state_identity), fresh=True)

    def read_observed(self, path):
        receipt = copy.deepcopy(self.saved[str(Path(path).resolve())])
        require(receipt['identity']['runtime'] == self.runtime_sha, 'GENERATION_RUNTIME_IDENTITY')
        receipt['work'] = dict(new_case_observations=0, cached_case_observations=len(receipt['rows']),
            generation_forwards=0, full_prefix_token_work=0, seconds=0.)
        return receipt

    def subset(self, observed, selected, endpoint, cohort=None):
        require(observed['identity']['runtime'] == self.runtime_sha, 'GENERATION_RUNTIME_IDENTITY')
        lookup = {row['occurrence']: row for row in observed['rows']}
        rows = [lookup[record['occurrence_index']] for record in selected]
        return self._receipt(rows, endpoint, cohort, observed['identity']['state_sha256'], fresh=False)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='generation-bridge-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = records()
        self.config = dict(cold_W={str(layer): 'cold-' + str(layer) for layer in range(3, 9)},
            generation=dict(schema=SCHEMA, profile=PROFILE, eval_seed=EVAL_SEED,
                W0_owner='BASE_MEMIT', W0_cache=str(self.root / 'shared-W0'),
                W0_state_identity={'full_model_cold_identity': 'stable-full-cold-hash'},
                model_identity='stable-model-runtime-tokenizer-hash', source_sha='a' * 64,
                generation_assets={'path': '/fixture/manifest.json'}, asset_paths={},
                reference_assets_sha256='b' * 64))
        MockSharedObserver.saved, MockSharedObserver.calls = {}, []
        MockSharedObserver.mutation, MockSharedObserver.failure = None, False
        self.assets = types.SimpleNamespace(sha='b' * 64)
        self.loader = patch.object(bridge, 'load_assets', return_value=self.assets).start()
        self.addCleanup(patch.stopall)
        patch.object(bridge, 'SharedObserver', MockSharedObserver).start()
        patch.object(bridge, 'tensor_sha', side_effect=lambda weight: weight).start()

    def observer(self, arm='BASE_MEMIT', config=None, selected=None):
        sites = (3, 8) if arm == 'ALPHAEDIT_BLUE' else tuple(range(3, 9))
        history = {} if arm == 'BASE_MEMIT' else {layer: torch.zeros((2, 2)) for layer in sites}
        engine = types.SimpleNamespace(module=types.SimpleNamespace(
            CONTEXT_TEMPLATES_CACHE=None if arm == 'BASE_MEMIT' else [['{}'], ['fixture. {}'] * 5],
            COV_CACHE={}), history=lambda: history, counts={'native_z': 0}, next_batch=1)
        view = types.SimpleNamespace(model=object(), sites=sites,
            weights={layer: 'cold-' + str(layer) for layer in sites})
        result = bridge.GenerationObserver(config or self.config, {}, view, engine, None,
            self.original if selected is None else selected, self.root / arm, arm)
        return result, view, engine

    def test_primary_cold_generation_atomic_ready_and_other_native_shapes_reuse(self):
        before = copy.deepcopy(self.original)
        primary, _, _ = self.observer()
        full = primary.load_W0()
        self.assertEqual(len(MockSharedObserver.calls), 1)
        self.assertEqual(full['summary']['planned_count'], 2000)
        self.assertEqual(full['work']['new_case_observations'], 2000)
        self.assertEqual(full['identity']['ordered_occurrences'], list(range(1, 2001)))
        ready = json.loads((self.root / 'shared-W0' / 'READY.json').read_text())
        self.assertEqual(ready['producer_arm'], 'BASE_MEMIT')
        self.assertEqual(ready['producer_work'], full['work'])
        for arm in ('CAKE', 'ALPHAEDIT_BLUE'):
            peer, _, _ = self.observer(arm)
            reused = peer.load_W0()
            self.assertEqual(reused['identity'], full['identity'])
            self.assertEqual(reused['shared_state_identity'], self.config['generation']['W0_state_identity'])
            self.assertEqual(reused['work']['generation_forwards'], 0)
            self.assertEqual(reused['work']['cached_case_observations'], 2000)
        self.assertEqual(len(MockSharedObserver.calls), 1)
        self.assertEqual(self.original, before)
        self.assertFalse(list((self.root / 'shared-W0').glob('*.tmp-*')))

    def test_nonowner_missing_ready_and_failed_primary_never_publish(self):
        peer, _, _ = self.observer('CAKE')
        with self.assertRaisesRegex(RuntimeError, 'W0_NOT_READY_NO_POLL'):
            peer.load_W0()
        self.assertFalse(MockSharedObserver.calls)
        primary, _, _ = self.observer()
        MockSharedObserver.failure = True
        with self.assertRaisesRegex(RuntimeError, 'FIXTURE_SHARED_OBSERVE_FAILED'):
            primary.load_W0()
        self.assertFalse((self.root / 'shared-W0' / 'READY.json').exists())

    def test_b1_pre_is_cpu_subset_and_milestone_current_keeps_exact_receipt(self):
        primary, view, _ = self.observer()
        primary.load_W0()
        cold = {'W': {str(layer): weight for layer, weight in view.weights.items()}, 'H': {}}
        pre = primary.endpoint(self.original[:100], out=self.root / 'pre', endpoint='B1_PRE',
            model_state=cold, cohort_label='CURRENT')
        self.assertEqual(len(MockSharedObserver.calls), 1)
        self.assertEqual(pre['work']['generation_forwards'], 0)
        self.assertEqual(pre['shared_state_identity'], self.config['generation']['W0_state_identity'])
        actual = {'W': {'3': 'actual-edited-hash'}, 'H': {'3': 'actual-history-hash'}}
        full = primary.endpoint(self.original[:500], out=self.root / 'post', endpoint='W5',
            model_state=actual, cohort_label='ALL_SEEN')
        current = primary.subset_receipt(full, self.original[400:500], endpoint='W5_CURRENT',
            out=self.root / 'current')
        self.assertEqual(current['summary']['planned_count'], 100)
        self.assertEqual(current['identity']['ordered_occurrences'], list(range(401, 501)))
        self.assertEqual(current['work']['generation_forwards'], 0)
        self.assertEqual(current['identity']['state_sha256'], digest(actual))
        self.assertEqual(primary.subset(full['cases'], self.original[400:500]), current['summary'])
        self.assertEqual(len(MockSharedObserver.calls), 2)
        persisted = json.loads(Path(current['receipt_path']).read_text())
        self.assertEqual(persisted['full_receipt'], current['full_receipt'])
        payload = primary.payload('current/post', current['summary'])
        self.assertEqual(payload['current/post/fluency/ngram_entropy'], 0.)
        self.assertNotIn('current/post/consistency/reference_score', payload)
        self.assertEqual(payload['current/post/generation/missing_missing_reference_count'], 100)
        self.assertEqual(full['shared_state_identity'], actual)

    def test_ready_identity_source_cohort_and_asset_mismatch_are_rejected(self):
        primary, _, _ = self.observer()
        primary.load_W0()
        for field, value in (('source_sha', 'c' * 64), ('model_identity', 'other-model'),
                             ('W0_state_identity', {'full_model_cold_identity': 'wrong'})):
            config = copy.deepcopy(self.config)
            config['generation'][field] = value
            peer, _, _ = self.observer('CAKE', config)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, 'W0_READY_IDENTITY'):
                peer.load_W0()
        self.assets.sha = 'wrong-assets'
        with self.assertRaisesRegex(RuntimeError, 'REFERENCE_IDENTITY'):
            self.observer('RECT')

    def test_exact_occurrence_order_record_and_cold_weight_guards(self):
        primary, view, _ = self.observer()
        self.assertEqual(primary.records[0]['case_id'], 5000)
        self.assertEqual(primary.records[-1]['case_id'], 3001)
        self.assertEqual(primary.records[-1]['occurrence_index'], 2000)
        changed = copy.deepcopy(self.original[:100])
        changed[0]['generation_prompts'] = ['changed fixture']
        with self.assertRaisesRegex(RuntimeError, 'RECORD_IDENTITY'):
            primary.endpoint(changed, out=self.root / 'bad', endpoint='W1', model_state={}, cohort_label='CURRENT')
        with self.assertRaisesRegex(RuntimeError, 'DUPLICATE_OCCURRENCE'):
            primary._selected([self.original[0], self.original[0]])
        wrong = copy.deepcopy(self.original)
        wrong[0]['occurrence_index'] = 0
        with self.assertRaisesRegex(RuntimeError, 'OCCURRENCE_ORDER'):
            self.observer('RECT', selected=wrong)
        view.weights[3] = 'not-cold'
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_COLD_W0'):
            primary.load_W0()

    def test_native_h_context_counter_signatures_and_mutation_failure(self):
        primary, _, engine = self.observer()
        before = primary._native_signature()
        engine.module.CONTEXT_TEMPLATES_CACHE = [['{}']]
        self.assertNotEqual(primary._native_signature(), before)
        engine.module.CONTEXT_TEMPLATES_CACHE = None
        engine.counts['native_z'] = 1
        self.assertNotEqual(primary._native_signature(), before)
        engine.counts['native_z'] = 0
        MockSharedObserver.mutation = lambda: engine.counts.update(native_z=1)
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_STATE_MUTATION'):
            primary.load_W0()
        self.assertFalse((self.root / 'shared-W0' / 'READY.json').exists())
        peer, _, peer_engine = self.observer('ALPHAEDIT_BLUE')
        signature = peer._native_signature()
        peer_engine.history()[3].add_(1.)
        self.assertNotEqual(peer._native_signature(), signature)


if __name__ == '__main__':
    unittest.main()
