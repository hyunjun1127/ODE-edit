"""Bounded CPU binding/lifecycle checks; no real model loader or CUDA calls."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from project.run_scripts.native_response_ode_v31.runtime import ObservedFamily
from project.run_scripts.ordered_response_barrier_ode.tests.test_artifacts import (
    _evaluation_with_canonical_ns,
)
from . import gpu_runtime as runtime
from .contracts import BindingBoundary
from .publication import member, write_once
from .sampling import canonical_hash, select_cohorts
from .test_shared import CPUFamily
from .test_sweep import _inventories, _records


class GPURuntimeBindingTests(unittest.TestCase):
    def test_launch_rejection_precedes_output_and_device_or_loader(self):
        loader = Mock()
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(runtime, 'validate_launch', side_effect=BindingBoundary('PREMODEL_LOCK')), \
             patch('torch.cuda.device_count') as cuda, \
             patch.dict('sys.modules', {'transformers': SimpleNamespace(AutoModelForCausalLM=loader)}):
            with self.assertRaisesRegex(BindingBoundary, 'PREMODEL_LOCK'):
                runtime.run_cell(folder, folder, 0)
            self.assertEqual(list(Path(folder).iterdir()), [])
            loader.from_pretrained.assert_not_called()
            cuda.assert_not_called()

    def test_sample_rejection_persists_premodel_failure_without_load(self):
        loader = Mock()
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(runtime, 'validate_launch', return_value={'model_alias': 'llama3-8b-inst'}), \
             patch.object(runtime.signal, 'signal'), \
             patch('torch.cuda.device_count') as cuda, \
             patch.dict('sys.modules', {'transformers': SimpleNamespace(AutoModelForCausalLM=loader)}):
            write_once(Path(folder)/'sample.lock.json', {}, root=folder)
            with self.assertRaisesRegex(BindingBoundary, 'SAMPLE_MANIFEST_IDENTITY'):
                runtime.run_cell(folder, folder, 0)
            failure = json.loads((Path(folder)/'cell-0/failure-boundary.json').read_text())
            self.assertEqual(failure['stage'], 'PREMODEL_ASSETS')
            self.assertEqual(failure['completed'], [])
            self.assertIsNone(failure['entry_restore'])
            loader.from_pretrained.assert_not_called()
            cuda.assert_not_called()

    def test_dev_exact_bytes_order_and_actual_prompt_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'cpu-dataset.json'
            records = list(reversed(_records()))
            write_once(path, records, root=folder)
            sample = select_cohorts(records, _inventories(), dataset_identity=member(path))
            loaded, cohort = runtime.bound_dev_records(sample, path)
            self.assertEqual([r['case_id'] for r in loaded],
                             [r['case_id'] for r in cohort['records']])
            expected = cohort['evaluation']['denominators']
            self.assertEqual(expected['RS'], len(loaded))
            self.assertEqual(expected['PS'], sum(len(r['paraphrase_prompts']) for r in loaded))
            self.assertEqual(expected['NS'], sum(len(r['neighborhood_prompts']) for r in loaded))
            self.assertGreater(expected['NS'], 1000)  # No forced locality denominator.
            wrong = copy.deepcopy(sample)
            wrong['cohorts']['S_DEV']['evaluation']['denominators']['NS'] += 1
            wrong['manifest_identity'] = canonical_hash({k:v for k,v in wrong.items() if k != 'manifest_identity'})
            with self.assertRaisesRegex(BindingBoundary, 'S_DEV_EVALUATOR_INPUT_DRIFT'):
                runtime.bound_dev_records(wrong, path)
            wrong = copy.deepcopy(sample)
            wrong['dataset']['sha256'] = '0'*64
            wrong['manifest_identity'] = canonical_hash({k:v for k,v in wrong.items() if k != 'manifest_identity'})
            with self.assertRaisesRegex(BindingBoundary, 'S_DATASET_BYTES_DRIFT'):
                runtime.bound_dev_records(wrong, path)

    def test_logits_observer_only_first_request_all_positions_full_vocabulary(self):
        class Batch(dict):
            def to(self, device):
                self.device = device
                return self
        batch = Batch(input_ids=torch.tensor([[1, 2, 3]]))
        tokenizer = Mock(return_value=batch)
        tokenizer.padding_side = 'right'
        logits = torch.arange(21, dtype=torch.float32).reshape(1, 3, 7)
        model = Mock(return_value=SimpleNamespace(logits=logits))
        family = SimpleNamespace(requests=[{'prompt':'{} is', 'subject':'first'},
                                           {'prompt':'{} never observed', 'subject':'second'}],
                                 tokenizer=tokenizer, model=model, device=torch.device('cpu'))
        actual = runtime.first_request_logits(family)
        tokenizer.assert_called_once_with(['first is'], padding=True, return_tensors='pt')
        model.assert_called_once_with(input_ids=batch['input_ids'], use_cache=False)
        self.assertEqual(actual.shape, (1, 3, 7))
        self.assertTrue(torch.equal(actual, logits))
        self.assertEqual(tokenizer.padding_side, 'right')

    def test_campaign_captures_actual_weight_cache_flag_and_prefix_activation(self):
        base = CPUFamily()
        campaign = runtime._family_type().__new__(runtime._family_type())
        campaign.__dict__.update(base.__dict__)
        with torch.no_grad():
            next(iter(campaign.parameters.values())).add_(.125)
            campaign.module.cache_c.add_(.25)
        campaign.module.cache_c_new = False
        campaign._capture_endpoint_for_persistence()
        name = next(iter(campaign.parameters))
        self.assertTrue(torch.equal(campaign._captured_endpoint_weights[name], campaign.parameters[name]))
        self.assertNotEqual(campaign._captured_endpoint_weights[name].data_ptr(), campaign.parameters[name].data_ptr())
        self.assertTrue(torch.equal(campaign._captured_endpoint_method_state, campaign.module.cache_c))
        self.assertFalse(campaign._captured_endpoint_cache_c_new)
        def evaluate(family):
            family.last_terminal = family.parameters[name].detach().clone()
            return {'scope':'CPU_FIXTURE_ACTUAL_ENDPOINT'}
        with patch.object(ObservedFamily, 'evaluate_endpoint', evaluate):
            result = campaign.evaluate_endpoint()
        actual = campaign.last_actual_activation.clone()
        campaign.last_terminal.zero_()  # Parent prefix finalize restores its own ledger.
        self.assertTrue(torch.equal(campaign.last_actual_activation, actual))
        self.assertGreater(float(actual.abs().sum()), 0.)
        self.assertEqual(result['scope'], 'CPU_FIXTURE_ACTUAL_ENDPOINT')

    def test_endpoint_denominators_use_bound_schema_and_strict_ns(self):
        ids = [12, 17]
        cohort = {'records':[{'case_id':i, 'request_sha256':canonical_hash({'case_id':i})} for i in ids],
                  'evaluation':{'denominators':{'RS':2, 'PS':4, 'NS':20}}}
        evaluation = _evaluation_with_canonical_ns(ids, entry=False)
        evaluation['locality_target_new'][0]['nll'] = evaluation['locality_target_true'][0]['nll']
        facts = runtime._validate_endpoint_counts(evaluation, cohort)
        self.assertEqual((facts['RS_d'], facts['PS_d'], facts['NS_d']), (2, 4, 20))
        self.assertEqual(facts['NS_n'], 19)  # Tie is not a locality success.
        wrong = copy.deepcopy(cohort)
        wrong['evaluation']['denominators']['NS'] = 21
        with self.assertRaisesRegex(BindingBoundary, 'ENDPOINT_DENOMINATOR_NS'):
            runtime._validate_endpoint_counts(evaluation, wrong)


if __name__ == '__main__':
    unittest.main()
