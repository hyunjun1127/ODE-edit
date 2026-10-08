"""Tiny model/CPU parity fixtures, explicitly not GPT-J/GPU certification."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from official.evaluation import factual
from official.experiments import checkpoint
from official.runners.server2 import parity


class Tokenizer:
    pad_token_id = eos_token_id = 0
    model_max_length = 1024
    def __call__(self,text,add_special_tokens=True,truncation=False):
        return {'input_ids':([1] if add_special_tokens else [])+[2+ord(char)%61 for char in text]}
    def encode(self,text):
        return self(text)['input_ids']


class Block(torch.nn.Module):
    def __init__(self,tensor_output=False):
        super().__init__()
        self.tensor_output = tensor_output
        self.mlp = torch.nn.Module()
        self.mlp.fc_in = torch.nn.Linear(4,6)
        self.mlp.fc_out = torch.nn.Linear(6,4)
    def forward(self,hidden_states):
        hidden = hidden_states+self.mlp.fc_out(torch.relu(self.mlp.fc_in(hidden_states))) / 100
        return hidden if self.tensor_output else (hidden,None)


class Config:
    max_position_embeddings = 1024
    def to_dict(self):
        return {'fixture_only':True,'model_type':'gptj-fixture','use_cache':False}


class Model(torch.nn.Module):
    def __init__(self,tensor_blocks=False):
        super().__init__()
        self.config = Config()
        self.transformer = torch.nn.Module()
        self.transformer.wte = torch.nn.Embedding(64,4)
        self.transformer.h = torch.nn.ModuleList([Block(tensor_blocks) for _ in range(28)])
        self.transformer.ln_f = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4,64)
        self.forward_calls = 0
    def get_input_embeddings(self):
        return self.transformer.wte
    def forward(self,input_ids,attention_mask,use_cache=False):
        self.forward_calls += 1
        hidden = self.transformer.wte(input_ids)
        for block in self.transformer.h:
            value = block(hidden_states=hidden)
            hidden = value if isinstance(value,torch.Tensor) else value[0]
        return SimpleNamespace(logits=self.lm_head(self.transformer.ln_f(hidden)))


def records(dataset):
    return [dict(case_id=1000-index,occurrence_index=index+1,
        requested_rewrite={'prompt':'{} likes','subject':'x','target_new':{'str':'y'},'target_true':{'str':'z'}},
        paraphrase_prompts=['x enjoys'],
        neighborhood_prompts=['x hates'] if dataset == 'cf' else [{'prompt':'x hates','target':'z'}])
        for index in range(100)]


def manifest(dataset):
    return dict(code_commit='a'*40,official_tree_sha256='b'*40,model_revision='r'*40,
        tokenizer_sha256='t'*64,streams={dataset:{'lock':{'stream_sha256':'s'*64}}})


class ParityFixtureTests(unittest.TestCase):
    def setUp(self):
        self.model,self.tok = Model(),Tokenizer()
        self.engine = SimpleNamespace(method='MEMIT',batch=1,hparams=SimpleNamespace(layers=[3,4,5,6,7,8]),
            state_identity=lambda:dict(batch=1,H='fixture-zero',contexts='unchanged'))
        self.shapes = [patch.object(parity,'HIDDEN',4),patch.object(parity,'KEY_WIDTH',6)]
        for item in self.shapes:
            item.start(); self.addCleanup(item.stop)

    def call(self,dataset='cf',frozen=None,alter=None):
        cohort,value = records(dataset),manifest(dataset)
        frozen = frozen or parity.plan(value,dataset,'MEMIT',cohort)
        def evaluation():
            observed = factual.evaluate(self.model,self.tok,cohort[:1],dataset,batch_size=16)
            return alter(observed) if alter else observed
        return parity.observe(self.model,self.tok,self.engine,value,cohort,frozen,evaluation)

    def test_frozen_plan_no_oracle_no_raw_data_and_fixed_tolerances(self):
        frozen = parity.plan(manifest('cf'),'cf','MEMIT',records('cf'))
        self.assertEqual(frozen['tolerance']['candidate_nll'],{'atol':1e-4,'rtol':0})
        self.assertEqual(frozen['independent_public_native_evaluator'],'NOT_AVAILABLE_IN_OFFICIAL_DISTRIBUTION')
        self.assertFalse(frozen['actual_GPU'])
        self.assertNotIn('case_id',repr(frozen))
        self.assertNotIn('input_token_ids',repr(frozen))
        wrong = deepcopy(frozen); wrong['tolerance']['candidate_nll']['atol'] = 1
        with self.assertRaisesRegex(ValueError,'PLAN_CHANGED'):
            self.call(frozen=wrong)
        self.assertEqual(self.model.forward_calls,0)

    def test_CF_existing_forward_same_call_bias_readout_token_NLL_controls(self):
        before,rng = parity._state(self.model,self.engine),checkpoint.rng_snapshot()
        observed,receipt = self.call()
        self.assertEqual(self.model.forward_calls,observed['work']['forward_calls'])
        self.assertEqual(receipt['extra_LM_forward_calls'],0)
        self.assertEqual(receipt['status'],'PASS_CPU_FORMULA_FIXTURE_NOT_ACTUAL_GPU')
        self.assertFalse(receipt['actual_GPU'])
        self.assertFalse(receipt['independent_oracle_PASS'])
        self.assertTrue(receipt['candidate_token_prefix_exact'])
        self.assertTrue(receipt['token_predictions_exact'])
        self.assertTrue(receipt['observer_no_mutation'])
        self.assertTrue(receipt['RNG_restored'])
        self.assertEqual(parity._state(self.model,self.engine),before)
        self.assertTrue(parity._equal(checkpoint.rng_snapshot(),rng))
        self.assertNotIn('input_token_ids',repr(receipt))
        self.assertEqual(receipt['block_output_schemas']['transformer.h.27']['container'],'tuple')

    def test_tensor_block_layout_supported_without_shared_source_patch(self):
        self.model = Model(tensor_blocks=True)
        _,receipt = self.call()
        self.assertEqual(receipt['block_output_schemas']['transformer.h.8']['container'],'Tensor')

    def test_ZSRE_exact_teacher_prefix_same_loc_ans_distinction(self):
        observed,receipt = self.call('zsre')
        self.assertEqual(receipt['dataset'],'zsre')
        self.assertTrue(receipt['candidate_nll_close'])
        self.assertIsNone(observed['cases'][0]['neighborhood_W0_agreement'])
        self.assertIn('Specificity_loc_ans',observed['summary'])

    def test_numerical_NLL_corruption_blocks_without_tolerance_relaxation(self):
        def corrupt(observed):
            observed['cases'][0]['rewrite_observations'][0]['target_new']['mean_nll'] += .01
            return observed
        before = parity._state(self.model,self.engine)
        with self.assertRaisesRegex(ValueError,'FP32_CANDIDATE_NLL'):
            self.call(alter=corrupt)
        self.assertEqual(parity._state(self.model,self.engine),before)

    def test_exact_token_prediction_not_substituted_by_close_NLL(self):
        def corrupt(observed):
            observed['cases'][0]['rewrite_observations'][0]['target_new']['predicted_token_ids'][0] ^= 1
            return observed
        with self.assertRaisesRegex(ValueError,'STORED_NATIVE_TOKEN_PREDICTIONS'):
            self.call(alter=corrupt)

    def test_first_B1_only_not_new_edit_pilot_or_later_state(self):
        self.engine.batch = 0
        with self.assertRaisesRegex(ValueError,'FIRST_REAL_NATIVE_BATCH'):
            self.call()
        self.assertEqual(self.model.forward_calls,0)

    def test_model_observer_error_preserved_and_temporary_hooks_removed(self):
        before = parity._state(self.model,self.engine)
        cohort,value = records('cf'),manifest('cf')
        frozen = parity.plan(value,'cf','MEMIT',cohort)
        def error():
            raise RuntimeError('ORIGINAL_SCORER_FAILURE')
        with self.assertRaisesRegex(RuntimeError,'ORIGINAL_SCORER_FAILURE'):
            parity.observe(self.model,self.tok,self.engine,value,cohort,frozen,error)
        self.assertEqual(parity._state(self.model,self.engine),before)


if __name__ == '__main__':
    unittest.main()
