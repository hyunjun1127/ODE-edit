import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

from project.run_scripts.baseline_mechanism_first.evaluation import (
    EvaluationBindingError, bind_evaluation_sources, diagnostic_layout,
    diagnostic_margin, evaluate_records,
)
from project.run_scripts.baseline_mechanism_first.signed_response import AllPositionContraction


HISTORY=Path('/mnt/raid5/janghj/.codex/worktrees/odeeditsh1-single-layer-cumulative-risk-abc-v1/local/single-layer-cumulative-risk-abc/20260910-v1/imports/historical/blue_alphaedit_sequential_comparison')


class Tokenizer:
    pad_token_id=0; eos_token_id=2; bos_token_id=1; unk_token_id=3; padding_side='right'
    def encode(self,text,add_special_tokens=False):
        ids=[4+ord(c)%12 for c in text.strip()]
        return [1]+ids if add_special_tokens else ids
    def __call__(self,text,add_special_tokens=True):
        return dict(input_ids=self.encode(text,add_special_tokens))


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.embedding=torch.nn.Embedding(16,5)
        self.write=torch.nn.Linear(5,5,bias=False)
        self.head=torch.nn.Linear(5,16,bias=False)
        self.calls=[]
    def forward(self,input_ids,attention_mask,use_cache=False):
        self.calls.append((tuple(input_ids.shape),use_cache))
        x=self.embedding(input_ids)*attention_mask[...,None]
        x=self.write(x).cumsum(1)
        return types.SimpleNamespace(logits=self.head(torch.tanh(x)))


def record(case=1):
    return dict(case_id=case,requested_rewrite=dict(prompt='{} lives',subject='AB',
        target_new=dict(str='xy'),target_true=dict(str='z')),
        paraphrase_prompts=['one','long other'],neighborhood_prompts=['n'+str(i) for i in range(10)])


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding=bind_evaluation_sources(HISTORY)

    def test_namespaces_do_not_execute_editor_initializers(self):
        self.assertEqual(self.binding['package_initializers_executed'],0)
        name='project.run_scripts.alphaedit_strength_neutral_barrier'
        self.assertFalse(hasattr(sys.modules[name],'__file__'))
        self.assertFalse(hasattr(sys.modules[name],'apply_alphaedit_strength_neutral_barrier'))

    def test_canonical_calls_original_kernel_and_reducer(self):
        model=Toy();tok=Tokenizer()
        evaluator=sys.modules['project.run_scripts.alphaedit_strength_neutral_barrier.evaluator']
        reducer=sys.modules['project.run_scripts.baseline_mechanism_first._historical_observation.evaluation']
        parameters={k:(p.data_ptr(),p._version,p.detach().clone(),p.grad) for k,p in model.named_parameters()}
        with patch.object(evaluator,'evaluate_pairs',wraps=evaluator.evaluate_pairs) as calls, patch.object(reducer,'reduce',wraps=reducer.reduce) as reduced:
            out=evaluate_records(model,tok,[record()])
        self.assertEqual(calls.call_count,6)
        self.assertTrue(all(c.kwargs['microbatch_size']==16 for c in calls.call_args_list))
        self.assertEqual(reduced.call_count,1)
        self.assertEqual([out['metrics'][tag]['denominator'] for tag in ('RS','PS','NS')],[1,2,10])
        for tag, group in out['metrics'].items():
            for row in group['rows']:
                self.assertEqual(row['margin'],row['true_nll']-row['new_nll'])
                self.assertEqual(row['success'],row['margin']<0 if tag=='NS' else row['margin']>0)
        for key,p in model.named_parameters():
            ptr,version,value,grad=parameters[key]
            self.assertEqual((p.data_ptr(),p._version),(ptr,version));self.assertTrue(torch.equal(p,value));self.assertIs(p.grad,grad)
        self.assertEqual(tok.padding_side,'right')

    def test_all_target_tokens_and_manual_padding_layout(self):
        batch=diagnostic_layout(Tokenizer(),record(),category='R')
        self.assertEqual(batch['target_prediction_mask'].sum(-1).tolist(),[2,1])
        self.assertEqual(batch['attention_mask'][1,0].item(),0)
        self.assertEqual(batch['labels'][0,batch['target_prediction_mask'][0]].tolist(),batch['target_token_ids'][0])
        self.assertFalse(batch['canonical_microbatch16_equivalence_claim'])

    def test_diagnostic_value_matches_same_kernel_on_toy(self):
        model=Toy();tok=Tokenizer()
        result=evaluate_records(model,tok,[record()])
        for category,tag in [('R','RS'),('P','PS'),('N','NS')]:
            got=diagnostic_margin(model,tok,record(),category)
            raw=result['metrics'][tag]['rows'][0]['margin']
            self.assertAlmostEqual(float(got.detach()),-raw if category=='N' else raw,places=6)

    def test_all_position_contraction_and_fd(self):
        model=Toy();tok=Tokenizer()
        for p in model.parameters():p.requires_grad_(False)
        delta=torch.arange(25,dtype=torch.float32).reshape(5,5)/100
        original=model.write.weight.detach().clone()
        with AllPositionContraction(model.write,delta) as capture:
            value=diagnostic_margin(model,tok,record())
            measured=capture.compute(value)
        eps=.01
        with torch.no_grad():
            model.write.weight.copy_(original+eps*delta)
            plus=float(diagnostic_margin(model,tok,record()))
            model.write.weight.copy_(original-eps*delta)
            minus=float(diagnostic_margin(model,tok,record()))
            model.write.weight.copy_(original)
        fd=(plus-minus)/(2*eps)
        self.assertAlmostEqual(measured['event_derivative'],fd,delta=3e-5)
        self.assertGreater(measured['raw_position_count'],3)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_invalid_selector_and_inventory_fail_close(self):
        with self.assertRaises(EvaluationBindingError): diagnostic_layout(Tokenizer(),record(),category='NS')
        with self.assertRaises(EvaluationBindingError): diagnostic_layout(Tokenizer(),record(),category='R',prompt_index=2)
        with self.assertRaises(EvaluationBindingError): evaluate_records(Toy(),Tokenizer(),[])


if __name__=='__main__':unittest.main()
