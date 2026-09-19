"""Synthetic CPU suffix fixtures; no real-model/GPU execution evidence."""
from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from .activation_lane import (activation_statistics, artifact_members, forward_group, pack_original,
                              save_path, selected_population, validate_dependencies)
from .common import WEIGHT, digest


@dataclass
class Pair:
    case_id: int
    prompt_index: int
    prompt: str
    target: str


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 0
    sequences = {'long':[1,2,3], 'short':[6], 'new':[4,5], 'true':[7,8]}


class Kernel:
    @staticmethod
    def _encode_pair(tok, pair):
        return tok.sequences[pair.prompt], tok.sequences[pair.target]


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(3,3,bias=False,dtype=torch.float64)

    def forward(self, x):
        return x + self.mlp.down_proj(x)


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Layer() for _ in range(5)])
        self.embedding = torch.nn.Embedding(16,3,dtype=torch.float64)
        self.head = torch.nn.Linear(3,16,bias=False,dtype=torch.float64)
        generator = torch.Generator().manual_seed(441)
        with torch.no_grad():
            for name,p in self.named_parameters():
                p.copy_(torch.randn(p.shape,generator=generator,dtype=p.dtype)*(.08 if 'down_proj' in name else .3))
        self.requires_grad_(False)
        self.fail_suffix = False

    def forward(self,input_ids,attention_mask,use_cache=False):
        if use_cache:raise ValueError('NO_CACHE_ALLOWED')
        x=self.embedding(input_ids)
        for layer in self.model.layers:x=layer(x)
        if self.fail_suffix:raise ValueError('SYNTHETIC_SUFFIX_FAILURE')
        # All preceding valid tokens influence every target prediction.
        mask=attention_mask[...,None].to(x.dtype)
        x=(x*mask).cumsum(1)/mask.cumsum(1).clamp_min(1)
        return SimpleNamespace(logits=self.head(x))


def fixture():
    pairs=[Pair(1,0,'long','new'),Pair(2,0,'short','true')]
    packed=pack_original(Kernel,Tokenizer(),pairs,torch.device('cpu'))
    return TinyCausalModel(),packed


class ActivationTests(unittest.TestCase):
    def test_original_left_padding_target_shift_and_implicit_positions(self):
        _,p=fixture()
        self.assertEqual(p['input_ids'].tolist(),[[1,2,3,4],[0,0,6,7]])
        self.assertEqual(p['attention_mask'].tolist(),[[1,1,1,1],[0,0,1,1]])
        self.assertEqual(p['target_positions'],[[2,3],[2,3]])
        self.assertEqual(p['target_ids'],[[4,5],[7,8]])
        self.assertEqual(p['position_ids'],'IMPLICIT_NOT_PASSED')

    def test_mean_nll_and_strict_equal_direct_full_forward(self):
        m,p=fixture();observed,ledger=forward_group(m,p,[0,1],gradient=False)
        logits=m(input_ids=p['input_ids'],attention_mask=p['attention_mask'],use_cache=False).logits.float()
        lp=torch.log_softmax(logits,dim=-1)
        for row in (0,1):
            target=torch.tensor(p['target_ids'][row]);pos=p['target_positions'][row]
            expected=-lp[row,pos,:].gather(1,target[:,None]).mean()
            self.assertEqual(observed[row]['nll'],float(expected))
            self.assertEqual(observed[row]['all_tokens_correct'],bool((logits[row,pos].argmax(-1)==target).all()))
        self.assertEqual(ledger['forward_calls'],1)
        self.assertEqual(ledger['vjp_calls'],0)
        self.assertEqual(ledger['processed_sequences'],2)

    def test_all_valid_tokens_have_gradient_not_subject_only(self):
        m,p=fixture();out,ledger=forward_group(m,p,[0,1],gradient=True,sign=1.)
        self.assertEqual(out[0]['K'].shape,(4,3))
        self.assertEqual(out[1]['K'].shape,(2,3))
        self.assertGreater(float(out[0]['margin_gradient'][0].norm()),0)
        self.assertGreater(float(out[0]['margin_gradient'][1].norm()),0)
        self.assertEqual(out[1]['valid_positions'],[2,3])
        self.assertEqual(out[1]['padding_gradient_norm'],0)
        self.assertEqual(out[0]['other_batch_gradient_norm'],0)
        self.assertEqual(out[1]['other_batch_gradient_norm'],0)
        self.assertEqual(ledger['vjp_calls'],2)
        self.assertTrue(all(param.grad is None for param in m.parameters()))

    def test_ns_true_path_has_negative_derivative(self):
        m,p=fixture()
        pos,_=forward_group(m,p,[0],gradient=True,sign=1.)
        neg,_=forward_group(m,p,[0],gradient=True,sign=-1.)
        self.assertTrue(torch.equal(pos[0]['margin_gradient'],-neg[0]['margin_gradient']))
        self.assertEqual(pos[0]['nll'],neg[0]['nll'])

    def test_output_activation_vjp_predicts_physical_weight_derivative(self):
        m,p=fixture();got,_=forward_group(m,p,[0],gradient=True)
        w=m.get_parameter(WEIGHT);entry=w.detach().clone()
        delta=torch.tensor([[.02,.01,-.01],[.01,-.03,.02],[0,.01,.02]],dtype=torch.float64)
        stats=activation_statistics(torch.zeros_like(delta),delta,got[0]['K'],got[0]['margin_gradient'])
        losses=[];step=.02
        with torch.no_grad():
            for sign in (-1,1):
                w.copy_(entry+sign*step*delta)
                value,_=forward_group(m,p,[0],gradient=False);losses.append(value[0]['nll'])
            w.copy_(entry)
        fd=(losses[1]-losses[0])/(2*step)
        self.assertAlmostEqual(fd,stats['predicted_margin_change'],delta=1e-5)

    def test_signed_activation_cross_term_not_energy_only(self):
        e=torch.tensor([[1.,0.]],dtype=torch.float64)
        d=torch.tensor([[-2.,0.]],dtype=torch.float64)
        k=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
        g=torch.tensor([[1.],[2.]],dtype=torch.float64)
        out=activation_statistics(e,d,k,g)
        self.assertEqual(out['EK_squared_norm'],10)
        self.assertEqual(out['DK_squared_norm'],40)
        self.assertEqual(out['EK_DK_signed_cross_term'],-40)
        self.assertEqual(out['activation_energy_change'],0)
        self.assertEqual(out['energy_identity_error'],0)
        self.assertEqual(out['predicted_margin_change'],-14)

    def test_failure_removes_temporary_output_hook(self):
        m,p=fixture();m.fail_suffix=True
        module=m.get_submodule(WEIGHT.rsplit('.',1)[0])
        with self.assertRaisesRegex(ValueError,'SYNTHETIC_SUFFIX_FAILURE'):
            forward_group(m,p,[0],gradient=True)
        self.assertEqual(len(module._forward_hooks),0)

    def test_gradient_forward_does_not_mutate_any_weight(self):
        m,p=fixture();before={n:(t.clone(),t._version) for n,t in m.named_parameters()}
        forward_group(m,p,[0,1],gradient=True)
        for name,parameter in m.named_parameters():
            self.assertTrue(torch.equal(before[name][0],parameter))
            self.assertEqual(before[name][1],parameter._version)
            self.assertIsNone(parameter.grad)

    def test_unfrozen_model_is_rejected(self):
        m,p=fixture();m.get_parameter(WEIGHT).requires_grad_(True)
        with self.assertRaisesRegex(ValueError,'MODEL_PARAMETERS_NOT_FROZEN'):
            forward_group(m,p,[0],gradient=True)

    def test_passed_cpu_contract_not_actual_C01(self):
        with self.assertRaisesRegex(ValueError,'ACTUAL_C01_NOT_PASS'):
            validate_dependencies({'status':'PASS'},{'status':'PASS'},{'status':'PASS'})
        with self.assertRaisesRegex(ValueError,'ACTIVATION_BOOKKEEPING_F00_NOT_PASS'):
            validate_dependencies({'status':'PASS','C01':'PASS'},{'status':'PASS'},{'status':'NOT_RUN'})
        validate_dependencies({'status':'PASS','C01':'PASS'},{'status':'PASS'},{'status':'PASS'})

    def test_analysis_tensor_whitelist_and_create_once(self):
        with tempfile.TemporaryDirectory(prefix='activation-unit-') as folder:
            path=Path(folder)/'key.pt'
            with self.assertRaisesRegex(ValueError,'UNAPPROVED_ANALYSIS_TENSOR_FIELDS'):
                save_path(path,{'weights':torch.ones(3,3)})
            value={'K':torch.ones(2,3),'margin_gradient':torch.ones(2,3),'metadata':{'purpose':'CPU_SYNTHETIC'}}
            receipt=save_path(path,value)
            self.assertEqual(receipt['kind'],'ALL_VALID_TOKEN_KEY_AND_MARGIN_GRADIENT_NOT_CHECKPOINT')
            with self.assertRaises(FileExistsError):save_path(path,value)

    def test_final_manifest_reuses_tensor_receipt_and_binds_csv(self):
        from .common import write_csv,sha256
        with tempfile.TemporaryDirectory(prefix='activation-members-') as folder:
            root=Path(folder)
            receipt=save_path(root/'keys.pt',{'K':torch.ones(1,3),'metadata':{}})
            write_csv(root/'activation_margin.csv',[{'actual_margin_change':-.1}])
            rows=artifact_members(root,[receipt])
            self.assertEqual(len(rows),2)
            self.assertEqual(next(r for r in rows if r['path'].endswith('.pt')),receipt)
            self.assertEqual(next(r for r in rows if r['path'].endswith('.csv'))['sha256'],sha256(root/'activation_margin.csv'))
            with self.assertRaisesRegex(ValueError,'UNBOUND_OR_UNAPPROVED_TENSOR_ARTIFACT'):
                artifact_members(root,[])

    def test_fixed_population_identity_join_not_index_only(self):
        new=[Pair(i//10,i%10,f'p{i}','n') for i in range(1000)]
        true=[Pair(i//10,i%10,f'p{i}','t') for i in range(1000)]
        kernel=SimpleNamespace(counterfact_pairs=lambda records:{'locality_target_true':true})
        locality=SimpleNamespace(counterfact_locality_target_new_pairs=lambda records:new)
        ident=digest([2,3,'p23','n','t'])
        panel=[dict(identity=ident,case_id=2,prompt_index=3,population='FIRST100_NS1000',metric_tag='NS')]
        _,selected=selected_population(kernel,locality,[],panel)
        self.assertEqual(list(selected),[23])
        panel[0]['case_id']=3
        with self.assertRaisesRegex(ValueError,'PANEL_CASE_PROMPT_MISMATCH'):
            selected_population(kernel,locality,[],panel)


if __name__=='__main__':unittest.main()
