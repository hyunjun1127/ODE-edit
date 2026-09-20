"""Bounded CPU fixtures for factor algebra, cold identity and neural wiring.

Tiny bank capacities in one fixture exercise overflow cheaply; production
constants remain 512/612 and no neural experiment tolerance is selected here.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from project.run_scripts.en_adapt_gss_history import factors as f
from project.run_scripts.en_adaptive_nullspace.json_io import save


class FactorAlgebraTest(unittest.TestCase):
    def test_sketch_matches_frozen_reference_including_prefix(self):
        rng = np.random.default_rng(77)
        basis = np.linalg.qr(rng.normal(size=(9,7)))[0]
        maps = f.make_maps(5,9,basis)
        keys = rng.normal(size=(9,6))
        activation = rng.normal(size=(5,6))
        projected = torch.from_numpy(np.stack([p.T@keys for _,_,p in maps]))
        actual = f.sketch_factors(torch.from_numpy(activation), projected,
                                  [torch.from_numpy(o) for o,_,_ in maps]).numpy()
        expected = f.REFERENCE.factor_sketch(activation,keys,basis@basis.T,[(o,v) for o,v,_ in maps])
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
        # Removing prefix factors must change this deliberately dense fixture.
        target_only = f.REFERENCE.factor_sketch(activation[:,-1:],keys[:,-1:],basis@basis.T,[(o,v) for o,v,_ in maps])
        self.assertGreater(np.linalg.norm(expected-target_only),1.)
        for pair, other in zip(maps,f.make_maps(5,9,basis)):
            for actual_map, expected_map in zip(pair,other):
                np.testing.assert_array_equal(actual_map,expected_map)

    def test_two_vjps_are_distinct_per_fact_means_and_aggregate(self):
        torch.manual_seed(21)
        keys = torch.randn(5,4)
        weight = torch.randn(3,5)
        activation = (weight@keys).T.detach().requires_grad_(True)
        # A dense causal linear suffix makes earlier prefix activations matter.
        causal = torch.tril(torch.ones(4,4))
        head = torch.randn(3,7)
        logits = (causal@activation)[2:]@head
        labels = torch.tensor([1,5])
        teacher = torch.randn(2,7).log_softmax(-1)
        _, nll, kl = f.activation_vjps(logits,activation,labels,teacher,selection=True)
        self.assertGreater(float(nll[:2].norm()),0)
        self.assertFalse(torch.equal(nll,kl))
        leaf = weight.detach().requires_grad_(True)
        direct_logits = (causal@(leaf@keys).T)[2:]@head
        direct_loss = f.signed_forward_kl(direct_logits,teacher)
        direct, = torch.autograd.grad(direct_loss,leaf)
        actual = f.aggregate_factors([kl.T],[keys],[1.],weight.shape,torch.device('cpu'))
        torch.testing.assert_close(actual,direct.double(),rtol=4e-7,atol=4e-7)
        with self.assertRaises(ValueError):
            f.aggregate_factors([kl.T],[keys[:,:1]],[1.],weight.shape,'cpu')

    def test_cold_publication_create_once_and_strict_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'teacher.npz'
            a = np.arange(12,dtype=np.float32).reshape(3,4)
            info = f._atomic_arrays(path,teacher_logp=a)
            before = path.read_bytes()
            self.assertEqual(hashlib.sha256(before).hexdigest(),info['sha256'])
            with self.assertRaises(FileExistsError):
                f._atomic_arrays(path,teacher_logp=a+1)
            self.assertEqual(path.read_bytes(),before)
            for bad in (np.asarray([np.nan]),np.asarray([np.inf])):
                with self.assertRaises(FloatingPointError):
                    f._atomic_arrays(Path(temporary)/'bad.npz',teacher_logp=bad)
            save(Path(temporary)/'receipt.json',dict(flag=np.bool_(True),count=np.int64(2)))
            self.assertEqual(json.loads((Path(temporary)/'receipt.json').read_text()),dict(flag=True,count=2))
            with self.assertRaises(TypeError):
                save(Path(temporary)/'forbidden.json',dict(weight=torch.ones(1)))
            self.assertFalse((Path(temporary)/'forbidden.json').exists())


def tiny_model():
    torch.manual_seed(91)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = LlamaConfig(vocab_size=31,hidden_size=12,intermediate_size=16,
                         num_hidden_layers=6,num_attention_heads=3,
                         num_key_value_heads=3,max_position_embeddings=64,
                         attention_dropout=0.,pretraining_tp=1)
    config._attn_implementation = 'eager'
    model = LlamaForCausalLM(config).eval().float()
    model.requires_grad_(False)
    return model


def canonical_contract():
    return types.SimpleNamespace(
        prompt_token_ids=lambda tokenizer,text:[1,3,4,5+int(text[-1])],
        target_token_ids=lambda tokenizer,text:[14,15+int(text[-1])])


def versions(n):
    return [dict(version_id=f'version-{i}',created_batch=1+i%2,
                 record=dict(case_id=i,requested_rewrite=dict(subject=str(i),relation_id='P1',
                             prompt='entity {}',target_new={'str':f'target{i}'}))) for i in range(n)]


class TinyNeuralTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_full_llama_prefix_factor_matches_physical_weight_gradient(self):
        model = tiny_model()
        weight = dict(model.named_parameters())['model.layers.4.mlp.down_proj.weight']
        with tempfile.TemporaryDirectory() as temporary, patch.object(f,'_token_contracts',canonical_contract):
            replay = f.HistoryReplay(model,object(),temporary,np.eye(16,dtype=np.float64))
            capture = replay.capture(versions(1),weight.detach().cpu().clone())
            self.assertEqual(capture['receipt']['new_versions'],1)
            with torch.no_grad():
                weight.add_(torch.randn_like(weight)*.025)
            replay.rebind(weight.detach())
            item = replay._load('version-0')
            cut, logits = replay._activation_logits(item,weight.detach())
            _, an, ak = f.activation_vjps(logits,cut,torch.tensor(item['metadata']['labels']),item['teacher'],selection=True)
            self.assertGreater(float(an[0,:3].norm()),0.)
            aggregate = f.aggregate_factors([ak[0].T],[item['cache'].keys[0].T],[1.],weight.shape,'cpu')
            weight.requires_grad_(True)
            packed = item['cache'].packed
            actual_logits = model(**packed,use_cache=False).logits[0,item['metadata']['positions']]
            direct, = torch.autograd.grad(f.signed_forward_kl(actual_logits,item['teacher']),weight)
            weight.requires_grad_(False)
            torch.testing.assert_close(aggregate,direct.double(),atol=5e-9,rtol=8e-5)

    def test_store_no_refresh_below_capacity_then_two_vjp_overflow(self):
        model = tiny_model()
        weight = dict(model.named_parameters())['model.layers.4.mlp.down_proj.weight']
        rows = versions(3)
        with tempfile.TemporaryDirectory() as temporary, patch.object(f,'_token_contracts',canonical_contract):
            replay = f.HistoryReplay(model,object(),temporary,np.eye(16,dtype=np.float64))
            empty = replay.prepare_pool(weight.detach(),[],select_gss=True,current_batch=1,recency=True)
            self.assertEqual(empty['receipt']['dense_gradient_D2H'],0)
            self.assertEqual(empty['receipt']['KL_VJPs'],0)
            self.assertEqual(float(empty['gradient'].norm()),0.)
            captured = replay.capture(rows,weight.detach().cpu().clone())
            before = {k:v['payload']['sha256'] for k,v in captured['bindings'].items()}
            with torch.no_grad():
                weight.add_(torch.randn_like(weight)*.015)
            again = replay.capture(rows,weight.detach().cpu().clone())
            self.assertEqual(again['receipt']['new_versions'],0)
            self.assertEqual(again['receipt']['reused_versions'],3)
            self.assertEqual(before,{k:v['payload']['sha256'] for k,v in again['bindings'].items()})
            normal = replay.prepare_pool(weight.detach(),rows[:2],select_gss=True,current_batch=4,recency=True)
            self.assertEqual(normal['receipt']['NLL_VJPs'],0)
            self.assertFalse(normal['receipt']['selection_exercised'])
            self.assertIsNone(replay.maps)
            with patch.object(f,'CAPACITY',2), patch.object(f,'POOL_MAX',3):
                overflow = replay.prepare_pool(weight.detach(),rows,select_gss=True,current_batch=4,recency=True)
                self.assertEqual(overflow['receipt']['NLL_VJPs'],3)
                self.assertEqual(overflow['receipt']['KL_VJPs'],3)
                self.assertEqual(len(overflow['selected_ids']),2)
                self.assertEqual(overflow['receipt']['microbatch_size'],1)
                self.assertTrue(overflow['receipt']['two_VJPs_share_graph'])
                value, receipt = replay.evaluate(weight.detach(),overflow['selected_ids'],overflow['weights'])
                self.assertAlmostEqual(value,overflow['L_H'],places=12)
                self.assertEqual(receipt['candidate_fact_forwards'],2)
            self.assertTrue((Path(temporary)/'sketch-diagnostic-once.json').exists())
            self.assertEqual(json.loads((Path(temporary)/'sketch-diagnostic-once.json').read_text())['status'],'SKETCH_SELECTION_UNRESOLVED')
            bad = dict(rows[0],created_batch=20)
            with self.assertRaisesRegex(ValueError,'REFRESH_ATTEMPT'):
                replay.capture([bad],weight.detach())
            # Reopening validates exact cold payload; corruption is not fallback.
            metadata = captured['bindings']['version-0']
            payload = Path(metadata['payload']['path'])
            raw = payload.read_bytes()
            payload.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            replay.hot.clear()
            with self.assertRaisesRegex(ValueError,'SHA_MISMATCH'):
                replay._load('version-0')

    def test_hash4_direct_aggregate_diagnostic_serializes(self):
        model = tiny_model()
        weight = dict(model.named_parameters())['model.layers.4.mlp.down_proj.weight']
        rows = versions(4)
        with tempfile.TemporaryDirectory() as temporary, patch.object(f,'_token_contracts',canonical_contract):
            replay = f.HistoryReplay(model,object(),temporary,np.eye(16,dtype=np.float64))
            replay.capture(rows,weight.detach())
            with torch.no_grad():
                weight.add_(torch.randn_like(weight)*.02)
            result = replay.prepare_pool(weight.detach(),rows,select_gss=False,current_batch=4,recency=False)
            diagnostic = result['receipt']['diagnostics']['factor']
            self.assertEqual(len(diagnostic['version_ids']),4)
            self.assertLess(diagnostic['relative_frobenius_error'],1e-5)
            self.assertEqual(diagnostic['precision'],'NOT_ESTABLISHED')
            save(Path(temporary)/'complete-pool-receipt.json',result['receipt'])
            self.assertEqual(replay.counts['diagnostic_backwards'],1)


if __name__ == '__main__':
    unittest.main()
