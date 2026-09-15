"""Tiny actual Llama architecture fixtures; NOT pretrained 8B GPU parity."""
import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from project.run_scripts.single_layer_zflow.llama_adapter import LlamaAffineOracle, model_guard


class LlamaAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(516); torch.set_num_threads(1)
        cfg = LlamaConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=8,
                          num_attention_heads=4, num_key_value_heads=2, vocab_size=43,
                          max_position_embeddings=128, attention_dropout=0., pad_token_id=0)
        cfg._attn_implementation = 'eager'
        self.model = LlamaForCausalLM(cfg).eval().requires_grad_(False)
        self.writer = torch.randn(2,24)*.03
        self.packed = dict(input_ids=torch.tensor([[1,2,3,4,5],[1,8,9,0,0]]),
             attention_mask=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]]),
             position_ids=torch.arange(5)[None], edit_rows=torch.tensor([0,0,1]),
             edit_cols=torch.tensor([3,4,2]), edit_labels=torch.tensor([7,8,9]),
             edit_weights=torch.tensor([.25,.25,.5],dtype=torch.float64),
             kl_rows=torch.tensor([0,1]),kl_cols=torch.tensor([1,1]),
             kl_weights=torch.tensor([.5,.5],dtype=torch.float64))

    def test_full_write_logits_nll_gradient_and_zero_teacher(self):
        oracle = LlamaAffineOracle(self.model,self.writer,[self.packed])
        x = torch.randn(16,2)*.01
        evidence = oracle.parity(x,backward=True)
        self.assertLess(evidence['max_logit_abs'],2e-6)
        self.assertLess(evidence['edit_abs_error'],1e-6)
        self.assertLess(evidence['gradient_relative_l2'],2e-5)
        oracle(torch.zeros_like(x))
        self.assertLess(abs(oracle.events[-1]['essence_kl']),1e-7)
        self.assertEqual(oracle.work['head_positions'],5)
        self.assertEqual(oracle.work['oracle_calls'],1)

    def test_microbatch_partition_global_gradient(self):
        parts=[]
        for row in (0,1):
            packed={k:v.clone() for k,v in self.packed.items()}
            for k in ('input_ids','attention_mask'):
                packed[k]=packed[k][row:row+1]
            for prefix in ('edit','kl'):
                mask=self.packed[prefix+'_rows']==row
                for key in list(packed):
                    if key.startswith(prefix+'_'):
                        packed[key]=packed[key][mask]
                packed[prefix+'_rows']=torch.zeros_like(packed[prefix+'_rows'])
            parts.append(packed)
        a=LlamaAffineOracle(self.model,self.writer,[self.packed])
        b=LlamaAffineOracle(self.model,self.writer,parts)
        x=torch.randn(16,2)*.1
        la,ga=a(x);lb,gb=b(x)
        self.assertAlmostEqual(la,lb,places=6)
        torch.testing.assert_close(ga,gb,rtol=2e-5,atol=2e-8)

    def test_no_mutation_selected_head_all_token_cache_and_stale_guard(self):
        before=model_guard(self.model)
        oracle=LlamaAffineOracle(self.model,self.writer,[self.packed])
        cache=oracle.caches[0]
        self.assertEqual(tuple(cache.a.shape),(2,5,2))
        self.assertTrue(bool((cache.a[0,:3].abs().sum(-1)>0).all()))
        oracle(torch.randn(16,2)*.1)
        self.assertEqual(before,model_guard(self.model))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        with torch.no_grad():
            self.model.model.layers[4].mlp.down_proj.weight.add_(1)
        with self.assertRaisesRegex(RuntimeError,'FROZEN_ENTRY'):
            oracle(torch.zeros(16,2))

    def test_materialized_fp32_candidate_parity(self):
        oracle=LlamaAffineOracle(self.model,self.writer,[self.packed])
        x=torch.randn(16,2)*.05
        candidate=self.model.model.layers[4].mlp.down_proj.weight+x@self.writer
        result=oracle.parity(x,candidate_weight=candidate)
        self.assertLess(result['max_logit_abs'],2e-6)


if __name__=='__main__': unittest.main()
