"""Exercise the actual pretrained driver oracle on a tiny random CPU Llama.

These tests cover functional updates and prefix replay in the real driver;
they make no pretrained editing-performance or convergence claim.
"""

import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from .prompts import entry_teacher, native_loss
from .run import LAYERS, Oracle, capture_keys
from .solver import solve
from .test_prompts import fixture


class DriverIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2107)
        config = LlamaConfig(vocab_size=128, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=10, num_attention_heads=4,
                             num_key_value_heads=2, max_position_embeddings=128,
                             attention_dropout=0.0, use_cache=False)
        config._attn_implementation = "eager"
        self.model = LlamaForCausalLM(config).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        _, _, self.contexts, self.spec = fixture()
        with torch.no_grad():
            logits = self.model(**self.spec["tokens"], use_cache=False).logits
        self.teacher = entry_teacher(logits, self.spec)
        self.keys = capture_keys(self.model, self.spec, self.contexts, microbatch=2)
        self.adj = {}
        for layer in LAYERS:
            key = self.keys[layer].double()
            self.adj[layer] = torch.linalg.solve(0.01 * torch.eye(24) + key @ key.T, key)
        self.active = torch.ones(2, dtype=torch.bool)
        self.point = 0.01 * torch.randn(10, 16)

    def oracle(self, microbatch):
        return Oracle(self.model, self.spec, self.teacher, self.keys, self.adj, self.active, microbatch)

    def test_driver_nonzero_full_suffix_loss_gradient_and_no_mutation(self):
        original = {layer: self.model.model.layers[layer].mlp.down_proj.weight.clone() for layer in LAYERS}
        oracle = self.oracle(microbatch=2)
        full, full_gradient, full_payload = oracle(self.point, route="full")
        suffix, suffix_gradient, suffix_payload = oracle(self.point, route="suffix")
        self.assertAlmostEqual(full, suffix, places=6)
        torch.testing.assert_close(full_gradient, suffix_gradient, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(torch.tensor(full_payload["kl"]), torch.tensor(suffix_payload["kl"]), atol=1e-7, rtol=1e-5)
        self.assertGreater(max(full_payload["kl"]), 1e-8)
        self.assertGreater(float(full_gradient.norm()), 0.0)
        for layer in LAYERS:
            torch.testing.assert_close(self.model.model.layers[layer].mlp.down_proj.weight, original[layer], rtol=0, atol=0)
            torch.testing.assert_close(full_payload["weights"][layer], suffix_payload["weights"][layer], rtol=0, atol=0)
        self.assertEqual(oracle.forward_calls, 14)
        self.assertEqual(oracle.backward_calls, 14)

    def test_driver_microbatch_loss_and_gradient(self):
        full = self.oracle(microbatch=14)(self.point)
        split = self.oracle(microbatch=1)(self.point)
        self.assertAlmostEqual(full[0], split[0], places=6)
        torch.testing.assert_close(full[1], split[1], atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(torch.tensor(full[2]["nll"]), torch.tensor(split[2]["nll"]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(torch.tensor(full[2]["kl"]), torch.tensor(split[2]["kl"]), atol=2e-7, rtol=2e-5)

    def test_actual_solver_payload_materialization_and_post_keys(self):
        oracle = self.oracle(microbatch=2)
        result = solve(oracle, torch.zeros_like(self.point), torch.zeros(10),
                       torch.full((10,), 0.1), torch.ones(10, dtype=torch.bool), cap=6, tol=1e-7)
        self.assertLessEqual(result["calls"], 6)
        self.assertTrue(result["final_recomputed"])
        self.assertGreater(int(torch.count_nonzero(result["x"])), 0)
        payload = result["final_payload"]
        direct = oracle.effective(result["x"])
        # Isolated diagnostic materialization, independent of production's
        # stricter CONVERGED-only commit policy.
        with torch.no_grad():
            for layer in LAYERS:
                torch.testing.assert_close(payload["weights"][layer], direct[layer], rtol=0, atol=0)
                self.model.model.layers[layer].mlp.down_proj.weight.copy_(payload["weights"][layer])
            logits = self.model(**self.spec["tokens"], use_cache=False).logits
            _, nll, kl = native_loss(logits, self.spec, self.teacher, self.active)
        torch.testing.assert_close(nll, torch.tensor(payload["nll"]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(kl, torch.tensor(payload["kl"]), atol=2e-7, rtol=2e-5)
        post = capture_keys(self.model, self.spec, self.contexts, microbatch=2)
        torch.testing.assert_close(post[4], self.keys[4], rtol=0, atol=0)
        self.assertGreater(float((post[8]-self.keys[8]).norm()), 1e-8)


if __name__ == "__main__":
    unittest.main()
