"""Tiny CPU Llama/FP64 scalar fixtures, NOT pretrained8B/actual GPU T evidence."""
import unittest
import torch
import transformers

from project.run_scripts.single_layer_edit_preserving_correction.alltoken import (
    FullWeightLlamaOracle, WEIGHT, model_guard, signed_forward_kl)


class KLArithmeticTests(unittest.TestCase):
    def test_signed_forward_kl_direction_reduction_and_gradient(self):
        logits = torch.tensor([[.2, -.5, .9], [.4, .7, -.2]], requires_grad=True)
        teacher = torch.tensor([[.9, -.3, .1], [.2, -.7, .5]]).log_softmax(-1)
        observed = signed_forward_kl(logits, teacher)
        t = teacher.double()
        expected = (t.exp()*(t-logits.log_softmax(-1).double())).sum(-1).mean()
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        grad, = torch.autograd.grad(observed, logits)
        expected_grad = ((logits.detach().softmax(-1).double()*t.exp().sum(-1, keepdim=True)
                         - t.exp())/2).float()
        torch.testing.assert_close(grad, expected_grad, rtol=2e-6, atol=2e-8)
        self.assertEqual(observed.dtype, torch.float64)

    def test_negative_roundoff_is_not_clamped_and_teacher_detached(self):
        logits = torch.zeros(1, 2, requires_grad=True)
        teacher = (logits.detach().log_softmax(-1)-1e-7).requires_grad_(True)
        loss = signed_forward_kl(logits, teacher)
        self.assertLess(float(loss.detach()), 0.)
        loss.backward()
        self.assertIsNone(teacher.grad)


@unittest.skipUnless(transformers.__version__ == "4.44.2", "pinned tiny architecture requires transformers4.44.2")
class AllTokenLlamaTests(unittest.TestCase):
    def setUp(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(718)
        self.old_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        config = LlamaConfig(hidden_size=16, intermediate_size=24, num_hidden_layers=7,
                             num_attention_heads=4, num_key_value_heads=2, vocab_size=43,
                             max_position_embeddings=512, attention_dropout=0., pad_token_id=0)
        config._attn_implementation = "eager"
        self.model = LlamaForCausalLM(config).eval().requires_grad_(False)
        self.packs = [dict(input_ids=torch.tensor([[0, 1, 2, 3, 4, 5]]),
                           attention_mask=torch.tensor([[0, 1, 1, 1, 1, 1]]),
                           position_ids=torch.tensor([[0, 0, 1, 2, 3, 4]])),
                      dict(input_ids=torch.tensor([[1, 8, 9, 10, 11, 0]]),
                           attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0]]),
                           position_ids=torch.tensor([[0, 1, 2, 3, 4, 0]]))]
        self.oracle = FullWeightLlamaOracle(self.model, self.packs, score_slice=(1, 5),
                                           require_reference_length=6, head_chunk_positions=2)
        self.entry = dict(self.model.named_parameters())[WEIGHT].detach().clone()
        teachers = []
        with torch.no_grad():
            for i in range(2):
                hidden = self.oracle.hidden(i, self.entry, route="physical")
                teachers.append(self.model.lm_head(hidden[0, 1:5]).float().log_softmax(-1).cpu())
        self.oracle.teacher_loader = lambda index, cache: teachers[index]

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.old_tf32

    def test_raw_key_dimensions_and_prefix_valid_positions(self):
        self.assertEqual(tuple(self.oracle.caches[0].keys.shape), (1, 6, 24))
        self.assertEqual(tuple(self.oracle.capture_keys(0).shape), (24, 5))
        self.assertEqual(self.oracle.caches[0].valid_positions().tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(self.oracle.caches[1].valid_positions().tolist(), [0, 1, 2, 3, 4])
        self.assertTrue(self.oracle.key_stationarity(0, self.entry+.01)["byte_equal"])

    def test_absolute_full_weight_logits_nll_gradient_parity(self):
        weight = self.entry+torch.randn_like(self.entry)*.03
        for i in range(2):
            result = self.oracle.compare_logits(i, weight, weight)
            self.assertLess(result["max_abs"], 2e-6)
            self.assertEqual(result["valid_positions"], 5)
        cached = self.oracle.sequence_nll(0, weight, [3, 4], [7, 8])
        actual = self.oracle.sequence_nll(0, weight, [3, 4], [7, 8], route="physical")
        self.assertAlmostEqual(cached["nll"], actual["nll"], places=6)
        self.assertEqual(cached["strict"], actual["strict"])
        lc, gc, rows = self.oracle.kl(weight, gradient=True)
        lp, gp, _ = self.oracle.kl(weight, gradient=True, route="physical")
        self.assertAlmostEqual(lc, lp, places=7)
        self.assertEqual(gc.dtype, torch.float64)
        self.assertEqual(gc.device.type, "cpu")
        self.assertLess(float((gc-gp).norm()/gp.norm()), 1e-5)
        self.assertEqual(len(rows), 2)
        self.assertTrue(torch.equal(self.entry, dict(self.model.named_parameters())[WEIGHT]))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_noop_signed_loss_and_document_mean_mass(self):
        loss, grad, _ = self.oracle.kl(self.entry, gradient=True)
        self.assertLess(abs(loss), 1e-7)
        candidate = self.entry+.02
        full_loss, full_grad, _ = self.oracle.kl(candidate, gradient=True)
        first = self.oracle.kl(candidate, gradient=True, indices=[0])
        second = self.oracle.kl(candidate, gradient=True, indices=[1])
        self.assertAlmostEqual(full_loss, (first[0]+second[0])/2, places=14)
        torch.testing.assert_close(full_grad, (first[1]+second[1])/2, rtol=0, atol=0)

    def test_physical_exception_exact_restore_and_no_leftover_hooks(self):
        before = model_guard(self.model)
        bad = lambda index, cache: (_ for _ in ()).throw(RuntimeError("loader failure"))
        with self.assertRaisesRegex(RuntimeError, "loader failure"):
            self.oracle.kl(self.entry+.03, gradient=True, route="physical", teacher_loader=bad)
        self.assertTrue(torch.equal(self.entry, dict(self.model.named_parameters())[WEIGHT]))
        after = model_guard(self.model)
        self.assertEqual(before[1], after[1])
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in self.model.parameters()))
        self.oracle.kl(self.entry)

    def test_cache_guard_rejects_physical_mutation(self):
        with torch.no_grad():
            dict(self.model.named_parameters())[WEIGHT].add_(.01)
        with self.assertRaisesRegex(RuntimeError, "FROZEN_MODEL"):
            self.oracle.kl(self.entry)

    def test_cache_guard_rejects_external_key_change(self):
        self.oracle.caches[0].keys.add_(1)
        with self.assertRaisesRegex(RuntimeError, "CPU_INPUT_KEY"):
            self.oracle.kl(self.entry)

    def test_explicit_selected_write_rebind_preserves_cache(self):
        candidate = self.entry+.01
        with torch.no_grad():
            dict(self.model.named_parameters())[WEIGHT].copy_(candidate)
        with self.assertRaisesRegex(RuntimeError, "EXPECTED_BYTES"):
            self.oracle.acknowledge_selected_write(self.entry)
        receipt = self.oracle.acknowledge_selected_write(candidate)
        self.assertTrue(receipt["expected_selected_bytes_equal"])
        direct = self.oracle.logits_at(0, candidate, [2, 4], route="physical")
        cached = self.oracle.logits_at(0, candidate, [2, 4])
        torch.testing.assert_close(cached, direct, rtol=1e-5, atol=1e-6)
        hidden = self.oracle.hidden(0, candidate)
        torch.testing.assert_close(self.oracle.logits_from_hidden(hidden, [2, 4]), cached,
                                   rtol=0, atol=0)

    def test_full_valid_logits_chunking_and_padding_exclusion(self):
        chunks = list(self.oracle.iter_logits(0, self.entry))
        self.assertEqual([len(p) for p, _ in chunks], [2, 2, 1])
        self.assertTrue(all(logit.shape[1] == 43 for _, logit in chunks))
        with self.assertRaisesRegex(ValueError, "PADDING"):
            self.oracle.sequence_nll(0, self.entry, [0], [1])
        with self.assertRaisesRegex(ValueError, "UNIQUE"):
            self.oracle.kl(self.entry, indices=[0, 0])


if __name__ == "__main__":
    unittest.main()
