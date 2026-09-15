"""CPU correctness tests, not Llama tokenizer/backend or GPU parity tests."""
from dataclasses import replace
import unittest

import torch
import torch.nn.functional as F

from project.run_scripts.single_layer_zflow.oracle import (
    CachedSuffixOracle, MicrobatchCache, make_affine_cache,
)
from project.run_scripts.single_layer_zflow.transaction import (
    CommitBudgetError, CommitConflict, CommitParityError,
    InMemoryBatchTransaction, materialized_cost,
)


class OracleTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(418)
        def rand(*shape):
            return torch.randn(*shape, generator=generator, dtype=torch.float64)
        self.keys = rand(2, 4, 5)
        self.weight = rand(3, 5)
        self.writer = rand(2, 5) * 0.15
        self.residual = rand(2, 4, 3)
        self.head = rand(3, 7).requires_grad_(True)
        self.x = rand(3, 2) * 0.1
        self.entry = self.residual + self.keys @ self.weight.T
        self.edit_positions = torch.tensor([[0, 1], [0, 2], [1, 3]])
        self.labels = torch.tensor([1, 4, 2])
        self.weights = torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64)
        self.kl_positions = torch.tensor([[0, 2], [1, 1]])
        with torch.no_grad():
            lp = self.suffix(self.entry).log_softmax(-1)
            self.teacher = lp[self.kl_positions[:, 0], self.kl_positions[:, 1]]
        self.cache = make_affine_cache(
            self.entry, self.keys, self.writer,
            edit_positions=self.edit_positions, edit_labels=self.labels,
            edit_weights=self.weights, kl_positions=self.kl_positions,
            teacher_log_probs=self.teacher,
            kl_weights=torch.tensor([0.5, 0.5], dtype=torch.float64),
        )

    def suffix(self, hidden):
        # Causal nonlinear dependence on earlier, unsupervised tokens.
        return torch.tanh(hidden + 0.3 * hidden.cumsum(dim=1)) @ self.head

    def direct_loss(self, x, beta=0.0625):
        actual_weight = self.weight + x @ self.writer
        logits = self.suffix(self.residual + self.keys @ actual_weight.T)
        pos = self.edit_positions
        edit = (F.cross_entropy(logits[pos[:, 0], pos[:, 1]], self.labels,
                                reduction="none") * self.weights).sum()
        pos = self.kl_positions
        current = logits[pos[:, 0], pos[:, 1]].log_softmax(-1)
        # Spell out p_current || p_entry independently of oracle's kl_div call.
        kl = (current.exp() * (current - self.teacher)).sum(-1).mean()
        return edit + beta * kl

    def split_caches(self):
        result = []
        for seq in range(2):
            edit = self.edit_positions[:, 0] == seq
            kl = self.kl_positions[:, 0] == seq
            ep = self.edit_positions[edit].clone()
            kp = self.kl_positions[kl].clone()
            ep[:, 0] = 0
            kp[:, 0] = 0
            result.append(MicrobatchCache(
                self.entry[seq:seq+1], self.cache.a[seq:seq+1], ep,
                self.labels[edit], self.weights[edit], kp, self.teacher[kl],
                self.cache.kl_weights[kl],
            ))
        return result

    def test_nonlinear_all_token_full_write_and_gradient_parity(self):
        x = self.x.clone().requires_grad_(True)
        expected = self.direct_loss(x)
        expected_gradient, = torch.autograd.grad(expected, x)
        loss, gradient = CachedSuffixOracle(self.suffix, [self.cache])(self.x)
        self.assertAlmostEqual(loss, float(expected.detach()), places=12)
        torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=1e-12)
        # A subject-only perturbation is a different intervention.
        partial = self.entry.clone()
        partial[:, 1] += self.cache.a[:, 1] @ self.x.T
        self.assertFalse(torch.allclose(self.suffix(partial), self.suffix(self.entry + self.cache.a @ self.x.T)))

    def test_global_weights_are_microbatch_partition_invariant(self):
        loss_whole, grad_whole = CachedSuffixOracle(self.suffix, [self.cache])(self.x)
        loss_split, grad_split = CachedSuffixOracle(self.suffix, self.split_caches())(self.x)
        self.assertAlmostEqual(loss_whole, loss_split, places=12)
        torch.testing.assert_close(grad_whole, grad_split, atol=1e-12, rtol=1e-12)
        # Local re-normalization to 1 per microbatch must be rejected.
        invalid = [replace(c, edit_weights=c.edit_weights/c.edit_weights.sum()) for c in self.split_caches()]
        with self.assertRaises(ValueError):
            CachedSuffixOracle(self.suffix, invalid)

    def test_oracle_owns_fixed_caches_and_does_not_mutate_external_state(self):
        oracle = CachedSuffixOracle(self.suffix, [self.cache])
        x = self.x.clone().requires_grad_(True)
        x.grad = torch.ones_like(x)
        before_x, before_w, before_head = x.detach().clone(), self.weight.clone(), self.head.detach().clone()
        baseline = oracle(x)
        self.cache.h_entry.add_(9)
        self.cache.a.zero_()
        self.cache.teacher_log_probs.zero_()
        after = oracle(x)
        self.assertEqual(baseline[0], after[0])
        torch.testing.assert_close(baseline[1], after[1], atol=0, rtol=0)
        torch.testing.assert_close(x, before_x, atol=0, rtol=0)
        torch.testing.assert_close(x.grad, torch.ones_like(x), atol=0, rtol=0)
        torch.testing.assert_close(self.weight, before_w, atol=0, rtol=0)
        torch.testing.assert_close(self.head, before_head, atol=0, rtol=0)
        self.assertIsNone(self.head.grad)

    def test_reverse_kl_and_optional_edit_only(self):
        edit_only = replace(self.cache, kl_positions=None, teacher_log_probs=None, kl_weights=None)
        loss, _ = CachedSuffixOracle(self.suffix, [edit_only])(self.x)
        self.assertAlmostEqual(loss, float(self.direct_loss(self.x, beta=0).detach()), places=12)
        loss_kl, _ = CachedSuffixOracle(self.suffix, [self.cache])(self.x)
        self.assertGreater(loss_kl, loss)
        self.assertAlmostEqual(loss_kl, float(self.direct_loss(self.x).detach()), places=12)


class TransactionTests(unittest.TestCase):
    def setUp(self):
        self.weight = torch.arange(15, dtype=torch.float32).reshape(3, 5) / 10
        self.history = torch.eye(5, dtype=torch.float32)
        self.x = torch.tensor([[.1, .2], [.3, -.1], [-.2, .4]], dtype=torch.float32)
        self.writer = torch.arange(10, dtype=torch.float32).reshape(2, 5) / 20
        self.keys = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 10
        self.entry_w, self.entry_m = self.weight.clone(), self.history.clone()
        self.receipts = {}
        self.transaction = InMemoryBatchTransaction("batch-1", self.weight, self.history, self.receipts)

    def assert_entry_unchanged(self):
        torch.testing.assert_close(self.weight, self.entry_w, atol=0, rtol=0)
        torch.testing.assert_close(self.history, self.entry_m, atol=0, rtol=0)
        self.assertEqual(self.receipts, {})

    def test_failed_parity_keeps_weight_history_and_receipts_unchanged(self):
        def reject(candidate, delta):
            self.assert_entry_unchanged()
            torch.testing.assert_close(candidate.double() - self.entry_w.double(), delta, atol=0, rtol=0)
            candidate.zero_()  # Checker sees private clones.
            return False
        with self.assertRaises(CommitParityError):
            self.transaction.commit(self.x, self.writer, self.keys, reject)
        self.assert_entry_unchanged()

    def test_checker_exception_restores_owned_weight_and_history(self):
        def failing_checker(*args):
            self.weight.add_(1)
            self.history.zero_()
            raise RuntimeError("injected checker failure")
        with self.assertRaises(CommitParityError):
            self.transaction.commit(self.x, self.writer, self.keys, failing_checker)
        self.assert_entry_unchanged()

    def test_checker_mutation_is_rolled_back_even_if_it_returns_true(self):
        def mutating_checker(*args):
            self.weight.zero_()
            self.history.add_(7)
            return True
        with self.assertRaises(CommitConflict):
            self.transaction.commit(self.x, self.writer, self.keys, mutating_checker)
        self.assert_entry_unchanged()

    def test_duplicate_commit_is_noop_and_conflicting_payload_is_refused(self):
        calls = []
        def parity(candidate, delta):
            self.assert_entry_unchanged()
            calls.append(True)
            return True
        first = self.transaction.commit(self.x, self.writer, self.keys, parity)
        expected_w = self.entry_w + self.x @ self.writer
        expected_m = self.entry_m + self.keys @ self.keys.T
        torch.testing.assert_close(self.weight, expected_w, atol=0, rtol=0)
        torch.testing.assert_close(self.history, expected_m, atol=0, rtol=0)
        second = self.transaction.commit(self.x, self.writer, self.keys, parity)
        self.assertFalse(first.replayed)
        self.assertTrue(second.replayed)
        self.assertEqual(len(calls), 1)
        self.assertEqual(second.history_sha256, first.history_sha256)
        torch.testing.assert_close(self.history, expected_m, atol=0, rtol=0)
        with self.assertRaises(CommitConflict):
            self.transaction.commit(self.x * 2, self.writer, self.keys, parity)
        torch.testing.assert_close(self.history, expected_m, atol=0, rtol=0)

    def test_materialized_budget_is_checked_before_parity_and_write(self):
        def should_not_run(*args):
            self.fail("parity should not run after a budget failure")
        with self.assertRaises(CommitBudgetError):
            self.transaction.commit(self.x, self.writer, self.keys, should_not_run, cost_budget=0)
        self.assert_entry_unchanged()
        actual_delta = (self.entry_w + self.x @ self.writer).double() - self.entry_w.double()
        cost = materialized_cost(actual_delta, self.entry_m, request_count=self.x.shape[1])
        receipt = self.transaction.commit(self.x, self.writer, self.keys, lambda *args: True,
                                          cost_budget=cost)
        self.assertEqual(receipt.actual_cost, cost)

    def test_cost_uses_rounded_actual_delta(self):
        weight = torch.full((1, 1), 1e8, dtype=torch.float32)
        history = torch.zeros(1, 1, dtype=torch.float32)
        tx = InMemoryBatchTransaction("rounding", weight, history)
        one = torch.ones(1, 1, dtype=torch.float32)
        receipt = tx.commit(one, one, one, lambda w, d: bool((d == 0).all()), cost_budget=0)
        self.assertEqual(receipt.actual_cost, 0.0)
        self.assertEqual(float(weight), 1e8)
        self.assertEqual(float(history), 1.0)

    def test_stale_entry_is_refused_without_overwriting_external_change(self):
        self.weight.add_(1)
        stale_live = self.weight.clone()
        with self.assertRaises(CommitConflict):
            self.transaction.commit(self.x, self.writer, self.keys, lambda *args: True)
        torch.testing.assert_close(self.weight, stale_live, atol=0, rtol=0)
        torch.testing.assert_close(self.history, self.entry_m, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
