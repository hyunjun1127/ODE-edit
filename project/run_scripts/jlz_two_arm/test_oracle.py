"""Tiny random CPU Llama integration; not pretrained/GPU numerical evidence.

Independent full-model functional_call differentiates all five FP32 materialized
weights and computes the explicit scalar formula. This exercises the production
oracle, live cross-layer dX, padding/target positions and reference loss means.
"""
import copy
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from project.run_scripts.jlz_pilot.prompts import prepare, entry_teacher
from .burden import build_burden_matrix
from .common import LAYERS
from .oracle import CommonOracle, prepare_teacher, reference_pack
from project.run_scripts.jlz_sequential.state import tensor_sha


class TinyTokenizer:
    padding_side = "right"
    bos_token_id = 1
    unk_token_id = 2
    pad_token_id = 0
    alphabet = " abcdefghijklmnopqrstuvwxyz.?"

    def encode(self, text, add_special_tokens=True):
        values = [self.alphabet.index(c) + 3 for c in text]
        return ([1] if add_special_tokens else []) + values

    def decode(self, ids):
        return "".join(self.alphabet[int(v) - 3] for v in ids if int(v) > 2)

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=True):
        if isinstance(text, str):
            ids = self.encode(text, add_special_tokens=add_special_tokens)
            if return_tensors:
                return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
            return {"input_ids": ids}
        rows = [self.encode(t, add_special_tokens=add_special_tokens) for t in text]
        width = max(map(len, rows))
        return {"input_ids": torch.tensor([r + [0] * (width - len(r)) for r in rows]),
                "attention_mask": torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])}


def record(case, subject, new, true, suffix=" de"):
    return {"case_id": case, "requested_rewrite": {
        "prompt": "{}" + suffix, "subject": subject, "target_new": {"str": new},
        "target_true": {"str": true}, "relation_id": "P1"}}


class CommonOracleCPUTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20261002)
        config = LlamaConfig(vocab_size=32, hidden_size=12, intermediate_size=24,
            num_hidden_layers=9, num_attention_heads=3, num_key_value_heads=1,
            max_position_embeddings=128, attention_dropout=0., use_cache=False,
            pad_token_id=0, bos_token_id=1, eos_token_id=2)
        config._attn_implementation = "eager"
        self.model = LlamaForCausalLM(config).float().eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tok = TinyTokenizer()
        self.current = [record(1, "ab", "fg", "h"), record(2, "c", "i", "jk l")]
        self.general = [record(10, "m", "n", "o"), record(11, "pq", "r", "st u"), record(12, "v", "w", "x")]
        self.replay = [record(20, "a", "bc", "d"), record(21, "ef", "g", "hi")]
        self.contexts = [["{}"], ["a. {}", "b. {}", "c. {}", "d. {}", "e. {}"]]
        self.spec = prepare(self.tok, [dict(r["requested_rewrite"], case_id=r["case_id"]) for r in self.current], self.contexts, "cpu")
        with torch.no_grad():
            logits = self.model(**self.spec["tokens"], use_cache=False).logits
        self.teacher = entry_teacher(logits, self.spec)
        self.w0teacher = prepare_teacher(self.model, self.tok, self.general, mb=2)
        self.adj, self.matrices, self.scales = {}, {}, {}
        for layer in LAYERS:
            A = torch.eye(24, dtype=torch.float64) * 1.5
            K = torch.randn(24, 2, dtype=torch.float64) * .1
            P = torch.linalg.solve(A + K @ K.T, K)
            self.adj[layer] = P
            self.matrices[layer] = build_burden_matrix(A, K, P)[0]
            self.scales[layer] = 1.3 + .1 * layer
        self.point = torch.randn(10, 12) * .02

    def oracle(self, eta=0, mb=2, replay=None, teacher=None, general_teacher=None):
        return CommonOracle(self.model, self.tok, self.spec,
            self.teacher if teacher is None else teacher,
            self.adj, self.matrices, self.scales, self.general,
            self.replay if replay is None else replay,
            self.w0teacher if general_teacher is None else general_teacher, eta, mb=mb)

    def independent(self, x, eta=0, replay=None, teacher=None, general_teacher=None):
        """Full logits/full model, no production suffix/loss/custom-backward code."""
        replay = self.replay if replay is None else replay
        teacher = self.teacher if teacher is None else teacher
        general_teacher = self.w0teacher if general_teacher is None else general_teacher
        parameters = {}
        for i, layer in enumerate(LAYERS):
            name = f"model.layers.{layer}.mlp.down_proj.weight"
            R = x[i * 2:(i + 1) * 2].T
            parameters[name] = self.model.get_parameter(name) + (R.double() @ self.adj[layer].T).float()

        def forward(tokens):
            return torch.func.functional_call(self.model, parameters, (), {**tokens, "use_cache": False}).logits

        logits = forward(self.spec["tokens"])
        nll, kl = [], []
        for request, spec in enumerate(self.spec["specs"]):
            row_nll = []
            for row in range(spec["offset"], spec["offset"] + 6):
                selected = self.spec["targets"][row] != -100
                labels = self.spec["targets"][row, selected]
                logp = logits[row, selected].log_softmax(-1)
                row_nll.append(-logp.gather(1, labels[:, None]).mean())
            nll.append(torch.stack(row_nll).mean())
            current = logits[self.spec["kl_rows"][request], self.spec["kl_cols"][request]].log_softmax(-1)
            kl.append((current.exp() * (current - teacher[request])).sum())
        current_loss = (torch.stack(nll) + .0625 * torch.stack(kl)).sum()
        general_values, replay_values = [], []
        for records, target, general in ((self.general, "target_true", True), (replay, "target_new", False)):
            if not records:
                continue
            pack = reference_pack(self.tok, records, target, "cpu")
            logits = forward(pack["tokens"])
            for i, rec in enumerate(records):
                mask = pack["targets"][i] != -100
                labels = pack["targets"][i, mask]
                logp = logits[i, mask].log_softmax(-1)
                if general:
                    teach = general_teacher[str(rec["case_id"])]["logp"]
                    general_values.append((logp.exp() * (logp - teach)).sum(-1).mean())
                else:
                    replay_values.append(-logp.gather(1, labels[:, None]).mean())
        general_mean = torch.stack(general_values).mean()
        replay_mean = torch.stack(replay_values).mean() if replay_values else general_mean * 0.
        omega = x.new_zeros((), dtype=torch.float64)
        for i, layer in enumerate(LAYERS):
            R = x[i * 2:(i + 1) * 2].T.double()
            omega = omega + .5 * ((R @ self.matrices[layer]) * R).sum() / self.scales[layer]
        value = current_loss + 2 * (.0625 * general_mean + replay_mean) + eta * omega
        return value, {"current": current_loss, "general": general_mean, "replay": replay_mean,
                       "nll": torch.stack(nll), "kl": torch.stack(kl), "omega": omega}

    def test_dense_original_direct_full_materialization_and_gradient(self):
        original = {n: p.clone() for n, p in self.model.named_parameters()}
        x = self.point.clone().requires_grad_()
        expected, components = self.independent(x)
        expected_grad, = torch.autograd.grad(expected, x)
        oracle = self.oracle(mb=2)
        records = {}
        for route in ("dense", "original", "direct"):
            value, grad, payload = oracle(self.point, route=route)
            self.assertAlmostEqual(value, float(expected.detach()), places=5)
            torch.testing.assert_close(grad, expected_grad, atol=3e-6, rtol=5e-4)
            torch.testing.assert_close(torch.tensor(payload["nll"]), components["nll"], atol=8e-7, rtol=2e-6)
            torch.testing.assert_close(torch.tensor(payload["kl"]), components["kl"], atol=8e-7, rtol=2e-6)
            for i, layer in enumerate(LAYERS):
                self.assertGreater(float(grad[2 * i:2 * (i + 1)].norm()), 1e-7)
                self.assertTrue(torch.equal(payload["weights"][layer], oracle.effective(self.point)[layer]))
            records[route] = payload
        self.assertEqual(oracle.calls, 3)
        for name, param in self.model.named_parameters():
            self.assertTrue(torch.equal(param, original[name]), name)
        self.assertTrue(all("forward" not in self.model.model.layers[l].mlp.down_proj.__dict__ for l in LAYERS))

    def test_uneven_microbatch_and_padding_keep_prompt_mean(self):
        one = self.oracle(mb=1)(self.point)
        wide = self.oracle(mb=4)(self.point)
        self.assertAlmostEqual(one[0], wide[0], places=5)
        torch.testing.assert_close(one[1], wide[1], atol=3e-6, rtol=5e-4)
        self.assertEqual(len(wide[2]["general"]), 3)
        self.assertEqual(len(wide[2]["replay"]), 2)
        self.assertEqual({r["case_id"] for r in wide[2]["general"]}, {10, 11, 12})

    def test_eta_only_adds_exact_burden_gradient_same_materialized_state(self):
        a, b = self.oracle(eta=0)(self.point), self.oracle(eta=1)(self.point)
        self.assertAlmostEqual(b[0] - a[0], b[2]["omega"], places=10)
        for i, layer in enumerate(LAYERS):
            # R's transpose convention: (R M).T = M @ x_block.
            expected = (self.matrices[layer] @ self.point[2 * i:2 * i + 2].double() / self.scales[layer]).float()
            torch.testing.assert_close(b[1][2 * i:2 * i + 2] - a[1][2 * i:2 * i + 2], expected, atol=1e-7, rtol=1e-5)
            self.assertTrue(torch.equal(a[2]["weights"][layer], b[2]["weights"][layer]))

    def test_inactive_columns_stay_zero_but_current_request_still_in_loss(self):
        point = self.point.clone()
        point[::2] = 0  # request 0 inactive in the solver, across all five layers.
        oracle = self.oracle()
        loss, gradient, payload = oracle(point)
        self.assertTrue(bool(oracle.active.all()))
        expected, parts = self.independent(point)
        self.assertAlmostEqual(loss, float(expected.detach()), places=5)
        self.assertGreater(payload["nll"][0], 0.)
        self.assertGreater(float(gradient[::2].norm()), 0.)  # oracle derivative retained; prox freezes columns.
        without_first = float(expected - parts["nll"][0] - .0625 * parts["kl"][0])
        self.assertGreater(abs(loss - without_first), 1.)

    def test_empty_replay_zero_and_general_weight_unchanged(self):
        loss, gradient, payload = self.oracle(replay=[])(self.point)
        x = self.point.clone().requires_grad_()
        expected, parts = self.independent(x, replay=[])
        expected_grad, = torch.autograd.grad(expected, x)
        self.assertAlmostEqual(loss, float(expected.detach()), places=5)
        self.assertEqual(float(parts["replay"].detach()), 0.)
        self.assertEqual(payload["replay"], [])
        torch.testing.assert_close(gradient, expected_grad, atol=3e-6, rtol=5e-4)

    def test_general_and_native_kl_direction_is_current_to_teacher(self):
        # Deliberately asymmetric synthetic log-probabilities test direction;
        # this is not a claim these synthetic rows are pretrained W0 teachers.
        native_teacher = (torch.randn_like(self.teacher) * 2).log_softmax(-1)
        general_teacher = copy.deepcopy(self.w0teacher)
        for item in general_teacher.values():
            item["logp"] = (torch.randn_like(item["logp"]) * 2).log_softmax(-1)
            item["sha256"] = tensor_sha(item["logp"])
        loss, gradient, payload = self.oracle(teacher=native_teacher, general_teacher=general_teacher)(self.point)
        x = self.point.clone().requires_grad_()
        expected, _ = self.independent(x, teacher=native_teacher, general_teacher=general_teacher)
        expected_grad, = torch.autograd.grad(expected, x)
        self.assertAlmostEqual(loss, float(expected.detach()), places=5)
        torch.testing.assert_close(gradient, expected_grad, atol=3e-6, rtol=5e-4)
        self.assertGreater(min(payload["kl"]), .1)
        self.assertGreater(min(r["value"] for r in payload["general"]), .1)

    def test_reference_identity_and_prediction_token_alignment(self):
        pack = reference_pack(self.tok, self.general, "target_true", "cpu")
        for i, row in enumerate(self.general):
            rw = row["requested_rewrite"]
            p = self.tok.encode(rw["prompt"].format(rw["subject"]))
            target = " " + rw["target_true"]["str"]
            y = self.tok.encode(target, add_special_tokens=False)
            size = len(p) + len(y) - 1
            self.assertEqual(pack["tokens"]["input_ids"][i, :size].tolist(), (p + y)[:-1])
            self.assertEqual(pack["targets"][i, len(p) - 1:size].tolist(), y)
            self.assertEqual(int((pack["targets"][i] != -100).sum()), len(y))
        teachers = copy.deepcopy(self.w0teacher)
        teachers["10"]["identity"] = "corrupt"
        with self.assertRaisesRegex(RuntimeError, "TEACHER_TOKEN_IDENTITY"):
            self.oracle(general_teacher=teachers)

    def test_committed_losses_equal_materialized_candidate_and_restore(self):
        oracle = self.oracle()
        _, _, payload = oracle(self.point)
        originals = {l: self.model.model.layers[l].mlp.down_proj.weight.clone() for l in LAYERS}
        try:
            with torch.no_grad():
                for layer in LAYERS:
                    self.model.model.layers[layer].mlp.down_proj.weight.copy_(payload["weights"][layer])
            nll, kl = oracle.committed_losses()
            torch.testing.assert_close(nll, torch.tensor(payload["nll"]), atol=8e-7, rtol=2e-6)
            torch.testing.assert_close(kl, torch.tensor(payload["kl"]), atol=8e-7, rtol=2e-6)
        finally:
            with torch.no_grad():
                for layer in LAYERS:
                    self.model.model.layers[layer].mlp.down_proj.weight.copy_(originals[layer])


if __name__ == "__main__":
    unittest.main()
