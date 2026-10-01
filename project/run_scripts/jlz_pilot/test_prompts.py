"""CPU checks for native prompt semantics and microbatch objective weights."""
from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from .prompts import entry_teacher, native_loss, prepare, subject_last


class CharacterTokenizer:
    """Deterministic tokenizer exposing BOS, padding, and multi-token targets."""
    padding_side = "right"
    bos_token_id = 1
    unk_token_id = 2
    pad_token_id = 0

    def encode(self, text):
        return [1] + [ord(character) + 3 for character in text]

    def decode(self, ids):
        return "".join(chr(int(token)-3) for token in ids if int(token) > 2)

    def __call__(self, texts, return_tensors=None, padding=False):
        if isinstance(texts, str):
            ids = self.encode(texts)
            return {"input_ids": torch.tensor([ids])} if return_tensors else {"input_ids": ids}
        rows = [self.encode(text) for text in texts]
        width = max(map(len, rows))
        return dict(input_ids=torch.tensor([row+[0]*(width-len(row)) for row in rows]),
                    attention_mask=torch.tensor([[1]*len(row)+[0]*(width-len(row)) for row in rows]))


def fixture():
    tok = CharacterTokenizer()
    requests = [dict(case_id=1, prompt="{} lives in", subject="Ada", target_new={"str": "Rome"}),
                dict(case_id=2, prompt="{} was born in", subject="Bo", target_new={"str": " X"})]
    contexts = [["{}"], ["A. {}", "B. {}", "C. {}", "D. {}", "E. {}"]]
    return tok, requests, contexts, prepare(tok, requests, contexts, "cpu")


class NativePromptsTest(unittest.TestCase):
    def test_target_forcing_lookup_and_target_free_keys(self):
        tok, requests, contexts, spec = fixture()
        self.assertEqual(requests[0]["target_new"]["str"], "Rome")
        self.assertEqual(spec["requests"][0]["target_new"]["str"], " Rome")
        self.assertEqual(spec["prompts"][0], "Ada lives in Rom")
        self.assertEqual(spec["key_prompts"][0], "Ada lives in")
        self.assertEqual(spec["prompts"][6], "Ada is a")
        self.assertEqual(spec["lookup"][0], len(tok.encode("Ada"))-1)
        self.assertEqual(subject_last(tok, "A. {} lives in", "Ada"), len(tok.encode("A. Ada"))-1)
        self.assertEqual(spec["canonical_rows"], [0, 7])
        self.assertEqual(spec["canonical_key_rows"], [0, 6])
        self.assertEqual(spec["key_context_weights"], [.5, .1, .1, .1, .1, .1]*2)
        target = tok.encode(" Rome")[1:]
        end = int(spec["tokens"]["attention_mask"][0].sum())
        self.assertEqual(spec["targets"][0, end-len(target):end].tolist(), target)
        self.assertEqual(int((spec["targets"][6] != -100).sum()), 0)

    def test_native_direction_matches_explicit_value_and_gradient(self):
        _, _, _, spec = fixture()
        torch.manual_seed(713)
        logits = torch.randn(*spec["targets"].shape, 128, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn(2, 128, dtype=torch.float64).log_softmax(-1)
        actual, nll, kl = native_loss(logits, spec, teacher, [True, True])
        expected_kl = []
        expected_nll = []
        for b, request in enumerate(spec["specs"]):
            rows = list(range(request["offset"], request["offset"]+6))
            full_logp = logits[rows].log_softmax(-1)
            labels = spec["targets"][rows]
            mask = labels != -100
            gathered = full_logp.gather(2, torch.where(mask, labels, 0).unsqueeze(2)).squeeze(2)
            expected_nll.append((-(gathered*mask).sum(1)/request["target"].numel()).mean())
            current = logits[spec["kl_rows"][b], spec["kl_cols"][b]].log_softmax(-1)
            expected_kl.append((current.exp()*(current-teacher[b])).sum())
        expected_nll, expected_kl = torch.stack(expected_nll), torch.stack(expected_kl)
        expected = (expected_nll+.0625*expected_kl).sum()
        torch.testing.assert_close(nll, expected_nll)
        torch.testing.assert_close(kl, expected_kl)
        torch.testing.assert_close(actual, expected)
        actual_grad, = torch.autograd.grad(actual, logits, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, logits)
        torch.testing.assert_close(actual_grad, expected_grad)
        reverse = F.kl_div(logits[spec["kl_rows"], spec["kl_cols"]].log_softmax(-1),
                           teacher, log_target=True, reduction="none").sum(1)
        self.assertGreater(float((reverse-kl).detach().abs().max()), 1e-3)

    def test_arbitrary_microbatch_partition_and_active_mask(self):
        _, _, _, spec = fixture()
        torch.manual_seed(118)
        logits = torch.randn(*spec["targets"].shape, 128, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn(2, 128, dtype=torch.float64).log_softmax(-1)
        full = native_loss(logits, spec, teacher, [False, True])
        chunks = [[13, 0, 8], [1, 2, 9], [3, 10], [4, 5, 6], [7, 11, 12]]
        partial = [native_loss(logits[rows], spec, teacher, [False, True], rows=rows) for rows in chunks]
        for index in range(3):
            torch.testing.assert_close(sum(part[index] for part in partial), full[index])
        full_grad, = torch.autograd.grad(full[0], logits, retain_graph=True)
        micro_grad, = torch.autograd.grad(sum(part[0] for part in partial), logits)
        torch.testing.assert_close(full_grad, micro_grad)
        self.assertEqual(int(torch.count_nonzero(full_grad[:7])), 0)
        self.assertGreater(int(torch.count_nonzero(full_grad[7:])), 0)

    def test_entry_teacher_and_all_inactive(self):
        _, _, _, spec = fixture()
        torch.manual_seed(119)
        logits = torch.randn(*spec["targets"].shape, 128, requires_grad=True)
        teacher = entry_teacher(logits, spec)
        self.assertFalse(teacher.requires_grad)
        loss, _, kl = native_loss(logits, spec, teacher, [False, False])
        self.assertEqual(float(loss.detach()), 0.)
        torch.testing.assert_close(kl, torch.zeros_like(kl))
        gradient, = torch.autograd.grad(loss, logits)
        self.assertEqual(int(torch.count_nonzero(gradient)), 0)

    def test_native_key_group_weighting(self):
        _, _, _, spec = fixture()
        fake_keys = torch.tensor([[10.], [1.], [2.], [3.], [4.], [5.]])
        weighted = (fake_keys * torch.tensor(spec["key_context_weights"][:6])[:, None]).sum(0)
        native = torch.stack([fake_keys[:1].mean(0), fake_keys[1:].mean(0)]).mean(0)
        torch.testing.assert_close(weighted, native)
        self.assertFalse(torch.allclose(weighted, fake_keys.mean(0)))

    def test_missing_teacher_is_nll_only_observation(self):
        _, _, _, spec = fixture()
        torch.manual_seed(211)
        logits = torch.randn(*spec["targets"].shape, 128, requires_grad=True)
        loss, nll, kl = native_loss(logits, spec, None, [True, False])
        teacher = entry_teacher(logits, spec)
        _, measured_nll, _ = native_loss(logits, spec, teacher, [True, False])
        torch.testing.assert_close(nll, measured_nll)
        torch.testing.assert_close(kl, torch.zeros_like(kl))
        torch.testing.assert_close(loss, nll[0])
        gradient, = torch.autograd.grad(loss, logits)
        self.assertEqual(int(torch.count_nonzero(gradient[spec["kl_rows"]])), 0)


if __name__ == "__main__":
    unittest.main()
